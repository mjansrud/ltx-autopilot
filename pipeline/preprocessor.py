"""
Dataset preprocessor — wraps the LTX trainer's process_dataset.py script
to precompute latents and text embeddings from captioned videos.
"""

import logging
import subprocess
from pathlib import Path

from .vram import log_vram

log = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, config: dict, training_config: dict, ltx_trainer_dir: str):
        self.trainer_dir = Path(ltx_trainer_dir)
        self.resolution_buckets = config.get("resolution_buckets", ["576x576x49"])
        self.with_audio = config.get("with_audio", True)
        self.model_path = training_config.get("model_checkpoint", "")
        self.text_encoder_path = training_config.get("text_encoder", "")

    def process(self, metadata_file: Path, output_dir: Path) -> Path:
        """
        Run the LTX process_dataset.py to precompute latents.

        Returns the precomputed data directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        log_vram("preprocess — start")

        script = self.trainer_dir / "scripts" / "process_dataset.py"
        if not script.exists():
            raise FileNotFoundError(f"LTX process_dataset.py not found at {script}")

        script_abs = script.resolve()
        metadata_abs = Path(metadata_file).resolve()
        output_abs = Path(output_dir).resolve()

        import sys
        # Use the main venv's python (torch 2.11+cu128 works on RTX 5090;
        # the LTX venv's torch 2.9.1 has a meta-device bug on Blackwell GPUs)
        ltx_python = sys.executable

        cmd = [
            str(ltx_python), str(script_abs),
            str(metadata_abs),  # positional: DATASET_PATH
            "--model-path", str(Path(self.model_path).resolve()),
            "--text-encoder-path", str(Path(self.text_encoder_path).resolve()),
            "--output-dir", str(output_abs),
        ]

        for bucket in self.resolution_buckets:
            cmd.extend(["--resolution-buckets", bucket])

        if self.with_audio:
            cmd.append("--with-audio")

        log.info("Running preprocessing: %s", " ".join(cmd))

        # The LTX scripts import sibling modules (decode_latents, process_captions, etc.)
        # so the scripts directory must be on PYTHONPATH
        import os
        env = os.environ.copy()
        scripts_dir = str(script.parent.resolve())
        env["PYTHONPATH"] = scripts_dir + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            cmd, cwd=str(self.trainer_dir.resolve()),
            capture_output=True, text=True, errors="replace",
            env=env,
        )

        if result.returncode != 0:
            err = (result.stderr or "no stderr")[-2000:]
            log.error("Preprocessing failed:\n%s", err)
            raise RuntimeError(f"process_dataset.py failed with code {result.returncode}")

        log.info("Preprocessing complete: %s", output_dir)
        log_vram("preprocess — end")
        return output_dir


class SceneSplitter:
    """Wraps the LTX split_scenes.py script."""

    def __init__(self, config: dict, ltx_trainer_dir: str):
        self.trainer_dir = Path(ltx_trainer_dir)
        self.enabled = config.get("enabled", True)
        self.min_duration = config.get("min_scene_duration", "3s")
        self.detector = config.get("detector", "content")

    def split(self, input_dir: Path, output_dir: Path) -> Path:
        """Split videos into scenes. Returns directory of split clips."""
        if not self.enabled:
            log.info("Scene splitting disabled, using raw videos")
            return input_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        script = self.trainer_dir / "scripts" / "split_scenes.py"
        if not script.exists():
            log.warning("split_scenes.py not found at %s — skipping", script)
            return input_dir

        import sys
        videos = sorted(input_dir.glob("*.mp4"))
        total_scenes = 0

        for video in videos:
            cmd = [
                sys.executable, str(script.resolve()),
                str(video.resolve()),           # positional: VIDEO_PATH
                str(output_dir.resolve()),       # positional: OUTPUT_DIR
                "--detector", self.detector,
                "--filter-shorter-than", self.min_duration,
            ]

            log.info("Splitting scenes: %s", video.name)
            result = subprocess.run(cmd, cwd=str(self.trainer_dir.resolve()),
                                    capture_output=True, text=True, errors="replace")

            if result.returncode != 0:
                err = (result.stderr or "")[-500:]
                log.warning("  Scene split failed for %s: %s", video.name, err.strip())
            else:
                new_scenes = len(list(output_dir.glob("*.mp4"))) - total_scenes
                total_scenes = len(list(output_dir.glob("*.mp4")))
                log.info("  %s -> %d scene clips", video.name, max(new_scenes, 0))

        scenes = list(output_dir.glob("*.mp4"))
        if not scenes:
            log.warning("No scenes produced, falling back to raw videos")
            return input_dir

        log.info("Split %d videos into %d scene clips", len(videos), len(scenes))
        return output_dir
