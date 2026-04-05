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
        self.lora_trigger = config.get("lora_trigger", None)
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

        # Low VRAM optimizations for 32GB GPUs
        cmd.append("--vae-tiling")
        cmd.append("--load-text-encoder-in-8bit")

        if self.lora_trigger:
            cmd.extend(["--lora-trigger", self.lora_trigger])

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
        self.max_duration = config.get("max_scene_duration", 30)
        self.max_scenes = config.get("max_scenes_per_video", None)
        self.detector = config.get("detector", "content")

    def _parse_min_duration(self) -> float:
        """Parse min_duration string like '8s' to seconds."""
        s = self.min_duration.strip().rstrip("s")
        return float(s)

    def split(self, input_dir: Path, output_dir: Path) -> Path:
        """Split videos into scenes. Returns directory of split clips."""
        if not self.enabled:
            log.info("Scene splitting disabled, using raw videos")
            return input_dir

        from concurrent.futures import ThreadPoolExecutor

        output_dir.mkdir(parents=True, exist_ok=True)
        videos = sorted(input_dir.glob("*.mp4"))
        min_dur = self._parse_min_duration()

        with ThreadPoolExecutor(max_workers=len(videos) or 1) as pool:
            futures = {pool.submit(self._split_video, v, output_dir, min_dur): v for v in videos}
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    log.warning("  Scene split failed for %s: %s", futures[future].name, e)

        scenes = list(output_dir.glob("*.mp4"))
        if not scenes:
            log.warning("No scenes produced, falling back to raw videos")
            return input_dir

        log.info("Split %d videos into %d scene clips", len(videos), len(scenes))
        return output_dir

    def _split_video(self, video: Path, output_dir: Path, min_dur: float):
        """Detect scenes, pick evenly spaced ones, cut only those with ffmpeg."""
        from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector

        log.info("Splitting scenes: %s", video.name)

        # 1. Detect scene boundaries (fast — just reads frames, no encoding)
        sv = open_video(str(video))
        sm = SceneManager()
        if self.detector == "adaptive":
            sm.add_detector(AdaptiveDetector())
        else:
            sm.add_detector(ContentDetector())
        sm.detect_scenes(sv)
        all_scenes = sm.get_scene_list()

        # 2. Filter by minimum duration
        max_dur = self.max_duration
        scenes = [(s, e) for s, e in all_scenes
                  if min_dur <= (e - s).get_seconds() <= max_dur]
        log.info("  %s: %d scenes detected, %d between %s-%ss", video.name, len(all_scenes), len(scenes), min_dur, max_dur)

        if not scenes:
            return

        # 3. Pick N evenly distributed scenes, skipping first/last (intro/outro)
        if self.max_scenes and len(scenes) > self.max_scenes:
            # Drop first and last scene (usually intro/credits)
            middle = scenes[1:-1] if len(scenes) > 2 else scenes
            if len(middle) >= self.max_scenes:
                n = len(middle)
                indices = [int(i * (n - 1) / (self.max_scenes - 1)) for i in range(self.max_scenes)]
                scenes = [middle[i] for i in indices]
            else:
                scenes = middle[:self.max_scenes]
            log.info("  Picked %d clips from middle of video", len(scenes))

        # 4. Cut only the selected scenes with ffmpeg (fast copy, no re-encode)
        stem = video.stem
        for i, (start, end) in enumerate(scenes):
            out_path = output_dir / f"{stem}-Scene-{i+1:03d}.mp4"
            ss = start.get_seconds()
            dur = (end - start).get_seconds()
            cmd = [
                "ffmpeg", "-y", "-ss", f"{ss:.3f}", "-i", str(video),
                "-t", f"{dur:.3f}", "-c", "copy", "-avoid_negative_ts", "1",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if result.returncode != 0:
                log.warning("  ffmpeg failed for scene %d: %s", i+1, (result.stderr or "")[-200:])

        log.info("  %s -> %d clips cut", video.name, len(scenes))
