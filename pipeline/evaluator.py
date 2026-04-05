"""
LoRA evaluator — runs inference with the current LoRA checkpoint to generate
test videos, optionally comparing against the base model (no LoRA).

This runs as a standalone step AFTER training (model unloaded), loading only
the inference pipeline, generating samples, then unloading again.
"""

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .vram import flush_vram, log_vram

log = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: dict, training_config: dict, ltx_trainer_dir: str):
        self.cfg = config
        self.training_cfg = training_config
        self.trainer_dir = Path(ltx_trainer_dir)
        self.output_dir = Path(config.get("output_dir", "./evaluations"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, checkpoint_path: str | None, batch_num: int,
                 total_steps: int, compare_base: bool = False):
        """
        Generate test videos using the current LoRA and optionally the base model.
        Saves videos to evaluations/<batch>_<steps>/.
        """
        if not checkpoint_path:
            log.warning("No checkpoint available for evaluation, skipping")
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        eval_dir = self.output_dir / f"batch{batch_num:04d}_step{total_steps:06d}_{timestamp}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        log_vram("evaluation — start")

        # Generate with LoRA
        lora_dir = eval_dir / "lora"
        self._run_inference(checkpoint_path, lora_dir, tag="LoRA")

        # Optionally generate with base model for comparison
        if compare_base:
            base_dir = eval_dir / "base"
            self._run_inference(None, base_dir, tag="Base (no LoRA)")

        flush_vram()
        log_vram("evaluation — end")

        log.info("Evaluation outputs saved to %s", eval_dir)
        self._write_eval_report(eval_dir, checkpoint_path, batch_num, total_steps, compare_base)

    def _run_inference(self, checkpoint_path: str | None, output_dir: Path, tag: str = ""):
        """Run inference.py from the LTX trainer."""
        output_dir.mkdir(parents=True, exist_ok=True)

        script = self.trainer_dir / "scripts" / "inference.py"

        # If inference.py doesn't exist, fall back to a direct pipeline approach
        if not script.exists():
            log.warning("inference.py not found, using direct pipeline inference")
            self._run_inference_direct(checkpoint_path, output_dir, tag)
            return

        for i, prompt in enumerate(self.cfg.get("prompts", [])):
            cmd = [
                "uv", "run", "python", str(script),
                "--checkpoint", self.training_cfg["model_checkpoint"],
                "--prompt", prompt,
                "--height", str(self.cfg.get("height", 576)),
                "--width", str(self.cfg.get("width", 576)),
                "--num-frames", str(self.cfg.get("num_frames", 49)),
                "--fps", str(self.cfg.get("fps", 25)),
                "--guidance-scale", str(self.cfg.get("guidance_scale", 4.0)),
                "--num-inference-steps", str(self.cfg.get("num_inference_steps", 40)),
                "--output", str(output_dir / f"sample_{i:02d}.mp4"),
            ]

            if checkpoint_path:
                cmd.extend(["--lora-path", checkpoint_path])

            log.info("[%s] Generating sample %d/%d: %.80s...",
                     tag, i + 1, len(self.cfg["prompts"]), prompt)

            result = subprocess.run(cmd, cwd=str(self.trainer_dir), capture_output=True, text=True)
            if result.returncode != 0:
                log.error("Inference failed for prompt %d: %s", i, result.stderr[-500:] if result.stderr else "")

    def _run_inference_direct(self, checkpoint_path: str | None, output_dir: Path, tag: str):
        """Direct inference using the LTX pipeline Python API."""
        try:
            import torch
            from diffusers import LTXPipeline

            log.info("[%s] Loading LTX pipeline for evaluation...", tag)
            pipe = LTXPipeline.from_pretrained(
                self.training_cfg["model_checkpoint"],
                torch_dtype=torch.bfloat16,
            ).to("cuda")

            if checkpoint_path:
                log.info("[%s] Loading LoRA from %s", tag, checkpoint_path)
                pipe.load_lora_weights(checkpoint_path)

            for i, prompt in enumerate(self.cfg.get("prompts", [])):
                log.info("[%s] Generating %d/%d: %.80s...",
                         tag, i + 1, len(self.cfg["prompts"]), prompt)

                result = pipe(
                    prompt=prompt,
                    height=self.cfg.get("height", 576),
                    width=self.cfg.get("width", 576),
                    num_frames=self.cfg.get("num_frames", 49),
                    guidance_scale=self.cfg.get("guidance_scale", 4.0),
                    num_inference_steps=self.cfg.get("num_inference_steps", 40),
                )

                # Save video frames
                out_path = output_dir / f"sample_{i:02d}.mp4"
                self._save_video(result.frames[0], out_path, self.cfg.get("fps", 25))

            # Unload pipeline
            del pipe
            flush_vram()

        except ImportError as e:
            log.error("Cannot run direct inference — missing dependency: %s", e)

    def _save_video(self, frames, output_path: Path, fps: int = 25):
        """Save a list of PIL images or tensors as an MP4."""
        try:
            import cv2
            import numpy as np

            if not frames:
                return

            first = frames[0]
            if hasattr(first, "numpy"):
                frames = [f.numpy() for f in frames]
            elif hasattr(first, "size"):  # PIL
                frames = [np.array(f) for f in frames]

            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            for frame in frames:
                if frame.shape[-1] == 3:  # RGB → BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            writer.release()
            log.info("Saved evaluation video: %s", output_path)

        except Exception as e:
            log.error("Failed to save video %s: %s", output_path, e)

    def _write_eval_report(self, eval_dir: Path, checkpoint: str,
                           batch_num: int, total_steps: int, compared_base: bool):
        """Write a small summary file alongside the generated videos."""
        report = {
            "batch": batch_num,
            "total_steps": total_steps,
            "checkpoint": checkpoint,
            "compared_base": compared_base,
            "prompts": self.cfg.get("prompts", []),
            "settings": {
                "height": self.cfg.get("height"),
                "width": self.cfg.get("width"),
                "num_frames": self.cfg.get("num_frames"),
                "guidance_scale": self.cfg.get("guidance_scale"),
                "num_inference_steps": self.cfg.get("num_inference_steps"),
            },
        }

        import json
        (eval_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
