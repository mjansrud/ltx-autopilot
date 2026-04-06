"""
Training wrapper — generates a per-batch YAML config, launches LTX training,
and handles checkpoint management.
"""

import logging
import subprocess
from pathlib import Path

import yaml

from .vram import log_vram

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: dict, ltx_trainer_dir: str):
        self.trainer_dir = Path(ltx_trainer_dir)
        self.cfg = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(self) -> str | None:
        """Find the most recent checkpoint directory."""
        candidates = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if candidates:
            latest = str(candidates[-1])
            log.info("Found latest checkpoint: %s", latest)
            return latest
        return None

    def build_config(self, precomputed_dir: Path, batch_config_path: Path,
                     resume_from: str | None = None) -> Path:
        """Write a YAML config matching the LTX trainer's Pydantic schema."""
        eval_cfg = self.cfg.get("_eval_config", {})
        lora_cfg = self.cfg.get("lora", {})

        config = {
            "model": {
                "model_path": str(Path(self.cfg["model_checkpoint"]).resolve()),
                "text_encoder_path": str(Path(self.cfg["text_encoder"]).resolve()),
                "training_mode": self.cfg["mode"],
            },
            "lora": {
                "rank": lora_cfg.get("rank", 32),
                "alpha": lora_cfg.get("alpha", 32),
                "dropout": lora_cfg.get("dropout", 0.0),
                "target_modules": lora_cfg.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"]),
            },
            "training_strategy": {
                "name": "text_to_video",
                "first_frame_conditioning_p": 0.5,
                "with_audio": True,
                "audio_latents_dir": "audio_latents",
            },
            "optimization": {
                "learning_rate": self.cfg["learning_rate"],
                "steps": self.cfg["steps_per_batch"],
                "batch_size": self.cfg.get("batch_size", 1),
                "gradient_accumulation_steps": self.cfg.get("gradient_accumulation_steps", 4),
                "max_grad_norm": self.cfg.get("max_grad_norm", 1.0),
                "optimizer_type": self.cfg.get("optimizer", "adamw"),
                "scheduler_type": self.cfg.get("lr_scheduler", "cosine"),
                "scheduler_params": {},
                "enable_gradient_checkpointing": self.cfg.get("gradient_checkpointing", True),
            },
            "acceleration": {
                "mixed_precision_mode": self.cfg.get("mixed_precision", "bf16"),
                "quantization": "int8-quanto",
                "load_text_encoder_in_8bit": True,
            },
            "data": {
                "preprocessed_data_root": str(Path(precomputed_dir).resolve()),
                "num_dataloader_workers": 0,  # Windows compatibility
            },
            "checkpoints": {
                "interval": self.cfg.get("save_every_n_steps", 250),
                "keep_last_n": -1,
                "precision": "bfloat16",
            },
            "flow_matching": {
                "timestep_sampling_mode": "shifted_logit_normal",
                "timestep_sampling_params": {},
            },
            "seed": 42,
            "output_dir": str(self.output_dir.resolve()),
        }

        if resume_from:
            config["model"]["load_checkpoint"] = str(Path(resume_from).resolve())

        # Validation config
        if eval_cfg.get("prompts"):
            w = eval_cfg.get("width", 576)
            h = eval_cfg.get("height", 576)
            f = eval_cfg.get("num_frames", 49)
            config["validation"] = {
                "prompts": eval_cfg["prompts"],
                "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
                "video_dims": [w, h, f],
                "frame_rate": eval_cfg.get("fps", 25.0),
                "seed": 42,
                "inference_steps": eval_cfg.get("num_inference_steps", 30),
                "interval": eval_cfg.get("every_n_steps", 100),
                "videos_per_prompt": 1,
                "guidance_scale": eval_cfg.get("guidance_scale", 4.0),
                "stg_scale": 1.0,
                "stg_blocks": [29],
                "stg_mode": "stg_av",
                "generate_audio": True,
                "skip_initial_validation": True,
            }

        batch_config_path.parent.mkdir(parents=True, exist_ok=True)
        batch_config_path.write_text(yaml.dump(config, default_flow_style=False))
        log.info("Wrote training config: %s", batch_config_path)
        return batch_config_path

    def train(self, precomputed_dir: Path, resume_from: str | None = None) -> str | None:
        """
        Run training for one batch. Returns path to latest checkpoint after training.
        """
        log_vram("training — start")

        # Auto-resume
        if resume_from is None and self.cfg.get("auto_resume", True):
            resume_from = self.find_latest_checkpoint()

        batch_config = self.output_dir / "_current_batch_config.yaml"
        self.build_config(precomputed_dir, batch_config, resume_from)

        script = self.trainer_dir / "scripts" / "train.py"
        if not script.exists():
            raise FileNotFoundError(f"LTX train.py not found at {script}")

        import sys, os
        # Use the main venv's python (torch 2.11+cu128 for RTX 5090 compatibility)
        ltx_python = sys.executable

        cmd = [
            str(ltx_python), str(script.resolve()),
            str(batch_config.resolve()),
        ]

        log.info("Starting training: %d steps", self.cfg["steps_per_batch"])
        log.info("Command: %s", " ".join(cmd))

        # Scripts dir on PYTHONPATH for sibling imports
        env = os.environ.copy()
        scripts_dir = str(script.parent.resolve())
        env["PYTHONPATH"] = scripts_dir + os.pathsep + env.get("PYTHONPATH", "")

        # Stream training output to log file + console
        train_log = self.output_dir / "training.log"
        log.info("Training log: %s", train_log)

        # Use PYTHONUNBUFFERED to force flushing
        env["PYTHONUNBUFFERED"] = "1"

        with open(train_log, "w", encoding="utf-8") as logf:
            process = subprocess.Popen(
                cmd, cwd=str(self.trainer_dir.resolve()),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
            )
            buf = b""
            while True:
                chunk = process.stdout.read(1024)
                if not chunk and process.poll() is not None:
                    break
                if chunk:
                    try:
                        text = chunk.decode("utf-8", errors="replace")
                    except Exception:
                        text = str(chunk)
                    logf.write(text)
                    logf.flush()
                    # Log lines (split by \n or \r)
                    buf += chunk
                    while b"\n" in buf or b"\r" in buf:
                        sep = b"\n" if b"\n" in buf else b"\r"
                        line_bytes, buf = buf.split(sep, 1)
                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if line and len(line) > 5:
                            log.info("  [train] %s", line[:200])
            process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {process.returncode}")

        checkpoint = self.find_latest_checkpoint()
        log.info("Training complete. Latest checkpoint: %s", checkpoint)
        log_vram("training — end")
        return checkpoint
