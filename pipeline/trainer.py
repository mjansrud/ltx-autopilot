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
        """Find the most recent LoRA checkpoint across all batch dirs."""
        import re
        all_lora_files = []
        # Search all batch-*/checkpoints/ directories
        for ckpt_dir in self.output_dir.glob("batch-*/checkpoints"):
            for f in ckpt_dir.glob("lora_weights_step_*.safetensors"):
                match = re.search(r"(\d+)", f.stem)
                if match:
                    all_lora_files.append((int(match.group(1)), f, ckpt_dir))

        if all_lora_files:
            all_lora_files.sort(key=lambda x: x[0])
            step, latest_file, ckpt_dir = all_lora_files[-1]
            log.info("Found checkpoint: step %d in %s", step, ckpt_dir)
            return str(ckpt_dir)
        return None

    def build_config(self, precomputed_dir: Path, batch_config_path: Path,
                     resume_from: str | None = None, batch_dir: Path | None = None,
                     i2v_refs: list[dict] | None = None) -> Path:
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
                "quantization": "nf4-bnb",
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
            "output_dir": str((batch_dir or self.output_dir).resolve()),
        }

        if resume_from:
            config["model"]["load_checkpoint"] = str(Path(resume_from).resolve())
            # Total steps = current step + steps_per_batch (cumulative)
            import re
            ckpt_path = Path(resume_from)
            lora_files = sorted(ckpt_path.glob("lora_weights_step_*.safetensors")) if ckpt_path.is_dir() else []
            if lora_files:
                match = re.search(r"(\d+)", lora_files[-1].stem)
                if match:
                    current_step = int(match.group(1))
                    new_total = current_step + self.cfg["steps_per_batch"]
                    config["optimization"]["steps"] = new_total
                    log.info("Resuming from step %d, will train to step %d",
                             current_step, new_total)

        # Validation config
        if eval_cfg.get("prompts"):
            w = eval_cfg.get("width", 576)
            h = eval_cfg.get("height", 576)
            f = eval_cfg.get("num_frames", 49)
            prompts = list(eval_cfg["prompts"])
            # Add i2v ref captions as extra t2v prompts (tests if model learned the content)
            if i2v_refs:
                for ref in i2v_refs[:2]:
                    prompts.append(ref["prompt"])
            config["validation"] = {
                "prompts": prompts,
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

    def train(self, precomputed_dir: Path, resume_from: str | None = None, batch_num: int = 0, i2v_refs: list[dict] | None = None) -> str | None:
        """
        Run training for one batch. Returns path to latest checkpoint after training.
        """
        log_vram("training — start")

        # Per-batch output directory
        batch_dir = self.output_dir / f"batch-{batch_num:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Auto-resume from latest checkpoint across all batches
        if resume_from is None and self.cfg.get("auto_resume", True):
            resume_from = self.find_latest_checkpoint()

        batch_config = batch_dir / "_current_batch_config.yaml"
        self.build_config(precomputed_dir, batch_config, resume_from, batch_dir, i2v_refs)

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
        # Ensure ninja is on PATH for quanto int4 CUDA kernel compilation
        venv_scripts = str(Path(sys.executable).parent)
        env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")

        # Training output goes directly to terminal (progress.jsonl tracks steps)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        progress_file = (batch_dir or self.output_dir) / "progress.jsonl"
        log.info("Training progress: %s", progress_file)

        # Tail progress.jsonl in background thread to show steps in terminal
        import threading, json as _json, time as _time
        _stop_tail = threading.Event()
        def _tail_progress():
            seen = 0
            while not _stop_tail.is_set():
                try:
                    lines = progress_file.read_text(encoding="utf-8").splitlines()
                    for line in lines[seen:]:
                        d = _json.loads(line)
                        log.info("[step %d/%d] loss=%.4f lr=%.6f time=%.1fs",
                                 d["step"], d["total_steps"], d["loss"], d["lr"], d["step_time"])
                    seen = len(lines)
                except Exception:
                    pass
                _stop_tail.wait(5)
        tail_thread = threading.Thread(target=_tail_progress, daemon=True)
        tail_thread.start()

        result = subprocess.run(
            cmd, cwd=str(self.trainer_dir.resolve()),
            stdout=None, stderr=None,  # inherit terminal
            env=env,
        )
        _stop_tail.set()
        tail_thread.join(timeout=2)

        if result.returncode != 0:
            # Log last 30 lines of training log for debugging
            try:
                lines = train_log.read_text(encoding="utf-8", errors="replace").splitlines()
                for line in lines[-30:]:
                    log.error("  [train] %s", line.strip()[:200])
            except Exception:
                pass
            raise RuntimeError(f"Training failed with exit code {result.returncode}")

        checkpoint = self.find_latest_checkpoint()
        log.info("Training complete. Latest checkpoint: %s", checkpoint)
        log_vram("training — end")
        return checkpoint
