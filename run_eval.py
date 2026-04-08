"""
Run evaluation (t2v + i2v) on the latest checkpoint without waiting for training.

Usage:
    python run_eval.py                  # auto-detect latest checkpoint
    python run_eval.py --step 6500      # specific step
    python run_eval.py --i2v-only       # only i2v eval
    python run_eval.py --t2v-only       # only t2v eval
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def find_latest_checkpoint(workspace: Path) -> tuple[Path | None, int]:
    """Find the latest LoRA checkpoint across all batch dirs."""
    import re
    best_step = 0
    best_file = None
    for ckpt in workspace.glob("batch-*/checkpoints/lora_weights_step_*.safetensors"):
        match = re.search(r"(\d+)", ckpt.stem)
        if match:
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_file = ckpt
    return best_file, best_step


def run_t2v_eval(cfg, checkpoint: Path, step: int):
    """Run t2v evaluation using the LTX trainer's built-in validation."""
    from pipeline.evaluator import Evaluator
    evaluator = Evaluator(cfg.get("evaluation", {}), cfg["training"], cfg["ltx_trainer_dir"])
    log.info("Running t2v evaluation at step %d with %s", step, checkpoint)
    evaluator.evaluate(str(checkpoint.parent), 0, step, False)
    log.info("t2v eval complete — check workspace for samples/")


def run_i2v_eval(cfg, checkpoint: Path, step: int):
    """Run i2v evaluation on latest i2v refs."""
    # Find i2v refs from any batch
    i2v_refs = []
    for meta in sorted(Path("workspace").glob("batch-*/i2v/metadata.jsonl"), reverse=True):
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ref = json.loads(line)
                # Verify image exists
                if Path(ref["image"]).exists():
                    i2v_refs.append(ref)
        if i2v_refs:
            log.info("Found %d i2v refs from %s", len(i2v_refs), meta.parent.parent.name)
            break

    if not i2v_refs:
        log.error("No i2v refs found in any batch-*/i2v/metadata.jsonl")
        return

    script = Path(cfg["ltx_trainer_dir"]) / "scripts" / "inference.py"
    if not script.exists():
        log.error("inference.py not found at %s", script)
        return

    eval_dir = Path("workspace") / f"i2v_eval_step{step:06d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg["training"]["model_checkpoint"]
    text_encoder = cfg["training"]["text_encoder"]

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script.parent.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    # Force correct venv (cwd inside LTX-2 would pick up LTX-2/.venv)
    venv_root = str(Path(sys.executable).parent.parent.resolve())
    env["VIRTUAL_ENV"] = venv_root

    for i, ref in enumerate(i2v_refs[:2]):
        out_path = eval_dir / f"i2v_{i:02d}.mp4"
        cmd = [
            sys.executable, str(script.resolve()),
            "--checkpoint", str(Path(model_path).resolve()),
            "--text-encoder-path", str(Path(text_encoder).resolve()),
            "--lora-path", str(checkpoint.resolve()),
            "--condition-image", ref["image"],
            "--prompt", ref["prompt"],
            "--output", str(out_path),
            "--height", "448", "--width", "768",
            "--num-frames", "41",
            "--guidance-scale", "4.0",
            "--num-inference-steps", "30",
        ]

        log.info("I2V eval %d: %s -> %s", i, Path(ref["image"]).name, out_path.name)
        log.info("Prompt: %.100s...", ref["prompt"])
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env)
        if result.returncode != 0:
            log.error("I2V eval failed:\n%s", (result.stderr or "")[-500:])
        else:
            log.info("I2V eval %d complete: %s (%.1f KB)", i, out_path, out_path.stat().st_size / 1024)

    log.info("I2V eval complete — results in %s", eval_dir)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on latest checkpoint")
    parser.add_argument("--step", type=int, help="Specific step to evaluate")
    parser.add_argument("--i2v-only", action="store_true", help="Only run i2v eval")
    parser.add_argument("--t2v-only", action="store_true", help="Only run t2v eval")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    workspace = Path("workspace")

    checkpoint, step = find_latest_checkpoint(workspace)
    if checkpoint is None:
        log.error("No checkpoints found in workspace/")
        return

    if args.step:
        # Find specific step
        for ckpt in workspace.glob(f"batch-*/checkpoints/lora_weights_step_{args.step:05d}.safetensors"):
            checkpoint = ckpt
            step = args.step
            break

    log.info("Using checkpoint: %s (step %d)", checkpoint, step)

    run_both = not args.i2v_only and not args.t2v_only

    if run_both or args.t2v_only:
        run_t2v_eval(cfg, checkpoint, step)

    if run_both or args.i2v_only:
        run_i2v_eval(cfg, checkpoint, step)


if __name__ == "__main__":
    main()
