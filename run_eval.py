"""
Run evaluation (t2v + i2v) on the latest checkpoint.

Usage:
    python run_eval.py                  # auto-detect latest checkpoint
    python run_eval.py --step 6500      # specific step
    python run_eval.py --i2v-only       # only i2v eval
    python run_eval.py --t2v-only       # only t2v eval
"""

from pipeline.eval_runner import run_eval, find_latest_checkpoint
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int)
parser.add_argument("--i2v-only", action="store_true")
parser.add_argument("--t2v-only", action="store_true")
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

workspace = Path("workspace")
ckpt, st, bdir = find_latest_checkpoint(workspace)
if args.step:
    for c in workspace.glob(f"batch-*/checkpoints/lora_weights_step_{args.step:05d}.safetensors"):
        ckpt, st, bdir = c, args.step, c.parent.parent

run_eval(
    config_path=args.config,
    checkpoint=ckpt, step=st, batch_dir=bdir,
    do_t2v=not args.i2v_only,
    do_i2v=not args.t2v_only,
)
