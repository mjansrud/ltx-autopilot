#!/usr/bin/env python3
"""
LTX Autopilot — Continuous self-learning video LoRA training pipeline.

Usage:
    python main.py                          # Run with default config.yaml
    python main.py --config my_config.yaml  # Custom config
    python main.py --max-batches 5          # Run 5 batches then stop
    python main.py --batch-once             # Run a single batch
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows to avoid cp1252 encoding errors
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pipeline.orchestrator import PipelineOrchestrator


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    # Log to both terminal and file
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = Path("workspace/pipeline.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    # Quiet noisy libraries
    for name in ["urllib3", "httpx", "httpcore", "filelock", "transformers.configuration_utils",
                  "numba", "numba.core", "numba.core.ssa", "numba.core.byteflow", "numba.core.interpreter"]:
        logging.getLogger(name).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="LTX Autopilot — Continuous LoRA Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after N batches (default: run forever)")
    parser.add_argument("--batch-once", action="store_true", help="Run a single batch then exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("autopilot")

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    max_batches = 1 if args.batch_once else args.max_batches

    # Save PID for clean shutdown (stop.sh)
    pid_file = Path("workspace/.pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    try:
        orchestrator = PipelineOrchestrator(config_path)
        orchestrator.run(max_batches=max_batches)
    finally:
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
