"""
Pipeline orchestrator — runs the full continuous loop:

    START LUSTPRESS → CRAWL → SPLIT → CAPTION (load→unload) → PREPROCESS → TRAIN → EVALUATE → CLEANUP → REPEAT

Models are loaded and unloaded between stages so captioning and training
can share the same GPU. Lustpress server is started automatically and
kept running across batches.
"""

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import yaml

from .captioner import create_captioner
from .crawler import VideoCrawler, LustpressServer
from .evaluator import Evaluator
from .preprocessor import Preprocessor, SceneSplitter
from .state import PipelineState
from .trainer import Trainer
from .vram import flush_vram, get_vram_usage, vram_stage
from . import dashboard as dash

log = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(self.config_path.read_text())

        self.work_dir = Path("./workspace")
        self.state = PipelineState(self.cfg["state_file"])

        # Lustpress server
        lustpress_dir = self.cfg["crawler"].get("lustpress_dir", "./lustpress")
        lustpress_port = self.cfg["crawler"].get("lustpress_port", 3000)
        self.lustpress = LustpressServer(lustpress_dir, lustpress_port)

        # Initialize components (no models loaded yet)
        self.crawler = VideoCrawler(self.cfg["crawler"], self.lustpress)
        self.scene_splitter = SceneSplitter(self.cfg.get("scene_split", {}), self.cfg["ltx_trainer_dir"])
        self.captioner = create_captioner(self.cfg["captioner"])
        self.preprocessor = Preprocessor(self.cfg["preprocessing"], self.cfg["training"], self.cfg["ltx_trainer_dir"])

        # Pass eval config into trainer so it can inject validation into the training YAML
        training_cfg = dict(self.cfg["training"])
        training_cfg["_eval_config"] = self.cfg.get("evaluation", {})
        self.trainer = Trainer(training_cfg, self.cfg["ltx_trainer_dir"])

        self.evaluator = Evaluator(
            self.cfg.get("evaluation", {}),
            self.cfg["training"],
            self.cfg["ltx_trainer_dir"],
        )

        self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_future: Future | None = None

    def _ensure_work_dir(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _prefetch_batch(self, batch_num: int, work_dir: Path) -> tuple[list[Path], list[Path]]:
        """Download + scene-split for a future batch (CPU only, no GPU)."""
        raw_dir = work_dir / "raw"
        scenes_dir = work_dir / "scenes"
        raw_dir.mkdir(parents=True, exist_ok=True)

        videos = self.crawler.crawl(batch_num, raw_dir)
        if not videos:
            return [], []

        split_dir = self.scene_splitter.split(raw_dir, scenes_dir)
        scene_videos = sorted(split_dir.glob("*.mp4"))
        if not scene_videos:
            scene_videos = videos

        log.info("[PREFETCH] Batch %d ready: %d videos, %d scenes", batch_num, len(videos), len(scene_videos))
        return videos, scene_videos

    def _start_prefetch(self, batch_num: int):
        """Start prefetching next batch in background thread."""
        next_work = Path(f"./workspace_next")
        if next_work.exists():
            shutil.rmtree(next_work, ignore_errors=True)
        self._prefetch_future = self._prefetch_executor.submit(self._prefetch_batch, batch_num, next_work)

    def _collect_prefetch(self) -> tuple[list[Path], list[Path]] | None:
        """Collect prefetched data if available."""
        if self._prefetch_future is None:
            return None
        try:
            result = self._prefetch_future.result(timeout=0)
            self._prefetch_future = None
            return result
        except Exception:
            self._prefetch_future = None
            return None

    def _cleanup(self):
        """Remove temporary batch data, keep checkpoints and evaluations."""
        cleanup_cfg = self.cfg.get("cleanup", {})

        if not cleanup_cfg.get("delete_after_training", True):
            return

        keep_metadata = cleanup_cfg.get("keep_metadata", True)

        if keep_metadata:
            metadata_dir = Path("./metadata_archive")
            metadata_dir.mkdir(exist_ok=True)
            for jsonl in self.work_dir.glob("**/*.jsonl"):
                dest = metadata_dir / f"batch{self.state.batch_num:04d}_{jsonl.name}"
                shutil.copy2(jsonl, dest)

        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

        dash.show_cleanup(keep_metadata)

    def run_batch(self) -> bool:
        """
        Run one complete batch. Returns True if successful.

        Memory lifecycle:
          1. Crawl + Split       — no GPU needed (Lustpress + yt-dlp + ffmpeg)
                                   (prefetched in background during previous batch's training)
          2. Caption             — captioner model LOADED then UNLOADED
          3. Preprocess          — uses LTX VAE/encoder (subprocess, separate process memory)
          4. Train               — training occupies GPU (subprocess)
                                   (prefetch NEXT batch starts here in background)
          5. Evaluate            — inference pipeline loaded then unloaded (subprocess)
          6. Cleanup             — free disk
        """
        self._ensure_work_dir()
        batch = self.state.batch_num
        raw_dir = self.work_dir / "raw"
        scenes_dir = self.work_dir / "scenes"
        precomputed_dir = self.work_dir / "precomputed"
        metadata_file = self.work_dir / "metadata.jsonl"

        # Show batch header
        query_idx = batch % len(self.cfg["crawler"]["search_queries"])
        query = self.cfg["crawler"]["search_queries"][query_idx]
        sources = self.cfg["crawler"].get("sources", ["xvideos", "xnxx"])
        dash.show_batch_header(batch, self.state.total_steps, query, sources)
        dash.show_vram_status("batch start", get_vram_usage())

        # ── 1+2. Crawl + Split (use prefetch if available) ────────
        prefetched = self._collect_prefetch()
        if prefetched and prefetched[0]:
            videos, scene_videos = prefetched
            # Move prefetched data into current workspace
            next_work = Path("./workspace_next")
            if (next_work / "raw").exists():
                shutil.copytree(next_work / "raw", raw_dir, dirs_exist_ok=True)
            if (next_work / "scenes").exists():
                shutil.copytree(next_work / "scenes", scenes_dir, dirs_exist_ok=True)
            shutil.rmtree(next_work, ignore_errors=True)
            # Update paths to point to current workspace
            scene_videos = sorted(scenes_dir.glob("*.mp4")) if scenes_dir.exists() else sorted(raw_dir.glob("*.mp4"))
            log.info("[1-2/6] Using prefetched data: %d videos, %d scenes", len(videos), len(scene_videos))
            dash.show_scene_split(len(videos), len(scene_videos), scenes_dir if scenes_dir.exists() else raw_dir)
        else:
            with vram_stage("crawl", False):
                log.info("[1/6] Crawling videos via Lustpress...")
                videos = self.crawler.crawl(batch, raw_dir)
                if not videos:
                    log.warning("No videos downloaded. Waiting before retry...")
                    return False
                log.info("Downloaded %d videos", len(videos))

            with vram_stage("scene_split", False):
                log.info("[2/6] Splitting %d videos into scenes...", len(videos))
                split_dir = self.scene_splitter.split(raw_dir, scenes_dir)
                scene_videos = sorted(split_dir.glob("*.mp4"))
                if not scene_videos:
                    scene_videos = videos
                dash.show_scene_split(len(videos), len(scene_videos), split_dir)

        # ── 3. Caption (MODEL LOADED → UNLOADED) ──────────────────
        with vram_stage("captioning"):
            log.info("[3/6] Captioning %d clips (loading model → GPU)...", len(scene_videos))
            dash.section("CAPTIONING")
            print(f"  Loading model: {self.cfg['captioner'].get('model_id', 'unknown')}")
            dash.show_vram_status("before caption model load", get_vram_usage())

            # caption_batch handles load() and unload() internally
            self.captioner.caption_batch(scene_videos, metadata_file)

            # Model is now UNLOADED — GPU is free
            flush_vram()
            dash.show_vram_status("after caption model unload", get_vram_usage())

            # Show the generated captions
            dash.show_captions(metadata_file)

        # ── 4. Preprocess (runs as subprocess) ─────────────────────
        with vram_stage("preprocessing"):
            log.info("[4/6] Preprocessing (computing latents)...")
            self.preprocessor.process(metadata_file, precomputed_dir)
            dash.show_preprocessing(precomputed_dir)

        # ── 5. Train (runs as subprocess, GPU occupied) ────────────
        # Start prefetching next batch while GPU is busy training
        next_batch = self.state.batch_num + 1
        log.info("[PREFETCH] Starting download + split for batch %d in background", next_batch)
        self._start_prefetch(next_batch)

        with vram_stage("training"):
            steps = self.cfg["training"]["steps_per_batch"]
            resume_from = self.state.last_checkpoint
            dash.show_training_start(steps, resume_from)

            log.info("[5/6] Training for %d steps...", steps)
            checkpoint = self.trainer.train(precomputed_dir, resume_from)

            new_total = self.state.total_steps + steps
            dash.show_training_complete(checkpoint, new_total)

        # ── 6. Evaluate (MODEL LOADED → UNLOADED) ─────────────────
        eval_cfg = self.cfg.get("evaluation", {})
        eval_every = eval_cfg.get("every_n_steps", 250)
        steps_this_batch = self.cfg["training"]["steps_per_batch"]
        new_total = self.state.total_steps + steps_this_batch

        should_eval = (eval_every > 0) and (
            new_total // eval_every > self.state.total_steps // eval_every
        )

        if should_eval and checkpoint:
            with vram_stage("evaluation"):
                log.info("[6/6] Evaluating LoRA at step %d...", new_total)
                compare_base = self.state.should_compare_base(
                    eval_cfg.get("compare_base_every_n_batches", 5)
                )
                self.evaluator.evaluate(checkpoint, batch, new_total, compare_base)

                eval_dir = self.evaluator.output_dir / f"batch{batch:04d}_step{new_total:06d}_*"
                # Find the actual eval dir (has timestamp)
                eval_dirs = sorted(self.evaluator.output_dir.glob(f"batch{batch:04d}_*"))
                if eval_dirs:
                    dash.show_evaluation(eval_dirs[-1], eval_cfg.get("prompts", []))
        else:
            log.info("[6/6] Skipping evaluation this batch (next at step %d)",
                     ((self.state.total_steps // eval_every) + 1) * eval_every if eval_every > 0 else 0)

        # ── Update state & cleanup ─────────────────────────────────
        videos_count = len(videos)
        captions_count = sum(1 for _ in open(metadata_file)) if metadata_file.exists() else 0
        self.state.advance_batch(steps_this_batch, checkpoint)
        self._cleanup()

        dash.show_batch_summary(batch, self.state.total_steps, checkpoint, videos_count, captions_count)
        return True

    def run(self, max_batches: int | None = None):
        """Run the continuous training loop."""
        log.info("Starting LTX Autopilot continuous training pipeline")
        log.info("Config: %s", self.config_path)

        # Start Lustpress server
        try:
            self.lustpress.start()
        except RuntimeError as e:
            log.error("Failed to start Lustpress: %s", e)
            log.error("Make sure lustpress is built: cd lustpress && npm run build")
            return

        print(f"\n  Lustpress API: {self.lustpress.base_url}")
        print(f"  State: batch={self.state.batch_num}, total_steps={self.state.total_steps}")
        print(f"  Checkpoint: {self.state.last_checkpoint or '(none)'}")

        batches_run = 0
        consecutive_failures = 0
        max_failures = 5

        try:
            while max_batches is None or batches_run < max_batches:
                try:
                    success = self.run_batch()

                    if success:
                        batches_run += 1
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        wait = min(60 * consecutive_failures, 300)
                        log.info("Waiting %ds before next attempt...", wait)
                        time.sleep(wait)

                    if consecutive_failures >= max_failures:
                        log.error("Too many consecutive failures (%d), stopping", max_failures)
                        break

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log.error("Batch failed with error: %s", e, exc_info=True)
                    consecutive_failures += 1
                    self._cleanup()
                    flush_vram()

                    if consecutive_failures >= max_failures:
                        log.error("Too many consecutive failures, stopping")
                        break

                    wait = min(30 * consecutive_failures, 120)
                    log.info("Retrying in %ds...", wait)
                    time.sleep(wait)

        except KeyboardInterrupt:
            log.info("\nInterrupted by user.")

        finally:
            # Stop Lustpress server on exit
            self.lustpress.stop()

        print(f"\nPipeline finished. {self.state.batch_num} batches, {self.state.total_steps} total steps.")
