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

        self.work_dir = Path("./workspace/current")
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

    def _restore_from_cache(self, scenes_dir: Path, metadata_file: Path) -> list[Path] | None:
        """Try to restore scenes + captions from cache. Returns scene clips if found."""
        cache_dir = Path("./workspace/history")
        cache_scenes = cache_dir / "scenes"
        cache_captions = cache_dir / "captions.jsonl"

        if not cache_scenes.exists() or not cache_captions.exists():
            return None

        cached_clips = sorted(cache_scenes.glob("*.mp4"))
        if not cached_clips:
            return None

        # Copy cached scenes into workspace
        scenes_dir.mkdir(parents=True, exist_ok=True)
        for clip in cached_clips:
            dst = scenes_dir / clip.name
            if not dst.exists():
                shutil.copy2(clip, dst)

        # Build metadata with only clips that have captions
        captioned = {}
        for line in cache_captions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                # Key by filename
                name = Path(entry["media_path"]).name
                captioned[name] = entry

        # Write metadata for clips we have captions for
        scene_clips = sorted(scenes_dir.glob("*.mp4"))
        matched = []
        with open(metadata_file, "w", encoding="utf-8") as f:
            for clip in scene_clips:
                if clip.name in captioned:
                    entry = captioned[clip.name]
                    # Update path to be relative to workspace
                    entry["media_path"] = str(clip.relative_to(self.work_dir))
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    matched.append(clip)

        if matched:
            log.info("[CACHE] Restored %d/%d clips with captions from cache", len(matched), len(scene_clips))
            return matched

        return None

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
        next_work = Path(f"./workspace/prefetch")
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
        """Remove precomputed latents but keep scenes + captions in cache."""
        cleanup_cfg = self.cfg.get("cleanup", {})

        if not cleanup_cfg.get("delete_after_training", True):
            return

        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

        # Prune history if it exceeds max size
        max_gb = cleanup_cfg.get("max_history_gb", 20)
        self._prune_history(max_gb)

        dash.show_cleanup(False)

    def _save_to_history(self):
        """Cache scenes + captions to history right after captioning."""
        history = Path("./workspace/history")
        history.mkdir(parents=True, exist_ok=True)

        for subdir in ["scenes"]:
            src = self.work_dir / subdir
            if src.exists():
                dst = history / subdir
                dst.mkdir(exist_ok=True)
                for f in src.glob("*.mp4"):
                    target = dst / f.name
                    if not target.exists():
                        shutil.copy2(f, target)

        meta_src = self.work_dir / "metadata.jsonl"
        if meta_src.exists():
            cache_meta = history / "captions.jsonl"
            with open(cache_meta, "a", encoding="utf-8") as out:
                out.write(meta_src.read_text(encoding="utf-8"))

        log.info("[HISTORY] Cached scenes + captions")

    def _prune_history(self, max_gb: float):
        """Delete oldest clips from history when it exceeds max_gb."""
        history = Path("./workspace/history")
        if not history.exists():
            return

        # Calculate total size
        all_files = sorted(history.rglob("*"), key=lambda f: f.stat().st_mtime if f.is_file() else 0)
        total = sum(f.stat().st_size for f in all_files if f.is_file())
        max_bytes = max_gb * 1024 ** 3

        if total <= max_bytes:
            return

        # Delete oldest files first (mp4s are the big ones)
        deleted = 0
        for f in all_files:
            if not f.is_file() or total <= max_bytes:
                break
            size = f.stat().st_size
            f.unlink()
            total -= size
            deleted += 1

        log.info("[PRUNE] Deleted %d old files from history (now %.1f GB)", deleted, total / 1024**3)

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

        # ── 1+2. Try history cache → prefetch → fresh download ─────
        cached = self._restore_from_cache(scenes_dir, metadata_file)
        prefetched = self._collect_prefetch() if not cached else None

        if cached:
            scene_videos = cached
            videos = cached
            log.info("[1-3/6] Restored %d clips from history (skip download+split+caption)", len(cached))
            dash.show_scene_split(len(cached), len(cached), scenes_dir)
        elif prefetched and prefetched[0]:
            videos, scene_videos = prefetched
            # Move prefetched data into current workspace
            next_work = Path("./workspace/prefetch")
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

        # ── 3. Caption (skip if restored from cache) ─────────────
        if not cached:
            with vram_stage("captioning"):
                log.info("[3/6] Captioning %d clips (loading model → GPU)...", len(scene_videos))
                dash.section("CAPTIONING")
                print(f"  Loading model: {self.cfg['captioner'].get('model_id', 'unknown')}")
                dash.show_vram_status("before caption model load", get_vram_usage())

                self.captioner.caption_batch(scene_videos, metadata_file)

                # Model is now UNLOADED — GPU is free
                flush_vram()
                dash.show_vram_status("after caption model unload", get_vram_usage())

                # Show the generated captions
                dash.show_captions(metadata_file)

            # Cache to history immediately (before preprocessing which may OOM)
            self._save_to_history()

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

        # ── Save state immediately after training (before eval) ───
        steps_this_batch = self.cfg["training"]["steps_per_batch"]
        self.state.advance_batch(steps_this_batch, checkpoint)

        # ── Save I2V reference frames for NEXT batch's eval ────────
        # (using current batch's frames for next batch = unseen data)
        i2v_dir = Path("./evaluations/i2v_refs")
        i2v_dir.mkdir(parents=True, exist_ok=True)
        next_refs_file = i2v_dir / "pending_refs.json"
        try:
            import cv2
            captions_by_path = {}
            if metadata_file.exists():
                for line in metadata_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        entry = json.loads(line)
                        captions_by_path[Path(entry["media_path"]).name] = entry.get("caption", "")

            clips = sorted(scenes_dir.glob("*.mp4")) if scenes_dir.exists() else []
            pending = []
            for clip in clips[:2]:
                cap = cv2.VideoCapture(str(clip))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    img_path = i2v_dir / f"batch{batch:04d}_{clip.stem}.png"
                    cv2.imwrite(str(img_path), frame)
                    caption = captions_by_path.get(clip.name, "")
                    if caption:
                        pending.append({"image": str(img_path.resolve()), "prompt": caption})
            # Save for next batch to pick up
            next_refs_file.write_text(json.dumps(pending, ensure_ascii=False), encoding="utf-8")
            log.info("Saved %d I2V refs for next batch's eval", len(pending))
        except Exception as e:
            log.debug("I2V ref frame save failed: %s", e)

        # ── Load I2V refs from PREVIOUS batch (unseen frames) ─────
        i2v_refs = []
        try:
            prev_refs_file = i2v_dir / "prev_refs.json"
            if prev_refs_file.exists():
                i2v_refs = json.loads(prev_refs_file.read_text(encoding="utf-8"))
                log.info("Loaded %d I2V refs from previous batch", len(i2v_refs))
        except Exception:
            pass

        # ── 6. Evaluate (MODEL LOADED → UNLOADED) ─────────────────
        eval_cfg = self.cfg.get("evaluation", {})
        eval_every = eval_cfg.get("every_n_steps", 250)
        # new_total already computed above; state already advanced
        prev_total = new_total - self.cfg["training"]["steps_per_batch"]

        should_eval = (eval_every > 0) and (
            new_total // eval_every > prev_total // eval_every
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
            # ── I2V eval: use frames from previous batch (unseen) ──
            if i2v_refs and checkpoint:
                log.info("[6b/6] Running I2V evaluation with %d unseen reference frames...", len(i2v_refs))
                self._run_i2v_eval(i2v_refs, checkpoint, batch, new_total)

        else:
            log.info("[6/6] Skipping evaluation this batch (next at step %d)",
                     ((self.state.total_steps // eval_every) + 1) * eval_every if eval_every > 0 else 0)

        # Rotate I2V refs: pending -> prev for next batch
        try:
            pending = i2v_dir / "pending_refs.json"
            prev = i2v_dir / "prev_refs.json"
            if pending.exists():
                shutil.copy2(pending, prev)
                pending.unlink()
        except Exception:
            pass

        # ── Cleanup ────────────────────────────────────────────────
        videos_count = len(videos)
        captions_count = sum(1 for _ in open(metadata_file)) if metadata_file.exists() else 0
        self._cleanup()

        dash.show_batch_summary(batch, self.state.total_steps, checkpoint, videos_count, captions_count)
        return True

    def _run_i2v_eval(self, i2v_refs: list[dict], checkpoint: str, batch: int, total_steps: int):
        """Run I2V inference using saved reference frames + their captions."""
        import sys, os

        script = Path(self.cfg["ltx_trainer_dir"]) / "scripts" / "inference.py"
        if not script.exists():
            log.warning("inference.py not found, skipping I2V eval")
            return

        eval_dir = Path("./workspace/evaluations") / f"i2v_batch{batch:04d}_step{total_steps:06d}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.cfg["training"]["model_checkpoint"]
        text_encoder = self.cfg["training"]["text_encoder"]

        env = os.environ.copy()
        scripts_dir = str(script.parent.resolve())
        env["PYTHONPATH"] = scripts_dir + os.pathsep + env.get("PYTHONPATH", "")

        for i, ref in enumerate(i2v_refs):
            out_path = eval_dir / f"i2v_{i:02d}.mp4"
            cmd = [
                sys.executable, str(script.resolve()),
                "--checkpoint", str(Path(model_path).resolve()),
                "--text-encoder-path", str(Path(text_encoder).resolve()),
                "--lora-path", str(Path(checkpoint).resolve()),
                "--condition-image", ref["image"],
                "--prompt", ref["prompt"],
                "--output", str(out_path),
                "--height", "576", "--width", "576",
                "--num-frames", "49",
                "--guidance-scale", "4.0",
                "--num-inference-steps", "30",
            ]

            log.info("  I2V eval %d: %s -> %s", i, Path(ref["image"]).name, out_path.name)
            result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env)
            if result.returncode != 0:
                log.warning("  I2V eval failed: %s", (result.stderr or "")[-300:])
            else:
                log.info("  I2V eval %d complete: %s", i, out_path)

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
            max_total_steps = self.cfg.get("training", {}).get("max_total_steps")
            while max_batches is None or batches_run < max_batches:
                # Check total step limit
                if max_total_steps and self.state.total_steps >= max_total_steps:
                    log.info("Reached max_total_steps (%d). Stopping.", max_total_steps)
                    break

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
