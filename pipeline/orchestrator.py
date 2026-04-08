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
import subprocess
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

        self.work_dir = None  # set per-batch in run_batch
        self.state = PipelineState(self.cfg["state_file"])

        # Lustpress server
        lustpress_dir = self.cfg["crawler"].get("lustpress_dir", "./lustpress")
        lustpress_port = self.cfg["crawler"].get("lustpress_port", 3000)
        self.lustpress = LustpressServer(lustpress_dir, lustpress_port)

        # Initialize components (no models loaded yet)
        self.captioner = create_captioner(self.cfg["captioner"])
        self.crawler = VideoCrawler(self.cfg["crawler"], self.lustpress, captioner=self.captioner)
        self.scene_splitter = SceneSplitter(self.cfg.get("scene_split", {}), self.cfg["ltx_trainer_dir"])
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

    def _restore_from_batch_dir(self, scenes_dir: Path, metadata_file: Path) -> list[Path] | None:
        """Restore scenes + captions from batch dir (crash recovery)."""
        batch = self.state.batch_num
        batch_dir = Path("./workspace") / f"batch-{batch:04d}"
        batch_scenes = batch_dir / "scenes"
        batch_captions = batch_dir / "metadata.jsonl"

        if not batch_scenes.exists() or not batch_captions.exists():
            return None

        cached_clips = sorted(batch_scenes.glob("*.mp4"))
        if not cached_clips:
            return None

        # Copy into workspace
        scenes_dir.mkdir(parents=True, exist_ok=True)
        for clip in cached_clips:
            dst = scenes_dir / clip.name
            if not dst.exists():
                shutil.copy2(clip, dst)

        # Build metadata
        captioned = {}
        for line in batch_captions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                name = Path(entry["media_path"]).name
                captioned[name] = entry

        scene_clips = sorted(scenes_dir.glob("*.mp4"))
        matched = []
        with open(metadata_file, "w", encoding="utf-8") as f:
            for clip in scene_clips:
                if clip.name in captioned:
                    entry = captioned[clip.name]
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
        next_work = Path(f"./workspace/batch-{batch_num:04d}")
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
        """Delete large temporary files, keep scenes + captions + checkpoints."""
        if self.work_dir:
            for d in ["raw", "precomputed"]:
                p = self.work_dir / d
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)

        # Prune old batch data if total exceeds limit
        cleanup_cfg = self.cfg.get("cleanup", {})
        max_gb = cleanup_cfg.get("max_history_gb", 20)
        self._prune_batches(max_gb)

        dash.show_cleanup(False)

    def _caption_prefetched_batch(self, next_batch_num: int):
        """Caption the prefetched next batch's scenes while Omni is still loaded."""
        # Wait for prefetch to complete (download + scene split runs in background thread)
        if self._prefetch_future is not None:
            log.info("[PREFETCH-CAPTION] Waiting for prefetch to complete...")
            try:
                self._prefetch_future.result(timeout=300)  # 5 min max
            except Exception as e:
                log.warning("[PREFETCH-CAPTION] Prefetch failed: %s", e)
            self._prefetch_future = None

        next_batch_dir = Path(f"./workspace/batch-{next_batch_num:04d}")
        next_scenes = next_batch_dir / "scenes"
        next_meta = next_batch_dir / "metadata.jsonl"

        if not next_scenes.exists():
            log.info("[PREFETCH-CAPTION] No prefetched scenes for batch %d", next_batch_num)
            return
        if next_meta.exists() and next_meta.stat().st_size > 0:
            log.info("[PREFETCH-CAPTION] Batch %d already captioned", next_batch_num)
            return

        clips = sorted(next_scenes.glob("*.mp4"))
        if not clips:
            return

        log.info("[PREFETCH-CAPTION] Captioning %d prefetched clips for batch %d", len(clips), next_batch_num)
        self.captioner.caption_batch(clips, next_meta)

    @staticmethod
    def _extract_mid_frame(clip_path):
        """Extract a frame from the middle of a clip (avoids intro/outro cards).

        Returns frame_bgr or None.
        """
        import cv2

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Seek to ~30% into the clip (past any intro card, before outro)
        target = max(1, int(total * 0.30))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _generate_i2v_refs(self, batch_num: int):
        """Generate i2v refs from NEXT batch's captioned clips (unseen data).

        Saves images + metadata.jsonl directly to batch_dir/i2v/.
        Only clips that passed Omni's SKIP filter (have captions) are eligible.
        Extracts a mid-clip frame to avoid title cards / ad screens at start/end.
        """
        import cv2, random

        next_batch_dir = Path(f"./workspace/batch-{batch_num + 1:04d}")
        metadata = next_batch_dir / "metadata.jsonl"
        if not metadata.exists() or metadata.stat().st_size == 0:
            log.info("[I2V] No captions yet, skipping")
            return

        entries = []
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))

        if len(entries) < 2:
            log.info("[I2V] Not enough clips for i2v (%d)", len(entries))
            return

        # Shuffle and pick 2 — all entries already passed Omni SKIP filter
        random.shuffle(entries)
        i2v_dir = self.work_dir / "i2v"
        i2v_dir.mkdir(parents=True, exist_ok=True)
        meta_lines = []

        for entry in entries:
            if len(meta_lines) >= 2:
                break
            clip_path = Path(entry["media_path"])
            if not clip_path.is_absolute():
                clip_path = next_batch_dir / clip_path

            frame = self._extract_mid_frame(clip_path)
            if frame is None:
                log.debug("[I2V] No valid frame from %s", clip_path.name)
                continue

            img_path = i2v_dir / f"{clip_path.stem}.png"
            cv2.imwrite(str(img_path), frame)
            meta_lines.append(json.dumps({"image": str(img_path.resolve()), "prompt": entry["caption"]}, ensure_ascii=False))
            log.info("[I2V] Ref: %s (%d chars)", clip_path.name, len(entry["caption"]))

        if meta_lines:
            meta_file = i2v_dir / "metadata.jsonl"
            meta_file.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
            log.info("[I2V] Saved %d refs to %s", len(meta_lines), i2v_dir)

    def _save_batch_data(self, batch_num: int):
        """Save scenes + captions into batch dir for recovery + archival."""
        batch_data = Path("./workspace") / f"batch-{batch_num:04d}" / "data"
        batch_data.mkdir(parents=True, exist_ok=True)

        for item in ["scenes", "metadata.jsonl"]:
            src = self.work_dir / item
            dst = batch_data / item
            if src.exists() and not dst.exists():
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        log.info("[HISTORY] Cached scenes + captions")

    def _prune_batches(self, max_gb: float):
        """Delete oldest batch data dirs when total exceeds max_gb."""
        batch_dirs = sorted(Path("./workspace").glob("batch-*"))
        if not batch_dirs:
            return

        total = sum(f.stat().st_size for d in batch_dirs for f in d.rglob("*") if f.is_file())
        max_bytes = max_gb * 1024 ** 3

        if total <= max_bytes:
            return

        # Delete oldest batch data dirs first (keep checkpoints)
        for batch_dir in batch_dirs:
            if total <= max_bytes:
                break
            data_dir = batch_dir / "data"
            if data_dir.exists():
                size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
                shutil.rmtree(data_dir, ignore_errors=True)
                total -= size
                log.info("[PRUNE] Deleted data from %s (freed %.1f MB)", batch_dir.name, size / 1024**2)

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
        batch = self.state.batch_num
        self.work_dir = Path(f"./workspace/batch-{batch:04d}")
        self._ensure_work_dir()
        raw_dir = self.work_dir / "raw"
        scenes_dir = self.work_dir / "scenes"
        precomputed_dir = self.work_dir / "precomputed"
        metadata_file = self.work_dir / "metadata.jsonl"
        i2v_refs = None
        checkpoint = None

        # Show batch header
        query_idx = batch % len(self.cfg["crawler"]["search_queries"])
        query = self.cfg["crawler"]["search_queries"][query_idx]
        sources = self.cfg["crawler"].get("sources", ["xvideos", "xnxx"])
        dash.show_batch_header(batch, self.state.total_steps, query, sources)
        dash.show_vram_status("batch start", get_vram_usage())

        # ── 1+2. Try history cache → prefetch → fresh download ─────
        cached = self._restore_from_batch_dir(scenes_dir, metadata_file)
        prefetched = self._collect_prefetch() if not cached else None

        if cached:
            scene_videos = cached
            videos = cached
            log.info("[1-3/6] Restored %d clips from batch dir (crash recovery)", len(cached))
            dash.show_scene_split(len(cached), len(cached), scenes_dir)
        elif prefetched and prefetched[0]:
            # Prefetch already wrote to this batch dir
            videos, scene_videos = prefetched
            scene_videos = sorted(scenes_dir.glob("*.mp4")) if scenes_dir.exists() else sorted(raw_dir.glob("*.mp4"))
            log.info("[1-2/6] Using prefetched data: %d videos, %d scenes", len(videos), len(scene_videos))
            dash.show_scene_split(len(videos), len(scene_videos), scenes_dir if scenes_dir.exists() else raw_dir)
        else:
            # Generate search query using captioner if no pre-generated one exists
            query_file = Path("./workspace") / "next_query.txt"
            if not query_file.exists():
                log.info("[QUERY-GEN] Loading captioner to generate search query...")
                self.captioner.load()
                self.crawler.generate_next_query(batch)
                self.captioner.unload()
                flush_vram()

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

        # ── Start prefetch early so scenes are ready by end of captioning ──
        next_batch = self.state.batch_num + 1
        log.info("[PREFETCH] Starting download + split for batch %d in background", next_batch)
        self._start_prefetch(next_batch)

        # ── 3. Caption (skip if restored from cache) ─────────────
        if not cached:
            with vram_stage("captioning"):
                log.info("[3/6] Captioning %d clips (loading model → GPU)...", len(scene_videos))
                dash.section("CAPTIONING")
                print(f"  Loading model: {self.cfg['captioner'].get('model_id', 'unknown')}")
                dash.show_vram_status("before caption model load", get_vram_usage())

                self.captioner.caption_batch(scene_videos, metadata_file)

                # Caption prefetched next batch while Omni is still loaded
                self._caption_prefetched_batch(batch + 1)

                # Generate i2v refs from next batch (unseen, already captioned)
                self._generate_i2v_refs(batch)

                # Generate search query for next crawl while model is still loaded
                self.crawler.generate_next_query(batch + 1)

                # Now unload captioner
                self.captioner.unload()
                flush_vram()
                dash.show_vram_status("after caption model unload", get_vram_usage())

                # Show the generated captions
                dash.show_captions(metadata_file)

            # Cache to history immediately (before preprocessing which may OOM)
            pass  # data already in batch dir

        # ── 4. Preprocess (runs as subprocess) ─────────────────────
        with vram_stage("preprocessing"):
            log.info("[4/6] Preprocessing (computing latents)...")
            self.preprocessor.process(metadata_file, precomputed_dir)
            dash.show_preprocessing(precomputed_dir)

        # ── 5. Train (runs as subprocess, GPU occupied) ────────────

        # Load i2v refs from i2v/metadata.jsonl (generated during captioning)
        i2v_refs = None
        i2v_meta = self.work_dir / "i2v" / "metadata.jsonl"
        if i2v_meta.exists():
            i2v_refs = []
            for line in i2v_meta.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    i2v_refs.append(json.loads(line))
            log.info("[I2V] Loaded %d refs for i2v evaluation", len(i2v_refs))

        with vram_stage("training"):
            steps = self.cfg["training"]["steps_per_batch"]
            # Use latest checkpoint from disk (more reliable than state after crashes)
            resume_from = self.trainer.find_latest_checkpoint() or self.state.last_checkpoint
            dash.show_training_start(steps, resume_from)

            log.info("[5/6] Training for %d steps...", steps)
            checkpoint = self.trainer.train(precomputed_dir, resume_from, batch_num=batch, i2v_refs=i2v_refs)

            new_total = self.state.total_steps + steps
            dash.show_training_complete(checkpoint, new_total)

        # ── Save state immediately after training (before eval) ───
        steps_this_batch = self.cfg["training"]["steps_per_batch"]
        self.state.advance_batch(steps_this_batch, checkpoint)

        # ── I2V refs from i2v_refs.json (generated during captioning from next batch) ──
        # These are unseen frames from the next batch's clips

        # ── 6. Evaluate (MODEL LOADED → UNLOADED) ─────────────────
        eval_cfg = self.cfg.get("evaluation", {})
        eval_every = eval_cfg.get("every_n_steps", 250)
        # new_total already computed above; state already advanced
        prev_total = new_total - self.cfg["training"]["steps_per_batch"]

        should_eval = (eval_every > 0) and (
            new_total // eval_every > prev_total // eval_every
        )

        if should_eval and checkpoint:
            log.info("[6/6] Evaluating LoRA at step %d...", new_total)
            ckpt_path = Path(checkpoint)
            if ckpt_path.is_dir():
                lora_files = sorted(ckpt_path.glob("lora_weights_*.safetensors"))
                ckpt_file = lora_files[-1] if lora_files else None
            else:
                ckpt_file = ckpt_path

            if ckpt_file:
                try:
                    from .comfyui_eval import run_eval as comfyui_eval
                    if True:  # comfyui_eval handles starting ComfyUI if needed
                        comfyui_eval(
                            checkpoint=ckpt_file,
                            step=new_total,
                            output_dir=self.work_dir / "samples",
                            prompts=eval_cfg.get("prompts", []),
                            i2v_refs=i2v_refs,
                            width=eval_cfg.get("width", 768),
                            height=eval_cfg.get("height", 448),
                            num_frames=eval_cfg.get("num_frames", 89),
                        )
                    else:
                        log.warning("ComfyUI not running — skipping eval")
                except Exception as e:
                    log.error("Eval failed: %s", e)

        else:
            log.info("[6/6] Skipping evaluation this batch (next at step %d)",
                     ((self.state.total_steps // eval_every) + 1) * eval_every if eval_every > 0 else 0)

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

        eval_dir = self.work_dir / f"i2v_step{total_steps:06d}"
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
