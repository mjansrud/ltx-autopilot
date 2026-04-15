"""
Pipeline orchestrator — runs the full continuous loop:

    START LUSTPRESS → CRAWL → SPLIT → CAPTION (load→unload) → PREPROCESS → TRAIN → EVALUATE → CLEANUP → REPEAT

Models are loaded and unloaded between stages so captioning and training
can share the same GPU. Lustpress server is started automatically and
kept running across batches.
"""

import json
import logging
import os
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
from .trainer import Trainer, SlowTrainingError
from .vram import flush_vram, get_vram_usage, vram_stage
from . import dashboard as dash

log = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Dicts merge, everything else replaces."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_config(path: Path) -> dict:
    """Load a YAML config, resolving `extends: <relative-path>` inheritance chains."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text()) or {}
    extends = cfg.pop("extends", None)
    if extends:
        parent_path = (path.parent / extends).resolve()
        parent = _load_config(parent_path)
        cfg = _deep_merge(parent, cfg)
    return cfg


class PipelineOrchestrator:
    def __init__(self, config_path: str | Path, start_fresh: bool = False):
        self.config_path = Path(config_path)
        self.cfg = _load_config(self.config_path)

        # If set, wipe the upcoming batch dir on the first run_batch() so we
        # discard cached clips/captions and force a fresh crawl. Only applies once.
        self._start_fresh_pending = start_fresh

        # Clear any stale pre-generated LLM artifacts from a previous run — they were
        # made under the previous session's focus and would leak into this one.
        for stale in (Path("./workspace") / "next_query.txt",
                      Path("./workspace") / "next_eval_prompts.json"):
            if stale.exists():
                log.info("[SESSION] Clearing stale %s from previous run", stale.name)
                stale.unlink()

        # Sweep orphaned GPU-holding processes from a previous crashed run. If
        # training BSOD'd or the user hard-killed the pipeline, ComfyUI (or its
        # python child) can linger holding ~20GB of VRAM. A fresh captioner load
        # on top of that causes fragmentation / TDRs / cascading driver crashes.
        self._sweep_orphan_processes()

        self.work_dir = None  # set per-batch in run_batch
        self.state = PipelineState(self.cfg["state_file"])

        # Lustpress server
        lustpress_dir = self.cfg["crawler"].get("lustpress_dir", "./lustpress")
        lustpress_port = self.cfg["crawler"].get("lustpress_port", 3000)
        self.lustpress = LustpressServer(lustpress_dir, lustpress_port)

        # Initialize components (no models loaded yet)
        self.captioner = create_captioner(self.cfg["captioner"])
        self.crawler = VideoCrawler(
            self.cfg["crawler"],
            self.lustpress,
            captioner=self.captioner,
            prompts=self.cfg.get("prompts", {}),
        )
        self.scene_splitter = SceneSplitter(
            self.cfg.get("scene_split", {}),
            self.cfg["ltx_trainer_dir"],
            preprocessing_config=self.cfg.get("preprocessing", {}),
        )
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

    def _assert_vram_near_zero(self, label: str, threshold_mb: int = 500):
        """Warn loudly if main-process VRAM is still high after a cleanup point.

        After captioner.unload() + flush_vram(), the main process's allocated
        VRAM should be near zero (typical observed: ~9 MB). If it's above the
        threshold, something leaked — fragmentation or an orphan tensor ref.
        This is diagnostic only; it does not block the pipeline.
        """
        info = get_vram_usage().get(0, {})
        alloc_mb = info.get("allocated_mb", 0)
        reserved_mb = info.get("reserved_mb", 0)
        if alloc_mb > threshold_mb:
            log.warning("[VRAM-LEAK] %s: %.0f MB allocated / %.0f MB reserved (expected < %d MB)",
                        label, alloc_mb, reserved_mb, threshold_mb)
        elif reserved_mb > threshold_mb * 4:
            log.warning("[VRAM-LEAK] %s: %.0f MB reserved (fragmentation?)", label, reserved_mb)

    def _sweep_orphan_processes(self):
        """Kill leftover ComfyUI processes from a previous crashed run.

        STRICTLY name-based — only targets the exact ComfyUI image names we
        spawn ourselves. Never kills by PID, never uses /T (which would kill
        child trees and can cascade into system processes on Windows). Safe
        to call before anything is loaded.

        Previous versions queried nvidia-smi for GPU-holding processes and
        killed arbitrary PIDs — that killed things like dwm.exe and browser
        GPU helpers and wrecked the desktop. Don't do that.
        """
        import subprocess
        killed = []
        # ONLY exact names we know we own. No /T flag. No PID-based kill.
        for image in ("ComfyUI.exe", "ComfyUI-python.exe"):
            try:
                r = subprocess.run(
                    ["taskkill", "/F", "/IM", image],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    killed.append(image)
            except Exception as e:
                log.debug("[SWEEP] taskkill %s failed: %s", image, e)

        # Diagnostic only — log what nvidia-smi sees on GPU 0, but NEVER kill
        # anything based on this list. Killing arbitrary GPU processes on
        # Windows can take down the desktop.
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                current_pid = os.getpid()
                for line in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        pid_s, name, mem = parts[0], parts[1], parts[2]
                        try:
                            pid = int(pid_s)
                        except ValueError:
                            continue
                        if pid == current_pid:
                            continue
                        log.info("[SWEEP] GPU-holding process present (not killing): "
                                 "pid=%d name=%s vram=%sMB", pid, name, mem)
        except FileNotFoundError:
            pass
        except Exception as e:
            log.debug("[SWEEP] nvidia-smi query failed: %s", e)

        if killed:
            log.info("[SWEEP] Killed orphan ComfyUI processes: %s", ", ".join(killed))

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
        """Download + scene-split for a future batch (CPU only, no GPU).

        Runs in a background ThreadPoolExecutor. Never touches the captioner:
        if `next_query.txt` isn't pre-populated by the main thread, the crawl
        refuses the query-gen path and returns empty. (Main thread calling
        into the captioner from a non-main thread races with WDDM.)
        """
        raw_dir = work_dir / "raw"
        scenes_dir = work_dir / "scenes"
        raw_dir.mkdir(parents=True, exist_ok=True)

        videos = self.crawler.crawl(batch_num, raw_dir, allow_query_gen=False)
        if not videos:
            return [], []

        split_dir = self.scene_splitter.split(raw_dir, scenes_dir)
        scene_videos = sorted(split_dir.glob("*.mp4"))
        if not scene_videos:
            scene_videos = videos

        log.info("[PREFETCH] Batch %d ready: %d videos, %d scenes", batch_num, len(videos), len(scene_videos))
        return videos, scene_videos

    # ── LLM-generated eval prompts (piggybacks on already-loaded captioner) ─────
    def _will_eval_this_batch(self) -> bool:
        """Predict whether this batch's training will cross an evaluation boundary."""
        eval_every = self.cfg.get("evaluation", {}).get("every_n_steps", 0)
        if eval_every <= 0:
            return False
        current = self.state.total_steps
        after = current + self.cfg["training"]["steps_per_batch"]
        return (after // eval_every) > (current // eval_every)

    def _generate_eval_prompts(self):
        """Generate eval scene prompts using the already-loaded captioner.

        Caller is responsible for ensuring the captioner is loaded. Writes the
        prompts to `./workspace/next_eval_prompts.json`, overwriting any prior file.
        """
        eval_cfg = self.cfg.get("evaluation", {})
        count = int(eval_cfg.get("num_prompts", 2))
        focus = (eval_cfg.get("focus") or self.cfg["crawler"].get("focus", "")).strip()
        log.info("[EVAL-GEN] Generating %d eval prompts from session focus", count)
        prompts = self.crawler.generate_eval_prompts(focus, count)
        out = Path("./workspace") / "next_eval_prompts.json"
        out.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("[EVAL-GEN] Wrote %d eval prompts to %s", len(prompts), out)

    def _consume_eval_prompts(self) -> list[str]:
        """Read and delete the pre-generated eval prompts file. Raises if missing."""
        f = Path("./workspace") / "next_eval_prompts.json"
        if not f.exists():
            raise RuntimeError(
                "[EVAL-GEN] next_eval_prompts.json missing at eval time — "
                "prompts are always LLM-generated. Check earlier captioning logs for errors."
            )
        prompts = json.loads(f.read_text(encoding="utf-8"))
        f.unlink()
        log.info("[EVAL-GEN] Consumed %d pre-generated eval prompts", len(prompts))
        return prompts

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

    def _ensure_next_batch_ready(self, next_batch_num: int) -> bool:
        """Guarantee batch N+1 has scenes on disk.

        Waits for the background prefetch future; if it failed or produced
        nothing, synchronously crawls + splits now (main thread, captioner
        still loaded so the crawler's query gen can use it). Returns True if
        scenes are available after this call.
        """
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
        next_raw = next_batch_dir / "raw"

        existing_scenes = sorted(next_scenes.glob("*.mp4")) if next_scenes.exists() else []
        if existing_scenes:
            return True

        # Prefetch thread produced nothing (common when the prefetch-time
        # captioner race condition makes query gen fail). Do a synchronous
        # crawl + split right here — we're inside the captioning block, so
        # the captioner is loaded and the crawler's query gen will succeed.
        log.info("[PREFETCH-SYNC] Prefetch gave no scenes for batch %d — crawling synchronously",
                 next_batch_num)
        next_raw.mkdir(parents=True, exist_ok=True)
        try:
            videos = self.crawler.crawl(next_batch_num, next_raw)
        except Exception as e:
            log.warning("[PREFETCH-SYNC] Crawl failed for batch %d: %s", next_batch_num, e)
            return False
        if not videos:
            log.warning("[PREFETCH-SYNC] No videos downloaded for batch %d", next_batch_num)
            return False

        try:
            split_dir = self.scene_splitter.split(next_raw, next_scenes)
        except Exception as e:
            log.warning("[PREFETCH-SYNC] Scene split failed for batch %d: %s", next_batch_num, e)
            return False

        scene_clips = sorted(split_dir.glob("*.mp4")) if split_dir.exists() else []
        if not scene_clips:
            log.warning("[PREFETCH-SYNC] No scenes extracted for batch %d", next_batch_num)
            return False

        log.info("[PREFETCH-SYNC] Prepared %d scenes for batch %d", len(scene_clips), next_batch_num)
        return True

    def _caption_prefetched_batch(self, next_batch_num: int):
        """Caption the prefetched next batch's scenes while Omni is still loaded.

        Ensures batch N+1 scenes exist first (synchronous crawl fallback if the
        background prefetch didn't deliver), then captions whichever clips end
        up on disk. i2v ref generation depends on the resulting metadata.
        """
        if not self._ensure_next_batch_ready(next_batch_num):
            log.info("[PREFETCH-CAPTION] No scenes available for batch %d", next_batch_num)
            return

        next_batch_dir = Path(f"./workspace/batch-{next_batch_num:04d}")
        next_scenes = next_batch_dir / "scenes"
        next_meta = next_batch_dir / "metadata.jsonl"

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

        Falls back to the most recent previous batch's i2v refs if the next
        batch isn't ready (prefetch failed, first batch, etc.) — avoids
        eval batches where i2v is silently skipped.
        """
        import cv2, random

        i2v_dir = self.work_dir / "i2v"
        meta_lines: list[str] = []

        next_batch_dir = Path(f"./workspace/batch-{batch_num + 1:04d}")
        metadata = next_batch_dir / "metadata.jsonl"

        if metadata.exists() and metadata.stat().st_size > 0:
            entries = []
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))

            if len(entries) >= 2:
                # Shuffle and pick 2 — all entries already passed Omni SKIP filter
                random.shuffle(entries)
                i2v_dir.mkdir(parents=True, exist_ok=True)

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
                    meta_lines.append(json.dumps(
                        {"image": str(img_path.resolve()), "prompt": entry["caption"]},
                        ensure_ascii=False,
                    ))
                    log.info("[I2V] Ref: %s (%d chars)", clip_path.name, len(entry["caption"]))
            else:
                log.info("[I2V] Not enough captioned clips in batch %d (%d)",
                         batch_num + 1, len(entries))
        else:
            log.info("[I2V] No captions yet for batch %d — will try fallback", batch_num + 1)

        # Fallback: reuse the most recent previous batch's i2v refs.
        if not meta_lines:
            prev_refs = self._find_latest_i2v_refs()
            if prev_refs is not None:
                src_dir, src_lines = prev_refs
                i2v_dir.mkdir(parents=True, exist_ok=True)
                for line in src_lines:
                    entry = json.loads(line)
                    src_img = Path(entry["image"])
                    if not src_img.exists():
                        log.debug("[I2V] Fallback image missing: %s", src_img)
                        continue
                    dst_img = i2v_dir / src_img.name
                    if not dst_img.exists():
                        shutil.copy2(src_img, dst_img)
                    entry["image"] = str(dst_img.resolve())
                    meta_lines.append(json.dumps(entry, ensure_ascii=False))
                if meta_lines:
                    log.info("[I2V] Reused %d refs from %s as fallback",
                             len(meta_lines), src_dir.name)
            else:
                log.info("[I2V] No previous i2v refs found — eval will skip i2v")

        if meta_lines:
            meta_file = i2v_dir / "metadata.jsonl"
            meta_file.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
            log.info("[I2V] Saved %d refs to %s", len(meta_lines), i2v_dir)

    def _find_latest_i2v_refs(self) -> tuple[Path, list[str]] | None:
        """Scan workspace for the most recent batch dir with a valid i2v/metadata.jsonl.

        Returns (batch_dir, non-empty metadata lines) or None if nothing found.
        Skips the current batch's own dir.
        """
        workspace = Path("./workspace")
        if not workspace.exists():
            return None
        batch_dirs = sorted(workspace.glob("batch-*"), reverse=True)
        for bd in batch_dirs:
            if self.work_dir is not None and bd.resolve() == self.work_dir.resolve():
                continue
            meta = bd / "i2v" / "metadata.jsonl"
            if not meta.exists() or meta.stat().st_size == 0:
                continue
            lines = [ln for ln in meta.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                return bd, lines
        return None

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

        Memory lifecycle (fresh-download path):
          1. Captioner LOAD       — covers query gen + eval prompts + LLM rank
                                    + crawl + split + captioning (single load)
          2. Crawl + Split        — CPU only (yt-dlp + ffmpeg); captioner idle in VRAM
          3. Caption              — same loaded captioner captions clips, then UNLOADS
          4. Preprocess           — uses LTX VAE/encoder (subprocess, separate memory)
          5. Train                — training occupies GPU (subprocess)
          6. Evaluate             — inference pipeline loaded then unloaded (subprocess)
          7. Cleanup              — free disk
        Cached-recovery path skips 1-3 and uses a smaller auxiliary load only if
        the LLM is needed for query/eval-prompt gen.
        """
        try:
            return self._run_batch_inner()
        finally:
            # Safety net: if anything raised between captioner.load() and the
            # captioning block's terminal unload, ensure the model isn't left
            # sitting in VRAM where it would conflict with training subprocesses.
            if self.captioner.is_loaded():
                log.warning("[SAFETY] Captioner still loaded at end of run_batch — unloading")
                self.captioner.unload()
                flush_vram()

    def _run_batch_inner(self) -> bool:
        batch = self.state.batch_num
        step = self.state.total_steps
        # Update root logger prefix so all sub-component logs include batch/step
        for handler in logging.getLogger().handlers:
            handler.setFormatter(logging.Formatter(
                f"%(asctime)s [%(levelname)s] [B{batch}/S{step}] %(name)s: %(message)s",
                datefmt="%H:%M:%S"))

        # Predict whether this batch's training will trigger an evaluation —
        # drives LLM eval-prompt generation during the captioner-loaded window.
        will_eval = self._will_eval_this_batch()
        eval_prompts_file = Path("./workspace") / "next_eval_prompts.json"
        self.work_dir = Path(f"./workspace/batch-{batch:04d}")

        # One-shot --fresh: wipe any cached clips/captions for this batch before starting
        if self._start_fresh_pending:
            if self.work_dir.exists():
                log.info("[FRESH] Wiping batch dir %s — will crawl fresh", self.work_dir)
                shutil.rmtree(self.work_dir, ignore_errors=True)
            self._start_fresh_pending = False

        self._ensure_work_dir()
        raw_dir = self.work_dir / "raw"
        scenes_dir = self.work_dir / "scenes"
        precomputed_dir = self.work_dir / "precomputed"
        metadata_file = self.work_dir / "metadata.jsonl"
        i2v_refs = None
        checkpoint = None

        # Show batch header — use pre-generated LLM query if available, else focus
        query_file = Path("./workspace") / "next_query.txt"
        if query_file.exists():
            display_query = query_file.read_text().strip()
        else:
            focus = self.cfg["crawler"].get("focus", "(no focus set)")
            display_query = f"[LLM will generate for: {focus[:80]}]"
        sources = self.cfg["crawler"].get("sources", ["xvideos", "xnxx"])
        dash.show_batch_header(batch, self.state.total_steps, display_query, sources)
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
            # Load captioner ONCE for the whole fresh-download flow: query gen,
            # eval prompts, LLM candidate ranking, AND captioning (later in
            # run_batch). It stays loaded across the CPU-only crawl + split phases
            # (~7 GB idle in VRAM, no GPU conflict since yt-dlp/ffmpeg are CPU).
            # The captioning block's terminal unload() is the single cleanup point;
            # run_batch's finally block is a safety net on exceptions.
            log.info("[LLM] Loading captioner for fresh-download (covers gen + rank + captioning)")
            self.captioner.load()

            query_file = Path("./workspace") / "next_query.txt"
            if not query_file.exists():
                self.crawler.generate_next_query(batch)
            if will_eval and not eval_prompts_file.exists():
                self._generate_eval_prompts()

            # Search + LLM-rank candidates while the captioner is loaded.
            # Loop: if the LLM finds fewer than top_n good matches in the pool,
            # generate a fresh query and search again. Up to 3 attempts total
            # so one off-focus query draw doesn't waste the batch.
            need_good = self.crawler.max_per_batch
            ranked_candidates: list[dict] = []
            best_good = -1
            for search_attempt in range(3):
                # After the first attempt, regenerate the query so we're
                # searching a different slice of the content space.
                if search_attempt > 0:
                    log.info("[SEARCH] Retrying with a fresh query (attempt %d/3)", search_attempt + 1)
                    self.crawler.generate_next_query(batch)

                pool = self.crawler.search_candidates(
                    batch, limit=self.crawler.max_per_batch * 4
                )
                if not pool:
                    log.warning("[SEARCH] No candidates for attempt %d", search_attempt + 1)
                    continue

                ranked, good_count = self.crawler.llm_rank_candidates(
                    pool, top_n=self.crawler.max_per_batch, consider=10
                )
                # Keep the best attempt (most good matches) across iterations.
                if good_count > best_good:
                    ranked_candidates = ranked
                    best_good = good_count
                if good_count >= need_good:
                    log.info("[SEARCH] Attempt %d: %d/%d good matches — accepting",
                             search_attempt + 1, good_count, need_good)
                    break
                log.info("[SEARCH] Attempt %d: only %d/%d good matches — retrying",
                         search_attempt + 1, good_count, need_good)

            if not ranked_candidates:
                log.warning("No candidates after 3 search attempts. Waiting before retry...")
                return False

            with vram_stage("crawl", False):
                log.info("[1/6] Downloading %d LLM-ranked candidates...", len(ranked_candidates))
                videos = self.crawler.download_candidates(ranked_candidates, raw_dir)
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

        # Cap total clips per batch
        max_clips = self.cfg.get("scene_split", {}).get("max_clips_per_batch", 0)
        if max_clips and len(scene_videos) > max_clips:
            import random
            random.shuffle(scene_videos)
            scene_videos = sorted(scene_videos[:max_clips])
            log.info("Capped to %d clips (from %d)", max_clips, len(scene_videos))

        # ── Cached-path auxiliary LLM gen ─────────────────────────
        # In the cached crash-recovery path, captioning is skipped, so the
        # captioner never loads. If we still need eval prompts (will_eval) or
        # a query file for the upcoming prefetch, load once here and handle both.
        if cached:
            need_query_for_prefetch = not (Path("./workspace") / "next_query.txt").exists()
            need_eval_prompts = will_eval and not eval_prompts_file.exists()
            if need_query_for_prefetch or need_eval_prompts:
                log.info("[LLM] Loading captioner for cached-path aux gen (query=%s, eval_prompts=%s)",
                         need_query_for_prefetch, need_eval_prompts)
                self.captioner.load()
                try:
                    if need_query_for_prefetch:
                        self.crawler.generate_next_query(batch + 1)
                    if need_eval_prompts:
                        self._generate_eval_prompts()
                finally:
                    self.captioner.unload()
                    flush_vram()

        # ── Pre-populate next_query.txt for the upcoming prefetch ─────
        # The prefetch thread runs with allow_query_gen=False (it can't touch
        # the main-thread captioner context without racing WDDM). So we must
        # write the query here, while we're still on the main thread with the
        # captioner loaded (fresh-download path). Cached path already handled
        # this above. If the captioner isn't loaded (prefetched-path batches),
        # we skip — prefetch-for-next-batch will then bail cleanly and the
        # next batch falls through to its own fresh-download.
        next_query_file = Path("./workspace") / "next_query.txt"
        if not next_query_file.exists() and self.captioner.is_loaded():
            self.crawler.generate_next_query(self.state.batch_num + 1)

        # ── Start prefetch early so scenes are ready by end of captioning ──
        next_batch = self.state.batch_num + 1
        log.info("[PREFETCH] Starting download + split for batch %d in background", next_batch)
        self._start_prefetch(next_batch)

        # ── 3. Caption (skip if restored from cache) ─────────────
        if not cached:
            with vram_stage("captioning"):
                log.info("[3/6] Captioning %d clips (captioner already loaded)...", len(scene_videos))
                dash.section("CAPTIONING")
                dash.show_vram_status("before captioning", get_vram_usage())

                self.captioner.caption_batch(scene_videos, metadata_file)

                # Caption prefetched next batch while Omni is still loaded
                self._caption_prefetched_batch(batch + 1)

                # Generate i2v refs from next batch (unseen, already captioned)
                self._generate_i2v_refs(batch)

                # Generate search query for next crawl while model is still loaded
                self.crawler.generate_next_query(batch + 1)

                # Piggyback eval-prompt generation on the same captioner load
                if will_eval and not eval_prompts_file.exists():
                    self._generate_eval_prompts()

                # Now unload captioner
                self.captioner.unload()
                flush_vram()
                dash.show_vram_status("after caption model unload", get_vram_usage())
                self._assert_vram_near_zero("after captioner unload")

                # Show the generated captions
                dash.show_captions(metadata_file)

            # Cache to history immediately (before preprocessing which may OOM)
            pass  # data already in batch dir

        # ── 4. Preprocess (runs as subprocess) ─────────────────────
        with vram_stage("preprocessing"):
            log.info("[4/6] Preprocessing (computing latents)...")
            self.preprocessor.process(metadata_file, precomputed_dir, batch_num=batch)
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
            try:
                checkpoint = self.trainer.train(precomputed_dir, resume_from, batch_num=batch, i2v_refs=i2v_refs)
            except SlowTrainingError:
                log.warning("[5/6] Training too slow — wiping batch %d and retrying with new videos", batch)
                shutil.rmtree(self.work_dir, ignore_errors=True)
                return False

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
                    eval_prompts = self._consume_eval_prompts()
                    if True:  # comfyui_eval handles starting ComfyUI if needed
                        comfyui_eval(
                            checkpoint=ckpt_file,
                            step=new_total,
                            output_dir=self.work_dir / "samples",
                            prompts=eval_prompts,
                            i2v_refs=i2v_refs,
                            width=eval_cfg.get("width", 768),
                            height=eval_cfg.get("height", 448),
                            num_frames=eval_cfg.get("num_frames", 89),
                        )
                    else:
                        log.warning("ComfyUI not running — skipping eval")
                except Exception as e:
                    log.error("Eval failed: %s", e)

            # ComfyUI should be killed by now — verify VRAM is actually free
            # before the next batch starts loading the captioner on top.
            flush_vram()
            self._assert_vram_near_zero("after eval + ComfyUI kill", threshold_mb=1000)

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
            # Shut down the prefetch thread pool so Ctrl+C doesn't hang on a
            # blocking yt-dlp or crawl call in the background worker.
            try:
                self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            # Final safety unload — if anything left the captioner loaded
            # (exception paths), free VRAM so the next launch starts clean.
            try:
                if self.captioner.is_loaded():
                    log.info("[SHUTDOWN] Unloading captioner on exit")
                    self.captioner.unload()
                    flush_vram()
            except Exception:
                pass

        print(f"\nPipeline finished. {self.state.batch_num} batches, {self.state.total_steps} total steps.")
