"""
Video crawler — uses Lustpress as a search engine to find videos,
then downloads them with yt-dlp.

Lustpress provides a unified REST API over multiple video sites.
The pipeline starts the Lustpress server automatically and queries it
for search results, then feeds the source URLs to yt-dlp for download.
"""

import json
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests

log = logging.getLogger(__name__)


class LustpressServer:
    """Manages the Lustpress Express server lifecycle."""

    def __init__(self, lustpress_dir: str, port: int = 3000):
        self.dir = Path(lustpress_dir)
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"

    def _is_port_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("localhost", self.port)) == 0

    def _is_server_alive(self) -> bool:
        """Check if Lustpress actually responds to HTTP requests."""
        try:
            import requests
            r = requests.get(f"http://localhost:{self.port}/xvideos/random", timeout=5)
            return r.status_code in (200, 301, 302, 404)
        except Exception:
            return False

    def start(self):
        """Start the Lustpress server if not already running."""
        if self._is_port_open() and self._is_server_alive():
            log.info("Lustpress already running on port %d", self.port)
            return

        log.info("Starting Lustpress server from %s ...", self.dir)

        env = os.environ.copy()
        env["PORT"] = str(self.port)

        self.process = subprocess.Popen(
            ["node", "build/src/index.js"],
            cwd=str(self.dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Windows needs CREATE_NEW_PROCESS_GROUP for clean shutdown
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

        # Wait for server to be ready
        for attempt in range(30):
            if self._is_port_open():
                log.info("Lustpress server ready on port %d (took %ds)", self.port, attempt)
                return
            time.sleep(1)

        raise RuntimeError(f"Lustpress server failed to start within 30s on port {self.port}")

    def stop(self):
        """Gracefully stop the server."""
        if self.process and self.process.poll() is None:
            log.info("Stopping Lustpress server (pid %d)...", self.process.pid)
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            log.info("Lustpress server stopped")

    def ensure_healthy(self):
        """Check if Lustpress is alive; restart if dead. Call before searching."""
        if self._is_port_open() and self._is_server_alive():
            return
        log.warning("[LUSTPRESS] Server is not responding — restarting")
        self.stop()
        try:
            self.start()
            log.info("[LUSTPRESS] Server restarted successfully")
        except RuntimeError:
            log.error("[LUSTPRESS] Failed to restart server")

    def search(self, source: str, query: str, page: int = 1,
               sort: str = "mr") -> list[dict]:
        """Search a source for videos. Returns list of video result dicts."""
        url = f"{self.base_url}/{source}/search"
        params = {"key": query, "page": page, "sort": sort}

        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("success", False):
                log.warning("Search failed for %s/%s: %s", source, query, data.get("message", "unknown"))
                return []

            results = data.get("data", [])
            log.info("  %s search '%s': %d results", source, query, len(results))
            return results

        except requests.ConnectionError:
            log.error("Lustpress connection refused for %s — server may be dead", source)
            return []
        except requests.RequestException as e:
            log.error("Lustpress request failed: %s", e)
            return []

    def get_video(self, source: str, video_id: str) -> dict | None:
        """Get detailed video info including assets/URLs."""
        url = f"{self.base_url}/{source}/get"
        params = {"id": video_id}

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success"):
                return data
            return None
        except requests.RequestException as e:
            log.error("Failed to get video %s/%s: %s", source, video_id, e)
            return None

    def random_video(self, source: str) -> dict | None:
        """Get a random video from a source."""
        url = f"{self.base_url}/{source}/random"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success"):
                return data
            return None
        except requests.RequestException as e:
            log.error("Random video failed for %s: %s", source, e)
            return None


class VideoCrawler:
    """
    Searches for videos via Lustpress, then downloads with yt-dlp.

    Flow:
      1. Query Lustpress search API across configured sources
      2. Collect video page URLs from results
      3. Filter out already-seen URLs
      4. Download with yt-dlp (handles the actual video extraction)
    """

    def __init__(self, config: dict, server: LustpressServer, captioner=None,
                 prompts: dict | None = None):
        self.server = server
        self._captioner = captioner
        self.focus = config.get("focus", "").strip()
        if not self.focus:
            raise RuntimeError(
                "crawler.focus is required in config — queries are always LLM-generated, "
                "no static fallback. Set 'focus' under 'crawler:' in your config/session."
            )
        # All LLM prompts come from config.yaml (gitignored) — no defaults in
        # code. Required keys: query_gen, eval_prompt, eval_prompt_example, rank.
        self._prompts = prompts or {}
        required = {"query_gen", "eval_prompt", "eval_prompt_example", "rank"}
        missing = required - set(self._prompts.keys())
        if missing:
            raise RuntimeError(
                f"[CONFIG] Missing required prompts in config.prompts: {sorted(missing)}. "
                "All LLM prompts must be defined in config.yaml under the top-level "
                "`prompts:` section."
            )
        self.sources = config.get("sources", ["xvideos", "xnxx", "eporner", "redtube"])
        self.max_per_batch = config.get("max_videos_per_batch", 8)
        self.min_dur = config.get("min_duration_sec", 5)
        self.max_dur = config.get("max_duration_sec", 90)
        self.max_res = config.get("max_resolution", 720)
        self.archive_path = config.get("download_archive", "./state/seen_videos.txt")
        self.use_random = config.get("include_random", True)
        self.custom_urls = config.get("custom_urls", [])
        # Substrings (case-insensitive) that cause a title to be rejected at
        # crawl time. Use this to block overrepresented uploaders / studios.
        self.title_blocklist = [
            s.lower() for s in config.get("title_blocklist", []) if s
        ]
        self._config = config  # keep ref for removing consumed URLs
        Path(self.archive_path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fill_prompt(template: str, **kwargs) -> str:
        """Substitute {key} placeholders in a prompt template via plain replace.

        Uses str.replace so users can freely use literal { and } in their prompt
        templates without worrying about .format() escaping.
        """
        result = template
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    def _generate_query(self, batch_num: int) -> str:
        """Use an LLM to generate a short search query based on session focus.

        Omni 7B tends to return keyword lists rather than compact queries, so we
        truncate aggressively to the first 6 tokens after ASCII cleanup. Retries
        a couple of times if the LLM returns something unusable.
        """
        if self._captioner is None:
            raise RuntimeError(
                "[QUERY-GEN] No captioner available — queries must be LLM-generated. "
                "Ensure the captioner is loaded before calling _generate_query."
            )

        history_file = Path(self.archive_path).parent / "query_history.txt"
        recent: list[str] = []
        if history_file.exists():
            recent = [ln for ln in history_file.read_text().splitlines() if ln.strip()][-20:]

        prompt = self._fill_prompt(self._prompts["query_gen"], focus=self.focus)

        import re as _re
        last_result = ""
        for attempt in range(4):
            # Higher temperature each retry to break out of repetition loops.
            temp = 0.7 + 0.15 * attempt
            # max_new_tokens=30 hard-caps the length so the LLM can't ramble
            # into a descriptive paragraph. 30 tokens ≈ 20-25 words worst case.
            result = self._captioner.generate_text(prompt, max_new_tokens=30, temperature=temp)
            last_result = result
            log.info("[QUERY-GEN] Attempt %d (temp=%.2f) raw LLM output: %r",
                     attempt + 1, temp, result[:200])

            # Minimal cleanup: lowercase, first line, strip quotes, drop non-ascii
            query = result.strip().strip('"').strip("'").lower().split("\n")[0]
            query = _re.sub(r'[^a-z0-9 ]', ' ', query)
            query = ' '.join(query.split())  # collapse whitespace

            if len(query.split()) < 2 or not (5 < len(query) <= 200):
                log.warning("[QUERY-GEN] Attempt %d unusable after cleanup: %r", attempt + 1, query)
                continue

            # The only validation: not an exact duplicate of a recent query.
            # Everything else is trusted to the LLM.
            if query in recent:
                log.warning("[QUERY-GEN] Attempt %d exact duplicate of recent query — retrying")
                continue

            with open(history_file, "a") as f:
                f.write(query + "\n")
            log.info("[QUERY-GEN] Final query: %r", query)
            return query

        raise RuntimeError(
            f"[QUERY-GEN] LLM failed after 3 attempts. Last raw output: {last_result!r}"
        )

    def generate_eval_prompts(self, focus: str, count: int) -> list[str]:
        """Generate N detailed LTX-2 scene prompts for t2v eval from a session focus.

        One LLM call per prompt; each call sees previous outputs so the LLM varies setups.
        Uses a concrete example to anchor the output format (Omni 7B tends toward keyword
        lists without one). Retries each slot a couple of times; raises after 3 failures.
        Captioner must already be loaded.
        """
        if self._captioner is None:
            raise RuntimeError("[EVAL-GEN] No captioner available — eval prompts must be LLM-generated.")
        if not focus or not focus.strip():
            raise RuntimeError("[EVAL-GEN] Empty focus — cannot generate eval prompts.")

        example = self._prompts["eval_prompt_example"]

        prompts: list[str] = []
        for i in range(count):
            prev_block = "\n".join(f"- {p[:120]}..." for p in prompts) if prompts else "(none yet)"
            llm_prompt = self._fill_prompt(
                self._prompts["eval_prompt"],
                focus=focus,
                example=example,
                previous=prev_block,
            )

            last_result = ""
            for attempt in range(3):
                result = self._captioner.generate_text(llm_prompt, max_new_tokens=400, temperature=0.8).strip()
                last_result = result
                # Strip wrappers
                result = result.strip('"').strip("'").strip()
                for prefix in ("Scene:", "Description:", "Prompt:", "Paragraph:", "Output:", "New scene:"):
                    if result.lower().startswith(prefix.lower()):
                        result = result[len(prefix):].strip()
                # Validate: sentence-like prose, not keyword soup
                sentence_count = result.count(".") + result.count("!") + result.count("?")
                word_count = len(result.split())
                if len(result) >= 150 and sentence_count >= 2 and word_count >= 25:
                    if len(result) > 2000:
                        result = result[:2000]
                    prompts.append(result)
                    log.info("[EVAL-GEN] Scene %d/%d ready (attempt %d, %d chars, %d sentences)",
                             i + 1, count, attempt + 1, len(result), sentence_count)
                    break
                log.warning(
                    "[EVAL-GEN] Scene %d attempt %d rejected (len=%d, sentences=%d, words=%d): %r",
                    i + 1, attempt + 1, len(result), sentence_count, word_count, result[:200],
                )
            else:
                raise RuntimeError(
                    f"[EVAL-GEN] Scene {i+1} failed after 3 attempts. Last raw output: {last_result!r}"
                )

        return prompts

    def _load_seen(self) -> set[str]:
        """Load seen URLs from archive file."""
        path = Path(self.archive_path)
        if path.exists():
            return set(line.strip() for line in path.read_text().splitlines() if line.strip())
        return set()

    def _mark_seen(self, url: str):
        with open(self.archive_path, "a") as f:
            f.write(url + "\n")

    def generate_next_query(self, next_batch_num: int):
        """Generate a search query via the LLM and stash it for the upcoming crawl.

        Called from the main thread while the captioner is loaded. The query
        is written to workspace/next_query.txt and consumed by the next call
        to search_candidates / crawl (main thread or prefetch).
        """
        query = self._generate_query(next_batch_num)
        query_file = Path("./workspace") / "next_query.txt"
        query_file.write_text(query)
        log.info("[QUERY-GEN] Wrote query for batch %d: %r", next_batch_num, query)

    def _parse_duration_seconds(self, duration_str: str) -> int | None:
        """Try to parse duration string to seconds."""
        if not duration_str:
            return None
        try:
            # Handle "XXmin, XXsec" format
            import re
            mins = re.search(r"(\d+)\s*min", duration_str)
            secs = re.search(r"(\d+)\s*sec", duration_str)
            total = 0
            if mins:
                total += int(mins.group(1)) * 60
            if secs:
                total += int(secs.group(1))
            if total > 0:
                return total

            # Handle "MM:SS" or "H:MM:SS"
            parts = duration_str.strip().split(":")
            parts = [int(p) for p in parts if p.isdigit()]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]

            # Handle raw seconds
            raw = int(duration_str.strip())
            return raw
        except (ValueError, AttributeError):
            return None

    def search_candidates(self, batch_num: int, limit: int | None = None) -> list[dict]:
        """Search Lustpress and return ranked candidate dicts (soft-scored).

        `limit` caps the returned pool. Defaults to `max_per_batch` (download budget).
        Pass a larger limit when you want a pool for LLM ranking.
        """
        return self._collect_video_urls(batch_num, limit=limit)

    def download_candidates(self, candidates: list[dict], output_dir: Path) -> list[Path]:
        """Public alias: download pre-selected candidates in parallel."""
        if not candidates:
            log.warning("No candidates to download")
            return []
        videos = self._download_videos(candidates, output_dir)
        log.info("Downloaded %d/%d videos to %s", len(videos), len(candidates), output_dir)
        return videos

    def llm_rank_candidates(self, candidates: list[dict], top_n: int,
                            focus: str | None = None, consider: int = 10
                            ) -> tuple[list[dict], int]:
        """Use the loaded captioner LLM to rank candidates by relevance to the focus.

        Only considers the top `consider` candidates from the soft-scored input
        (keeps the LLM task small for a small model). Returns the LLM-ranked top_n
        followed by the remainder in their original order as tail-fill.

        Returns a tuple (ranked_top_n, good_count) where good_count is how many
        of the ranked candidates the LLM considered a good match (score >= 3).
        The orchestrator uses good_count to decide whether to retry with a new
        query when the pool doesn't contain enough matches.

        Falls back to the input order on ANY failure (missing captioner, parse error,
        empty pool). Never raises.
        """
        if not candidates:
            return candidates, 0
        if self._captioner is None or not self._captioner.is_loaded():
            log.debug("[LLM-RANK] Captioner not loaded — using soft-score order")
            return candidates[:top_n], 0

        focus = (focus or self.focus).strip()
        if not focus:
            return candidates[:top_n], 0

        # LLM sees only the head of the soft-scored pool — small models do badly
        # on long ranking tasks, so cap at `consider`.
        head = candidates[:consider]
        tail = candidates[consider:]

        title_block = "\n".join(
            f"{i+1}. {(c.get('title') or '').strip()[:120]}"
            for i, c in enumerate(head)
        )
        prompt = self._fill_prompt(
            self._prompts["rank"],
            focus=focus,
            titles=title_block,
        )

        # We want enough perfect-match titles to actually fill the top_n slots
        # with LLM-endorsed picks, not fall back to search-order tiebreakers.
        # Require at least top_n titles with score 5 (perfect match). Retry up
        # to 3 attempts at escalating temperature if the LLM returns sparse.
        good_threshold = 5
        min_good = top_n
        scores = None
        best_scores = None  # highest-good-count result across attempts
        best_good_count = -1

        for attempt in range(3):
            temp = 0.2 + 0.2 * attempt
            scores = self._try_rank_call(prompt, temperature=temp, head_len=len(head))
            if not scores:
                log.info("[LLM-RANK] Attempt %d returned no parseable scores — retrying", attempt + 1)
                continue
            good_count = sum(1 for v in scores.values() if v >= good_threshold)
            if good_count > best_good_count:
                best_scores = scores
                best_good_count = good_count
            if good_count >= min_good:
                break
            log.info("[LLM-RANK] Attempt %d sparse: %d titles scored >= %d (need %d) — retrying",
                     attempt + 1, good_count, good_threshold, min_good)

        # Always use whichever attempt gave us the most good matches.
        scores = best_scores
        if not scores:
            log.warning("[LLM-RANK] No parseable scores after 3 attempts — falling back to search order")
            return candidates[:top_n], 0

        good_count = sum(1 for v in scores.values() if v >= good_threshold)
        if good_count < min_good:
            log.warning("[LLM-RANK] Only %d/%d titles scored >= %d after retries — "
                        "using sparse ranking anyway (still better than raw search order)",
                        good_count, min_good, good_threshold)

        # Rank head by LLM score desc, stable on original order for ties.
        # Even when scoring is sparse, the high-scored titles get surfaced and
        # zero-scored tail preserves search order — strictly better than passthrough.
        indexed = list(enumerate(head))
        indexed.sort(key=lambda x: scores.get(x[0], 0), reverse=True)
        ranked_head = [c for _, c in indexed]

        self._log_score_table(head, scores, tag=f"ranked ({good_count}/{len(head)} good)")

        combined = ranked_head + tail
        return combined[:top_n], good_count

    def _log_score_table(self, head: list[dict], scores: dict[int, int], tag: str = ""):
        """Log a sorted score table so you can see which titles got which scores."""
        rows = sorted(
            ((scores.get(i, 0), c.get("title", "") or "") for i, c in enumerate(head)),
            key=lambda r: r[0],
            reverse=True,
        )
        header = f"[LLM-RANK] Score table ({tag}):" if tag else "[LLM-RANK] Score table:"
        log.info(header)
        for score, title in rows:
            log.info("  [%d] %s", score, title[:80])

    def _try_rank_call(self, prompt: str, temperature: float, head_len: int) -> dict[int, int] | None:
        """Single LLM call + parse for llm_rank_candidates. Returns None on failure."""
        import re as _re
        try:
            result = self._captioner.generate_text(prompt, max_new_tokens=200, temperature=temperature)
        except Exception as e:
            log.warning("[LLM-RANK] generate_text failed (temp=%.1f): %s", temperature, e)
            return None
        scores: dict[int, int] = {}
        for line in result.splitlines():
            m = _re.search(r"(\d+)\s*[:\-=]\s*(\d+)", line)
            if m:
                idx = int(m.group(1)) - 1
                score = int(m.group(2))
                if 0 <= idx < head_len:
                    scores[idx] = max(0, min(5, score))
        if not scores:
            log.debug("[LLM-RANK] Parsed 0 scores at temp=%.1f from: %r", temperature, result[:200])
            return None
        return scores

    def _collect_video_urls(self, batch_num: int, limit: int | None = None,
                            allow_query_gen: bool = True) -> list[dict]:
        """Search Lustpress and collect video page URLs.

        If `allow_query_gen=False` and there's no pre-generated query on disk,
        returns [] instead of calling the captioner LLM. Use this from background
        threads (prefetch) where touching the main-thread CUDA context is unsafe.
        """
        import random
        seen = self._load_seen()
        candidates = []

        # Consume the query the main thread already wrote via generate_next_query.
        # If missing (edge cases: first batch, prefetch path, manual intervention),
        # generate inline unless explicitly disabled (prefetch thread must pass
        # allow_query_gen=False to avoid touching the main-thread captioner).
        query_file = Path("./workspace") / "next_query.txt"
        if query_file.exists():
            query = query_file.read_text().strip()
            query_file.unlink()
            log.info("[QUERY] Search query: %r", query)
        else:
            if not allow_query_gen:
                log.warning("[QUERY] next_query.txt missing and LLM query-gen disabled "
                            "(prefetch path). Skipping this crawl.")
                return []
            query = self._generate_query(batch_num)

        # Ensure Lustpress is alive before we fire search requests. If it
        # died (node crash, OOM, etc.), restart it now instead of burning all
        # 3 search attempts on connection-refused errors.
        self.server.ensure_healthy()

        # Stay near the top of search results so query relevance dominates.
        # Pages >5 rapidly degrade into long-tail keyword matches on porn sites.
        page = random.randint(1, 5)
        # Only use sorts that keep query relevance as a strong signal.
        # Dropped "tr" (top rated — uncorrelated with query) and "lg" (longest —
        # surfaces hour-long compilations that happen to contain one tag word).
        sort = random.choice(["mr", "mv"])  # most recent, most viewed
        log.info("Search params: query='%s', page=%d, sort=%s", query, page, sort)

        # Search across configured sources in parallel (I/O bound HTTP calls —
        # threads give ~N× speedup over the sequential loop). Ranking happens
        # in llm_rank_candidates() in the caller; this method only collects
        # the raw candidate pool.
        sources = list(self.sources)
        random.shuffle(sources)

        from concurrent.futures import ThreadPoolExecutor

        def _search_one(src: str) -> tuple[str, list]:
            try:
                return src, self.server.search(src, query, page=page, sort=sort)
            except Exception as e:
                log.warning("  %s search failed: %s", src, e)
                return src, []

        with ThreadPoolExecutor(max_workers=len(sources) or 1) as pool:
            source_results = list(pool.map(_search_one, sources))

        for source, results in source_results:
            if len(candidates) >= self.max_per_batch * 4:
                break
            log.info("  %s search '%s': %d results", source, query, len(results))

            for item in results:
                link = item.get("link", "")
                vid_id = item.get("id", "")

                if not link or link in seen:
                    continue

                title = item.get("title", "") or ""

                # Title blocklist — reject overrepresented uploaders/studios.
                title_lc = title.lower()
                blocked = next((b for b in self.title_blocklist if b in title_lc), None)
                if blocked:
                    log.info("[BLOCKLIST] Rejecting %r (matched %r)", title[:80], blocked)
                    continue

                # Duration filtering
                dur = self._parse_duration_seconds(item.get("duration", ""))
                if dur is not None:
                    if dur < self.min_dur or dur > self.max_dur:
                        continue

                candidates.append({
                    "url": link,
                    "id": vid_id,
                    "title": title or "unknown",
                    "source": source,
                    "duration": dur,
                })

                # Over-collect so scoring has something to rank from.
                if len(candidates) >= self.max_per_batch * 4:
                    break

            if len(candidates) >= self.max_per_batch * 4:
                break

        # Also grab some random videos if configured
        if self.use_random and len(candidates) < self.max_per_batch:
            for source in self.sources[:2]:
                try:
                    rnd = self.server.random_video(source)
                    if rnd and rnd.get("source"):
                        src_url = rnd["source"]
                        if src_url not in seen:
                            candidates.append({
                                "url": src_url,
                                "id": rnd.get("data", {}).get("id", ""),
                                "title": rnd.get("data", {}).get("title", "random"),
                                "source": source,
                                "duration": None,
                            })
                except Exception as e:
                    log.debug("Random fetch failed: %s", e)

        # Inject custom URLs (priority — always included)
        if self.custom_urls:
            for url in list(self.custom_urls):
                if url not in seen:
                    candidates.insert(0, {
                        "url": url,
                        "id": url.split("/")[-1],
                        "title": "custom",
                        "source": "custom",
                        "duration": None,
                    })
                    log.info("[CUSTOM] Injecting: %s", url[:80])

        # Deduplicate
        unique = {}
        for c in candidates:
            if c["url"] not in unique:
                unique[c["url"]] = c
        deduped = list(unique.values())

        # Apply the caller's cap. Default is the download budget (max_per_batch);
        # callers that want a larger pool for LLM ranking pass a higher limit.
        cap = limit if limit is not None else self.max_per_batch
        deduped = deduped[:cap]

        log.info("Collected %d candidate video URLs (cap=%d)", len(deduped), cap)
        return deduped

    def _download_one(self, item: dict, output_dir: Path) -> Path | None:
        """Download a single video. Returns path if successful, None otherwise."""
        url = item["url"]
        output_template = str(output_dir / f"{item['source']}_%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            url,
            "-f", f"bestvideo[height<={self.max_res}][ext=mp4]+bestaudio[ext=m4a]/best[height<={self.max_res}][ext=mp4]/best[height<={self.max_res}]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--no-playlist",
            "--no-overwrites",
            "--retries", "3",
            "--socket-timeout", "30",
            "--max-filesize", "100m",
            "--match-filter", f"duration>={self.min_dur} & duration<={self.max_dur}",
            "--download-archive", str(Path(self.archive_path).resolve()),
        ]

        log.info("  Downloading: %s [%s]", item["title"][:60], item["source"])

        existing_before = set(output_dir.glob("*.mp4"))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                    errors="replace")
            if result.returncode == 0:
                new_files = set(output_dir.glob("*.mp4")) - existing_before
                if new_files:
                    new_file = next(iter(new_files))
                    log.info("    OK: %s", new_file.name)
                    return new_file
            else:
                log.warning("    Failed: %s", (result.stderr or "")[-200:].strip())
        except subprocess.TimeoutExpired:
            log.warning("    Timeout: %s", item["title"][:60])
        finally:
            self._mark_seen(url)

        return None

    def _download_videos(self, candidates: list[dict], output_dir: Path) -> list[Path]:
        """Download videos in parallel using yt-dlp."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {pool.submit(self._download_one, item, output_dir): item for item in candidates}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    downloaded.append(result)

        return downloaded

    def crawl(self, batch_num: int, output_dir: Path,
              allow_query_gen: bool = True) -> list[Path]:
        """
        Full crawl cycle: search via Lustpress → download with yt-dlp.
        Returns list of downloaded MP4 paths.

        Pass `allow_query_gen=False` when calling from a background thread so
        the crawler refuses to touch the main-thread captioner context.
        """
        log.info("Crawling batch %d — sources: %s (query generated by LLM)",
                 batch_num, self.sources)

        candidates = self._collect_video_urls(batch_num, allow_query_gen=allow_query_gen)
        if not candidates:
            log.warning("No candidate videos found")
            return []

        videos = self._download_videos(candidates, output_dir)
        log.info("Downloaded %d/%d videos to %s", len(videos), len(candidates), output_dir)
        return videos
