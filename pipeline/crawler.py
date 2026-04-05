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

    def start(self):
        """Start the Lustpress server if not already running."""
        if self._is_port_open():
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

    def search(self, source: str, query: str, page: int = 1,
               sort: str = "mr") -> list[dict]:
        """Search a source for videos. Returns list of video result dicts."""
        url = f"{self.base_url}/{source}/search"
        params = {"key": query, "page": page, "sort": sort}

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("success", False):
                log.warning("Search failed for %s/%s: %s", source, query, data.get("message", "unknown"))
                return []

            results = data.get("data", [])
            log.info("  %s search '%s': %d results", source, query, len(results))
            return results

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
            resp = requests.get(url, timeout=30)
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

    def __init__(self, config: dict, server: LustpressServer):
        self.server = server
        self.queries = config.get("search_queries", ["amateur"])
        self.sources = config.get("sources", ["xvideos", "xnxx", "eporner", "redtube"])
        self.max_per_batch = config.get("max_videos_per_batch", 8)
        self.min_dur = config.get("min_duration_sec", 5)
        self.max_dur = config.get("max_duration_sec", 90)
        self.max_res = config.get("max_resolution", 720)
        self.archive_path = config.get("download_archive", "./state/seen_videos.txt")
        self.use_random = config.get("include_random", True)
        Path(self.archive_path).parent.mkdir(parents=True, exist_ok=True)

    def _load_seen(self) -> set[str]:
        path = Path(self.archive_path)
        if path.exists():
            return set(line.strip() for line in path.read_text().splitlines() if line.strip())
        return set()

    def _mark_seen(self, url: str):
        with open(self.archive_path, "a") as f:
            f.write(url + "\n")

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

    def _collect_video_urls(self, batch_num: int) -> list[dict]:
        """Search Lustpress and collect video page URLs."""
        seen = self._load_seen()
        candidates = []

        # Pick query for this batch
        query_idx = batch_num % len(self.queries)
        query = self.queries[query_idx]

        # Search across configured sources
        for source in self.sources:
            results = self.server.search(source, query, page=1, sort="mr")

            for item in results:
                link = item.get("link", "")
                vid_id = item.get("id", "")

                if not link or link in seen:
                    continue

                # Duration filtering
                dur = self._parse_duration_seconds(item.get("duration", ""))
                if dur is not None:
                    if dur < self.min_dur or dur > self.max_dur:
                        continue

                candidates.append({
                    "url": link,
                    "id": vid_id,
                    "title": item.get("title", "unknown"),
                    "source": source,
                    "duration": dur,
                })

                if len(candidates) >= self.max_per_batch * 2:
                    break

            if len(candidates) >= self.max_per_batch * 2:
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

        # Deduplicate and limit
        unique = {}
        for c in candidates:
            if c["url"] not in unique:
                unique[c["url"]] = c
        candidates = list(unique.values())[:self.max_per_batch]

        log.info("Collected %d candidate video URLs", len(candidates))
        return candidates

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

    def crawl(self, batch_num: int, output_dir: Path) -> list[Path]:
        """
        Full crawl cycle: search via Lustpress → download with yt-dlp.
        Returns list of downloaded MP4 paths.
        """
        query_idx = batch_num % len(self.queries)
        log.info("Crawling batch %d — query: '%s', sources: %s",
                 batch_num, self.queries[query_idx], self.sources)

        candidates = self._collect_video_urls(batch_num)
        if not candidates:
            log.warning("No candidate videos found")
            return []

        videos = self._download_videos(candidates, output_dir)
        log.info("Downloaded %d/%d videos to %s", len(videos), len(candidates), output_dir)
        return videos
