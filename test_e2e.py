#!/usr/bin/env python3
"""
End-to-end test — runs the crawl + caption stages with 2 videos.
Skips training (no LTX model needed) to verify the pipeline works.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
for name in ["urllib3", "httpx", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)

log = logging.getLogger("test_e2e")

from pipeline.crawler import LustpressServer, VideoCrawler
from pipeline.captioner import create_captioner, extract_frames
from pipeline import dashboard as dash


def test_lustpress_search(server: LustpressServer):
    """Test that Lustpress can search and return results."""
    dash.banner("TEST: Lustpress Search API")

    sources = ["xvideos", "xnxx", "eporner"]
    for source in sources:
        results = server.search(source, "amateur", page=1, sort="mr")
        print(f"  {source}: {len(results)} results")
        if results:
            r = results[0]
            print(f"    First: {r.get('title', '?')[:60]}")
            print(f"    Link:  {r.get('link', '?')}")
            print(f"    Dur:   {r.get('duration', '?')}")
        print()

    return len(results) > 0


def test_crawl_download(server: LustpressServer, work_dir: Path):
    """Test downloading 2 videos via Lustpress + yt-dlp."""
    dash.banner("TEST: Crawl + Download (2 videos)")

    crawler = VideoCrawler(
        config={
            "search_queries": ["amateur"],
            "sources": ["xvideos"],
            "max_videos_per_batch": 2,
            "min_duration_sec": 30,
            "max_duration_sec": 300,
            "max_resolution": 480,
            "download_archive": str(work_dir / "test_seen.txt"),
            "include_random": False,
        },
        server=server,
    )

    raw_dir = work_dir / "raw"
    videos = crawler.crawl(batch_num=0, output_dir=raw_dir)

    print(f"\n  Downloaded {len(videos)} videos:")
    for v in videos:
        size_mb = v.stat().st_size / 1024 / 1024
        print(f"    {v.name}  ({size_mb:.1f} MB)")

    return videos


def test_frame_extraction(videos: list[Path]):
    """Test extracting frames from downloaded videos."""
    dash.banner("TEST: Frame Extraction")

    for video in videos:
        frames = extract_frames(video, fps=2, max_frames=8)
        print(f"  {video.name}: extracted {len(frames)} frames")
        if frames:
            h, w = frames[0].shape[:2]
            print(f"    Resolution: {w}x{h}")

    return True


def test_caption_mock(videos: list[Path], work_dir: Path):
    """Test caption generation with a mock (no GPU needed)."""
    dash.banner("TEST: Caption Generation (mock)")

    metadata_file = work_dir / "test_metadata.jsonl"

    # Write mock captions in LTX 2.3 format to verify the display
    entries = []
    for video in videos:
        frames = extract_frames(video, fps=1, max_frames=4)
        h, w = (frames[0].shape[:2]) if frames else (0, 0)

        caption = (
            f"[VISUAL]: A video at {w}x{h} resolution showing various scenes. "
            f"The camera captures natural movement with ambient lighting. "
            f"The scene progresses through multiple angles with warm color tones. "
            f"Subjects are visible in the frame with natural body language and expressions. "
            f"The environment appears to be an indoor setting with soft diffused lighting.\n"
            f"[SPEECH]: None.\n"
            f"[SOUNDS]: Ambient room sounds, soft background music with a slow tempo.\n"
            f"[TEXT]: None."
        )
        entries.append({"media_path": str(video), "caption": caption})

    with open(metadata_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Display the captions like the real pipeline would
    dash.show_captions(metadata_file)
    return metadata_file


def main():
    work_dir = Path("./test_workspace")
    work_dir.mkdir(exist_ok=True)

    lustpress_dir = Path("./lustpress")
    server = LustpressServer(str(lustpress_dir), port=3000)

    try:
        # 1. Start Lustpress
        dash.banner("STARTING LUSTPRESS SERVER")
        server.start()
        print(f"  Server ready at {server.base_url}")

        # 2. Test search
        if not test_lustpress_search(server):
            log.error("Search returned no results, aborting")
            return

        # 3. Test crawl + download
        videos = test_crawl_download(server, work_dir)
        if not videos:
            log.error("No videos downloaded, aborting")
            return

        # 4. Test frame extraction
        test_frame_extraction(videos)

        # 5. Test captions (mock — no model loaded)
        test_caption_mock(videos, work_dir)

        # Summary
        dash.banner("ALL TESTS PASSED")
        print(f"  Videos downloaded: {len(videos)}")
        print(f"  Working dir: {work_dir}")
        print(f"  To run with real captioning, use: python main.py --batch-once")
        print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
