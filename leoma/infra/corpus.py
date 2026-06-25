"""
Video corpus curation and management for Leoma.

Provides functions for downloading videos from YouTube, validating them,
and uploading to the Hippius source bucket (default: "videos").
"""

from __future__ import annotations

import os
import asyncio
import tempfile
import subprocess
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

from minio import Minio

from leoma.bootstrap import (
    SOURCE_BUCKET,
    CORPUS_MIN_DURATION,
    CORPUS_MAX_DURATION,
    CORPUS_TARGET_RESOLUTION,
    CORPUS_MAX_FILESIZE,
)
from leoma.bootstrap import emit_log as log
from leoma.infra.storage_backend import ensure_bucket_exists

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")
MIN_VIDEO_FILESIZE_BYTES = 100000
YT_DLP_INFO_TIMEOUT_SECONDS = 30
YT_DLP_DOWNLOAD_TIMEOUT_SECONDS = 600
YT_DLP_SEARCH_TIMEOUT_SECONDS = 60


def _is_supported_video_file(path: str) -> bool:
    """Return whether a path has a supported video file extension."""
    return path.endswith(VIDEO_EXTENSIONS)


async def _run_subprocess(
    command: List[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Run subprocess command in worker thread with captured output."""
    return await asyncio.to_thread(
        subprocess.run,
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _parse_json_lines(raw_output: str) -> List[Dict[str, Any]]:
    """Parse newline-delimited JSON objects from command output."""
    items: List[Dict[str, Any]] = []
    for line in raw_output.splitlines():
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def _find_downloaded_video(output_dir: str) -> Optional[str]:
    """Find first downloaded video file in output directory."""
    for filename in os.listdir(output_dir):
        if _is_supported_video_file(filename):
            return os.path.join(output_dir, filename)
    return None


def _build_ingest_metadata(info: Dict[str, Any], url: str, video_id: str) -> Dict[str, Any]:
    """Build metadata payload for uploaded corpus entry."""
    return {
        "source": "youtube",
        "video_id": video_id,
        "url": url,
        "title": info.get("title"),
        "channel": info.get("channel"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "ingested_at": datetime.now().isoformat(),
    }


def _new_batch_results(total: int) -> Dict[str, Any]:
    """Create initialized ingest batch result structure."""
    return {
        "total": total,
        "success": 0,
        "failed": 0,
        "uploaded": [],
        "errors": [],
    }


async def get_video_info(url: str) -> Optional[Dict[str, Any]]:
    """Get video metadata from a YouTube URL using yt-dlp (None if failed)."""
    try:
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            url,
        ]
        result = await _run_subprocess(cmd, timeout=YT_DLP_INFO_TIMEOUT_SECONDS)
        if result.returncode != 0:
            log(f"yt-dlp info failed: {result.stderr[:200]}", "warn")
            return None
        return json.loads(result.stdout)
    except Exception as e:
        log(f"Failed to get video info: {e}", "warn")
        return None


def validate_video_metadata(info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate yt-dlp video metadata meets corpus requirements. Returns (is_valid, error_reason)."""
    duration = info.get("duration", 0)
    if duration < CORPUS_MIN_DURATION:
        return False, f"too_short:{duration}s<{CORPUS_MIN_DURATION}s"
    if duration > CORPUS_MAX_DURATION:
        return False, f"too_long:{duration}s>{CORPUS_MAX_DURATION}s"

    formats = info.get("formats", [])
    has_video = any(f.get("vcodec", "none") != "none" for f in formats)
    if not has_video:
        return False, "no_video_stream"

    if info.get("age_limit", 0) > 0:
        return False, "age_restricted"

    if info.get("is_live", False):
        return False, "live_stream"

    return True, None


async def download_video(
    url: str,
    output_dir: str,
    filename: Optional[str] = None,
) -> Optional[str]:
    """Download a YouTube video using yt-dlp into output_dir. Returns the path or None if failed.

    ``filename`` is an optional name without extension.
    """
    if filename is None:
        filename = "%(id)s"
    
    output_template = os.path.join(output_dir, f"{filename}.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={CORPUS_TARGET_RESOLUTION}]+bestaudio/best[height<={CORPUS_TARGET_RESOLUTION}]/best",
        "--merge-output-format", "mp4",
        "--recode-video", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--no-overwrites",
        url,
    ]
    
    try:
        result = await _run_subprocess(cmd, timeout=YT_DLP_DOWNLOAD_TIMEOUT_SECONDS)
        if result.returncode != 0:
            log(f"yt-dlp download failed: {result.stderr[:200]}", "warn")
            return None
        
        downloaded_path = _find_downloaded_video(output_dir)
        if downloaded_path:
            return downloaded_path
        
        log("Downloaded file not found", "warn")
        return None
        
    except subprocess.TimeoutExpired:
        log("Download timeout (10 min)", "warn")
        return None
    except Exception as e:
        log(f"Download failed: {e}", "warn")
        return None


def _check_ffprobe_available() -> bool:
    """Check if ffprobe is available on the system."""
    import shutil
    return shutil.which("ffprobe") is not None


async def validate_downloaded_video(video_path: str) -> Tuple[bool, Optional[str]]:
    """Validate a downloaded video file. Returns (is_valid, error_reason)."""
    if not os.path.exists(video_path):
        return False, "file_not_found"
    
    filesize = os.path.getsize(video_path)
    if filesize > CORPUS_MAX_FILESIZE:
        return False, f"file_too_large:{filesize}>{CORPUS_MAX_FILESIZE}"
    if filesize < MIN_VIDEO_FILESIZE_BYTES:  # 100KB minimum
        return False, f"file_too_small:{filesize}<100KB"
    
    # If ffprobe is not available, just validate by file size and extension
    if not _check_ffprobe_available():
        if _is_supported_video_file(video_path):
            return True, None
        return False, "invalid_extension"
    
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        result = await _run_subprocess(cmd, timeout=30)
        if result.returncode != 0:
            # ffprobe failed but file exists with valid size - accept it
            return True, None
        
        probe = json.loads(result.stdout)

        streams = probe.get("streams", [])
        has_video = any(s.get("codec_type") == "video" for s in streams)
        if not has_video:
            return False, "no_video_stream"

        duration = float(probe.get("format", {}).get("duration", 0))
        if duration < CORPUS_MIN_DURATION:
            return False, f"too_short:{duration:.1f}s"
        
        return True, None
        
    except Exception as e:
        # If validation fails but file exists with good size, accept it
        log(f"ffprobe validation skipped: {e}", "warn")
        return True, None


async def upload_to_corpus(
    minio_client: Minio,
    video_path: str,
    video_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Upload a video to the corpus bucket. Returns the object key or None if failed."""
    await ensure_bucket_exists(minio_client, SOURCE_BUCKET)

    # Date-prefixed object key for organization.
    date_prefix = datetime.now().strftime("%Y/%m")
    object_key = f"{date_prefix}/{video_id}.mp4"
    
    try:
        await asyncio.to_thread(
            minio_client.fput_object,
            SOURCE_BUCKET,
            object_key,
            video_path,
            content_type="video/mp4",
        )
        
        filesize_mb = os.path.getsize(video_path) / (1024 * 1024)
        log(f"Uploaded: s3://{SOURCE_BUCKET}/{object_key} ({filesize_mb:.1f} MB)", "success")

        if metadata:
            metadata_key = f"{date_prefix}/{video_id}.json"
            metadata_path = f"/tmp/{video_id}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            await asyncio.to_thread(
                minio_client.fput_object,
                SOURCE_BUCKET,
                metadata_key,
                metadata_path,
                content_type="application/json",
            )
            os.remove(metadata_path)
        
        return object_key
        
    except Exception as e:
        log(f"Upload failed: {e}", "error")
        return None


async def ingest_youtube_video(
    minio_client: Minio,
    url: str,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Ingest a single YouTube video into the corpus. Returns (success, object_key, error_reason)."""
    log(f"Fetching info: {url}", "info")
    info = await get_video_info(url)
    if not info:
        return False, None, "info_fetch_failed"

    video_id = info.get("id", "unknown")
    title = info.get("title", "Unknown")[:50]
    duration = info.get("duration", 0)

    log(f"Video: {title} ({duration}s)", "info")

    is_valid, reason = validate_video_metadata(info)
    if not is_valid:
        return False, None, reason

    with tempfile.TemporaryDirectory() as tmpdir:
        log(f"Downloading: {video_id}", "info")
        video_path = await download_video(url, tmpdir, video_id)
        if not video_path:
            return False, None, "download_failed"

        is_valid, reason = await validate_downloaded_video(video_path)
        if not is_valid:
            return False, None, reason

        metadata = _build_ingest_metadata(info, url, video_id)

        object_key = await upload_to_corpus(minio_client, video_path, video_id, metadata)
        if not object_key:
            return False, None, "upload_failed"
        
        return True, object_key, None


async def ingest_youtube_batch(
    minio_client: Minio,
    urls: List[str],
    max_concurrent: int = 2,
) -> Dict[str, Any]:
    """Ingest multiple YouTube videos concurrently. Returns a success/failure summary dict."""
    results = _new_batch_results(total=len(urls))
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_url(url: str):
        async with semaphore:
            success, object_key, error = await ingest_youtube_video(minio_client, url)
            if success:
                results["success"] += 1
                results["uploaded"].append(object_key)
            else:
                results["failed"] += 1
                results["errors"].append({"url": url, "error": error})
    
    tasks = [process_url(url) for url in urls]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    return results


# Default search queries for diverse video content suitable for I2V
DEFAULT_SEARCH_QUERIES = [
    "nature documentary 4k",
    "cooking tutorial",
    "travel vlog",
    "dance performance",
    "sports highlights",
    "city timelapse",
    "animal behavior",
    "art tutorial painting",
    "fitness workout",
    "music performance live",
    "ocean underwater footage",
    "drone footage landscape",
    "street food cooking",
    "crafts diy tutorial",
    "science experiment",
]


async def search_youtube(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Search YouTube for videos matching a query. Returns a list of metadata dicts."""
    try:
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--flat-playlist",
            "--no-download",
            f"ytsearch{max_results}:{query}",
        ]
        result = await _run_subprocess(cmd, timeout=YT_DLP_SEARCH_TIMEOUT_SECONDS)
        if result.returncode != 0:
            log(f"YouTube search failed: {result.stderr[:200]}", "warn")
            return []
        
        return _parse_json_lines(result.stdout)
        
    except Exception as e:
        log(f"Search failed: {e}", "warn")
        return []


async def discover_random_videos(
    queries: Optional[List[str]] = None,
    videos_per_query: int = 5,
    total_limit: int = 50,
) -> List[str]:
    """Discover random YouTube video URLs via search queries (defaults to DEFAULT_SEARCH_QUERIES)."""
    import random

    if queries is None:
        queries = DEFAULT_SEARCH_QUERIES.copy()

    random.shuffle(queries)

    all_urls = []
    seen_ids = set()
    
    for query in queries:
        if len(all_urls) >= total_limit:
            break
        
        log(f"Searching: {query}", "info")
        videos = await search_youtube(query, videos_per_query * 2)  # Fetch extra for filtering
        
        for video in videos:
            if len(all_urls) >= total_limit:
                break
            
            video_id = video.get("id")
            if not video_id or video_id in seen_ids:
                continue
            
            duration = video.get("duration")
            if duration and (duration < CORPUS_MIN_DURATION or duration > CORPUS_MAX_DURATION):
                continue
            
            seen_ids.add(video_id)
            all_urls.append(f"https://www.youtube.com/watch?v={video_id}")
            
            if len(all_urls) >= total_limit:
                break
    
    log(f"Discovered {len(all_urls)} video URLs", "success")
    return all_urls


async def expand_corpus_random(
    minio_client: Minio,
    count: int = 10,
    queries: Optional[List[str]] = None,
    max_concurrent: int = 2,
) -> Dict[str, Any]:
    """Expand the corpus by searching diverse queries and ingesting suitable videos."""
    log(f"Discovering {count} random videos...", "info")

    # Discover more URLs than needed (some will fail validation).
    urls = await discover_random_videos(
        queries=queries,
        videos_per_query=max(3, count // 5),
        total_limit=count * 2,
    )
    
    if not urls:
        result = _new_batch_results(total=0)
        result["errors"].append({"error": "no_videos_discovered"})
        return result

    urls = urls[:count]

    log(f"Ingesting {len(urls)} videos...", "info")
    return await ingest_youtube_batch(minio_client, urls, max_concurrent)

