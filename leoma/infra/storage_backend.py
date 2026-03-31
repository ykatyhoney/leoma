"""
S3-compatible object storage (Hippius or Cloudflare R2): clients, buckets, sample uploads.

Backend is selected with OBJECT_STORAGE_BACKEND=r2|hippius (default r2; see env.example).
"""
import io
import os
import json
import asyncio
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

from minio import Minio

from leoma.bootstrap import SAMPLES_BUCKET, emit_log
from leoma.bootstrap.runtime import normalize_s3_endpoint_host, settings


def _active_s3_host_and_region() -> Tuple[str, str]:
    if settings.object_storage_backend == "r2":
        raw = settings.r2_endpoint_raw
        if not raw or not raw.strip():
            raise ValueError(
                "R2_ENDPOINT is required when OBJECT_STORAGE_BACKEND=r2 "
                "(e.g. https://<ACCOUNT_ID>.r2.cloudflarestorage.com)"
            )
        return normalize_s3_endpoint_host(raw), settings.r2_region
    return settings.hippius_endpoint, settings.hippius_region


def _create_minio_client(
    access_key: str | None,
    secret_key: str | None,
    *,
    purpose: str,
) -> Minio:
    if not access_key or not secret_key:
        backend = settings.object_storage_backend
        raise ValueError(
            f"Missing object storage credentials for {purpose} (backend={backend}). "
            "Set the matching access key and secret key environment variables "
            f"({'R2_*' if backend == 'r2' else 'HIPPIUS_*'})."
        )
    endpoint, region = _active_s3_host_and_region()
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        region=region,
    )


def _is_non_empty_file(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


async def _upload_file(
    minio_client: Minio,
    bucket: str,
    object_name: str,
    local_path: str,
) -> None:
    await asyncio.to_thread(minio_client.fput_object, bucket, object_name, local_path)


def _write_metadata_file(prefix: str, metadata: Dict[str, Any]) -> str:
    metadata_path = f"/tmp/metadata_{prefix}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def create_source_read_client() -> Minio:
    if settings.object_storage_backend == "r2":
        ak, sk = settings.r2_videos_read_access_key, settings.r2_videos_read_secret_key
    else:
        ak, sk = settings.hippius_videos_read_access_key, settings.hippius_videos_read_secret_key
    return _create_minio_client(ak, sk, purpose="source bucket read access")


def create_source_write_client() -> Minio:
    if settings.object_storage_backend == "r2":
        ak, sk = settings.r2_videos_write_access_key, settings.r2_videos_write_secret_key
    else:
        ak, sk = settings.hippius_videos_write_access_key, settings.hippius_videos_write_secret_key
    return _create_minio_client(ak, sk, purpose="source bucket write access")


def create_samples_write_client() -> Minio:
    if settings.object_storage_backend == "r2":
        ak, sk = settings.r2_samples_write_access_key, settings.r2_samples_write_secret_key
    else:
        ak, sk = settings.hippius_samples_write_access_key, settings.hippius_samples_write_secret_key
    return _create_minio_client(ak, sk, purpose="samples bucket write access")


def create_samples_read_client() -> Minio:
    if settings.object_storage_backend == "r2":
        ak = settings.r2_samples_read_access_key or settings.r2_samples_write_access_key
        sk = settings.r2_samples_read_secret_key or settings.r2_samples_write_secret_key
    else:
        ak = settings.hippius_samples_read_access_key or settings.hippius_samples_write_access_key
        sk = settings.hippius_samples_read_secret_key or settings.hippius_samples_write_secret_key
    return _create_minio_client(ak, sk, purpose="samples bucket read access")


def get_presigned_get_url(
    minio_client: Minio,
    bucket: str,
    object_name: str,
    expires: Optional[timedelta] = None,
) -> str:
    if expires is None:
        expires = timedelta(hours=1)
    return minio_client.presigned_get_object(bucket, object_name, expires=expires)


async def get_task_media_presigned_urls(
    task_id: int,
    miner_hotkey: str,
    *,
    bucket: str = SAMPLES_BUCKET,
    expires: Optional[timedelta] = None,
) -> Optional[Dict[str, str]]:
    try:
        client = create_samples_read_client()
    except ValueError:
        return None
    prefix = str(task_id)
    safe_hotkey = miner_hotkey.replace("/", "_").replace("\\", "_")
    keys = {
        "first_frame_url": f"{prefix}/first_frame.png",
        "original_clip_url": f"{prefix}/original_clip.mp4",
        "generated_video_url": f"{prefix}/generated_videos/{safe_hotkey}.mp4",
    }
    result: Dict[str, str] = {}
    for name, object_name in keys.items():
        try:
            url = await asyncio.to_thread(
                get_presigned_get_url, client, bucket, object_name, expires
            )
            result[name] = url
        except Exception:
            pass
    return result if result else None


async def download_task_artifacts(
    minio_client: Minio,
    bucket: str,
    task_id: int,
    dest_dir: str,
    *,
    include_original_clip: bool = True,
) -> Dict[str, Any]:
    prefix = f"{task_id}/"
    meta_key = f"{task_id}/metadata.json"
    clip_key = f"{task_id}/original_clip.mp4"
    frame_key = f"{task_id}/first_frame.png"
    meta_path = os.path.join(dest_dir, "metadata.json")
    clip_path = os.path.join(dest_dir, "original_clip.mp4")
    frame_path = os.path.join(dest_dir, "first_frame.png")
    os.makedirs(dest_dir, exist_ok=True)
    await asyncio.to_thread(minio_client.fget_object, bucket, meta_key, meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if include_original_clip:
        await asyncio.to_thread(minio_client.fget_object, bucket, clip_key, clip_path)
    await asyncio.to_thread(minio_client.fget_object, bucket, frame_key, frame_path)
    miners = metadata.get("miners", [])
    miner_hotkeys = miners if isinstance(miners, list) else (list(miners.keys()) if isinstance(miners, dict) else [])
    generated = {}
    for hotkey in miner_hotkeys:
        safe = hotkey.replace("/", "_").replace("\\", "_")
        key = f"{task_id}/generated_videos/{safe}.mp4"
        local = os.path.join(dest_dir, f"generated_{safe}.mp4")
        try:
            await asyncio.to_thread(minio_client.fget_object, bucket, key, local)
            generated[hotkey] = local
        except Exception:
            pass
    return {
        "metadata": metadata,
        "original_clip": clip_path if include_original_clip else None,
        "first_frame": frame_path,
        "generated_videos": generated,
    }


async def list_evaluated_task_ids(
    minio_client: Minio,
    bucket: str,
    validator_hotkey: str,
    max_tasks: int = 100,
) -> list[int]:
    safe_hotkey = validator_hotkey.replace("/", "_").replace("\\", "_")
    suffix = f"evaluation_results/{safe_hotkey}.json"
    task_ids: list[int] = []
    objects = await asyncio.to_thread(
        lambda: list(minio_client.list_objects(bucket, prefix="", recursive=True))
    )
    for obj in objects:
        key = obj.object_name
        if key.endswith(suffix):
            try:
                tid = int(key.split("/")[0])
                task_ids.append(tid)
            except (ValueError, IndexError):
                continue
    task_ids = sorted(set(task_ids), reverse=True)[:max_tasks]
    return task_ids


def create_minio_client() -> Minio:
    return create_samples_read_client()


async def ensure_bucket_exists(minio_client: Minio, bucket_name: str) -> None:
    exists = await asyncio.to_thread(minio_client.bucket_exists, bucket_name)
    if not exists:
        await asyncio.to_thread(minio_client.make_bucket, bucket_name)
        emit_log(f"Created bucket: {bucket_name}", "success")


async def upload_evaluation_result_json(
    minio_client: Minio,
    task_id: int,
    validator_hotkey: str,
    payload: list,
    signature: str | None = None,
) -> str:
    safe_hotkey = validator_hotkey.replace("/", "_").replace("\\", "_")
    object_name = f"{task_id}/evaluation_results/{safe_hotkey}.json"
    wrapper = {"signature": signature or "", "data": payload}
    body = json.dumps(wrapper, indent=2).encode("utf-8")
    await asyncio.to_thread(
        minio_client.put_object,
        SAMPLES_BUCKET,
        object_name,
        io.BytesIO(body),
        len(body),
        content_type="application/json",
    )
    emit_log(f"Uploaded evaluation result: {object_name}", "info")
    return object_name


async def upload_task_artifacts(
    minio_client: Minio,
    task_id: int,
    original_clip_path: str,
    first_frame_path: str,
    metadata: Dict[str, Any],
    miner_videos: Dict[str, str],
) -> str:
    prefix = str(task_id)
    if _is_non_empty_file(original_clip_path):
        await _upload_file(
            minio_client, SAMPLES_BUCKET, f"{prefix}/original_clip.mp4", original_clip_path
        )
    if _is_non_empty_file(first_frame_path):
        await _upload_file(
            minio_client, SAMPLES_BUCKET, f"{prefix}/first_frame.png", first_frame_path
        )
    metadata_path = _write_metadata_file(prefix, metadata)
    try:
        await _upload_file(
            minio_client, SAMPLES_BUCKET, f"{prefix}/metadata.json", metadata_path
        )
    finally:
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    for hotkey, local_path in miner_videos.items():
        if not _is_non_empty_file(local_path):
            continue
        safe_hotkey = hotkey.replace("/", "_").replace("\\", "_")
        object_name = f"{prefix}/generated_videos/{safe_hotkey}.mp4"
        await _upload_file(minio_client, SAMPLES_BUCKET, object_name, local_path)
        emit_log(f"Uploaded: {object_name}", "info")
    return prefix

