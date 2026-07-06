"""
S3-compatible object storage (Hippius or Cloudflare R2): clients + buckets.

Backend is selected with OBJECT_STORAGE_BACKEND=r2|hippius (default r2; see
env.example). Used for the source-video corpus (read/write) and each
validator's own king-state bucket (write).
"""
import asyncio
from typing import Tuple

from minio import Minio

from leoma.bootstrap import emit_log
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


def _build_minio_client(
    endpoint_raw: str | None,
    region: str | None,
    access_key: str | None,
    secret_key: str | None,
    *,
    purpose: str,
) -> Minio:
    """Build a Minio client from an explicit endpoint/region/creds (own bucket)."""
    if not endpoint_raw or not endpoint_raw.strip():
        raise ValueError(f"Missing object storage endpoint for {purpose}")
    if not access_key or not secret_key:
        raise ValueError(
            f"Missing object storage credentials for {purpose}. "
            "Set the matching access key and secret key (R2_OWN_*)."
        )
    return Minio(
        normalize_s3_endpoint_host(endpoint_raw),
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        region=region or "auto",
    )


def create_own_write_client() -> Minio:
    """Minio client for this validator's OWN state bucket (write creds from env)."""
    return _build_minio_client(
        settings.r2_own_endpoint,
        settings.r2_own_region,
        settings.r2_own_write_access_key,
        settings.r2_own_write_secret_key,
        purpose="own state bucket write access",
    )


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


async def ensure_bucket_exists(minio_client: Minio, bucket_name: str) -> None:
    exists = await asyncio.to_thread(minio_client.bucket_exists, bucket_name)
    if not exists:
        await asyncio.to_thread(minio_client.make_bucket, bucket_name)
        emit_log(f"Created bucket: {bucket_name}", "success")
