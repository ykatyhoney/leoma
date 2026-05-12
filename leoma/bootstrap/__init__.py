# Bootstrap: runtime config and logging.

from leoma.bootstrap.runtime import (
    # Core logger
    logger,
    log_exception,
    
    # New logging functions
    emit_log,
    emit_header,
    emit_section,
    log_debug,
    log_info,
    log_success,
    log_warning,
    log_error,
    log_critical,
    log_start,
    
    # Log context
    set_log_context,
    clear_log_context,
    get_log_context,
    
    # Timing
    LogTimer,
    timed,
    
    # Log levels
    LogLevel,
    
    # Legacy
    USED_VIDEOS,
    
    # Settings
    CHUTES_API_KEY,
    CHUTES_API_URL,
    CLIP_DURATION,
    CHUTE_CACHE_TTL,
    CORPUS_MAX_DURATION,
    CORPUS_MAX_FILESIZE,
    CORPUS_MIN_DURATION,
    CORPUS_TARGET_RESOLUTION,
    DATABASE_URL,
    EPOCH_LEN,
    HF_TOKEN,
    HIPPIUS_ENDPOINT,
    HIPPIUS_REGION,
    HIPPIUS_SAMPLES_READ_ACCESS_KEY,
    HIPPIUS_SAMPLES_READ_SECRET_KEY,
    HIPPIUS_SAMPLES_WRITE_ACCESS_KEY,
    HIPPIUS_SAMPLES_WRITE_SECRET_KEY,
    HIPPIUS_VIDEOS_READ_ACCESS_KEY,
    HIPPIUS_VIDEOS_READ_SECRET_KEY,
    HIPPIUS_VIDEOS_WRITE_ACCESS_KEY,
    HIPPIUS_VIDEOS_WRITE_SECRET_KEY,
    HOTKEY_NAME,
    MAX_CONCURRENT_MINERS,
    MAX_VIDEO_HISTORY,
    MAX_VIDEO_SIZE,
    MIN_VALIDATOR_STAKE,
    MIN_VIDEO_SIZE,
    REQUIRED_VIDEO_HEIGHT,
    VALIDATOR_SYNC_INTERVAL,
    MODEL_HASH_CACHE_TTL,
    NETUID,
    NETWORK,
    OBJECT_STORAGE_BACKEND,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    REQUEST_TIMEOUT,
    SAMPLES_BUCKET,
    SOURCE_BUCKET,
    WALLET_NAME,
    settings,
)

__all__ = [
    # Settings
    "settings",
    "USED_VIDEOS",
    "CHUTES_API_KEY",
    "CHUTES_API_URL",
    "CLIP_DURATION",
    "CHUTE_CACHE_TTL",
    "CORPUS_MAX_DURATION",
    "CORPUS_MAX_FILESIZE",
    "CORPUS_MIN_DURATION",
    "CORPUS_TARGET_RESOLUTION",
    "DATABASE_URL",
    "EPOCH_LEN",
    "HF_TOKEN",
    "HIPPIUS_ENDPOINT",
    "HIPPIUS_REGION",
    "HIPPIUS_SAMPLES_READ_ACCESS_KEY",
    "HIPPIUS_SAMPLES_READ_SECRET_KEY",
    "HIPPIUS_SAMPLES_WRITE_ACCESS_KEY",
    "HIPPIUS_SAMPLES_WRITE_SECRET_KEY",
    "HIPPIUS_VIDEOS_READ_ACCESS_KEY",
    "HIPPIUS_VIDEOS_READ_SECRET_KEY",
    "HIPPIUS_VIDEOS_WRITE_ACCESS_KEY",
    "HIPPIUS_VIDEOS_WRITE_SECRET_KEY",
    "HOTKEY_NAME",
    "MAX_CONCURRENT_MINERS",
    "MAX_VIDEO_HISTORY",
    "MAX_VIDEO_SIZE",
    "MIN_VALIDATOR_STAKE",
    "MIN_VIDEO_SIZE",
    "REQUIRED_VIDEO_HEIGHT",
    "VALIDATOR_SYNC_INTERVAL",
    "MODEL_HASH_CACHE_TTL",
    "NETUID",
    "NETWORK",
    "OBJECT_STORAGE_BACKEND",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "POSTGRES_DB",
    "POSTGRES_HOST",
    "POSTGRES_PASSWORD",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "REQUEST_TIMEOUT",
    "SAMPLES_BUCKET",
    "SOURCE_BUCKET",
    "WALLET_NAME",
    
    # Core logging
    "logger",
    "log_exception",
    
    # New logging functions
    "emit_log",
    "emit_header",
    "emit_section",
    "log_debug",
    "log_info",
    "log_success",
    "log_warning",
    "log_error",
    "log_critical",
    "log_start",
    
    # Log context
    "set_log_context",
    "clear_log_context",
    "get_log_context",
    
    # Timing
    "LogTimer",
    "timed",
    
    # Log levels
    "LogLevel",
]
