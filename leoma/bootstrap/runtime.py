"""
Runtime environment and logging.

Loads configuration from the environment and provides structured logging helpers
with component context, log levels, and detailed metadata.
"""

import inspect
import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, TypeVar, ParamSpec

from dotenv import load_dotenv

load_dotenv()

# Standard logger for leoma; use log_exception() in production to avoid full tracebacks
logger = logging.getLogger("leoma")

# Context variable for request/correlation ID tracking
_request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})

P = ParamSpec("P")
R = TypeVar("R")


def _is_production() -> bool:
    env = os.environ.get("LEOMA_ENV", os.environ.get("ENVIRONMENT", "")).lower()
    return env == "production"


def _is_debug() -> bool:
    return os.environ.get("LOG_LEVEL", "INFO").upper() == "DEBUG"


# ============================================================================
# ANSI Color Codes - Professional Palette
# ============================================================================

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[90m"      # Dark gray (subtle)

# Professional color palette
_BLACK = "\033[30m"
_DARK_GRAY = "\033[90m"
_RED = "\033[31m"       # For errors
_GREEN = "\033[32m"     # For success
_YELLOW = "\033[33m"    # For warnings
_BLUE = "\033[34m"      # For info
_MAGENTA = "\033[35m"   # For component names
_CYAN = "\033[36m"      # For debug
_WHITE = "\033[37m"

# Bright versions for emphasis
_BRIGHT_RED = "\033[91m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_BLUE = "\033[94m"

_HEADER_WIDTH = 80

# Level-specific styling (professional, high contrast)
_LEVEL_COLORS = {
    "DEBUG": _DIM,              # Subtle gray for debug
    "INFO": _BLUE,              # Blue for informational
    "SUCCESS": _GREEN,          # Green for success
    "WARNING": _YELLOW,         # Yellow for warnings
    "ERROR": _RED,              # Red for errors
    "CRITICAL": f"{_BOLD}{_RED}",  # Bold red for critical
}

# Level tokens with consistent styling
_LEVEL_TOKENS = {
    "DEBUG": f"{_DIM}··{_RESET}",
    "INFO": f"{_BLUE}●{_RESET}",
    "SUCCESS": f"{_GREEN}✓{_RESET}",
    "WARNING": f"{_YELLOW}▲{_RESET}",
    "ERROR": f"{_RED}✗{_RESET}",
    "CRITICAL": f"{_BOLD}{_RED}✗{_RESET}",
    "START": f"{_BLUE}▶{_RESET}",
}


# ============================================================================
# Log Level Configuration
# ============================================================================

class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    START = "START"


# ============================================================================
# Caller Information
# ============================================================================

def _get_caller_info(skip_frames: int = 2) -> Dict[str, Any]:
    """Get detailed caller information (file, function, line)."""
    for i, frame_info in enumerate(inspect.stack()):
        if i < skip_frames:
            continue
        # Skip frames inside runtime.py
        if "runtime.py" in frame_info.filename:
            continue
        # Get relative path from workspace
        filename = frame_info.filename
        if "/leoma_subnet/" in filename:
            filename = filename.split("/leoma_subnet/")[-1]
        elif "/leoma/" in filename:
            filename = filename.split("/leoma/")[-1]
        
        return {
            "file": filename,
            "function": frame_info.function,
            "line": frame_info.lineno,
        }
    return {"file": "unknown", "function": "unknown", "line": 0}


def _get_component_name(caller_info: Dict[str, Any]) -> str:
    """Extract component/module name from caller file path."""
    filepath = caller_info.get("file", "")
    # Extract meaningful component names
    parts = filepath.split("/")
    if len(parts) >= 2:
        # Return something like "app.validator.main" or "infra.judge"
        return ".".join(parts[-3:]) if len(parts) >= 3 else ".".join(parts[-2:])
    return filepath


# ============================================================================
# Timestamp and Formatting
# ============================================================================

def _wall_clock() -> str:
    """Return full datetime string: YYYY-MM-DD HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"


def _format_prefix(level: str, ts: str, caller_info: Dict[str, Any], component: str) -> str:
    """Format log line prefix with timestamp, component, and level.
    
    Format: [TIMESTAMP] COMP_MODULE  LEVEL_TOKEN  MESSAGE
    """
    level_token = _LEVEL_TOKENS.get(level, " ")
    
    # Timestamp in brackets (dimmed)
    timestamp = f"{_DIM}[{ts}]{_RESET}"
    
    # Component name in subtle color
    component_str = f"{_MAGENTA}{component}{_RESET}"
    
    # Line number (subtle)
    line_str = f"{_DIM}:{caller_info['line']}{_RESET}"
    
    return f"{timestamp} {component_str}{line_str} {level_token}"


# ============================================================================
# Structured Data Support
# ============================================================================

def _format_data(data: Optional[Dict[str, Any]], indent: bool = False) -> str:
    """Format structured data for logging."""
    if not data:
        return ""
    
    if _is_debug() or indent:
        try:
            return "\n  " + json.dumps(data, indent=2, default=str).replace("\n", "\n  ")
        except Exception:
            return f" {data}"
    else:
        try:
            return f" | {json.dumps(data, default=str)}"
        except Exception:
            return f" | {data}"


# ============================================================================
# Context Management
# ============================================================================

def set_log_context(**kwargs: Any) -> None:
    """Set context variables to be included in all subsequent logs."""
    current = _request_context.get().copy()
    current.update(kwargs)
    _request_context.set(current)


def clear_log_context() -> None:
    """Clear all context variables."""
    _request_context.set({})


def get_log_context() -> Dict[str, Any]:
    """Get current log context."""
    return _request_context.get().copy()


# ============================================================================
# Main Logging Functions
# ============================================================================

def emit_log(
    msg: str,
    level: str = "INFO",
    data: Optional[Dict[str, Any]] = None,
    exc: Optional[Exception] = None,
) -> None:
    """Print a structured log line with timestamp, component, level, and optional data.
    
    Args:
        msg: The log message
        level: Log level (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        data: Optional structured data to include
        exc: Optional exception to include in log
    """
    caller_info = _get_caller_info()
    component = _get_component_name(caller_info)
    ts = _wall_clock()
    
    # Get context
    context = get_log_context()
    all_data = {**context, **(data or {})}
    if exc:
        all_data["exception"] = str(exc)
    
    # Format prefix
    prefix = _format_prefix(level, ts, caller_info, component)
    
    # Format data
    data_str = _format_data(all_data) if all_data else ""
    
    # Print log line
    print(f"{prefix} {msg}{data_str}")
    
    # Also log to standard logger for ERROR/CRITICAL
    if level in ("ERROR", "CRITICAL"):
        logger.error("%s %s", msg, data_str)


def emit_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a bold section header with optional subtitle.
    
    Compact format: **** TITLE ****
    """
    if subtitle:
        print(f"\n{_BOLD}{_BLUE}#### {title} ####{_RESET}")
        print(f"{_DIM}    {subtitle}{_RESET}")
    else:
        print(f"\n{_BOLD}{_BLUE}#### {title} ####{_RESET}")


def emit_section(title: str) -> None:
    """Print a minor section header.
    
    Compact format: ---- TITLE ----
    """
    print(f"\n{_WHITE}---- {title} ----{_RESET}")


# ============================================================================
# Convenience Logging Functions
# ============================================================================

def log_debug(msg: str, **data: Any) -> None:
    """Log a debug message."""
    emit_log(msg, level=LogLevel.DEBUG, data=data if data else None)


def log_info(msg: str, **data: Any) -> None:
    """Log an info message."""
    emit_log(msg, level=LogLevel.INFO, data=data if data else None)


def log_success(msg: str, **data: Any) -> None:
    """Log a success message."""
    emit_log(msg, level=LogLevel.SUCCESS, data=data if data else None)


def log_warning(msg: str, **data: Any) -> None:
    """Log a warning message."""
    emit_log(msg, level=LogLevel.WARNING, data=data if data else None)


def log_error(msg: str, exc: Optional[Exception] = None, **data: Any) -> None:
    """Log an error message."""
    emit_log(msg, level=LogLevel.ERROR, data=data if data else None, exc=exc)


def log_critical(msg: str, exc: Optional[Exception] = None, **data: Any) -> None:
    """Log a critical message."""
    emit_log(msg, level=LogLevel.CRITICAL, data=data if data else None, exc=exc)


def log_start(msg: str, **data: Any) -> None:
    """Log a start message."""
    emit_log(msg, level=LogLevel.START, data=data if data else None)


# ============================================================================
# Exception Logging
# ============================================================================

def log_exception(message: str, exc: Optional[BaseException] = None) -> None:
    """Log an exception; in production omit full traceback to avoid leaking paths."""
    if _is_production():
        detail = str(exc) if exc else ""
        emit_log(message, level=LogLevel.ERROR, data={"exception": detail})
    else:
        emit_log(message, level=LogLevel.ERROR, exc=exc if isinstance(exc, Exception) else None)
        # Also log to standard logger for full traceback in development
        logger.exception("%s", message, exc_info=exc is None or True)


# ============================================================================
# Performance Timing
# ============================================================================

class LogTimer:
    """Context manager for timing operations and logging the duration."""
    
    def __init__(self, operation: str, **data: Any):
        self.operation = operation
        self.data = data
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
    
    def __enter__(self) -> "LogTimer":
        self.start_time = time.monotonic()
        log_debug(f"Starting: {self.operation}", **self.data)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.duration_ms = (time.monotonic() - (self.start_time or 0)) * 1000
        
        if exc_type:
            log_error(
                f"Failed: {self.operation}",
                exc=exc_val,
                duration_ms=f"{self.duration_ms:.1f}",
                **self.data
            )
        else:
            log_success(
                f"Completed: {self.operation}",
                duration_ms=f"{self.duration_ms:.1f}",
                **self.data
            )


def timed(operation: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to time a function and log the duration."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with LogTimer(operation, function=func.__name__):
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with LogTimer(operation, function=func.__name__):
                return await func(*args, **kwargs)  # type: ignore
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper
    
    return decorator


# ============================================================================
# Legacy Compatibility
# ============================================================================

# Keep these for backward compatibility
USED_VIDEOS: List[str] = []


# ============================================================================
# Settings
# ============================================================================

def _read_str(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def _read_int(name: str, fallback: int) -> int:
    return int(os.environ.get(name, str(fallback)))


def _read_float(name: str, fallback: float) -> float:
    return float(os.environ.get(name, str(fallback)))


def _read_optional_str(name: str) -> Optional[str]:
    return os.environ.get(name)


class Settings:
    """Central settings loaded from the environment."""

    def __init__(self) -> None:
        self.netuid = _read_int("NETUID", 99)
        self.epoch_len = _read_int("EPOCH_LEN", 180)
        self.request_timeout = _read_int("REQUEST_TIMEOUT", 300)
        self.chutes_api_url = _read_str("CHUTES_API_URL", "https://api.chutes.ai")
        self.chutes_api_key = _read_optional_str("CHUTES_API_KEY")
        self.wallet_name = _read_str("WALLET_NAME", "default")
        self.hotkey_name = _read_str("HOTKEY_NAME", "default")
        self.network = _read_str("NETWORK", "finney")
        self.hippius_endpoint = _read_str("HIPPIUS_ENDPOINT", "s3.hippius.com")
        self.hippius_region = _read_str("HIPPIUS_REGION", "decentralized")
        self.hippius_videos_read_access_key = _read_optional_str("HIPPIUS_VIDEOS_READ_ACCESS_KEY")
        self.hippius_videos_read_secret_key = _read_optional_str("HIPPIUS_VIDEOS_READ_SECRET_KEY")
        self.hippius_videos_write_access_key = _read_optional_str("HIPPIUS_VIDEOS_WRITE_ACCESS_KEY")
        self.hippius_videos_write_secret_key = _read_optional_str("HIPPIUS_VIDEOS_WRITE_SECRET_KEY")
        self.hippius_samples_read_access_key = _read_optional_str("HIPPIUS_SAMPLES_READ_ACCESS_KEY")
        self.hippius_samples_read_secret_key = _read_optional_str("HIPPIUS_SAMPLES_READ_SECRET_KEY")
        self.hippius_samples_write_access_key = _read_optional_str("HIPPIUS_SAMPLES_WRITE_ACCESS_KEY")
        self.hippius_samples_write_secret_key = _read_optional_str("HIPPIUS_SAMPLES_WRITE_SECRET_KEY")
        self.source_bucket = _read_str("HIPPIUS_SOURCE_BUCKET", "videos")
        self.samples_bucket = _read_str("HIPPIUS_SAMPLES_BUCKET", "samples")
        self.openai_api_key = _read_optional_str("OPENAI_API_KEY")
        self.min_video_size = _read_int("MIN_VIDEO_SIZE", 1_000_000)
        self.max_video_size = _read_int("MAX_VIDEO_SIZE", 200_000_000)
        self.clip_duration = _read_int("CLIP_DURATION", 5)
        self.max_video_history = _read_int("MAX_VIDEO_HISTORY", 50)
        self.chute_cache_ttl = _read_int("CHUTE_CACHE_TTL", 300)
        self.max_concurrent_miners = _read_int("MAX_CONCURRENT_MINERS", 5)
        self.database_url = _read_optional_str("DATABASE_URL")
        self.postgres_host = _read_str("POSTGRES_HOST", "localhost")
        self.postgres_port = _read_str("POSTGRES_PORT", "5432")
        self.postgres_user = _read_str("POSTGRES_USER", "leoma")
        self.postgres_password = _read_str("POSTGRES_PASSWORD", "leoma")
        self.postgres_db = _read_str("POSTGRES_DB", "leoma")
        self.hf_token = _read_optional_str("HF_TOKEN")
        self.model_hash_cache_ttl = _read_int("MODEL_HASH_CACHE_TTL", 3600)
        self.corpus_min_duration = _read_int("CORPUS_MIN_DURATION", 5)
        self.corpus_max_duration = _read_int("CORPUS_MAX_DURATION", 300)
        self.corpus_target_resolution = _read_str("CORPUS_TARGET_RESOLUTION", "720")
        self.corpus_max_filesize = _read_int("CORPUS_MAX_FILESIZE", 200_000_000)
        self.min_validator_stake = _read_float("MIN_VALIDATOR_STAKE", 1000.0)
        self.validator_sync_interval = _read_int("VALIDATOR_SYNC_INTERVAL", 600)


_settings_instance = Settings()
settings = _settings_instance

NETUID = settings.netuid
EPOCH_LEN = settings.epoch_len
REQUEST_TIMEOUT = settings.request_timeout
CHUTES_API_URL = settings.chutes_api_url
CHUTES_API_KEY = settings.chutes_api_key
WALLET_NAME = settings.wallet_name
HOTKEY_NAME = settings.hotkey_name
NETWORK = settings.network
HIPPIUS_ENDPOINT = settings.hippius_endpoint
HIPPIUS_REGION = settings.hippius_region
HIPPIUS_VIDEOS_READ_ACCESS_KEY = settings.hippius_videos_read_access_key
HIPPIUS_VIDEOS_READ_SECRET_KEY = settings.hippius_videos_read_secret_key
HIPPIUS_VIDEOS_WRITE_ACCESS_KEY = settings.hippius_videos_write_access_key
HIPPIUS_VIDEOS_WRITE_SECRET_KEY = settings.hippius_videos_write_secret_key
HIPPIUS_SAMPLES_READ_ACCESS_KEY = settings.hippius_samples_read_access_key
HIPPIUS_SAMPLES_READ_SECRET_KEY = settings.hippius_samples_read_secret_key
HIPPIUS_SAMPLES_WRITE_ACCESS_KEY = settings.hippius_samples_write_access_key
HIPPIUS_SAMPLES_WRITE_SECRET_KEY = settings.hippius_samples_write_secret_key
SOURCE_BUCKET = settings.source_bucket
SAMPLES_BUCKET = settings.samples_bucket
OPENAI_API_KEY = settings.openai_api_key
MIN_VIDEO_SIZE = settings.min_video_size
MAX_VIDEO_SIZE = settings.max_video_size
CLIP_DURATION = settings.clip_duration
MAX_VIDEO_HISTORY = settings.max_video_history
CHUTE_CACHE_TTL = settings.chute_cache_ttl
MAX_CONCURRENT_MINERS = settings.max_concurrent_miners
DATABASE_URL = settings.database_url
POSTGRES_HOST = settings.postgres_host
POSTGRES_PORT = settings.postgres_port
POSTGRES_USER = settings.postgres_user
POSTGRES_PASSWORD = settings.postgres_password
POSTGRES_DB = settings.postgres_db
HF_TOKEN = settings.hf_token
MODEL_HASH_CACHE_TTL = settings.model_hash_cache_ttl
CORPUS_MIN_DURATION = settings.corpus_min_duration
CORPUS_MAX_DURATION = settings.corpus_max_duration
CORPUS_TARGET_RESOLUTION = settings.corpus_target_resolution
CORPUS_MAX_FILESIZE = settings.corpus_max_filesize
MIN_VALIDATOR_STAKE = settings.min_validator_stake
VALIDATOR_SYNC_INTERVAL = settings.validator_sync_interval

# Ensure leoma logger has a handler when not configured by application
if not logger.handlers:
    _log_level = getattr(
        logging,
        os.environ.get("LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.setLevel(_log_level)
    logger.addHandler(_handler)
