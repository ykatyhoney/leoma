FROM python:3.12-slim

# ffprobe/ffmpeg for video processing (eval server); curl for healthchecks; build-essential for compiling Python deps if needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY leoma.py ./
COPY leoma ./leoma

# Install package (non-editable for production image)
RUN uv pip install --system --no-cache .

# Override in compose: leoma serve (validator) or leoma api (API service)
CMD ["leoma"]
