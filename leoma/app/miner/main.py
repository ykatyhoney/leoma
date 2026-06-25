"""Miner command implementations: push (deploy I2V model to Chutes) and commit (model info on-chain)."""

from __future__ import annotations

import os
import sys
import json
import asyncio
import re
import textwrap
from pathlib import Path
from typing import Optional, Dict, Any

import aiohttp

from leoma.bootstrap import NETUID, CHUTES_API_KEY, WALLET_NAME, HOTKEY_NAME, NETWORK
from leoma.bootstrap import emit_log as log

_CHUTES_API_BASE = "https://api.chutes.ai"
_CHUTES_TIMEOUT_SECONDS = 10


def _chutes_headers(api_key: str) -> Dict[str, str]:
    """Build Chutes API auth headers."""
    return {"Authorization": f"Bearer {api_key}"}


def _trim_chute_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Remove large non-essential fields from chute payload."""
    for key in ("readme", "cords", "tagline", "instances"):
        info.pop(key, None)
    info.get("image", {}).pop("readme", None)
    return info


def _deploy_output_has_error(output: str) -> bool:
    """Detect deploy failure marker in CLI output."""
    match = re.search(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)",
        output,
    )
    return bool(match and match.group(2) == "ERROR")


def _commit_payload(model_name: str, model_revision: str, chute_id: str) -> str:
    """Build on-chain commitment payload."""
    return json.dumps(
        {
            "model_name": model_name,
            "model_revision": model_revision,
            "chute_id": chute_id,
        }
    )


async def get_chute_info(chute_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Get chute info from Chutes API, or None if the request fails."""
    url = f"{_CHUTES_API_BASE}/chutes/{chute_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=_chutes_headers(api_key),
                timeout=aiohttp.ClientTimeout(total=_CHUTES_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status != 200:
                    return None
                
                info = await resp.json()
                return _trim_chute_info(info)
    except Exception as e:
        log(f"Failed to fetch chute {chute_id}: {e}", "warn")
        return None


async def get_latest_chute_id(model_name: str, api_key: str) -> Optional[str]:
    """Get latest chute ID for a HuggingFace repository, or None if not found."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_CHUTES_API_BASE}/chutes/",
                headers=_chutes_headers(api_key),
                timeout=aiohttp.ClientTimeout(total=_CHUTES_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
    except Exception:
        return None
    
    chutes = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(chutes, list):
        return None

    for chute in reversed(chutes):
        if any(chute.get(k) == model_name for k in ("tagline", "readme", "name")):
            return chute.get("chute_id")
    return None


async def push_command(
    model_name: str,
    model_revision: str,
    chutes_api_key: Optional[str] = None,
    chute_user: Optional[str] = None,
) -> Dict[str, Any]:
    """Deploy I2V model to Chutes. Returns a result dict with success status and chute_id."""
    chutes_api_key = chutes_api_key or CHUTES_API_KEY
    chute_user = chute_user
    
    if not chutes_api_key:
        log("CHUTES_API_KEY not configured", "error")
        return {"success": False, "error": "CHUTES_API_KEY not configured"}
    
    if not chute_user:
        log("CHUTE_USER not configured", "error")
        return {"success": False, "error": "CHUTE_USER not configured"}
    
    log(f"Building Chute config for model_name={model_name} model_revision={model_revision}", "info")

    chutes_config = textwrap.dedent(f'''
import os
import uuid
import base64
import asyncio
import aiohttp
import tempfile
from typing import Optional, Literal
from io import BytesIO
from PIL import Image
from loguru import logger
from pydantic import BaseModel, Field
from fastapi import Response, HTTPException, status

from chutes.chute import Chute, NodeSelector


chute = Chute(
    username="{chute_user}",
    name="{model_name}",
    tagline="{model_name}",
    readme="{model_name}",
    revision="{model_revision}",
    image="leoma/video-gen:1.0",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=80,
    ),
    concurrency=4,
    shutdown_after_seconds=86400,
    allow_external_egress=False,
)


class I2VArgs(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the desired video.")
    image: str = Field(..., description="Image, either https URL or base64 encoded data.")
    negative_prompt: Optional[str] = Field("low quality, blurry, distorted", description="Negative prompt.")
    num_frames: int = Field(81, ge=1, le=161, description="Number of frames to generate.")
    fps: Optional[int] = Field(16, ge=1, le=30, description="Output FPS (default 16)")
    seed: Optional[int] = Field(None, description="Generation seed.")
    guidance_scale: Optional[float] = Field(5.0, ge=1.0, le=20.0, description="Guidance scale.")
    num_inference_steps: Optional[int] = Field(30, ge=1, le=100, description="Number of inference steps.")


async def download_image(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download from {{url}}")
            return await resp.read()


async def get_image(image_data: str):
    image_bytes = None
    if image_data.startswith("http"):
        image_bytes = await download_image(image_data)
    else:
        image_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


class Predictor:
    def __init__(self):
        import torch
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video

        self.torch = torch
        self.export_to_video = export_to_video
        self.pipe = DiffusionPipeline.from_pretrained("{model_name}", revision="{model_revision}", torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")

    def predict(self, prompt, image, negative_prompt, num_frames, fps, seed, guidance_scale, num_inference_steps):
        generator = (
            self.torch.Generator("cuda").manual_seed(seed)
            if isinstance(seed, int)
            else self.torch.Generator(device="cuda").manual_seed(42)
        )
        width, height = 832, 480
        image = image.resize((width, height))
        
        with self.torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
            frames = output.frames[0]

        output_file = f"/tmp/{{uuid.uuid4()}}.mp4"
        try:
            self.export_to_video(frames, output_file, fps=fps)
            buffer = BytesIO()
            with open(output_file, "rb") as infile:
                buffer.write(infile.read())
            buffer.seek(0)
            return Response(
                content=buffer.getvalue(),
                media_type="video/mp4",
                headers={{"Content-Disposition": f"attachment; filename=\\"{{os.path.basename(output_file)}}\\""}}
            )
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)


@chute.on_startup()
async def initialize(self):
    self.predictor = Predictor()
    self.lock = asyncio.Lock()

@chute.cord(
    public_api_path="/generate",
    public_api_method="POST",
    stream=False,
    output_content_type="video/mp4",
)
async def generate(self, args: I2VArgs):
    try:
        image_data = await get_image(args.image)
    except Exception as exc:
        logger.error(f"Failed to extract image data: {{str(exc)}}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to extract image: {{str(exc)}}")

    async with self.lock:
        return self.predictor.predict(
            args.prompt, image_data, args.negative_prompt, args.num_frames,
            args.fps, args.seed, args.guidance_scale, args.num_inference_steps,
        )
''')
    
    tmp_file = Path("tmp_chute.py")
    tmp_file.write_text(chutes_config)
    log(f"Wrote Chute config to {tmp_file}", "info")

    cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
    env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()
        
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="ignore")
        log(output, "info")
        
        if _deploy_output_has_error(output):
            log("Chutes deploy failed with error log", "error")
            tmp_file.unlink(missing_ok=True)
            return {"success": False, "error": "Chutes deploy failed"}
        
        if proc.returncode != 0:
            log(f"Chutes deploy failed with code {proc.returncode}", "error")
            tmp_file.unlink(missing_ok=True)
            return {"success": False, "error": f"Exit code {proc.returncode}"}
        
        tmp_file.unlink(missing_ok=True)
        log("Chute deployment successful", "success")

        chute_id = await get_latest_chute_id(model_name, api_key=chutes_api_key)
        log(f"Chute ID: {chute_id}", "info")
        
        chute_info = await get_chute_info(chute_id, chutes_api_key) if chute_id else None
        
        result = {
            "success": bool(chute_id),
            "chute_id": chute_id,
            "chute": chute_info,
            "model_name": model_name,
            "model_revision": model_revision,
        }
        log(f"Deployed to Chutes: {chute_id}", "success")
        return result
    
    except Exception as e:
        log(f"Chutes deployment failed: {e}", "error")
        tmp_file.unlink(missing_ok=True)
        return {"success": False, "error": str(e)}


async def commit_command(
    model_name: str,
    model_revision: str,
    chute_id: str,
    coldkey: Optional[str] = None,
    hotkey: Optional[str] = None,
) -> Dict[str, Any]:
    """Commit model info to the blockchain. Returns a result dict with success status."""
    import bittensor as bt
    
    cold = coldkey or WALLET_NAME
    hot = hotkey or HOTKEY_NAME
    wallet = bt.Wallet(name=cold, hotkey=hot)
    
    log(f"Committing: {model_name}@{model_revision[:8]} (chute: {chute_id})", "info")
    log(f"Using wallet: {wallet.hotkey.ss58_address[:16]}...", "info")
    
    async def _commit():
        subtensor = bt.AsyncSubtensor(network=NETWORK)

        log(f"Subtensor network configured to {NETWORK}", "info")
        data = _commit_payload(model_name, model_revision, chute_id)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await subtensor.set_reveal_commitment(
                    wallet=wallet,
                    netuid=NETUID,
                    data=data,
                    blocks_until_reveal=1,
                )
                return True
            except Exception as e:
                if "SpaceLimitExceeded" in str(e):
                    log("Space limit exceeded, waiting for next block...", "warn")
                    await asyncio.sleep(12)
                elif attempt < max_retries - 1:
                    log(f"Commit attempt {attempt + 1} failed: {e}", "warn")
                    await asyncio.sleep(6)
                else:
                    raise
        return False
    
    try:
        success = await _commit()
        
        if success:
            result = {
                "success": True,
                "model_name": model_name,
                "model_revision": model_revision,
                "chute_id": chute_id,
            }
            log("Commit successful", "success")
        else:
            result = {"success": False, "error": "Commit failed after retries"}
            log("Commit failed", "error")
        
        return result
    
    except Exception as e:
        log(f"Commit failed: {e}", "error")
        return {"success": False, "error": str(e)}
