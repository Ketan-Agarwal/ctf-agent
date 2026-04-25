"""VLM sidecar — uses Gemini Flash Lite to describe images for non-vision solvers."""

from __future__ import annotations

import base64
import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Model to use for VLM sidecar
VLM_MODEL = "gemini-3.1-flash-lite-preview"

VLM_PROMPT = (
    "You are analyzing an image from a CTF (Capture The Flag) challenge. "
    "Describe the image in detail, focusing on:\n"
    "- Any visible text, characters, or symbols\n"
    "- Hidden or steganographic content (unusual patterns, noise, LSB artifacts)\n"
    "- Color patterns, layers, or overlays that might encode data\n"
    "- QR codes, barcodes, or encoded visual elements\n"
    "- Metadata clues visible in the image\n"
    "- Any anomalies or suspicious patterns\n\n"
    "Be thorough — the flag may be hidden in the image."
)


async def describe_image(image_bytes: bytes, media_type: str) -> str:
    """Send image to Gemini Flash Lite and get a text description.

    Returns a text description of the image, or an error message.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "VLM sidecar: GEMINI_API_KEY not set. Use bash tools (steghide, zsteg, exiftool, strings) instead."

    b64_data = base64.b64encode(image_bytes).decode()

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": VLM_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": media_type,
                            "data": b64_data,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 4096,
            "temperature": 0.2,
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{VLM_MODEL}"
        f":generateContent?key={api_key}"
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            error_body = resp.text[:500]
            logger.warning(f"VLM sidecar error: HTTP {resp.status_code} — {error_body}")
            return f"VLM sidecar error (HTTP {resp.status_code}). Use bash tools instead."

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "VLM sidecar: no response from model. Use bash tools instead."

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p["text"] for p in parts if "text" in p]
        description = "\n".join(text_parts)

        if not description:
            return "VLM sidecar: empty response. Use bash tools instead."

        logger.info(f"VLM sidecar described image ({len(image_bytes)} bytes, {media_type})")
        return f"[VLM Image Analysis — {VLM_MODEL}]\n\n{description}"

    except Exception as e:
        logger.warning(f"VLM sidecar failed: {e}")
        return f"VLM sidecar error: {e}. Use bash tools instead."
