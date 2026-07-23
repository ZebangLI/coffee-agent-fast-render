from __future__ import annotations

import json
import os
import uuid
import urllib.error
import urllib.request
from typing import Any

from .models import DrinkIntent

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
DEFAULT_TRANSCRIPTION_PROMPT = (
    "Coffee ordering app. Common phrases include: I want a latte, "
    "iced americano, cold brew, Starbucks, Campus Cafe, Local Cafe, order this."
)

SYSTEM_PROMPT = """
You extract coffee ordering intent from a short user message.
Return only JSON. Do not create orders. Do not call tools.

Allowed drink values:
- latte
- americano
- cold brew

If the user is not asking for coffee, return {"drink": null}.
If the drink is unclear, return {"drink": null}.
Use null for unknown optional fields.
"""


def parse_intent(message: str) -> DrinkIntent | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    payload = {
        "model": os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract coffee intent as JSON with keys: "
                    "drink, temperature, size, milk, budget, pickup_time. "
                    f"Message: {message}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "coffee-agent-fast-render/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(os.environ.get("GROQ_TIMEOUT_SECONDS", "12"))) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return _intent_from_json(json.loads(_strip_code_fence(content)))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, IndexError):
        return None


def transcribe_audio(audio: bytes, filename: str, content_type: str) -> str | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    boundary = f"----coffee-agent-{uuid.uuid4().hex}"
    fields = {
        "model": os.environ.get("GROQ_TRANSCRIPTION_MODEL", DEFAULT_GROQ_TRANSCRIPTION_MODEL),
        "response_format": "json",
        "temperature": "0",
        "prompt": os.environ.get("GROQ_TRANSCRIPTION_PROMPT", DEFAULT_TRANSCRIPTION_PROMPT),
    }
    language = os.environ.get("GROQ_TRANSCRIPTION_LANGUAGE", "en").strip()
    if language:
        fields["language"] = language

    body = _multipart_body(
        boundary,
        fields=fields,
        files={
            "file": {
                "filename": filename or "voice.webm",
                "content_type": content_type or "audio/webm",
                "content": audio,
            }
        },
    )

    try:
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "coffee-agent-fast-render/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=float(os.environ.get("GROQ_TRANSCRIPTION_TIMEOUT_SECONDS", "20")),
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = str(data.get("text") or "").strip()
        return text or None
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError):
        return None


def llm_status() -> dict[str, Any]:
    return {
        "enabled": bool(os.environ.get("GROQ_API_KEY")),
        "provider": "groq",
        "model": os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "transcription_model": os.environ.get("GROQ_TRANSCRIPTION_MODEL", DEFAULT_GROQ_TRANSCRIPTION_MODEL),
        "transcription_language": os.environ.get("GROQ_TRANSCRIPTION_LANGUAGE", "en"),
    }


def _intent_from_json(data: dict[str, Any]) -> DrinkIntent | None:
    raw_drink = data.get("drink")
    if raw_drink is None:
        return None

    drink = str(raw_drink).lower()
    if drink in {"", "none", "null", "unknown", "unclear"}:
        return None

    if drink not in {"latte", "americano", "cold brew"}:
        return None

    temperature = data.get("temperature")
    return DrinkIntent(
        drink=drink,
        temperature=str(temperature).lower() if temperature else None,
        size=str(data.get("size") or "medium").lower(),
    )


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def _multipart_body(
    boundary: str,
    fields: dict[str, str],
    files: dict[str, dict[str, bytes | str]],
) -> bytes:
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                f"{value}\r\n".encode("utf-8"),
            ]
        )
    for name, file_info in files.items():
        filename = str(file_info["filename"])
        content_type = str(file_info["content_type"])
        content = file_info["content"]
        if not isinstance(content, bytes):
            raise TypeError("Multipart file content must be bytes")
        lines.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                content,
                b"\r\n",
            ]
        )
    lines.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(lines)
