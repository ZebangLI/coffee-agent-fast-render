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

Return these keys:
- drink: one of latte, americano, cold brew, or null
- quantity: integer from 1 to 10
- temperature: hot, iced, or null
- size: small, medium, large, or medium if not mentioned

Map natural coffee requests to the closest available drink:
- coffee, black coffee, espresso, drip coffee, regular coffee, 美式, 黑咖啡, 咖啡 -> americano
- latte, cappuccino, flat white, mocha, macchiato, milk coffee, 拿铁, 卡布奇诺, 摩卡, 奶咖 -> latte
- cold brew, iced coffee, 冷萃, 冰咖啡 -> cold brew

If the user is not asking for coffee, return {"drink": null}.
If the user asks for coffee but the exact product is unclear, choose the closest available drink.
Use null for unknown optional fields.
"""

SELECTION_PROMPT = """
You interpret whether a user is selecting one item from a numbered recommendation list.
Return only JSON with this exact key: selected_index.
selected_index is zero-based. Examples:
- first one, order the first, number one -> 0
- second one, buy option 2, the Starbucks one -> 1
- third, last one -> 2
If the user is asking for a new coffee search instead of selecting an existing option, return {"selected_index": null}.
If unclear, return {"selected_index": null}.
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
                    "drink, quantity, temperature, size, milk, budget, pickup_time. "
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


def parse_selection(message: str, option_count: int) -> int | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    payload = {
        "model": os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SELECTION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"There are {option_count} numbered coffee options. "
                    f"User message: {message}"
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
        data = json.loads(_strip_code_fence(content))
        raw_index = data.get("selected_index")
        if raw_index is None:
            return None
        selected_index = int(raw_index)
        if 0 <= selected_index < option_count:
            return selected_index
        return None
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError):
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

    drink = _normalize_drink(str(raw_drink))
    if drink in {"", "none", "null", "unknown", "unclear"}:
        return None

    if drink not in {"latte", "americano", "cold brew"}:
        return None

    temperature = data.get("temperature")
    return DrinkIntent(
        drink=drink,
        temperature=str(temperature).lower() if temperature else None,
        size=str(data.get("size") or "medium").lower(),
        quantity=_parse_quantity(data.get("quantity")),
    )


def _normalize_drink(raw_drink: str) -> str:
    drink = raw_drink.strip().lower()
    if drink in {"", "none", "null", "unknown", "unclear"}:
        return drink

    if any(term in drink for term in ("cold brew", "iced coffee", "冷萃", "冰咖啡")):
        return "cold brew"
    if any(
        term in drink
        for term in (
            "latte",
            "cappuccino",
            "flat white",
            "mocha",
            "macchiato",
            "milk coffee",
            "拿铁",
            "卡布奇诺",
            "摩卡",
            "奶咖",
        )
    ):
        return "latte"
    if any(
        term in drink
        for term in (
            "americano",
            "coffee",
            "espresso",
            "drip",
            "black coffee",
            "regular coffee",
            "美式",
            "黑咖啡",
            "咖啡",
        )
    ):
        return "americano"
    return drink


def _parse_quantity(raw_quantity: Any) -> int:
    if raw_quantity is None:
        return 1

    chinese_numbers = {
        "一": 1,
        "两": 2,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if isinstance(raw_quantity, str):
        cleaned = raw_quantity.strip().lower()
        if cleaned in chinese_numbers:
            return chinese_numbers[cleaned]
        cleaned = "".join(ch for ch in cleaned if ch.isdigit())
        if not cleaned:
            return 1
        raw_quantity = cleaned

    try:
        quantity = int(raw_quantity)
    except (TypeError, ValueError):
        return 1
    return max(1, min(quantity, 10))


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
