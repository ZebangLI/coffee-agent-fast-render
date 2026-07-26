from __future__ import annotations

import json
import os
import re
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
Return only JSON with these exact keys: selected_index, quantity.
selected_index is zero-based. Examples:
- first one, order the first, number one -> 0
- second one, buy option 2, the Starbucks one -> 1
- third, last one -> 2
quantity is the number of drinks the user explicitly mentions, from 1 to 10.
If the user does not mention a quantity, use null.
Examples:
- 2 coffee at third store -> {"selected_index": 2, "quantity": 2}
- order the second one -> {"selected_index": 1, "quantity": null}
- buy three from first -> {"selected_index": 0, "quantity": 3}
If the user is asking for a new coffee search instead of selecting an existing option, return {"selected_index": null, "quantity": null}.
If unclear, return {"selected_index": null, "quantity": null}.
"""

AGENT_PROMPT = """
You are the decision layer for a coffee ordering agent.
Return only JSON. Do not create orders. Do not invent stores or products.

Available product families are only:
- latte
- americano
- cold brew

Use the provided context:
- options: current recommendation cards, if any
- pending_quantity: quantity already chosen on the current cards
- last_order: the most recent completed order, if any

Return these keys:
- action: one of search, select_option, update_quantity, reorder_last, unsupported, chat
- drink: latte, americano, cold brew, or null
- selected_index: zero-based option index, or null
- quantity: integer 1-10 only if the user explicitly mentions a new quantity, otherwise null
- message: short user-facing message when action is chat or unsupported

Rules:
- If the user asks for coffee, map it to the closest available drink.
- If the user names a current option's store or product, use select_option.
- If the user says first/second/third option, use select_option.
- If the user only changes quantity while options exist, use update_quantity.
- If there are no options and the user says one more, again, same one, reorder, or similar, use reorder_last when last_order exists.
- If the user asks for a product outside coffee, use unsupported.
- If the user is just chatting, use chat.
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


def parse_selection(message: str, option_count: int) -> tuple[int | None, int | None]:
    fallback_index = _selection_from_text(message, option_count)
    fallback_quantity = _quantity_from_text(message)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return fallback_index, fallback_quantity

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
        explicit_quantity = fallback_quantity
        raw_quantity = explicit_quantity if explicit_quantity is not None else data.get("quantity")
        quantity = _parse_quantity(raw_quantity) if raw_quantity not in (None, "", "null") else None
        raw_index = data.get("selected_index")
        if raw_index is None:
            return fallback_index, quantity
        selected_index = int(raw_index)
        if 0 <= selected_index < option_count:
            return selected_index, quantity
        return fallback_index, quantity
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return fallback_index, fallback_quantity


def decide_agent_action(message: str, context: dict[str, Any]) -> dict[str, Any]:
    fallback = _fallback_agent_action(message, context)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return fallback

    safe_context = {
        "options": context.get("options") or [],
        "pending_quantity": context.get("pending_quantity") or 1,
        "last_order": context.get("last_order"),
    }
    payload = {
        "model": os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": AGENT_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"message": message, "context": safe_context},
                    ensure_ascii=False,
                    separators=(",", ":"),
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
        decision = json.loads(_strip_code_fence(content))
        return _sanitize_agent_decision(decision, message, context, fallback)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, IndexError):
        return fallback


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


def _quantity_from_text(message: str) -> int | None:
    text = message.lower()
    for pattern in (
        r"\b([1-9]|10)\s*(?:cups?|coffees?|drinks?|lattes?|americanos?)\b",
        r"\b([1-9]|10)\s+more\b",
        r"\b(?:buy|order|get|want)\s+([1-9]|10)\b",
        r"\b(?:change|switch|make|update)\s+(?:it\s+)?(?:to\s+)?([1-9]|10)\b",
    ):
        match = re.search(pattern, text)
        if match:
            return _parse_quantity(match.group(1))

    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for word, quantity in word_numbers.items():
        if re.search(rf"\b{word}\s+(?:cups?|coffees?|drinks?|lattes?|americanos?)\b", text):
            return quantity
        if re.search(rf"\b{word}\s+more\b", text):
            return quantity
        if re.search(rf"\b(?:buy|order|get|want)\s+{word}\b", text):
            return quantity

    chinese_numbers = {
        "\u4e00": 1,
        "\u4e24": 2,
        "\u4e8c": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
        "\u5341": 10,
    }
    for word, quantity in chinese_numbers.items():
        if re.search(rf"(?<!\u7b2c){word}\s*(?:\u676f|\u4e2a)", text):
            return quantity
    return None


def _fallback_agent_action(message: str, context: dict[str, Any]) -> dict[str, Any]:
    options = context.get("options") or []
    option_count = len(options)
    quantity = _quantity_from_text(message)

    if option_count:
        selected_index = _option_index_from_context(message, options)
        if selected_index is None:
            selected_index = _selection_from_text(message, option_count)
        if selected_index is not None:
            return {
                "action": "select_option",
                "selected_index": selected_index,
                "quantity": quantity,
            }
        if quantity is not None and _looks_like_quantity_change(message):
            return {"action": "update_quantity", "quantity": quantity}

    last_order = context.get("last_order") or {}
    if last_order and _looks_like_reorder(message):
        return {
            "action": "reorder_last",
            "product_id": last_order.get("product_id"),
            "quantity": quantity or 1,
        }

    if _looks_like_unsupported_product(message):
        return {
            "action": "unsupported",
            "message": "Sorry, this demo can only order coffee right now. That product is not available yet.",
        }

    drink = _fallback_drink_from_text(message)
    if drink:
        return {
            "action": "search",
            "drink": drink,
            "quantity": quantity or 1,
        }

    return {
        "action": "chat",
        "message": "No problem. Tell me what coffee you want when you are ready.",
    }


def _sanitize_agent_decision(
    decision: dict[str, Any],
    message: str,
    context: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    action = str(decision.get("action") or "").strip().lower()
    if action not in {"search", "select_option", "update_quantity", "reorder_last", "unsupported", "chat"}:
        return fallback

    # Local context wins for high-confidence actions. Small models sometimes
    # treats short order phrases like "cold brew 3 cups" as chatter, but the
    # deterministic parser can identify supported products and follow-ups.
    fallback_action = fallback.get("action")
    if fallback_action in {"search", "select_option", "update_quantity", "reorder_last", "unsupported"} and action != fallback_action:
        return fallback

    quantity = decision.get("quantity")
    parsed_quantity = None if quantity in (None, "", "null") else _parse_quantity(quantity)
    fallback_quantity = fallback.get("quantity")
    options = context.get("options") or []

    if action == "select_option":
        selected_index = decision.get("selected_index")
        try:
            selected_index = int(selected_index)
        except (TypeError, ValueError):
            selected_index = _option_index_from_context(message, options)
        if selected_index is None or not (0 <= selected_index < len(options)):
            return fallback
        return {"action": action, "selected_index": selected_index, "quantity": parsed_quantity or fallback_quantity}

    if action == "update_quantity":
        if parsed_quantity is None:
            if fallback_quantity is not None:
                return {"action": action, "quantity": fallback_quantity}
            return fallback
        return {"action": action, "quantity": parsed_quantity}

    if action == "reorder_last":
        last_order = context.get("last_order") or {}
        product_id = decision.get("product_id") or last_order.get("product_id")
        if not product_id:
            return fallback
        return {"action": action, "product_id": product_id, "quantity": parsed_quantity or fallback_quantity or 1}

    if action == "search":
        drink = _normalize_drink(str(decision.get("drink") or ""))
        if drink not in {"latte", "americano", "cold brew"}:
            return fallback
        return {"action": action, "drink": drink, "quantity": parsed_quantity or fallback_quantity or 1}

    if action == "unsupported":
        return {
            "action": action,
            "message": str(
                decision.get("message")
                or "Sorry, this demo can only order coffee right now. That product is not available yet."
            ),
        }

    return {
        "action": "chat",
        "message": str(decision.get("message") or fallback.get("message") or "Tell me what coffee you want."),
    }


def _option_index_from_context(message: str, options: list[dict[str, Any]]) -> int | None:
    text = _compact_text(message)
    if not text:
        return None
    for index, option in enumerate(options):
        for key in ("shop_name", "store_name", "product_name", "shop_id", "product_id"):
            value = option.get(key)
            if value and _compact_text(str(value)) in text:
                return index
    return None


def _fallback_drink_from_text(message: str) -> str | None:
    text = message.lower()
    if any(term in text for term in ("burger", "hamburger", "pizza", "sandwich", "tea")):
        return None
    if any(term in text for term in ("cold brew", "iced coffee")):
        return "cold brew"
    if any(term in text for term in ("latte", "mocha", "cappuccino", "flat white", "macchiato")):
        return "latte"
    if any(term in text for term in ("coffee", "americano", "espresso", "black coffee", "drip")):
        return "americano"
    if any(term in message for term in ("\u5496\u5561", "\u7f8e\u5f0f", "\u62ff\u94c1", "\u51b7\u8403", "\u51b0\u5496\u5561")):
        if "\u62ff\u94c1" in message:
            return "latte"
        if "\u51b7\u8403" in message or "\u51b0\u5496\u5561" in message:
            return "cold brew"
        return "americano"
    return None


def _looks_like_quantity_change(message: str) -> bool:
    text = message.lower()
    return any(term in text for term in ("change", "switch", "make", "update")) or "\u6539" in message


def _looks_like_reorder(message: str) -> bool:
    text = message.lower()
    number_words = "one|two|three|four|five|six|seven|eight|nine|ten"
    if re.search(rf"\b(?:[1-9]|10|{number_words})\s+more\b", text):
        return True
    return any(
        term in text
        for term in (
            "one more",
            "another",
            "again",
            "same",
            "reorder",
            "more coffee",
            "more latte",
        )
    ) or any(term in message for term in ("\u518d\u6765", "\u518d\u8981", "\u4e00\u6837", "\u8fd8\u8981"))


def _looks_like_unsupported_product(message: str) -> bool:
    text = message.lower()
    return any(term in text for term in ("burger", "hamburger", "pizza", "sandwich", "tea")) or any(
        term in message for term in ("\u6c49\u5821", "\u62ab\u8428", "\u4e09\u660e\u6cbb", "\u5976\u8336")
    )


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def _selection_from_text(message: str, option_count: int) -> int | None:
    text = message.lower()
    ordinals = (
        ("first", "1st"),
        ("second", "2nd"),
        ("third", "3rd"),
        ("fourth", "4th"),
        ("fifth", "5th"),
        ("sixth", "6th"),
        ("seventh", "7th"),
        ("eighth", "8th"),
        ("ninth", "9th"),
        ("tenth", "10th"),
    )
    for index, words in enumerate(ordinals[:option_count]):
        if any(re.search(rf"\b{re.escape(word)}\b", text) for word in words):
            return index

    chinese_ordinals = (
        "\u7b2c\u4e00",
        "\u7b2c\u4e8c",
        "\u7b2c\u4e09",
        "\u7b2c\u56db",
        "\u7b2c\u4e94",
        "\u7b2c\u516d",
        "\u7b2c\u4e03",
        "\u7b2c\u516b",
        "\u7b2c\u4e5d",
        "\u7b2c\u5341",
    )
    for index, word in enumerate(chinese_ordinals[:option_count]):
        if word in text:
            return index

    match = re.search(r"\b(?:option|store|number|#)\s*([1-9]|10)\b", text)
    if match:
        selected_index = int(match.group(1)) - 1
        if 0 <= selected_index < option_count:
            return selected_index
    match = re.search(r"(?:\u9009|\u7b2c)\s*([1-9]|10)\s*(?:\u4e2a|\u9879|\u9009\u9879|\u5bb6|\u5e97)", text)
    if match:
        selected_index = int(match.group(1)) - 1
        if 0 <= selected_index < option_count:
            return selected_index
    return None


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
