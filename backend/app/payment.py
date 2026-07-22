from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any
from uuid import uuid4

DEFAULT_AIGENTIC_PAY_URL = "https://aigenticpay.onrender.com"


def record_payment(payload: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("AIGENTIC_PAY_ENABLED", "0") == "0":
        return {"payment_status": "mock_paid", "tx_hash": f"mock_tx_{uuid4().hex[:12]}"}

    card = request_virtual_card(payload) if os.environ.get("AIGENTIC_PAY_USE_VIRTUAL_CARD", "0") == "1" else None
    order_hash = hash_payload({**payload, "virtual_card_last4": _card_last4(card) if card else None})
    audit = post_onchain_hash(order_hash)
    if audit is None:
        if os.environ.get("AIGENTIC_PAY_REQUIRE_ONCHAIN", "0") == "1":
            raise ValueError("AigenticPay on-chain audit is unavailable")
        return {"payment_status": "mock_paid", "tx_hash": f"order_hash_{order_hash}"}

    result = {
        "payment_status": "onchain_recorded",
        "tx_hash": audit["tx_hash"],
        "explorer_url": audit.get("explorer_url"),
    }
    if card:
        result["virtual_card_last4"] = _card_last4(card)
    return result


def request_virtual_card(payload: dict[str, Any]) -> dict[str, Any]:
    email = os.environ.get("AIGENTIC_PAY_USER_EMAIL")
    if not email:
        raise ValueError("AIGENTIC_PAY_USER_EMAIL is required when virtual card is enabled")

    request = urllib.request.Request(
        f"{_base_url()}/api/virtual_card",
        data=json.dumps(
            {
                "email": email,
                "merchant_name": payload["shop_name"],
                "amount": payload["amount"],
            }
        ).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.environ.get("AIGENTIC_PAY_TIMEOUT_SECONDS", "30"))) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ValueError(f"AigenticPay virtual card rejected the order: {exc.read().decode(errors='replace')}") from exc
    except (OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"AigenticPay virtual card is unavailable: {exc}") from exc


def post_onchain_hash(data_hash: str) -> dict[str, Any] | None:
    attempts = int(os.environ.get("AIGENTIC_PAY_RETRY_ATTEMPTS", "2"))
    timeout = float(os.environ.get("AIGENTIC_PAY_TIMEOUT_SECONDS", "30"))
    for attempt in range(attempts):
        request = urllib.request.Request(
            f"{_base_url()}/api/onchain_audit",
            data=json.dumps({"hash": data_hash}).encode("utf-8"),
            headers=_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body if body.get("tx_hash") else None
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                return None
        except (OSError, TimeoutError, json.JSONDecodeError):
            pass
        if attempt + 1 < attempts:
            time.sleep(1)
    return None


def hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "0x" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-API-Key": os.environ.get("AIGENTIC_PAY_API_KEY", ""),
    }


def _base_url() -> str:
    return os.environ.get("AIGENTIC_PAY_BASE_URL", DEFAULT_AIGENTIC_PAY_URL).rstrip("/")


def _card_last4(card: dict[str, Any] | None) -> str | None:
    if not card:
        return None
    number = str(card.get("card_number", "")).replace(" ", "")
    return number[-4:] if len(number) >= 4 else None
