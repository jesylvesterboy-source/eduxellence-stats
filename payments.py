"""
Eduxellence Payments Engine — Phase 5
========================================
Routes Nigerian users to Paystack, international users to Lemon Squeezy.
Initializes transactions, verifies webhooks (HMAC signature checks),
and credits the user's wallet on confirmed payment.

Both gateways called via urllib (stdlib) — zero new pip dependencies.

Flow:
  1. POST /api/payments/initiate  -> returns a checkout URL for the right gateway
  2. User pays on Paystack/Lemon Squeezy's hosted page
  3. Gateway calls our webhook -> we verify signature -> credit the wallet
  4. User is redirected back to a success page

by Eduxellence Analytics · https://eduxellence.org
"""

import os, json, hmac, hashlib, uuid, ssl, urllib.request, urllib.error
from typing import Optional

PAYSTACK_SECRET_KEY = os.environ.get("PAYSTACK_SECRET_KEY", "")
PAYSTACK_PUBLIC_KEY = os.environ.get("PAYSTACK_PUBLIC_KEY", "")
LEMONSQUEEZY_API_KEY = os.environ.get("LEMONSQUEEZY_API_KEY", "")
LEMONSQUEEZY_STORE_ID = os.environ.get("LEMONSQUEEZY_STORE_ID", "")
LEMONSQUEEZY_WEBHOOK_SECRET = os.environ.get("LEMONSQUEEZY_WEBHOOK_SECRET", "")
LEMONSQUEEZY_VARIANT_ID = os.environ.get("LEMONSQUEEZY_VARIANT_ID", "")

SITE_URL = os.environ.get("SITE_URL", "https://eduxellence.org")

CREDIT_PACKS = [
    {"id": "pack_5",  "credits": 5,  "usd": 5,  "label": "5 Credits"},
    {"id": "pack_15", "credits": 15, "usd": 13, "label": "15 Credits", "badge": "Save $2"},
    {"id": "pack_40", "credits": 40, "usd": 32, "label": "40 Credits", "badge": "Save $8 — Best Value"},
    {"id": "pack_100","credits": 100,"usd": 75, "label": "100 Credits", "badge": "Save $25"},
]

def get_pack(pack_id: str) -> Optional[dict]:
    for p in CREDIT_PACKS:
        if p["id"] == pack_id:
            return p
    return None


def _http_json(url: str, method: str = "GET", payload: Optional[dict] = None,
                headers: Optional[dict] = None, timeout: int = 12):
    ctx = ssl.create_default_context()
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method,
                                  headers=headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
        except Exception:
            err_body = {"message": str(e)}
        return e.code, err_body
    except Exception as e:
        return 0, {"message": str(e)}


def select_gateway(country_code: Optional[str]) -> str:
    """Nigeria -> Paystack. Everyone else -> Lemon Squeezy."""
    if country_code and country_code.upper() == "NG":
        return "paystack"
    return "lemonsqueezy"


# ── Paystack ─────────────────────────────────────────────────────────────

def paystack_initiate(email: str, amount_usd: float, user_id: str,
                       pack_id: str, ngn_amount: Optional[float] = None) -> dict:
    if not PAYSTACK_SECRET_KEY:
        return {"ok": False, "error": "Paystack is not configured (missing PAYSTACK_SECRET_KEY)."}

    if ngn_amount is None:
        from pricing import convert_usd
        ngn_amount = convert_usd(amount_usd, "NGN")["converted"]

    reference = f"edux_{uuid.uuid4().hex[:16]}"
    payload = {
        "email": email,
        "amount": int(round(ngn_amount * 100)),
        "currency": "NGN",
        "reference": reference,
        "callback_url": f"{SITE_URL}/payment/success",
        "metadata": {
            "user_id": user_id,
            "pack_id": pack_id,
            "amount_usd": amount_usd,
        },
    }
    status, body = _http_json(
        "https://api.paystack.co/transaction/initialize",
        method="POST", payload=payload,
        headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
                 "Content-Type": "application/json"},
    )
    if status == 200 and body.get("status"):
        data = body["data"]
        return {"ok": True, "gateway": "paystack",
                "checkout_url": data["authorization_url"], "reference": data["reference"]}
    return {"ok": False, "error": body.get("message", "Paystack initialization failed."), "raw": body}


def paystack_verify(reference: str) -> dict:
    if not PAYSTACK_SECRET_KEY:
        return {"ok": False, "error": "Paystack is not configured."}
    status, body = _http_json(
        f"https://api.paystack.co/transaction/verify/{reference}",
        method="GET",
        headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"},
    )
    if status == 200 and body.get("status") and body["data"]["status"] == "success":
        data = body["data"]
        return {"ok": True, "amount_kobo": data["amount"], "currency": data["currency"],
                "metadata": data.get("metadata", {}), "reference": reference,
                "paid_at": data.get("paid_at")}
    return {"ok": False, "error": "Payment not verified or not successful.", "raw": body}


def verify_paystack_webhook_signature(raw_body: bytes, signature_header: str) -> bool:
    if not PAYSTACK_SECRET_KEY or not signature_header:
        return False
    computed = hmac.new(PAYSTACK_SECRET_KEY.encode("utf-8"), raw_body, hashlib.sha512).hexdigest()
    return hmac.compare_digest(computed, signature_header)


# ── Lemon Squeezy ────────────────────────────────────────────────────────

def lemonsqueezy_initiate(email: str, amount_usd: float, user_id: str,
                           pack_id: str) -> dict:
    if not LEMONSQUEEZY_API_KEY or not LEMONSQUEEZY_STORE_ID:
        return {"ok": False, "error": "Lemon Squeezy is not configured (missing API key or store ID)."}
    variant_id = LEMONSQUEEZY_VARIANT_ID
    if not variant_id:
        return {"ok": False, "error": "Lemon Squeezy variant ID not configured."}

    payload = {
        "data": {
            "type": "checkouts",
            "attributes": {
                "custom_price": int(round(amount_usd * 100)),
                "product_options": {
                    "name": f"Eduxellence Credits — {pack_id}",
                    "redirect_url": f"{SITE_URL}/payment/success",
                },
                "checkout_data": {
                    "email": email,
                    "custom": {"user_id": user_id, "pack_id": pack_id, "amount_usd": str(amount_usd)},
                },
            },
            "relationships": {
                "store":   {"data": {"type": "stores",   "id": str(LEMONSQUEEZY_STORE_ID)}},
                "variant": {"data": {"type": "variants",  "id": str(variant_id)}},
            },
        }
    }
    status, body = _http_json(
        "https://api.lemonsqueezy.com/v1/checkouts",
        method="POST", payload=payload,
        headers={"Authorization": f"Bearer {LEMONSQUEEZY_API_KEY}",
                 "Content-Type": "application/vnd.api+json",
                 "Accept": "application/vnd.api+json"},
    )
    if status in (200, 201) and body.get("data"):
        checkout_url = body["data"]["attributes"]["url"]
        return {"ok": True, "gateway": "lemonsqueezy", "checkout_url": checkout_url,
                "reference": body["data"]["id"]}
    return {"ok": False,
            "error": body.get("errors", [{}])[0].get("detail", "Lemon Squeezy checkout failed."),
            "raw": body}


def verify_lemonsqueezy_webhook_signature(raw_body: bytes, signature_header: str) -> bool:
    if not LEMONSQUEEZY_WEBHOOK_SECRET or not signature_header:
        return False
    computed = hmac.new(LEMONSQUEEZY_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature_header)


# ── Unified entry point ──────────────────────────────────────────────────

def initiate_payment(email: str, pack_id: str, user_id: str,
                      country_code: Optional[str] = None) -> dict:
    pack = get_pack(pack_id)
    if not pack:
        return {"ok": False, "error": f"Unknown credit pack: {pack_id}"}
    gateway = select_gateway(country_code)
    if gateway == "paystack":
        return paystack_initiate(email, pack["usd"], user_id, pack_id)
    return lemonsqueezy_initiate(email, pack["usd"], user_id, pack_id)


def process_webhook(gateway: str, raw_body: bytes, signature: str, payload: dict) -> dict:
    """Verify + parse a webhook into a normalised result."""
    if gateway == "paystack":
        if not verify_paystack_webhook_signature(raw_body, signature):
            return {"ok": False, "verified": False, "error": "Invalid Paystack signature."}
        event = payload.get("event")
        if event != "charge.success":
            return {"ok": True, "verified": True, "ignored": True, "event": event}
        data = payload.get("data", {})
        meta = data.get("metadata", {})
        return {"ok": True, "verified": True, "user_id": meta.get("user_id"),
                "pack_id": meta.get("pack_id"), "amount_usd": meta.get("amount_usd"),
                "reference": data.get("reference"), "gateway": "paystack"}

    elif gateway == "lemonsqueezy":
        if not verify_lemonsqueezy_webhook_signature(raw_body, signature):
            return {"ok": False, "verified": False, "error": "Invalid Lemon Squeezy signature."}
        event_name = payload.get("meta", {}).get("event_name")
        if event_name != "order_created":
            return {"ok": True, "verified": True, "ignored": True, "event": event_name}
        custom = payload.get("meta", {}).get("custom_data", {})
        return {"ok": True, "verified": True, "user_id": custom.get("user_id"),
                "pack_id": custom.get("pack_id"), "amount_usd": custom.get("amount_usd"),
                "reference": str(payload.get("data", {}).get("id", "")), "gateway": "lemonsqueezy"}

    return {"ok": False, "verified": False, "error": f"Unknown gateway: {gateway}"}
