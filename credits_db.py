"""
Eduxellence Credits Ledger — Phase 5
======================================
Manages the `credits` table (balance + full transaction history) on top
of the existing Supabase schema. Uses the same client/mock pattern as
auth_db.py for consistency and local testability.

New table this phase introduces:

  credits(
    id            UUID (PK)
    user_id       UUID (FK -> users.id)
    balance       INTEGER          -- current credit balance (denormalised for fast reads)
    delta         INTEGER          -- + for top-up/grant, - for spend (this row's change)
    reason        TEXT             -- 'topup' | 'spend' | 'referral_bonus' | 'subscription_grant' | 'admin_adjust'
    reference     TEXT             -- payment reference / analysis_id / referral_id
    gateway       TEXT             -- 'paystack' | 'lemonsqueezy' | NULL (non-payment entries)
    amount_usd    NUMERIC          -- USD value of this transaction (NULL for spend entries)
    created_at    TIMESTAMP
  )

Each row is an immutable ledger entry. Current balance = running sum of
`delta`, but we also store the post-transaction `balance` on each row so
reads never require summing the whole table.

by Eduxellence Analytics · https://eduxellence.org
"""

import os, json, uuid
from datetime import datetime, timezone
from typing import Optional

from auth_db import get_client, is_mock_mode, _mock_store, _now_iso

# ══════════════════════════════════════════════════════════════════════════
#  MOCK STORE EXTENSION (credits ledger)
# ══════════════════════════════════════════════════════════════════════════

if "credits" not in _mock_store:
    _mock_store["credits"] = {}     # ledger rows, keyed by row id
if "credit_balances" not in _mock_store:
    _mock_store["credit_balances"] = {}   # user_id -> current balance (fast lookup)


# ══════════════════════════════════════════════════════════════════════════
#  BALANCE READ
# ══════════════════════════════════════════════════════════════════════════

def get_balance(user_id: str) -> int:
    """Return the current credit balance for a user. 0 if no history."""
    client = get_client()

    if is_mock_mode():
        return _mock_store["credit_balances"].get(user_id, 0)

    try:
        resp = (client.table("credits")
                .select("balance")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute())
        if resp.data:
            return int(resp.data[0]["balance"])
        return 0
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════
#  LEDGER WRITE (the only way balance ever changes)
# ══════════════════════════════════════════════════════════════════════════

def _write_ledger_entry(user_id: str, delta: int, reason: str,
                         reference: str = "", gateway: Optional[str] = None,
                         amount_usd: Optional[float] = None) -> dict:
    """
    Internal: append one immutable ledger row and return the new balance.
    This is the single choke point for every balance mutation.
    """
    current = get_balance(user_id)
    new_balance = current + delta

    row = {
        "id":          str(uuid.uuid4()),
        "user_id":     user_id,
        "balance":     new_balance,
        "delta":       delta,
        "reason":      reason,
        "reference":   reference,
        "gateway":     gateway,
        "amount_usd":  amount_usd,
        "created_at":  _now_iso(),
    }

    client = get_client()
    if is_mock_mode():
        _mock_store["credits"][row["id"]] = row
        _mock_store["credit_balances"][user_id] = new_balance
        return {"ok": True, "balance": new_balance, "entry": row, "mock": True}

    try:
        client.table("credits").insert(row).execute()
        return {"ok": True, "balance": new_balance, "entry": row}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC OPERATIONS
# ══════════════════════════════════════════════════════════════════════════

def grant_credits(user_id: str, amount: int, reason: str,
                   reference: str = "", gateway: Optional[str] = None,
                   amount_usd: Optional[float] = None) -> dict:
    """Add credits to a user's balance (top-up, referral bonus, admin grant, etc.)."""
    if amount <= 0:
        return {"ok": False, "error": "Grant amount must be positive."}
    return _write_ledger_entry(user_id, amount, reason, reference, gateway, amount_usd)


def spend_credits(user_id: str, amount: int, reason: str = "spend",
                   reference: str = "") -> dict:
    """
    Deduct credits for a paid analysis. Atomically checks sufficient balance
    first (best-effort; for true atomicity under high concurrency, wrap this
    in a Postgres function/RPC on the Supabase side).
    """
    if amount <= 0:
        return {"ok": True, "balance": get_balance(user_id), "skipped": "free_analysis"}

    current = get_balance(user_id)
    if current < amount:
        return {
            "ok": False,
            "error": "insufficient_credits",
            "balance": current,
            "required": amount,
            "shortfall": amount - current,
        }
    result = _write_ledger_entry(user_id, -amount, reason, reference)
    return result


def refund_credits(user_id: str, amount: int, reference: str,
                    reason: str = "refund") -> dict:
    """Refund credits (e.g. analysis failed after deduction)."""
    if amount <= 0:
        return {"ok": False, "error": "Refund amount must be positive."}
    return _write_ledger_entry(user_id, amount, reason, reference)


def get_transaction_history(user_id: str, limit: int = 50) -> dict:
    """Return the most recent ledger entries for a user."""
    client = get_client()

    if is_mock_mode():
        rows = [r for r in _mock_store["credits"].values() if r["user_id"] == user_id]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return {"ok": True, "transactions": rows[:limit]}

    try:
        resp = (client.table("credits")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute())
        return {"ok": True, "transactions": resp.data or []}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
#  GUARD: check-then-spend wrapper for the analyze route
# ══════════════════════════════════════════════════════════════════════════

def check_and_charge(user_id: Optional[str], analysis_type: str,
                      params: Optional[dict] = None,
                      role: str = "free") -> dict:
    """
    The core gate used by /api/analyze. Decides whether an analysis is
    allowed to proceed, and if it costs credits, deducts them.

    Returns:
      { allowed: bool, cost: int, balance: int|None, error: str|None }
    """
    from pricing import get_credit_cost, is_free_analysis

    cost = get_credit_cost(analysis_type, params)

    if cost == 0:
        return {"allowed": True, "cost": 0, "balance": None, "error": None}

    # Professional plan = unlimited fair-use, no per-analysis charge
    if role == "professional_subscriber":
        return {"allowed": True, "cost": 0, "balance": None, "error": None,
                "note": "Covered by Professional subscription (unlimited fair-use)."}

    if not user_id:
        return {
            "allowed": False, "cost": cost, "balance": None,
            "error": f"This analysis costs {cost} credit(s). Please sign in and "
                     f"top up to continue, or stay on free tools (cleaning + descriptive stats).",
            "code": "signin_required",
        }

    spend_result = spend_credits(user_id, cost, reason="spend",
                                  reference=f"{analysis_type}:{params}")
    if not spend_result.get("ok"):
        return {
            "allowed": False, "cost": cost,
            "balance": spend_result.get("balance", 0),
            "error": f"Insufficient credits. This analysis costs {cost}, "
                     f"you have {spend_result.get('balance',0)}. Top up to continue.",
            "code": "insufficient_credits",
        }

    return {"allowed": True, "cost": cost, "balance": spend_result["balance"], "error": None}
