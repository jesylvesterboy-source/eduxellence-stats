"""
Eduxellence Auth & Database Layer — Phase 4
=============================================
Wraps Supabase access for:
  • User authentication (registered vs anonymous)
  • Saving analyses
  • Shareable result links (shared_links table)
  • Audit logging

Uses supabase-py (`from supabase import create_client, Client`) in production.
Falls back to an in-memory mock store when SUPABASE_URL/SUPABASE_KEY are not
set or the supabase package is unavailable — this lets the rest of the app
run and be tested locally without a live Supabase project.

Schema assumed (as provided):
  users(id, email, display_name, created_at, last_login, is_active, role)
  analyses(id, user_id, analysis_type, dataset_name, variables, parameters,
            results, interpretation, created_at, is_shared, share_id)
  charts(id, analysis_id, chart_name, chart_url, chart_data, created_at)
  datasets(id, user_id, name, file_url, file_size, rows, columns,
            column_types, created_at, last_used)
  shared_links(id, analysis_id, user_id, share_token, created_at,
                expires_at, view_count)
  audit_logs(id, user_id, action, ip_address, user_agent, details, created_at)

by Eduxellence Analytics · https://eduxellence.org
"""

import os, json, uuid, secrets, string
from datetime import datetime, timedelta, timezone
from typing import Optional

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")  # anon/public key for client-side-safe ops
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")  # service role, server-only

SHARE_LINK_EXPIRY_DAYS = 30
SHARE_TOKEN_LENGTH = 10  # short, URL-friendly


# ══════════════════════════════════════════════════════════════════════════
#  CLIENT INITIALISATION (real Supabase OR mock fallback)
# ══════════════════════════════════════════════════════════════════════════

_client = None
_mock_mode = False
_mock_store = {
    "users": {},
    "analyses": {},
    "charts": {},
    "datasets": {},
    "shared_links": {},
    "audit_logs": {},
}

def get_client():
    """Return a live Supabase client, or None if running in mock mode."""
    global _client, _mock_mode
    if _client is not None or _mock_mode:
        return _client

    if not SUPABASE_URL or not SUPABASE_KEY:
        _mock_mode = True
        return None

    try:
        from supabase import create_client, Client
        key = SUPABASE_SERVICE_KEY or SUPABASE_KEY
        _client = create_client(SUPABASE_URL, key)
        return _client
    except Exception:
        _mock_mode = True
        return None


def is_mock_mode() -> bool:
    get_client()
    return _mock_mode


# ══════════════════════════════════════════════════════════════════════════
#  TOKEN GENERATION
# ══════════════════════════════════════════════════════════════════════════

_ALPHABET = string.ascii_letters + string.digits

def generate_share_token(length: int = SHARE_TOKEN_LENGTH) -> str:
    """Generate a short, URL-safe, cryptographically random token."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expiry_iso(days: int = SHARE_LINK_EXPIRY_DAYS) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


# ══════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_user_from_request(auth_header: Optional[str]) -> Optional[dict]:
    """
    Resolve the current user from a Bearer token (Supabase JWT).
    Returns the user dict, or None if anonymous / invalid.

    In mock mode, a header of 'Bearer mock:<user_id>' resolves to that
    mock user — used purely for local testing.
    """
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:].strip()
    if not token:
        return None

    client = get_client()

    if is_mock_mode():
        if token.startswith("mock:"):
            uid = token[5:]
            return _mock_store["users"].get(uid)
        return None

    try:
        resp = client.auth.get_user(token)
        user_obj = getattr(resp, "user", None)
        if not user_obj:
            return None
        uid = user_obj.id
        # Fetch profile row from users table (created via trigger on signup)
        prof = client.table("users").select("*").eq("id", uid).single().execute()
        return prof.data if prof.data else {
            "id": uid, "email": user_obj.email, "role": "free"
        }
    except Exception:
        return None


def require_registered_user(auth_header: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    """
    Returns (user, error). If error is not None, the caller should
    reject the request (used to enforce 'only registered users can create/edit/delete').
    """
    user = get_user_from_request(auth_header)
    if not user:
        return None, "You must be signed in to perform this action. Anonymous users can view shared results but cannot create them."
    return user, None


# ══════════════════════════════════════════════════════════════════════════
#  ANALYSES — save a completed analysis
# ══════════════════════════════════════════════════════════════════════════

def save_analysis(
    results: dict,
    analysis_type: str,
    params: dict,
    dataset_name: str = "",
    user_id: Optional[str] = None,
) -> dict:
    """
    Persist a completed analysis to the `analyses` table.
    user_id may be None (anonymous analyses are still saved so a share
    link can later be created by a registered user who claims it — but
    per the access rules, only registered users can CREATE share links,
    so anonymous saves exist only transiently for the session).
    """
    record_id = str(uuid.uuid4())
    row = {
        "id":              record_id,
        "user_id":         user_id,
        "analysis_type":   analysis_type,
        "dataset_name":    dataset_name or "Untitled dataset",
        "variables":       json.dumps(_extract_variables(params)),
        "parameters":      json.dumps(params),
        "results":         json.dumps(_strip_charts_for_db(results)),
        "interpretation":  results.get("interpretation", ""),
        "created_at":      _now_iso(),
        "is_shared":       False,
        "share_id":        None,
    }

    client = get_client()
    if is_mock_mode():
        _mock_store["analyses"][record_id] = row
        return {"ok": True, "id": record_id, "mock": True}

    try:
        resp = client.table("analyses").insert(row).execute()
        return {"ok": True, "id": record_id, "data": resp.data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _extract_variables(params: dict) -> list:
    """Pull variable names out of a params dict for the `variables` JSONB column."""
    keys = ["numeric_var", "group_var", "var1", "var2", "dependent",
            "predictors", "columns"]
    out = []
    for k in keys:
        v = params.get(k)
        if isinstance(v, list):
            out.extend(v)
        elif v:
            out.append(v)
    return list(dict.fromkeys(out))  # dedupe, preserve order


def _strip_charts_for_db(results: dict) -> dict:
    """
    Charts (base64 PNGs) are large — store them in the `charts` table /
    Supabase Storage instead of inline in the analyses.results JSONB.
    Returns a copy of results with chart images replaced by placeholders.
    """
    stripped = {k: v for k, v in results.items() if k != "charts"}
    if results.get("charts"):
        stripped["chart_count"] = len(results["charts"])
        stripped["chart_titles"] = [c.get("title", "") for c in results["charts"]]
    return stripped


# ══════════════════════════════════════════════════════════════════════════
#  CHARTS — store chart images (Supabase Storage bucket: eduxellence-charts)
# ══════════════════════════════════════════════════════════════════════════

def save_charts(analysis_id: str, charts: list) -> dict:
    """
    Upload each chart's base64 PNG to the public 'eduxellence-charts' bucket
    and insert a row per chart into the `charts` table.
    Returns list of public URLs.
    """
    import base64

    client = get_client()
    urls = []

    if is_mock_mode():
        for i, chart in enumerate(charts):
            chart_id = str(uuid.uuid4())
            fake_url = f"/mock-storage/charts/{analysis_id}/{i}.png"
            _mock_store["charts"][chart_id] = {
                "id": chart_id, "analysis_id": analysis_id,
                "chart_name": chart.get("title", f"chart_{i}"),
                "chart_url": fake_url, "chart_data": json.dumps({}),
                "created_at": _now_iso(),
            }
            urls.append(fake_url)
        return {"ok": True, "urls": urls, "mock": True}

    try:
        for i, chart in enumerate(charts):
            img_bytes = base64.b64decode(chart["img"])
            path = f"{analysis_id}/{i}_{chart.get('type','chart')}.png"

            client.storage.from_("eduxellence-charts").upload(
                path, img_bytes,
                file_options={"content-type": "image/png", "upsert": "true"}
            )
            public_url = client.storage.from_("eduxellence-charts").get_public_url(path)

            client.table("charts").insert({
                "id":           str(uuid.uuid4()),
                "analysis_id":  analysis_id,
                "chart_name":   chart.get("title", f"chart_{i}"),
                "chart_url":    public_url,
                "chart_data":   json.dumps({"type": chart.get("type", "")}),
                "created_at":   _now_iso(),
            }).execute()

            urls.append(public_url)

        return {"ok": True, "urls": urls}
    except Exception as e:
        return {"ok": False, "error": str(e), "urls": []}


# ══════════════════════════════════════════════════════════════════════════
#  SHARED LINKS — create / fetch / list / delete / edit
# ══════════════════════════════════════════════════════════════════════════

def create_share_link(analysis_id: str, user_id: str,
                       expiry_days: int = SHARE_LINK_EXPIRY_DAYS) -> dict:
    """
    Create a shared_links row. ONLY callable for registered users
    (enforced by require_registered_user at the route level).
    """
    token   = generate_share_token()
    link_id = str(uuid.uuid4())
    row = {
        "id":          link_id,
        "analysis_id": analysis_id,
        "user_id":     user_id,
        "share_token": token,
        "created_at":  _now_iso(),
        "expires_at":  _expiry_iso(expiry_days),
        "view_count":  0,
    }

    client = get_client()
    if is_mock_mode():
        _mock_store["shared_links"][token] = row
        # Also flip is_shared/share_id on the analysis if present
        a = _mock_store["analyses"].get(analysis_id)
        if a:
            a["is_shared"] = True
            a["share_id"]  = token
        return {"ok": True, "token": token, "id": link_id,
                "expires_at": row["expires_at"], "mock": True}

    try:
        resp = client.table("shared_links").insert(row).execute()
        client.table("analyses").update({
            "is_shared": True, "share_id": token
        }).eq("id", analysis_id).execute()
        return {"ok": True, "token": token, "id": link_id,
                "expires_at": row["expires_at"], "data": resp.data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_shared_analysis(token: str) -> dict:
    """
    Fetch a shared analysis by token. Open to ANYONE (anonymous included).
    Increments view_count. Returns {"ok": False, "error": ...} if not
    found or expired.
    """
    client = get_client()

    if is_mock_mode():
        link = _mock_store["shared_links"].get(token)
        if not link:
            return {"ok": False, "error": "Share link not found.", "code": 404}
        if _is_expired(link["expires_at"]):
            return {"ok": False, "error": "This share link has expired.", "code": 410}
        link["view_count"] = link.get("view_count", 0) + 1
        analysis = _mock_store["analyses"].get(link["analysis_id"])
        if not analysis:
            return {"ok": False, "error": "Linked analysis no longer exists.", "code": 404}
        return {
            "ok": True,
            "analysis": _deserialize_analysis(analysis),
            "view_count": link["view_count"],
            "created_at": link["created_at"],
            "expires_at": link["expires_at"],
        }

    try:
        link_resp = client.table("shared_links").select("*").eq("share_token", token).single().execute()
        link = link_resp.data
        if not link:
            return {"ok": False, "error": "Share link not found.", "code": 404}
        if _is_expired(link["expires_at"]):
            return {"ok": False, "error": "This share link has expired.", "code": 410}

        # Increment view count
        client.table("shared_links").update({
            "view_count": link.get("view_count", 0) + 1
        }).eq("share_token", token).execute()

        a_resp = client.table("analyses").select("*").eq("id", link["analysis_id"]).single().execute()
        analysis = a_resp.data
        if not analysis:
            return {"ok": False, "error": "Linked analysis no longer exists.", "code": 404}

        # Fetch charts
        c_resp = client.table("charts").select("*").eq("analysis_id", link["analysis_id"]).execute()
        charts = c_resp.data or []

        return {
            "ok": True,
            "analysis": _deserialize_analysis(analysis, charts),
            "view_count": link.get("view_count", 0) + 1,
            "created_at": link["created_at"],
            "expires_at": link["expires_at"],
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "code": 500}


def list_share_links(user_id: str) -> dict:
    """List all share links owned by a registered user."""
    client = get_client()
    if is_mock_mode():
        links = [l for l in _mock_store["shared_links"].values() if l["user_id"] == user_id]
        return {"ok": True, "links": links}
    try:
        resp = client.table("shared_links").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return {"ok": True, "links": resp.data or []}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def delete_share_link(token: str, user_id: str) -> dict:
    """Delete a share link. Only the owning registered user may delete it."""
    client = get_client()

    if is_mock_mode():
        link = _mock_store["shared_links"].get(token)
        if not link:
            return {"ok": False, "error": "Link not found.", "code": 404}
        if link["user_id"] != user_id:
            return {"ok": False, "error": "You do not own this share link.", "code": 403}
        del _mock_store["shared_links"][token]
        return {"ok": True, "mock": True}

    try:
        existing = client.table("shared_links").select("user_id").eq("share_token", token).single().execute()
        if not existing.data:
            return {"ok": False, "error": "Link not found.", "code": 404}
        if existing.data["user_id"] != user_id:
            return {"ok": False, "error": "You do not own this share link.", "code": 403}
        client.table("shared_links").delete().eq("share_token", token).execute()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def update_share_link(token: str, user_id: str, expiry_days: Optional[int] = None) -> dict:
    """Edit a share link (currently: extend/shorten expiry). Owner-only."""
    client = get_client()
    new_expiry = _expiry_iso(expiry_days) if expiry_days is not None else None

    if is_mock_mode():
        link = _mock_store["shared_links"].get(token)
        if not link:
            return {"ok": False, "error": "Link not found.", "code": 404}
        if link["user_id"] != user_id:
            return {"ok": False, "error": "You do not own this share link.", "code": 403}
        if new_expiry:
            link["expires_at"] = new_expiry
        return {"ok": True, "expires_at": link["expires_at"], "mock": True}

    try:
        existing = client.table("shared_links").select("user_id").eq("share_token", token).single().execute()
        if not existing.data:
            return {"ok": False, "error": "Link not found.", "code": 404}
        if existing.data["user_id"] != user_id:
            return {"ok": False, "error": "You do not own this share link.", "code": 403}
        update_fields = {}
        if new_expiry:
            update_fields["expires_at"] = new_expiry
        if update_fields:
            client.table("shared_links").update(update_fields).eq("share_token", token).execute()
        return {"ok": True, "expires_at": new_expiry}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _is_expired(expires_at_iso: str) -> bool:
    try:
        exp = datetime.fromisoformat(expires_at_iso.replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > exp
    except Exception:
        return False


def _deserialize_analysis(analysis: dict, charts: Optional[list] = None) -> dict:
    """Parse JSONB string fields back into dicts for the API response."""
    out = dict(analysis)
    for field in ("variables", "parameters", "results"):
        val = out.get(field)
        if isinstance(val, str):
            try:
                out[field] = json.loads(val)
            except Exception:
                pass
    if charts is not None:
        out["chart_urls"] = [c.get("chart_url") for c in charts]
    return out


# ══════════════════════════════════════════════════════════════════════════
#  AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════

def log_action(action: str, user_id: Optional[str] = None,
                ip_address: str = "", user_agent: str = "",
                details: Optional[dict] = None) -> None:
    """Fire-and-forget audit log entry. Never raises — logging must not break the app."""
    row = {
        "id":         str(uuid.uuid4()),
        "user_id":    user_id,
        "action":     action,
        "ip_address": ip_address,
        "user_agent": user_agent[:300] if user_agent else "",
        "details":    json.dumps(details or {}),
        "created_at": _now_iso(),
    }
    try:
        client = get_client()
        if is_mock_mode():
            _mock_store["audit_logs"][row["id"]] = row
            return
        client.table("audit_logs").insert(row).execute()
    except Exception:
        pass  # audit logging is best-effort only


# ══════════════════════════════════════════════════════════════════════════
#  MOCK SEEDING (test/dev helper)
# ══════════════════════════════════════════════════════════════════════════

def _seed_mock_user(user_id: str, email: str = "test@eduxellence.org",
                     role: str = "free") -> dict:
    """Create a mock registered user for local testing (mock mode only)."""
    row = {
        "id": user_id, "email": email, "display_name": email.split("@")[0],
        "created_at": _now_iso(), "last_login": _now_iso(),
        "is_active": True, "role": role,
    }
    _mock_store["users"][user_id] = row
    return row


def _reset_mock_store():
    """Clear all mock data — used between test runs."""
    for k in _mock_store:
        _mock_store[k].clear()
