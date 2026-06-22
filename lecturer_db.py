"""
Eduxellence Lecturer Dashboard Engine — Phase 8
==================================================
Lets approved lecturers (role='lecturer', admin-gated per the confirmed
flow) create class codes. Students who enter a code get a free monthly
credit allowance (5 credits/month) for as long as they remain enrolled.

New tables this phase introduces:

  lecturer_classes(
    id              UUID (PK)
    lecturer_id     UUID (FK -> users.id)
    class_name      TEXT
    class_code      TEXT (unique, student-facing)
    monthly_allowance INTEGER   -- credits granted per student per month (default 5)
    is_active       BOOLEAN
    created_at      TIMESTAMP
  )

  class_enrollments(
    id                UUID (PK)
    class_id          UUID (FK -> lecturer_classes.id)
    student_id        UUID (FK -> users.id)
    enrolled_at       TIMESTAMP
    last_allowance_at TIMESTAMP (nullable)  -- last time the monthly credit was granted
    is_active         BOOLEAN
  )

Monthly allowance mechanics:
  - On enrollment, the student gets their first allowance immediately.
  - On any login/analyze call (checked lazily, no cron needed on Vercel
    free tier), if 30+ days have passed since last_allowance_at, grant
    the next month's allowance automatically.
  - Lecturer dashboard shows: students enrolled, total credits granted,
    total analyses run by class students, per-student activity.

Uses the same supabase-py client / mock-mode pattern as the other
*_db.py modules for consistency and local testability.

by Eduxellence Analytics · https://eduxellence.org
"""

import os, json, uuid, secrets, string
from datetime import datetime, timedelta, timezone
from typing import Optional

from auth_db import get_client, is_mock_mode, _mock_store, _now_iso
import credits_db as cr

ALLOWANCE_PERIOD_DAYS = 30
DEFAULT_MONTHLY_ALLOWANCE = 5
CLASS_CODE_LENGTH = 7
_ALPHABET = string.ascii_uppercase + string.digits

# ══════════════════════════════════════════════════════════════════════════
#  MOCK STORE EXTENSION
# ══════════════════════════════════════════════════════════════════════════

if "lecturer_classes" not in _mock_store:
    _mock_store["lecturer_classes"] = {}     # class_id -> row
if "class_codes" not in _mock_store:
    _mock_store["class_codes"] = {}          # code -> class_id  (fast lookup)
if "class_enrollments" not in _mock_store:
    _mock_store["class_enrollments"] = {}    # enrollment_id -> row


# ══════════════════════════════════════════════════════════════════════════
#  ROLE GATE — lecturer access requires admin-approved role
# ══════════════════════════════════════════════════════════════════════════

def require_lecturer(user: Optional[dict]) -> tuple[bool, Optional[str]]:
    """
    Per the confirmed flow: lecturer status requires admin approval —
    i.e. role must already be 'lecturer' (or 'admin', who can do anything).
    There is no self-serve upgrade path; an admin manually sets this in
    Supabase (or via a future admin panel).
    """
    if not user:
        return False, "You must be signed in to access the lecturer dashboard."
    role = user.get("role", "free")
    if role not in ("lecturer", "admin"):
        return False, (
            "Lecturer access requires admin approval. Your current role is "
            f"'{role}'. Contact support@eduxellence.org to request lecturer status "
            "for your institution."
        )
    return True, None


# ══════════════════════════════════════════════════════════════════════════
#  CLASS CODE GENERATION
# ══════════════════════════════════════════════════════════════════════════

def _generate_unique_class_code() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(CLASS_CODE_LENGTH))


def create_class(lecturer_id: str, class_name: str,
                  monthly_allowance: int = DEFAULT_MONTHLY_ALLOWANCE) -> dict:
    """Create a new class with a unique, student-shareable join code."""
    if not class_name or not class_name.strip():
        return {"ok": False, "error": "Class name is required."}
    if monthly_allowance < 0:
        return {"ok": False, "error": "Monthly allowance cannot be negative."}

    class_id = str(uuid.uuid4())
    code = _generate_unique_class_code()

    row = {
        "id": class_id,
        "lecturer_id": lecturer_id,
        "class_name": class_name.strip(),
        "class_code": code,
        "monthly_allowance": monthly_allowance,
        "is_active": True,
        "created_at": _now_iso(),
    }

    client = get_client()
    if is_mock_mode():
        _mock_store["lecturer_classes"][class_id] = row
        _mock_store["class_codes"][code] = class_id
        return {"ok": True, "class_id": class_id, "class_code": code, "mock": True}

    try:
        client.table("lecturer_classes").insert(row).execute()
        return {"ok": True, "class_id": class_id, "class_code": code}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_class_by_code(code: str) -> Optional[dict]:
    """Resolve a class code to its full class row."""
    if not code:
        return None
    client = get_client()
    code = code.upper().strip()

    if is_mock_mode():
        class_id = _mock_store["class_codes"].get(code)
        if not class_id:
            return None
        return _mock_store["lecturer_classes"].get(class_id)

    try:
        resp = client.table("lecturer_classes").select("*").eq("class_code", code).single().execute()
        return resp.data
    except Exception:
        return None


def get_lecturer_classes(lecturer_id: str) -> dict:
    """List all classes created by this lecturer."""
    client = get_client()
    if is_mock_mode():
        rows = [c for c in _mock_store["lecturer_classes"].values() if c["lecturer_id"] == lecturer_id]
        return {"ok": True, "classes": rows}
    try:
        resp = client.table("lecturer_classes").select("*").eq("lecturer_id", lecturer_id).execute()
        return {"ok": True, "classes": resp.data or []}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
#  STUDENT ENROLLMENT
# ══════════════════════════════════════════════════════════════════════════

def enroll_student(class_code: str, student_id: str) -> dict:
    """
    Student enters a class code to enroll. Grants the first month's
    allowance immediately on successful enrollment.
    """
    cls = get_class_by_code(class_code)
    if not cls:
        return {"ok": False, "error": "Invalid or unknown class code."}
    if not cls.get("is_active", True):
        return {"ok": False, "error": "This class is no longer active."}

    client = get_client()

    if is_mock_mode():
        # Prevent duplicate enrollment in the same class
        for e in _mock_store["class_enrollments"].values():
            if e["class_id"] == cls["id"] and e["student_id"] == student_id and e["is_active"]:
                return {"ok": False, "error": "You are already enrolled in this class."}

        enrollment_id = str(uuid.uuid4())
        row = {
            "id": enrollment_id, "class_id": cls["id"], "student_id": student_id,
            "enrolled_at": _now_iso(), "last_allowance_at": None, "is_active": True,
        }
        _mock_store["class_enrollments"][enrollment_id] = row
    else:
        try:
            existing = (client.table("class_enrollments").select("id")
                        .eq("class_id", cls["id"]).eq("student_id", student_id)
                        .eq("is_active", True).execute())
            if existing.data:
                return {"ok": False, "error": "You are already enrolled in this class."}

            enrollment_id = str(uuid.uuid4())
            row = {
                "id": enrollment_id, "class_id": cls["id"], "student_id": student_id,
                "enrolled_at": _now_iso(), "last_allowance_at": None, "is_active": True,
            }
            client.table("class_enrollments").insert(row).execute()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # Grant the first month's allowance immediately
    allowance_result = grant_monthly_allowance(enrollment_id)

    return {
        "ok": True, "enrollment_id": enrollment_id,
        "class_name": cls["class_name"], "lecturer_id": cls["lecturer_id"],
        "allowance_granted": allowance_result.get("granted", 0),
    }


# ══════════════════════════════════════════════════════════════════════════
#  MONTHLY ALLOWANCE (lazy refresh — no cron needed)
# ══════════════════════════════════════════════════════════════════════════

def _get_enrollment(enrollment_id: str) -> Optional[dict]:
    client = get_client()
    if is_mock_mode():
        return _mock_store["class_enrollments"].get(enrollment_id)
    try:
        resp = client.table("class_enrollments").select("*").eq("id", enrollment_id).single().execute()
        return resp.data
    except Exception:
        return None


def grant_monthly_allowance(enrollment_id: str) -> dict:
    """
    Grants the monthly credit allowance if the student hasn't received
    one in the last ALLOWANCE_PERIOD_DAYS. Called on enrollment and
    lazily checked on every analyze call for enrolled students (no
    cron job required — fits Vercel free tier).
    """
    enrollment = _get_enrollment(enrollment_id)
    if not enrollment or not enrollment.get("is_active", True):
        return {"ok": False, "granted": 0, "error": "Enrollment not found or inactive."}

    last = enrollment.get("last_allowance_at")
    if last:
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00")) if isinstance(last, str) else last
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        days_since = (datetime.now(timezone.utc) - last_dt).days
        if days_since < ALLOWANCE_PERIOD_DAYS:
            return {"ok": True, "granted": 0, "next_in_days": ALLOWANCE_PERIOD_DAYS - days_since}

    cls = _get_class(enrollment["class_id"])
    if not cls:
        return {"ok": False, "granted": 0, "error": "Class not found."}

    amount = cls.get("monthly_allowance", DEFAULT_MONTHLY_ALLOWANCE)
    if amount > 0:
        cr.grant_credits(
            enrollment["student_id"], amount, reason="class_allowance",
            reference=f"class_{cls['id']}_{datetime.now(timezone.utc).strftime('%Y%m')}",
        )

    _update_last_allowance(enrollment_id)
    return {"ok": True, "granted": amount}


def _get_class(class_id: str) -> Optional[dict]:
    client = get_client()
    if is_mock_mode():
        return _mock_store["lecturer_classes"].get(class_id)
    try:
        resp = client.table("lecturer_classes").select("*").eq("id", class_id).single().execute()
        return resp.data
    except Exception:
        return None


def _update_last_allowance(enrollment_id: str) -> None:
    client = get_client()
    now = _now_iso()
    if is_mock_mode():
        if enrollment_id in _mock_store["class_enrollments"]:
            _mock_store["class_enrollments"][enrollment_id]["last_allowance_at"] = now
        return
    try:
        client.table("class_enrollments").update({"last_allowance_at": now}).eq("id", enrollment_id).execute()
    except Exception:
        pass


def check_and_refresh_student_allowances(student_id: str) -> dict:
    """
    Called lazily (e.g. at the top of /api/analyze) for any registered
    student — checks ALL their active enrollments and grants any
    allowance that's come due. Cheap no-op for students with no
    enrollments or who already received this month's credit.
    """
    client = get_client()

    if is_mock_mode():
        enrollments = [e for e in _mock_store["class_enrollments"].values()
                       if e["student_id"] == student_id and e["is_active"]]
    else:
        try:
            resp = (client.table("class_enrollments").select("*")
                    .eq("student_id", student_id).eq("is_active", True).execute())
            enrollments = resp.data or []
        except Exception:
            enrollments = []

    total_granted = 0
    for e in enrollments:
        result = grant_monthly_allowance(e["id"])
        total_granted += result.get("granted", 0)

    return {"ok": True, "total_granted": total_granted, "enrollments_checked": len(enrollments)}


# ══════════════════════════════════════════════════════════════════════════
#  LECTURER DASHBOARD DATA
# ══════════════════════════════════════════════════════════════════════════

def get_class_dashboard(class_id: str, lecturer_id: str) -> dict:
    """
    Full dashboard payload for one class: roster, enrollment dates,
    allowance status, and (best-effort) recent analysis activity per
    student pulled from audit_logs.
    """
    cls = _get_class(class_id)
    if not cls:
        return {"ok": False, "error": "Class not found."}
    if cls["lecturer_id"] != lecturer_id:
        return {"ok": False, "error": "You do not own this class."}

    client = get_client()
    if is_mock_mode():
        enrollments = [e for e in _mock_store["class_enrollments"].values()
                       if e["class_id"] == class_id]
    else:
        try:
            resp = client.table("class_enrollments").select("*").eq("class_id", class_id).execute()
            enrollments = resp.data or []
        except Exception:
            enrollments = []

    roster = []
    for e in enrollments:
        student = _get_user_basic(e["student_id"])
        analyses_count = _count_student_analyses(e["student_id"])
        roster.append({
            "student_id": e["student_id"],
            "email": student.get("email", "—") if student else "—",
            "enrolled_at": e["enrolled_at"],
            "is_active": e.get("is_active", True),
            "last_allowance_at": e.get("last_allowance_at"),
            "analyses_run": analyses_count,
        })

    return {
        "ok": True,
        "class_name": cls["class_name"],
        "class_code": cls["class_code"],
        "monthly_allowance": cls["monthly_allowance"],
        "is_active": cls.get("is_active", True),
        "created_at": cls["created_at"],
        "total_students": len(roster),
        "active_students": sum(1 for r in roster if r["is_active"]),
        "total_analyses": sum(r["analyses_run"] for r in roster),
        "roster": sorted(roster, key=lambda r: r["enrolled_at"], reverse=True),
    }


def _get_user_basic(user_id: str) -> Optional[dict]:
    client = get_client()
    if is_mock_mode():
        return _mock_store["users"].get(user_id)
    try:
        resp = client.table("users").select("id,email,display_name").eq("id", user_id).single().execute()
        return resp.data
    except Exception:
        return None


def _count_student_analyses(student_id: str) -> int:
    """Best-effort count of analyses run by a student, from audit_logs."""
    client = get_client()
    if is_mock_mode():
        return sum(1 for log in _mock_store["audit_logs"].values()
                   if log.get("user_id") == student_id
                   and log.get("action") in ("analysis_run", "analysis_saved", "thesis_package_generated"))
    try:
        resp = (client.table("audit_logs").select("id", count="exact")
                .eq("user_id", student_id)
                .in_("action", ["analysis_run", "analysis_saved", "thesis_package_generated"])
                .execute())
        return resp.count or 0
    except Exception:
        return 0


def deactivate_class(class_id: str, lecturer_id: str) -> dict:
    """Lecturer closes a class — stops future allowance grants, keeps history."""
    cls = _get_class(class_id)
    if not cls:
        return {"ok": False, "error": "Class not found."}
    if cls["lecturer_id"] != lecturer_id:
        return {"ok": False, "error": "You do not own this class."}

    client = get_client()
    if is_mock_mode():
        _mock_store["lecturer_classes"][class_id]["is_active"] = False
        return {"ok": True}
    try:
        client.table("lecturer_classes").update({"is_active": False}).eq("id", class_id).execute()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
