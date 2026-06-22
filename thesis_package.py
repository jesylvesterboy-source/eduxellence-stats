"""
Eduxellence Thesis Package Engine — Phase 7
==============================================
Bundles 7 statistical tests (Descriptive, Chi-Square, T-Test, ANOVA,
Correlation, Regression, Reliability) into one purchase, with a
hybrid auto-suggest + manual-review variable selection flow, and a
single combined APA-formatted report (Word + PDF) at the end.

Flow (per the confirmed spec):
  1. suggest_package_variables(df, columns) -> auto-picks variables
     for each of the 7 component tests where the data allows it.
  2. Frontend shows these suggestions to the user for review/edit
     (the "hybrid" requirement).
  3. validate_package_config(df, config) -> BLOCKS generation entirely
     if ANY of the 7 tests has no valid variable selection — per the
     confirmed "block until fixed" rule. No silent skipping.
  4. run_thesis_package(df, config) -> runs all 7 tests via the
     existing stats_engine dispatcher, compiles one combined results
     dict, and exports it as a single Word + PDF report covering
     every section in order.

Credit cost: charged ONCE, only after all 7 tests complete
successfully (per the confirmed "no refund logic needed" rule —
we simply never charge until success is already confirmed).

by Eduxellence Analytics · https://eduxellence.org
"""

import io
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from stats_engine import run_analysis, ANALYSIS_LABELS

# ── The 7 component tests, in report order ──────────────────────────────
PACKAGE_TESTS = [
    "descriptive",
    "chi_square",
    "t_test",
    "anova",
    "correlation",
    "regression",
    "reliability",   # Cronbach's Alpha — new, added below
]

PACKAGE_TEST_LABELS = {
    "descriptive":  "Descriptive Statistics",
    "chi_square":   "Chi-Square Test of Independence",
    "t_test":       "Independent Samples T-Test",
    "anova":        "One-Way ANOVA",
    "correlation":  "Correlation Analysis",
    "regression":   "Linear Regression",
    "reliability":  "Reliability Analysis (Cronbach's Alpha)",
}


# ══════════════════════════════════════════════════════════════════════════
#  RELIABILITY TEST (Cronbach's Alpha) — not in the original 8, added for
#  the Thesis Package per the pricing table's "Reliability tests" line item.
# ══════════════════════════════════════════════════════════════════════════

def reliability_analysis(df: pd.DataFrame, columns: list) -> dict:
    """
    Cronbach's Alpha for a set of Likert/scale items (e.g. survey questions
    meant to measure the same underlying construct).
    """
    sub = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 2 or len(columns) < 2:
        return {"error": "Reliability analysis needs at least 2 items and 2 valid rows."}

    item_vars = sub.var(axis=0, ddof=1)
    total_var = sub.sum(axis=1).var(ddof=1)
    k = len(columns)

    if total_var == 0:
        return {"error": "Total variance is zero — cannot compute Cronbach's Alpha."}

    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)

    interpretation_label = (
        "excellent" if alpha >= 0.9 else
        "good" if alpha >= 0.8 else
        "acceptable" if alpha >= 0.7 else
        "questionable" if alpha >= 0.6 else
        "poor" if alpha >= 0.5 else "unacceptable"
    )

    # Item-total correlations (corrected) for a diagnostic table
    item_table = []
    for col in columns:
        rest = sub.drop(columns=[col]).sum(axis=1)
        try:
            r = float(np.corrcoef(sub[col], rest)[0, 1])
        except Exception:
            r = float("nan")
        item_table.append({
            "Item": col,
            "Item Variance": round(float(item_vars[col]), 4),
            "Item-Total Correlation": round(r, 4) if not np.isnan(r) else "—",
        })

    return {
        "test": "Reliability Analysis (Cronbach's Alpha)",
        "alpha": round(float(alpha), 4),
        "n_items": k,
        "n": int(len(sub)),
        "effect_size": interpretation_label,
        "item_table": item_table,
        "p_value": None, "p_display": "—", "significance": "—",
        "apa_citation": f"Cronbach's α = {alpha:.2f} (N = {len(sub)}, k = {k} items)",
        "interpretation": (
            f"Reliability analysis was conducted on {k} items using Cronbach's Alpha. "
            f"Internal consistency was {interpretation_label} (α = {alpha:.2f}). "
            + ("Values above .70 are generally considered acceptable for research purposes."
               if alpha >= 0.7 else
               "Values below .70 suggest the items may not be measuring a single "
               "consistent underlying construct; consider reviewing or removing weak items.")
        ),
        "charts": [],
    }


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 — AUTO-SUGGEST VARIABLES (hybrid flow, step one)
# ══════════════════════════════════════════════════════════════════════════

def suggest_package_variables(df: pd.DataFrame, columns: list) -> dict:
    """
    For each of the 7 component tests, propose a best-guess variable
    configuration based on column types. Returns None for a test's
    'params' if no valid configuration could be found — the frontend
    must show this as needing manual fix (per the "block until fixed" rule).
    """
    num_cols = [c["name"] for c in columns if c["dtype"] == "numeric"]
    cat_cols = [c["name"] for c in columns if c["dtype"] == "categorical"]
    cat2     = [c for c in cat_cols if df[c].dropna().nunique() == 2]
    cat3p    = [c for c in cat_cols if df[c].dropna().nunique() >= 3]

    # Filter out identifier-looking columns (student_id, order_id, etc.) —
    # these are technically numeric but meaningless as analysis variables.
    _ID_PATTERNS = ("id", "code", "ref", "number", "no", "num")
    def _looks_like_id(name: str) -> bool:
        lname = name.lower().replace("_", " ").replace("-", " ")
        tokens = lname.split()
        return any(tok in _ID_PATTERNS for tok in tokens)

    num_cols_analytical = [c for c in num_cols if not _looks_like_id(c)] or num_cols

    suggestions = {}

    # 1. Descriptive — every numeric + a couple categoricals, capped for readability
    desc_cols = (num_cols_analytical[:6] + cat_cols[:2]) if (num_cols_analytical or cat_cols) else []
    suggestions["descriptive"] = {
        "params": {"columns": desc_cols} if desc_cols else None,
        "reason": f"{len(desc_cols)} columns selected for summary statistics.",
    }

    # 2. Chi-Square — first two categorical columns with 2+ levels each
    chi_cands = [c for c in cat_cols if df[c].dropna().nunique() >= 2]
    if len(chi_cands) >= 2:
        suggestions["chi_square"] = {
            "params": {"var1": chi_cands[0], "var2": chi_cands[1]},
            "reason": f"Using '{chi_cands[0]}' and '{chi_cands[1]}' — your two most relevant categorical variables.",
        }
    else:
        suggestions["chi_square"] = {"params": None,
            "reason": "Needs at least 2 categorical columns. Your dataset doesn't have enough — please select manually."}

    # 3. T-Test — first numeric + first 2-group categorical
    if num_cols_analytical and cat2:
        suggestions["t_test"] = {
            "params": {"numeric_var": num_cols_analytical[0], "group_var": cat2[0]},
            "reason": f"Comparing '{num_cols_analytical[0]}' between the two groups in '{cat2[0]}'.",
        }
    else:
        suggestions["t_test"] = {"params": None,
            "reason": "Needs 1 numeric column + 1 categorical column with exactly 2 groups. Please select manually."}

    # 4. ANOVA — first numeric + first 3+-group categorical
    if num_cols_analytical and cat3p:
        suggestions["anova"] = {
            "params": {"numeric_var": num_cols_analytical[0], "group_var": cat3p[0]},
            "reason": f"Comparing '{num_cols_analytical[0]}' across the {df[cat3p[0]].nunique()} groups in '{cat3p[0]}'.",
        }
    else:
        suggestions["anova"] = {"params": None,
            "reason": "Needs 1 numeric column + 1 categorical column with 3+ groups. Please select manually."}

    # 5. Correlation — all numeric columns (min 2)
    if len(num_cols_analytical) >= 2:
        suggestions["correlation"] = {
            "params": {"columns": num_cols_analytical[:6], "method": "pearson"},
            "reason": f"Correlating {min(len(num_cols_analytical),6)} numeric variables.",
        }
    else:
        suggestions["correlation"] = {"params": None,
            "reason": "Needs at least 2 numeric columns. Please select manually."}

    # 6. Regression — first numeric as dependent, next 1-2 as predictors
    if len(num_cols_analytical) >= 2:
        dep = num_cols_analytical[0]
        preds = num_cols_analytical[1:3]
        suggestions["regression"] = {
            "params": {"dependent": dep, "predictors": preds},
            "reason": f"Predicting '{dep}' from {', '.join(preds)}.",
        }
    else:
        suggestions["regression"] = {"params": None,
            "reason": "Needs at least 2 numeric columns (1 outcome + 1+ predictors). Please select manually."}

    # 7. Reliability — numeric-coercible columns that look like Likert items
    # (low cardinality, e.g. 1-5 or 1-7 scale). These often get bucketed as
    # "categorical" upstream because of low unique-value counts, so we scan
    # ALL columns here by attempting numeric coercion directly, not just
    # the pre-classified num_cols list.
    likert_cands = []
    for c in [col["name"] for col in columns]:
        coerced = pd.to_numeric(df[c], errors="coerce")
        coercion_rate = coerced.notna().sum() / max(len(df), 1)
        nun = coerced.dropna().nunique()
        if coercion_rate > 0.9 and 3 <= nun <= 11:
            likert_cands.append(c)
    if len(likert_cands) >= 2:
        suggestions["reliability"] = {
            "params": {"columns": likert_cands[:8]},
            "reason": f"{min(len(likert_cands),8)} scale-like items detected for internal consistency check.",
        }
    else:
        suggestions["reliability"] = {"params": None,
            "reason": "Needs 2+ scale/Likert-style numeric columns (e.g. survey items rated 1-5). Please select manually."}

    return suggestions


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 — VALIDATE (block-until-fixed rule)
# ══════════════════════════════════════════════════════════════════════════

def validate_package_config(df: pd.DataFrame, config: dict) -> dict:
    """
    Validates a user-confirmed (possibly hand-edited) config dict of
    { test_key: params }. Per the confirmed rule: package generation
    is BLOCKED entirely if any of the 7 tests has missing/invalid params.

    Returns { ok: bool, errors: [ {test, message}, ... ] }
    """
    errors = []

    for test_key in PACKAGE_TESTS:
        params = config.get(test_key)
        if not params:
            errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                            "message": "No variables selected for this test."})
            continue

        # Per-test structural validation
        if test_key == "descriptive":
            if not params.get("columns"):
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": "Select at least 1 column."})

        elif test_key == "chi_square":
            v1, v2 = params.get("var1"), params.get("var2")
            if not v1 or not v2:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": "Select two categorical variables."})
            elif v1 == v2:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": "The two variables must be different."})

        elif test_key in ("t_test",):
            nv, gv = params.get("numeric_var"), params.get("group_var")
            if not nv or not gv:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": "Select a numeric variable and a grouping variable."})
            elif gv in df.columns and df[gv].dropna().nunique() != 2:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": f"'{gv}' must have exactly 2 groups (has {df[gv].dropna().nunique()})."})

        elif test_key == "anova":
            nv, gv = params.get("numeric_var"), params.get("group_var")
            if not nv or not gv:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": "Select a numeric variable and a grouping variable."})
            elif gv in df.columns and df[gv].dropna().nunique() < 3:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": f"'{gv}' must have 3+ groups (has {df[gv].dropna().nunique()})."})

        elif test_key == "correlation":
            cols = params.get("columns", [])
            if len(cols) < 2:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": "Select at least 2 numeric columns."})

        elif test_key == "regression":
            dep = params.get("dependent")
            preds = params.get("predictors", [])
            if not dep or not preds:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": "Select a dependent variable and at least 1 predictor."})
            elif dep in preds:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": "Dependent variable cannot also be a predictor."})

        elif test_key == "reliability":
            cols = params.get("columns", [])
            if len(cols) < 2:
                errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                "message": "Select at least 2 scale items."})

    return {"ok": len(errors) == 0, "errors": errors}


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 — RUN THE FULL PACKAGE
# ══════════════════════════════════════════════════════════════════════════

def run_thesis_package(df: pd.DataFrame, config: dict) -> dict:
    """
    Runs all 7 tests using the validated config. Assumes
    validate_package_config() has already returned ok=True — this
    function does NOT re-validate (caller's responsibility, per the
    confirmed "block until fixed" flow happening before this is called).

    Returns:
      { ok, results: { test_key: result_dict, ... }, errors: [...] }
    Note: even after passing structural validation, a test can still
    fail at runtime (e.g. degenerate data). Those are surfaced in
    `errors` but do not block the rest of the package from completing —
    this is a runtime safety net, distinct from the pre-flight block rule.
    """
    results = {}
    runtime_errors = []

    for test_key in PACKAGE_TESTS:
        params = config.get(test_key, {})
        try:
            if test_key == "reliability":
                r = reliability_analysis(df, params.get("columns", []))
            else:
                r = run_analysis(df, test_key, params)

            if "error" in r:
                runtime_errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                        "message": r["error"]})
                continue

            r["analysis_label"] = PACKAGE_TEST_LABELS[test_key]
            results[test_key] = r

        except Exception as e:
            runtime_errors.append({"test": test_key, "label": PACKAGE_TEST_LABELS[test_key],
                                    "message": str(e)})

    return {
        "ok": len(runtime_errors) == 0,
        "results": results,
        "errors": runtime_errors,
        "completed_count": len(results),
        "total_count": len(PACKAGE_TESTS),
    }


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4 — COMBINED REPORT (Word + PDF, one document covering all 7)
# ══════════════════════════════════════════════════════════════════════════

def generate_combined_docx(package_result: dict, dataset_name: str = "") -> bytes:
    """Builds one APA-style Word document covering all 7 completed tests in order."""
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from export_engine import (_apa_heading, _apa_table, _add_hor_rule, _strip_md,
                                 BRAND_NAVY, BRAND_BLUE, BRAND_SLATE, SITE, NOW)

    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Inches(1)

    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)

    # Cover page
    title_p = doc.add_paragraph(); title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title_p.add_run("Complete Statistical Analysis Report")
    r.bold = True; r.font.size = Pt(18); r.font.name = 'Times New Roman'; r.font.color.rgb = BRAND_NAVY

    doc.add_paragraph()
    sub_p = doc.add_paragraph(); sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub_p.add_run(f"Thesis / Research Package — {dataset_name or 'Dataset'}")
    sr.bold = True; sr.font.size = Pt(13); sr.font.color.rgb = BRAND_BLUE

    meta_p = doc.add_paragraph(); meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = meta_p.add_run(f"Generated: {NOW}\nPowered by Eduxellence Analytics — {SITE}")
    mr.font.size = Pt(10); mr.font.color.rgb = BRAND_SLATE; mr.font.italic = True

    _add_hor_rule(doc)
    doc.add_paragraph()

    # Table of contents
    _apa_heading(doc, "Contents", level=1)
    for i, key in enumerate(PACKAGE_TESTS, 1):
        if key in package_result["results"]:
            p = doc.add_paragraph()
            p.add_run(f"{i}. {PACKAGE_TEST_LABELS[key]}").font.size = Pt(11)
    doc.add_paragraph()
    _add_hor_rule(doc)

    # Each test section
    section_num = 1
    for key in PACKAGE_TESTS:
        result = package_result["results"].get(key)
        if not result:
            continue

        doc.add_page_break()
        _apa_heading(doc, f"{section_num}. {PACKAGE_TEST_LABELS[key]}", level=1)
        section_num += 1

        if result.get("apa_citation"):
            cite_p = doc.add_paragraph()
            cite_p.paragraph_format.left_indent = Inches(0.5)
            cr_ = cite_p.add_run(result["apa_citation"])
            cr_.italic = True; cr_.font.size = Pt(12)
            doc.add_paragraph()

        # Tables (reuse export_engine table renderer)
        for tkey in ("numeric_summary", "summary_table", "coef_table", "pairs_table",
                     "posthoc_table", "item_table"):
            if result.get(tkey):
                _apa_table(doc, result[tkey])
                doc.add_paragraph()

        if result.get("categorical_summary"):
            for cs in result["categorical_summary"]:
                p = doc.add_paragraph(); p.add_run(f"Frequency Table: {cs['variable']}").bold = True
                _apa_table(doc, cs["table"])
                doc.add_paragraph()

        if result.get("interpretation"):
            clean = _strip_md(result["interpretation"])
            for para_text in clean.split("\n\n"):
                if para_text.strip():
                    p = doc.add_paragraph()
                    p.paragraph_format.first_line_indent = Inches(0.5)
                    p.add_run(para_text.strip().replace("\n", " ")).font.size = Pt(12)
            doc.add_paragraph()

    # Footer
    doc.add_page_break()
    _apa_heading(doc, "Methodology Note", level=1)
    method_p = doc.add_paragraph()
    method_p.paragraph_format.first_line_indent = Inches(0.5)
    method_p.add_run(
        f"This complete statistical package was generated using the Eduxellence Analytics "
        f"platform ({SITE}). All {package_result['completed_count']} of {package_result['total_count']} "
        f"requested analyses completed successfully. Computations used Python's SciPy library; "
        f"results follow APA 7th Edition conventions. Generated on {NOW}."
    ).font.size = Pt(12)

    _add_hor_rule(doc)
    footer_p = doc.add_paragraph(); footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer_p.add_run(f"Generated by Eduxellence Analytics · {SITE} · {NOW}")
    fr.font.size = Pt(9); fr.font.italic = True; fr.font.color.rgb = BRAND_SLATE

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def generate_combined_pdf(package_result: dict, dataset_name: str = "") -> bytes:
    """Builds one branded PDF report covering all 7 completed tests, including charts."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                      HRFlowable, Image, KeepTogether, PageBreak)
    from export_engine import (_pdf_table, _strip_md, RL_NAVY, RL_BLUE, RL_SLATE,
                                 SITE, NOW)
    import base64

    buf = io.BytesIO()

    title_style = ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=20,
        textColor=RL_NAVY, alignment=TA_CENTER, spaceAfter=6)
    sub_style = ParagraphStyle('Sub', fontName='Helvetica-Bold', fontSize=13,
        textColor=RL_BLUE, alignment=TA_CENTER, spaceAfter=4)
    meta_style = ParagraphStyle('Meta', fontName='Helvetica-Oblique', fontSize=9,
        textColor=RL_SLATE, alignment=TA_CENTER, spaceAfter=16)
    h1_style = ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=14,
        textColor=RL_NAVY, spaceBefore=14, spaceAfter=8)
    body_style = ParagraphStyle('Body', fontName='Helvetica', fontSize=10,
        leading=16, alignment=TA_JUSTIFY, spaceAfter=8, firstLineIndent=18,
        textColor=colors.HexColor('#1E293B'))
    apa_style = ParagraphStyle('APA', fontName='Helvetica-Oblique', fontSize=10.5,
        textColor=RL_NAVY, leading=16, spaceAfter=10, leftIndent=18, rightIndent=18,
        borderColor=colors.HexColor('#C4B5FD'), borderWidth=1, borderPad=8,
        backColor=colors.HexColor('#F8F7FF'))
    footer_style = ParagraphStyle('Footer', fontName='Helvetica-Oblique', fontSize=8,
        textColor=RL_SLATE, alignment=TA_CENTER)

    story = []
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Complete Statistical Analysis Report", title_style))
    story.append(Paragraph(f"Thesis / Research Package — {dataset_name or 'Dataset'}", sub_style))
    story.append(Paragraph(f"Generated: {NOW} &nbsp;·&nbsp; Powered by Eduxellence Analytics &nbsp;·&nbsp; {SITE}", meta_style))
    story.append(HRFlowable(width="100%", thickness=2, color=RL_BLUE, spaceAfter=14))

    section_num = 1
    for key in PACKAGE_TESTS:
        result = package_result["results"].get(key)
        if not result:
            continue

        if section_num > 1:
            story.append(PageBreak())

        story.append(Paragraph(f"{section_num}. {PACKAGE_TEST_LABELS[key]}", h1_style))
        section_num += 1

        if result.get("apa_citation"):
            story.append(Paragraph(result["apa_citation"], apa_style))

        for tkey in ("numeric_summary", "summary_table", "coef_table", "pairs_table",
                     "posthoc_table", "item_table"):
            if result.get(tkey):
                story.append(_pdf_table(result[tkey]))
                story.append(Spacer(1, 10))

        if result.get("interpretation"):
            clean = _strip_md(result["interpretation"])
            for para_text in clean.split("\n\n"):
                if para_text.strip():
                    story.append(Paragraph(para_text.strip().replace("\n"," "), body_style))

        if result.get("charts"):
            for chart in result["charts"][:2]:   # cap at 2 charts/test to keep PDF size sane
                try:
                    img_bytes = base64.b64decode(chart["img"])
                    img = Image(io.BytesIO(img_bytes), width=6*inch, height=3.6*inch)
                    cap_style = ParagraphStyle('Cap', fontName='Helvetica-Oblique', fontSize=9,
                        textColor=RL_SLATE, alignment=TA_CENTER, spaceAfter=10)
                    story.append(KeepTogether([img, Paragraph(f"Figure. {chart.get('title','Chart')}", cap_style), Spacer(1,8)]))
                except Exception:
                    pass

    story.append(PageBreak())
    story.append(Paragraph("Methodology Note", h1_style))
    story.append(Paragraph(
        f"This complete statistical package was generated using the Eduxellence Analytics "
        f"platform ({SITE}). All {package_result['completed_count']} of {package_result['total_count']} "
        f"requested analyses completed successfully. Results follow APA 7th Edition conventions. "
        f"Generated on {NOW}.", body_style))
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0'), spaceAfter=6))
    story.append(Paragraph(f"Generated by Eduxellence Analytics · {SITE} · {NOW}", footer_style))

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=inch, rightMargin=inch,
        topMargin=inch, bottomMargin=inch,
        title="Complete Statistical Analysis Report — Eduxellence Analytics",
        author="Eduxellence Analytics")
    doc.build(story)
    buf.seek(0)
    return buf.read()
