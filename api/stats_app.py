"""
Eduxellence Statistical Platform — Flask API v2.0
Full-featured: upload, clean, analyse, export Excel, assumption checking, smart recommender.
Free tier Vercel compatible.  https://eduxellence.org
"""
import os, sys, json, uuid, tempfile, traceback, io
from io import BytesIO
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent.parent))
from stats_engine import run_analysis, check_assumptions, recommend_tests, ANALYSIS_LABELS, apply_log_transform
from export_engine import generate_docx, generate_pdf, generate_excel

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024   # 15 MB hard limit

ALLOWED = {"csv", "xlsx", "xls"}

def ok_ext(n):
    return "." in n and n.rsplit(".", 1)[1].lower() in ALLOWED

def load_df(path):
    ext = Path(path).suffix.lower()
    try:
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        for enc in ("utf-8", "latin-1", "cp1252", "utf-8-sig"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                pass
        return pd.read_csv(path, encoding="utf-8", errors="replace")
    except Exception as e:
        raise ValueError(f"Could not read file: {str(e)}")

def classify(df):
    out = []
    for c in df.columns:
        num = pd.to_numeric(df[c], errors="coerce")
        is_num = num.notna().sum() > len(df) * 0.5 and df[c].nunique() > 5
        out.append({
            "name": c,
            "dtype": "numeric" if is_num else "categorical",
            "n_unique": int(df[c].nunique()),
            "missing": int(df[c].isna().sum()),
            "missing_pct": round(100 * df[c].isna().sum() / max(len(df), 1), 1),
            "sample": [str(v) for v in df[c].dropna().head(4).tolist()]
        })
    return out

def to_excel(results, analysis_type):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        # Summary
        meta = {
            "Analysis": results.get("test", ""),
            "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Website": "https://eduxellence.org",
            "p-value": results.get("p_display", "—"),
            "Significance": results.get("significance", "—"),
            "APA Citation": results.get("apa_citation", "—")
        }
        pd.DataFrame([meta]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"}).to_excel(w, sheet_name="Summary", index=False)
        # Tables
        for key, sheet in [("numeric_summary", "Descriptive"), ("summary_table", "Group Summary"),
                           ("coef_table", "Coefficients"), ("pairs_table", "Correlations"), ("posthoc_table", "Post Hoc")]:
            if results.get(key):
                pd.DataFrame(results[key]).to_excel(w, sheet_name=sheet, index=False)
        if results.get("categorical_summary"):
            for cs in results["categorical_summary"]:
                pd.DataFrame(cs["table"]).to_excel(w, sheet_name=f"Freq_{cs['variable'][:20]}", index=False)
        # Interpretation
        pd.DataFrame({"Interpretation": [results.get("interpretation", "")]}).to_excel(w, sheet_name="Interpretation", index=False)
    buf.seek(0)
    return buf.read()

# ── Helper for "Too Large" Response ──────────────────────────────────────────
def large_file_response(message=None, cta_url="https://eduxellence.org/#contact", cta_label="📞 Book a Free Expert Consultation"):
    """Return a standardized response for analyses that exceed free tier limits."""
    return {
        "error": "analysis_too_large",
        "message": message or "This dataset exceeds the capacity of our free tool. For larger datasets, our expert consulting team can provide a full analysis with publication-ready output.",
        "cta_text": cta_label,
        "cta_url": cta_url
    }, 413

# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    p = Path(__file__).parent.parent / "public" / "index.html"
    if p.exists():
        return p.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}
    p = Path(__file__).parent.parent / "public" / "stats.html"
    return p.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}

@app.route("/stats", methods=["GET"])
def stats_page():
    p = Path(__file__).parent.parent / "public" / "stats.html"
    return p.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}

@app.route("/upload", methods=["GET"])
def upload_page():
    p = Path(__file__).parent.parent / "public" / "upload.html"
    if p.exists():
        return p.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}
    from flask import redirect
    return redirect("/stats")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "Eduxellence Stats v2", "site": "eduxellence.org"})

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file."}), 400
    f = request.files["file"]
    if not ok_ext(f.filename):
        return jsonify({"error": "Use .csv, .xlsx or .xls"}), 400

    # Size guard — redirect large files to consultation
    raw = f.read()
    file_size = len(raw)
    if file_size > 10 * 1024 * 1024:
        return jsonify({
            "error": "large_file",
            "message": "Your file exceeds 10 MB. For datasets of this size, our expert team can run a full analysis for you.",
            "cta_url": "https://eduxellence.org/#contact",
            "cta_label": "Book a Free Expert Consultation"
        }), 413
    f.stream = BytesIO(raw)  # reset stream for save()

    session_id = str(uuid.uuid4())
    tmp = tempfile.mkdtemp(prefix=f"edux_{session_id}_")
    path = os.path.join(tmp, secure_filename(f.filename))
    f.save(path)
    try:
        df = load_df(path)
        cols = classify(df)
        prev = df.head(6).where(df.head(6).notna(), other=None).to_dict(orient="records")
        meta_path = path + ".meta"
        with open(meta_path, "w") as mf:
            json.dump({"path": path}, mf)
        recs = recommend_tests(df, cols)
        return jsonify({
            "session_id": session_id,
            "meta_path": meta_path,
            "filename": f.filename,
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "columns": cols,
            "preview": prev,
            "recommendations": recs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/assumptions", methods=["POST"])
def assumptions():
    data = request.get_json()
    meta_path = data.get("meta_path")
    analysis_type = data.get("analysis_type")
    params = data.get("params", {})
    if not meta_path or not os.path.exists(meta_path):
        return jsonify({"error": "Session expired."}), 400
    try:
        with open(meta_path) as mf:
            meta = json.load(mf)
        df = load_df(meta["path"])
        checks = check_assumptions(df, analysis_type, params)
        return jsonify({"checks": checks, "all_passed": all(c["passed"] for c in checks)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/transform", methods=["POST"])
def transform():
    data = request.get_json()
    meta_path = data.get("meta_path")
    column = data.get("column")
    transform_type = data.get("transform", "log")
    if not meta_path or not os.path.exists(meta_path):
        return jsonify({"error": "Session expired."}), 400
    try:
        with open(meta_path) as mf:
            meta = json.load(mf)
        df = load_df(meta["path"])
        if transform_type == "log":
            df_new, new_col = apply_log_transform(df, column)
        else:
            return jsonify({"error": "Unknown transform."}), 400
        new_path = meta["path"].replace(".csv", "_transformed.csv").replace(".xlsx", "_transformed.csv")
        df_new.to_csv(new_path, index=False)
        new_meta = new_path + ".meta"
        with open(new_meta, "w") as mf:
            json.dump({"path": new_path}, mf)
        return jsonify({
            "meta_path": new_meta,
            "new_column": new_col,
            "message": f"Log transform applied. New column '{new_col}' added.",
            "columns": classify(df_new)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    meta_path = data.get("meta_path")
    analysis_type = data.get("analysis_type")
    params = data.get("params", {})

    if not meta_path or not os.path.exists(meta_path):
        return jsonify({"error": "Session expired. Re-upload your file."}), 400

    if analysis_type not in ANALYSIS_LABELS:
        return jsonify({"error": f"Unknown analysis."}), 400

    try:
        with open(meta_path) as mf:
            meta = json.load(mf)
        df = load_df(meta["path"])
    except Exception as e:
        return jsonify({"error": f"Cannot load file: {e}"}), 500

    try:
        results = run_analysis(df, analysis_type, params)
    except Exception as e:
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500

    # ── Handle analysis_too_large from stats_engine ──
    if isinstance(results, dict) and results.get("error") == "analysis_too_large":
        return jsonify({
            "error": "analysis_too_large",
            "message": results.get("message", "This analysis exceeds the capacity of our free tool."),
            "cta_text": results.get("cta_text", "📞 Book a Free Expert Consultation"),
            "cta_url": results.get("cta_url", "https://eduxellence.org/#contact")
        }), 413

    if "error" in results:
        return jsonify(results), 422

    results["analysis_label"] = ANALYSIS_LABELS[analysis_type]
    results["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["powered_by"] = "Eduxellence Analytics — https://eduxellence.org"

    return jsonify(results)

@app.route("/api/export", methods=["POST"])
def export():
    """
    POST /api/export
    Body: { meta_path, analysis_type, params, format }
    format: "xlsx" (default) | "docx" | "pdf"
    """
    data = request.get_json()
    meta_path = data.get("meta_path")
    analysis_type = data.get("analysis_type")
    params = data.get("params", {})
    fmt = data.get("format", "xlsx").lower()

    if not meta_path or not os.path.exists(meta_path):
        return jsonify({"error": "Session expired."}), 400

    try:
        with open(meta_path) as mf:
            meta = json.load(mf)
        df = load_df(meta["path"])
        results = run_analysis(df, analysis_type, params)

        # ── Handle analysis_too_large from stats_engine ──
        if isinstance(results, dict) and results.get("error") == "analysis_too_large":
            return jsonify({
                "error": "analysis_too_large",
                "message": results.get("message", "This analysis exceeds the capacity of our free tool."),
                "cta_text": results.get("cta_text", "📞 Book a Free Expert Consultation"),
                "cta_url": results.get("cta_url", "https://eduxellence.org/#contact")
            }), 413

        results["analysis_label"] = ANALYSIS_LABELS.get(analysis_type, analysis_type)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"eduxellence_{analysis_type}_{ts}"

        if fmt == "docx":
            file_bytes = generate_docx(results, stem)
            return send_file(
                io.BytesIO(file_bytes),
                as_attachment=True,
                download_name=f"{stem}.docx",
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        elif fmt == "pdf":
            file_bytes = generate_pdf(results, stem)
            return send_file(
                io.BytesIO(file_bytes),
                as_attachment=True,
                download_name=f"{stem}.pdf",
                mimetype="application/pdf"
            )
        else:  # xlsx default
            file_bytes = generate_excel(results)
            return send_file(
                io.BytesIO(file_bytes),
                as_attachment=True,
                download_name=f"{stem}.xlsx",
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  Eduxellence Stats Platform — http://localhost:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)
