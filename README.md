# Eduxellence Analytics — Free Data Tools

> **Live at:** [eduxellence.org](https://eduxellence.org)

A fully free, browser-based platform for professional data cleaning and statistical analysis. No login. No installation. No Python knowledge required.

---

## Tools Included

### 📊 Statistical Analysis (`/stats`)
Run 8 professional statistical tests with automatic assumption checking, APA-formatted output, publication-ready charts, and plain-English interpretation.

| Test | Type | What It Does |
|------|------|-------------|
| Descriptive Statistics | Descriptive | Mean, SD, skewness, histograms, Q-Q plots, pie charts |
| Chi-Square Test | Parametric | Independence between two categorical variables + Cramér's V |
| Independent T-Test | Parametric | Compare means between 2 groups + Cohen's d, 95% CI |
| One-Way ANOVA | Parametric | Compare means across 3+ groups + η², Bonferroni post-hoc |
| Correlation Analysis | Parametric | Pearson or Spearman, heatmap, scatter matrix |
| Linear Regression | Parametric | Simple or multiple, R², coefficients, residuals |
| Mann-Whitney U | Non-Parametric | Non-parametric T-Test alternative |
| Kruskal-Wallis | Non-Parametric | Non-parametric ANOVA alternative |

### 🧹 Data Cleaner (`/upload`)
Upload any messy CSV or Excel file. Get a clean version back in seconds.

11 automated cleaning stages: whitespace stripping · null standardisation · blank removal · type detection · category standardisation · date standardisation · currency parsing · duplicate removal · outlier flagging · missing value imputation · column intelligence report.

---

## Project Structure

```
eduxellence_stats/
├── api/
│   └── stats_app.py        ← Flask backend (all API routes)
├── public/
│   ├── index.html          ← Landing page
│   ├── stats.html          ← Statistical analysis tool UI
│   └── upload.html         ← Data cleaner tool UI (optional)
├── data/
│   └── samples/
│       └── student_survey.csv   ← Demo dataset (120 rows)
├── stats_engine.py         ← Core statistical engine
├── dataclean_pro.py        ← Core data cleaning engine
├── vercel.json             ← Vercel deployment config
├── requirements.txt        ← Python dependencies
└── README.md
```

---

## Local Development

```bash
# 1. Clone
git clone https://github.com/eduxellence/eduxellence-stats.git
cd eduxellence-stats

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python api/stats_app.py

# 4. Open in browser
# http://localhost:5001
```

---

## Deploy to Vercel (Free)

```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Login
vercel login

# 3. Deploy
vercel

# 4. Add your custom domain
# Vercel dashboard → Settings → Domains → stats.eduxellence.org
```

Vercel reads `vercel.json` automatically. No extra configuration needed.

**Free tier limits:**
- 10 MB file upload limit (larger files → consultation prompt)
- 30 second function execution timeout
- 1 GB memory per function
- 100 GB bandwidth/month

---

## API Reference

### `POST /api/upload`
Upload a CSV or Excel file.

**Body:** `multipart/form-data` with `file` field

**Returns:**
```json
{
  "session_id": "uuid",
  "meta_path": "/tmp/...",
  "rows": 120,
  "cols": 12,
  "columns": [...],
  "preview": [...],
  "recommendations": [...]
}
```

### `POST /api/assumptions`
Check statistical assumptions before running a test.

**Body:**
```json
{
  "meta_path": "/tmp/...",
  "analysis_type": "t_test",
  "params": { "numeric_var": "score", "group_var": "gender" }
}
```

### `POST /api/analyze`
Run a statistical test.

**Body:**
```json
{
  "meta_path": "/tmp/...",
  "analysis_type": "chi_square",
  "params": { "var1": "gender", "var2": "satisfied" }
}
```

**Analysis types:** `descriptive` · `chi_square` · `t_test` · `anova` · `correlation` · `regression` · `mann_whitney` · `kruskal_wallis`

### `POST /api/export`
Export results as a formatted Excel workbook (.xlsx).

### `POST /api/transform`
Apply a log transformation to a column.

**Body:** `{ "meta_path": "...", "column": "income", "transform": "log" }`

### `GET /api/health`
Returns `{ "status": "ok" }`

---

## Large File Notice

Files over 10 MB trigger a consultation prompt that redirects users to [eduxellence.org/#contact](https://eduxellence.org/#contact). For large-scale analysis, the expert team handles R, Python, SPSS, and EViews pipelines with full report delivery.

---

## About Eduxellence

Built and maintained by [Eduxellence Analytics](https://eduxellence.org) — elite data solutions for research teams, academic institutions, and enterprises.

**Services:** Advanced statistical analysis · Econometric modelling (EViews, R) · Data pipeline architecture · Thesis statistical support · Custom dashboards

📞 [Book a Free 15-Minute Data Strategy Audit](https://eduxellence.org/#contact)

---

## Licence

MIT — free to use, modify, and distribute. Attribution appreciated.

---

<div align="center">

**Built with ❤️ by [Eduxellence Analytics](https://eduxellence.org)**

[![Website](https://img.shields.io/badge/Website-eduxellence.org-blue?style=for-the-badge)](https://eduxellence.org)
[![Tool](https://img.shields.io/badge/Stats_Tool-Free-green?style=for-the-badge)](https://eduxellence.org/stats)

</div>
