"""
Eduxellence Statistical Analysis Engine v2.0
============================================
8 statistical tests · Auto assumption checking · Smart test recommender
APA-formatted output · Publication-ready charts via matplotlib/seaborn
Zero paid dependencies. Vercel free-tier compatible.
Supabase Storage integration for charts to prevent payload size limits.

by Eduxellence Analytics · https://eduxellence.org
"""

import io, base64, warnings, traceback, os, json, uuid, logging, sys, signal
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# ── Version Metadata ──────────────────────────────────────────────────────
ENGINE_VERSION = "2.0.1"
ENGINE_BUILD = datetime.now().strftime("%Y%m%d")

# ── Configure Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
MAX_RUNTIME_SECONDS = 25  # Leave buffer before Vercel's 30s limit
MAX_MEMORY_MB = 500  # Conservative memory limit
MAX_CHARTS = 10
MAX_PAIRPLOT_VARS = 4
MAX_PAIRPLOT_SAMPLE = 3000
MAX_ROWS = 50000
MAX_COLS = 100
MAX_DESCRIPTIVE_COLS = 15
MAX_PREDICTORS = 10
RANDOM_SEED = 42  # For reproducibility
API_RATE_LIMIT = 10  # Requests per minute per IP

# ── Error Codes ────────────────────────────────────────────────────────────
ERROR_CODES = {
    "VAL_001": "Column not found in dataset",
    "VAL_002": "Column contains no valid numeric data",
    "VAL_003": "Column has insufficient unique values",
    "VAL_004": "Dataset has insufficient rows",
    "VAL_005": "Too many predictors for regression",
    "VAL_006": "Duplicate predictor names found",
    "VAL_007": "Dataset exceeds maximum allowed rows",
    "VAL_008": "Dataset exceeds maximum allowed columns",
    "VAL_009": "All predictors are constant",
    "VAL_010": "Insufficient sample size for regression",
    "LIM_001": "Analysis exceeded time limit",
    "LIM_002": "Analysis exceeded memory limit",
    "LIM_003": "Analysis too large for free tier",
    "LIM_004": "API rate limit exceeded",
    "ERR_001": "Internal server error",
    "ERR_002": "Chi-square test failed",
    "ERR_003": "Regression matrix is singular",
    "ERR_004": "Shapiro-Wilk test failed",
}

# ── Targeted Warnings Filtering ──────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# ── Supabase Storage Integration ──────────────────────────────────────────
try:
    from supabase import create_client, Client
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "eduxellence-charts")
    supabase_available = bool(SUPABASE_URL and SUPABASE_KEY)
    if supabase_available:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    else:
        logger.warning("Supabase credentials not found. Charts will use base64 encoding.")
except ImportError:
    supabase_available = False
    supabase = None
    logger.warning("Supabase package not installed. Charts will use base64 encoding.")

# ── Custom Exceptions ──────────────────────────────────────────────────────
class AnalysisTooLargeError(Exception):
    """Raised when analysis exceeds free tier limits."""
    pass

class AnalysisTimeoutError(Exception):
    """Raised when analysis exceeds time limits."""
    pass

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass

class MemoryLimitError(Exception):
    """Raised when analysis exceeds memory limits."""
    pass

class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass

class SingularMatrixError(Exception):
    """Raised when regression matrix is singular."""
    pass

# ── API Rate Limiting ────────────────────────────────────────────────────
_rate_limit_store = {}

def check_rate_limit(ip_address, limit=API_RATE_LIMIT, window_seconds=60):
    """Check if IP has exceeded rate limit."""
    from collections import defaultdict
    from datetime import datetime, timedelta
    
    current_time = datetime.now()
    
    if ip_address not in _rate_limit_store:
        _rate_limit_store[ip_address] = []
    
    # Clean old entries
    _rate_limit_store[ip_address] = [
        t for t in _rate_limit_store[ip_address] 
        if current_time - t < timedelta(seconds=window_seconds)
    ]
    
    if len(_rate_limit_store[ip_address]) >= limit:
        raise RateLimitError(ERROR_CODES["LIM_004"])
    
    _rate_limit_store[ip_address].append(current_time)
    return True

# ── Memory Check ──────────────────────────────────────────────────────────
def check_memory_usage(df, max_memory_mb=MAX_MEMORY_MB):
    """Check if dataset memory usage exceeds limit."""
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if memory_mb > max_memory_mb:
        raise MemoryLimitError(
            f"Dataset memory usage ({memory_mb:.1f} MB) exceeds limit ({max_memory_mb} MB). "
            f"Please reduce dataset size."
        )
    return memory_mb

# ── Timeout Protection ────────────────────────────────────────────────────
def with_timeout(timeout_seconds=MAX_RUNTIME_SECONDS):
    """Decorator to enforce runtime limits using ThreadPoolExecutor."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout_seconds)
                    return result
                except FutureTimeoutError:
                    raise AnalysisTimeoutError(ERROR_CODES["LIM_001"])
                except Exception as e:
                    raise e
        return wrapper
    return decorator

# ── Brand palette ──────────────────────────────────────────────────────────
P = {
    "navy":"#0B1829","navy2":"#112240","blue":"#1E6BFF","blue2":"#4B8AFF",
    "teal":"#0FC9A0","teal2":"#09A882","white":"#FFFFFF","off":"#F7F9FC",
    "slate":"#64748B","slateL":"#CBD5E1","border":"#E2E8F0",
    "ok":"#16A34A","err":"#DC2626","warn":"#D97706","text":"#0F172A",
}
PAL = ["#1E6BFF","#0FC9A0","#F59E0B","#8B5CF6","#EC4899","#EF4444","#10B981","#F97316","#06B6D4","#84CC16"]

# ── Chart Storage Handler ──────────────────────────────────────────────────
def _store_chart(fig, chart_name, dpi=110, use_storage=True):
    """Store chart either as base64 (small) or Supabase Storage (large)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    img_data = buf.read()
    
    size_mb = len(img_data) / (1024 * 1024)
    
    if use_storage and size_mb > 1.0 and supabase_available and supabase:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"charts/{timestamp}_{unique_id}_{chart_name.replace(' ', '_')}.png"
            
            # ── FIXED: Supabase upload API compatibility ──────────────────
            try:
                # Try newer SDK format
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    file=img_data,
                    path=filename,
                    file_options={"content-type": "image/png"}
                )
            except TypeError:
                # Fallback to older SDK format
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    filename,
                    img_data,
                    {"content-type": "image/png"}
                )
            
            public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
            
            plt.close(fig)
            return {"type": "url", "url": public_url, "size_mb": round(size_mb, 2)}
            
        except Exception as e:
            logger.error(f"Supabase upload failed: {str(e)}. Falling back to base64.")
            result = base64.b64encode(img_data).decode()
            plt.close(fig)
            return {"type": "base64", "data": result, "size_mb": round(size_mb, 2)}
    
    result = base64.b64encode(img_data).decode()
    plt.close(fig)
    return {"type": "base64", "data": result, "size_mb": round(size_mb, 2)}

def _b64(fig, dpi=110):
    """Legacy function - maintains compatibility."""
    result = _store_chart(fig, "chart", dpi)
    return result["data"] if result["type"] == "base64" else result["url"]

def _style(ax_list):
    for ax in (ax_list if hasattr(ax_list,"__iter__") and not isinstance(ax_list,plt.Axes) else [ax_list]):
        ax.set_facecolor(P["off"]); ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(P["border"])
        ax.tick_params(colors=P["slate"],labelsize=9)
        for lbl in [ax.xaxis.label, ax.yaxis.label]: lbl.set_color(P["slate"])
        if ax.get_title(): ax.title.set_color(P["navy"])

def sp(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
def fp(p): return "p < .001" if p<.001 else f"p = {p:.3f}"
def verdict(p,a=.05): sig=p<a; return ("statistically significant" if sig else "not statistically significant"),("reject" if sig else "fail to reject")

# ── FIXED: Input Validation Layer ─────────────────────────────────────────
def validate_numeric_column(df, column, context=""):
    """Validate that a column exists and is numeric."""
    if column not in df.columns:
        raise ValidationError(f"{ERROR_CODES['VAL_001']}: {column}{': ' + context if context else ''}")
    
    test_series = pd.to_numeric(df[column], errors='coerce')
    if test_series.isna().all():
        raise ValidationError(f"{ERROR_CODES['VAL_002']}: {column}{': ' + context if context else ''}")
    
    return True

def validate_categorical_column(df, column, context=""):
    """Validate that a column exists and can be treated as categorical."""
    if column not in df.columns:
        raise ValidationError(f"{ERROR_CODES['VAL_001']}: {column}{': ' + context if context else ''}")
    
    n_unique = df[column].nunique()
    if n_unique < 2:
        raise ValidationError(f"{ERROR_CODES['VAL_003']}: {column} has only {n_unique} unique value(s){': ' + context if context else ''}")
    
    return True

def validate_row_count(df, min_rows=3, context=""):
    """Validate that dataset has sufficient rows."""
    if len(df) < min_rows:
        raise ValidationError(f"{ERROR_CODES['VAL_004']}: Dataset has only {len(df)} rows. Minimum required: {min_rows}{': ' + context if context else ''}")
    return True

# ── FIXED: validate_predictor_count with correct parameter order ──────────
def validate_predictor_count(predictors, max_predictors=MAX_PREDICTORS, context=""):
    """Validate number of predictors."""
    if len(predictors) > max_predictors:
        raise ValidationError(f"{ERROR_CODES['VAL_005']}: {len(predictors)} predictors. Maximum allowed: {max_predictors}{': ' + context if context else ''}")
    if len(set(predictors)) < len(predictors):
        raise ValidationError(f"{ERROR_CODES['VAL_006']}: Duplicate predictor names found{': ' + context if context else ''}")
    return True

def validate_no_empty_columns(df):
    """Check for completely empty columns."""
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        warnings.warn(f"Found completely empty columns: {empty_cols}. They will be dropped.")
        df = df.drop(columns=empty_cols)
    return df

def validate_no_duplicate_columns(df):
    """Check for duplicate column names."""
    if len(df.columns) != len(set(df.columns)):
        warnings.warn("Duplicate column names detected. Renaming to unique names.")
        df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df

# ── Global Data Cleaning Function ──────────────────────────────────────────
def prepare_dataframe(df):
    """Standardize and clean dataset before analysis."""
    logger.info(f"Preparing dataframe with {len(df)} rows and {len(df.columns)} columns")
    
    # Remove infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove empty columns
    df = validate_no_empty_columns(df)
    
    # Handle duplicate columns
    df = validate_no_duplicate_columns(df)
    
    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    
    # Convert obvious numerics
    for col in df.columns:
        if df[col].dtype == 'object':
            converted = pd.to_numeric(df[col], errors='coerce')
            if not converted.isna().all():
                df[col] = converted
    
    # Drop constant columns with warning
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        warnings.warn(f"Dropped constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    
    logger.info(f"Dataframe prepared: {len(df)} rows, {len(df.columns)} columns")
    return df

# ── Helper for Performance Checks ──────────────────────────────────────────
def check_analysis_limits(df, analysis_type, params):
    """Raises AnalysisTooLargeError if dataset exceeds free tier limits."""
    rows = len(df)
    if rows > MAX_ROWS:
        raise AnalysisTooLargeError(
            f"{ERROR_CODES['LIM_003']}: {rows:,} rows exceeds limit of {MAX_ROWS:,} rows. "
            "For larger datasets, our expert consulting team can provide a full analysis."
        )

    if analysis_type == "descriptive":
        cols = params.get("columns", [])
        if len(cols) > MAX_DESCRIPTIVE_COLS:
            raise AnalysisTooLargeError(
                f"{ERROR_CODES['LIM_003']}: {len(cols)} columns exceeds limit of {MAX_DESCRIPTIVE_COLS} columns. "
                "For such a comprehensive analysis, our expert team can generate a full report with all charts and statistics."
            )

    if analysis_type == "regression":
        predictors = params.get("predictors", [])
        if len(predictors) > MAX_PREDICTORS:
            raise AnalysisTooLargeError(
                f"{ERROR_CODES['LIM_003']}: {len(predictors)} predictors exceeds limit of {MAX_PREDICTORS} predictors. "
                "Our expert consulting team can build and interpret more complex models for you."
            )
    
    # Check memory
    check_memory_usage(df)

# ── Security: CSV Bomb Protection ──────────────────────────────────────────
def validate_dataset(df, max_rows=MAX_ROWS, max_cols=MAX_COLS):
    """Validate dataset to prevent CSV bombs and malicious data."""
    if len(df) > max_rows:
        raise AnalysisTooLargeError(
            f"{ERROR_CODES['VAL_007']}: {len(df):,} rows. Maximum allowed is {max_rows:,}. "
            "For larger datasets, our expert consulting team can provide a full analysis."
        )
    if len(df.columns) > max_cols:
        raise AnalysisTooLargeError(
            f"{ERROR_CODES['VAL_008']}: {len(df.columns)} columns. Maximum allowed is {max_cols}. "
            "For such comprehensive data, our expert consulting team can provide a full analysis."
        )
    
    df = prepare_dataframe(df)
    return df

# ── Assumption Checker ──────────────────────────────────────────────────────
def check_assumptions(df, test_type, params):
    """Returns list of assumption check dicts with status, message, suggestion, fix_available."""
    checks = []
    def chk(name, passed, note, suggestion="", fix=None):
        checks.append({"name":name,"passed":bool(passed),"note":note,"suggestion":suggestion,"fix":fix})

    num_var  = params.get("numeric_var") or params.get("dependent")
    grp_var  = params.get("group_var")

    if test_type in ("t_test","anova","mann_whitney","kruskal_wallis") and num_var and grp_var:
        series = pd.to_numeric(df[num_var], errors="coerce").dropna()

        # 1. Sample size
        n = len(series)
        chk("Adequate sample size", n >= 30,
            f"N = {n}. {'Sufficient (≥ 30).' if n>=30 else 'Small sample — results may be unreliable.'}",
            "" if n>=30 else "Interpret results with caution for small samples.")

        # 2. Normality - FIXED: Check for constant values first
        if n >= 3:
            # ── FIXED: Check if series has enough variation ──────────────
            if series.nunique() > 2:
                try:
                    if n < 5000:
                        sw_stat, sw_p = stats.shapiro(series)
                        normal = sw_p > 0.05
                        chk("Normality (Shapiro-Wilk)", normal,
                            f"W = {sw_stat:.3f}, {fp(sw_p)}. {'Distribution appears normal.' if normal else 'Distribution is not normal.'}",
                            "" if normal else "Consider Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups) as non-parametric alternatives.",
                            "log_transform" if not normal else None)
                    else:
                        from scipy.stats import normaltest
                        k2_stat, k2_p = normaltest(series)
                        normal = k2_p > 0.05
                        chk("Normality (D'Agostino K²)", normal,
                            f"K² = {k2_stat:.3f}, {fp(k2_p)}. {'Distribution appears normal.' if normal else 'Distribution is not normal.'}",
                            "" if normal else "Consider Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups) as non-parametric alternatives.",
                            "log_transform" if not normal else None)
                except Exception as e:
                    chk("Normality check", False,
                        f"Could not compute normality test: {str(e)}",
                        "Consider using non-parametric alternatives.")
            else:
                chk("Normality check", False,
                    "Variable has insufficient variation for normality test.",
                    "Consider using non-parametric alternatives.")

        # 3. Outliers
        Q1,Q3=series.quantile(.25),series.quantile(.75); IQR=Q3-Q1
        n_out=int(((series<Q1-3*IQR)|(series>Q3+3*IQR)).sum())
        chk("No extreme outliers", n_out==0,
            f"{n_out} extreme outlier(s) detected (IQR×3 method)." if n_out else "No extreme outliers found.",
            "Review extreme values and consider whether they are valid data points." if n_out else "")

        # 4. Equal variances (Levene) — for t-test/anova
        if test_type in ("t_test","anova") and grp_var:
            groups = [pd.to_numeric(df.loc[df[grp_var]==g,num_var],errors="coerce").dropna() for g in df[grp_var].dropna().unique()]
            if len(groups)>=2 and all(len(g)>=2 for g in groups):
                lev_s, lev_p = stats.levene(*groups)
                eq_var = lev_p > 0.05
                chk("Equal variances (Levene's test)", eq_var,
                    f"Levene's F = {lev_s:.3f}, {fp(lev_p)}. {'Variances are equal.' if eq_var else 'Unequal variances detected — Welch correction applied automatically.'}",
                    "" if eq_var else "Welch's t-test is applied automatically when variances are unequal.")

    if test_type == "chi_square":
        v1,v2 = params.get("var1",""),params.get("var2","")
        if v1 and v2:
            ct = pd.crosstab(df[v1],df[v2])
            if ct.empty or ct.sum().sum() == 0:
                chk("Valid contingency table", False,
                    "Contingency table is empty or has zero totals.",
                    "Check categorical variables for missing or invalid values.")
            else:
                try:
                    chi2, p, dof, exp = stats.chi2_contingency(ct)
                    exp_low = (exp < 5).any()
                    chk("Expected frequencies ≥ 5", not exp_low,
                        "All expected cell frequencies are ≥ 5." if not exp_low else "Some expected frequencies < 5. Chi-square may not be reliable.",
                        "Consider collapsing categories or using Fisher's exact test." if exp_low else "")
                except Exception as e:
                    chk("Valid chi-square test", False,
                        f"Could not compute chi-square: {str(e)}",
                        "Check data for issues and try again.")

    if test_type == "regression":
        predictors = params.get("predictors", [])
        if num_var and predictors:
            sub = df[[num_var]+predictors].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) > len(predictors)+1:
                import statsmodels.api as sm
                X = sm.add_constant(sub[predictors].values)
                
                # ── FIXED: Check for singular matrix ──────────────────────
                if np.linalg.matrix_rank(X) < X.shape[1]:
                    chk("Non-singular matrix", False,
                        "Design matrix is singular. Perfect collinearity detected.",
                        "Remove redundant predictors or combine correlated ones.")
                    return checks
                
                model = sm.OLS(sub[num_var].values, X).fit()
                
                # Normality of residuals
                if len(model.resid) < 5000:
                    try:
                        sw_s, sw_p2 = stats.shapiro(model.resid)
                        chk("Normality of residuals", sw_p2>.05,
                            f"Shapiro-Wilk on residuals: W = {sw_s:.3f}, {fp(sw_p2)}",
                            "Non-normal residuals suggest the linear model may not be the best fit." if sw_p2<=.05 else "")
                    except:
                        chk("Normality of residuals", False,
                            "Could not compute Shapiro-Wilk test.",
                            "Consider using non-parametric alternatives.")
                else:
                    from scipy.stats import normaltest
                    k2_stat, k2_p = normaltest(model.resid)
                    chk("Normality of residuals", k2_p>.05,
                        f"D'Agostino K² on residuals: K² = {k2_stat:.3f}, {fp(k2_p)}",
                        "Non-normal residuals suggest the linear model may not be the best fit." if k2_p<=.05 else "")

                # Homoscedasticity (Breusch-Pagan)
                from statsmodels.stats.diagnostic import het_breuschpagan
                bp_test = het_breuschpagan(model.resid, model.model.exog)
                chk("Homoscedasticity", bp_test[1] > 0.05,
                    f"Breusch-Pagan p = {bp_test[1]:.4f}. {'Homoscedastic.' if bp_test[1] > 0.05 else 'Heteroscedasticity detected.'}",
                    "Consider robust standard errors or data transformation." if bp_test[1] <= 0.05 else "")

                # VIF for multicollinearity
                if len(predictors) > 1:
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    try:
                        X_vif = sub[predictors].values
                        vif_data = pd.DataFrame()
                        vif_data["Variable"] = predictors
                        vif_data["VIF"] = [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
                        max_vif = vif_data["VIF"].max()
                        chk("No multicollinearity (VIF)", float(max_vif) < 10,
                            f"Max VIF = {max_vif:.3f}. {'Acceptable (<10).' if max_vif < 10 else 'High multicollinearity detected (>10).'}",
                            "Remove or combine highly correlated predictors." if max_vif >= 10 else "")
                    except Exception as e:
                        corr_mat = sub[predictors].corr()
                        max_corr = corr_mat.where(~np.eye(len(predictors),dtype=bool)).abs().max().max()
                        chk("No multicollinearity (Correlation)", float(max_corr)<0.9,
                            f"Max predictor correlation = {max_corr:.3f}. {'Acceptable.' if max_corr<0.9 else 'High multicollinearity detected.'}",
                            "Remove or combine highly correlated predictors." if max_corr>=0.9 else "")

    return checks


# ══════════════════════════════════════════════════════════════════════════════
# SMART TEST RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
def recommend_tests(df, columns):
    """Given column metadata, suggest appropriate statistical tests."""
    cols = {c["name"]: c for c in columns}
    num  = [c["name"] for c in columns if c["dtype"].startswith("num") and c["n_unique"]>5]
    cat  = [c["name"] for c in columns if not c["dtype"].startswith("num") or c["n_unique"]<=10]
    cat2 = [c for c in cat if df[c].nunique()==2]
    cat3p= [c for c in cat if df[c].nunique()>=3]

    suggestions = []
    if len(num)>=1:
        suggestions.append({"test":"descriptive","reason":"Summarise your numeric variables with mean, SD, skewness and distribution charts.","priority":1})
    if len(cat)>=2:
        suggestions.append({"test":"chi_square","reason":f"Test association between categorical variables (e.g. {cat[0]} × {cat[1] if len(cat)>1 else '…'}).","priority":2})
    if num and cat2:
        suggestions.append({"test":"t_test","reason":f"Compare {num[0]} between two groups in {cat2[0]}.","priority":2})
    if num and cat3p:
        suggestions.append({"test":"anova","reason":f"Compare {num[0]} across 3+ groups in {cat3p[0]}.","priority":3})
    if len(num)>=2:
        suggestions.append({"test":"correlation","reason":f"Explore relationships between numeric variables ({', '.join(num[:3])}).","priority":2})
    if len(num)>=2:
        suggestions.append({"test":"regression","reason":f"Predict {num[0]} from other numeric variables.","priority":3})
    if num and cat2:
        suggestions.append({"test":"mann_whitney","reason":"Non-parametric alternative if normality assumption is violated.","priority":4})

    return sorted(suggestions, key=lambda x: x["priority"])


# ══════════════════════════════════════════════════════════════════════════════
# LOG TRANSFORM HELPER
# ══════════════════════════════════════════════════════════════════════════════
def apply_log_transform(df, column):
    s = pd.to_numeric(df[column], errors="coerce")
    if (s<=0).any(): s = s - s.min() + 1
    df_out = df.copy()
    df_out[column+"_log"] = np.log(s)
    return df_out, column+"_log"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
@with_timeout(MAX_RUNTIME_SECONDS)
def descriptive_statistics(df, columns):
    # ── Performance Guard ──────────────────────────────────────────────────
    check_analysis_limits(df, "descriptive", {"columns": columns})
    
    # ── Input Validation ──────────────────────────────────────────────────
    for col in columns:
        if col not in df.columns:
            raise ValidationError(f"{ERROR_CODES['VAL_001']}: {col}")

    num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c],errors="coerce"))]
    num_cols = [c for c in columns if pd.to_numeric(df[c],errors="coerce").notna().sum() > len(df)*0.5]
    cat_cols = [c for c in columns if c not in num_cols]
    charts=[]; numeric_summary=[]; categorical_summary=[]
    
    # ── FIXED: Track data cleaning statistics ────────────────────────────
    original_rows = len(df)
    original_cols = len(columns)

    if num_cols:
        for c in num_cols:
            s=pd.to_numeric(df[c],errors="coerce").dropna()
            if len(s)==0: continue
            # ── FIXED: Correct missing value count ──────────────────────
            missing_mask = (
                df[c].isna() |
                df[c].astype(str).str.strip().isin(["", "N/A"])
            )
            missing_count = missing_mask.sum()
            
            # Shapiro-Wilk with fallback - check variation first
            if len(s) >= 3 and s.nunique() > 2:
                try:
                    if len(s) < 5000:
                        sw_s,sw_p=stats.shapiro(s)
                    else:
                        from scipy.stats import normaltest
                        sw_s, sw_p = normaltest(s)
                except:
                    sw_s, sw_p = np.nan, np.nan
            else:
                sw_s,sw_p = np.nan,np.nan
            numeric_summary.append({"Variable":c,"N":int(s.count()),"Missing":int(missing_count),
                "Mean":round(float(s.mean()),4),"Median":round(float(s.median()),4),"Std Dev":round(float(s.std()),4),
                "Min":round(float(s.min()),4),"Max":round(float(s.max()),4),"Q1":round(float(s.quantile(.25)),4),
                "Q3":round(float(s.quantile(.75)),4),"Skewness":round(float(s.skew()),4),
                "Kurtosis":round(float(s.kurt()),4),"Shapiro-Wilk p":round(float(sw_p),4) if not np.isnan(sw_p) else "—"})

        # ── Limit Charts to Top 15 Selected Columns ──────────────────────
        selected_num_cols = num_cols[:15]
        num_charts = len(selected_num_cols)

        # ── FIXED: Chart Generation Limits using global constant ─────────
        chart_count = 0

        # Histogram grid
        if chart_count < MAX_CHARTS and num_charts > 0:
            nc=min(3, num_charts); nr=(num_charts+nc-1)//nc
            fig,axes=plt.subplots(nr,nc,figsize=(5.5*nc,4*nr)); fig.patch.set_facecolor(P["white"])
            axf=np.array(axes).flatten() if num_charts>1 else [axes]
            for i,c in enumerate(selected_num_cols):
                ax=axf[i]; s=pd.to_numeric(df[c],errors="coerce").dropna()
                ax.hist(s,bins="auto",color=PAL[i%len(PAL)],alpha=.85,edgecolor="white",linewidth=.5)
                ax.axvline(s.mean(),color=P["navy"],ls="--",lw=1.5,label=f"M={s.mean():.2f}")
                ax.axvline(s.median(),color=P["err"],ls=":",lw=1.5,label=f"Mdn={s.median():.2f}")
                ax.set_title(c,fontsize=11,fontweight="bold",color=P["navy"]); ax.legend(fontsize=8)
                ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
            for j in range(i+1,len(axf)): axf[j].set_visible(False)
            _style(axf[:num_charts]); fig.suptitle("Distribution Histograms (First 15 Variables)",fontsize=13,fontweight="bold",color=P["navy"],y=1.01)
            plt.tight_layout(); charts.append({"title":"Histograms (First 15)","img":_store_chart(fig, "histograms", 110)})
            chart_count += 1

            # Boxplot grid
            if num_charts > 1 and chart_count < MAX_CHARTS:
                fig2,ax2=plt.subplots(figsize=(max(9, num_charts*1.6),5)); fig2.patch.set_facecolor(P["white"])
                data=[pd.to_numeric(df[c],errors="coerce").dropna().values for c in selected_num_cols]
                bp=ax2.boxplot(data,patch_artist=True,medianprops=dict(color=P["navy"],lw=2),flierprops=dict(marker="o",ms=4,alpha=.5))
                for patch,col in zip(bp["boxes"],PAL): patch.set_facecolor(col); patch.set_alpha(.75)
                ax2.set_xticklabels(selected_num_cols,rotation=30,ha="right",fontsize=9)
                ax2.set_title("Comparative Boxplots (First 15 Variables)",fontweight="bold",color=P["navy"])
                _style(ax2); plt.tight_layout(); charts.append({"title":"Boxplots (First 15)","img":_store_chart(fig2, "boxplots", 110)})
                chart_count += 1

            # Q-Q plots
            if chart_count < MAX_CHARTS:
                fig3,axes3=plt.subplots(nr,nc,figsize=(5*nc,4*nr)); fig3.patch.set_facecolor(P["white"])
                axf3=np.array(axes3).flatten() if num_charts>1 else [axes3]
                for i,c in enumerate(selected_num_cols):
                    s=pd.to_numeric(df[c],errors="coerce").dropna()
                    stats.probplot(s,dist="norm",plot=axf3[i])
                    axf3[i].set_title(f"Q-Q: {c}",fontsize=10,fontweight="bold",color=P["navy"])
                    axf3[i].get_lines()[0].set(color=PAL[i%len(PAL)],alpha=.7,ms=4)
                    axf3[i].get_lines()[1].set(color=P["err"],lw=1.5)
                for j in range(i+1,len(axf3)): axf3[j].set_visible(False)
                _style(axf3[:num_charts]); fig3.suptitle("Q-Q Plots (Normality Check, First 15)",fontsize=13,fontweight="bold",color=P["navy"],y=1.01)
                plt.tight_layout(); charts.append({"title":"Q-Q Plots (First 15)","img":_store_chart(fig3, "qqplots", 110)})
                chart_count += 1

    if cat_cols:
        selected_cat_cols = cat_cols[:10]
        for c in selected_cat_cols:
            if chart_count >= MAX_CHARTS:
                break
            vc=df[c].value_counts(); pct=df[c].value_counts(normalize=True)*100
            categorical_summary.append({"variable":c,"table":[{"Category":str(k),"Count":int(v),"Percent (%)":round(float(pct[k]),2)} for k,v in vc.items()]})
            fig4,ax4=plt.subplots(figsize=(max(6,len(vc)*.9+2),5)); fig4.patch.set_facecolor(P["white"])
            bars=ax4.bar(vc.index.astype(str),vc.values,color=PAL[:len(vc)],alpha=.88,edgecolor="white",lw=.5)
            for b,v in zip(bars,vc.values):
                ax4.text(b.get_x()+b.get_width()/2,b.get_height()+.3,f"{v}\n({v/len(df)*100:.1f}%)",ha="center",va="bottom",fontsize=9,color=P["navy"])
            ax4.set_title(f"Frequency: {c}",fontweight="bold",color=P["navy"]); ax4.tick_params(axis="x",rotation=30)
            _style(ax4); plt.tight_layout(); charts.append({"title":f"Bar Chart: {c}","img":_store_chart(fig4, f"barchart_{c}", 110)})
            chart_count += 1

            if chart_count < MAX_CHARTS:
                fig5,ax5=plt.subplots(figsize=(6,5)); fig5.patch.set_facecolor(P["white"])
                ax5.pie(vc.values,labels=vc.index.astype(str),colors=PAL[:len(vc)],autopct="%1.1f%%",startangle=140,pctdistance=.82,
                        wedgeprops=dict(edgecolor="white",linewidth=1.5))
                ax5.set_title(f"Proportion: {c}",fontweight="bold",color=P["navy"])
                charts.append({"title":f"Pie Chart: {c}","img":_store_chart(fig5, f"piechart_{c}", 110)})
                chart_count += 1

    interp=[]
    for r in numeric_summary:
        sk="approximately symmetric" if abs(r["Skewness"])<.5 else ("positively skewed" if r["Skewness"]>0 else "negatively skewed")
        interp.append(f"**{r['Variable']}**: N={r['N']}, M={r['Mean']}, SD={r['Std Dev']}, Median={r['Median']}. Range=[{r['Min']},{r['Max']}]. Distribution is {sk} (skewness={r['Skewness']}).")

    # ── FIXED: Add data cleaning report ──────────────────────────────────
    rows_removed = original_rows - len(df)
    
    return {
        "test":"Descriptive Statistics",
        "numeric_summary":numeric_summary,
        "categorical_summary":categorical_summary,
        "charts":charts,
        "interpretation":"\n\n".join(interp) if interp else "Selected columns analysed.",
        "apa_citation":"Descriptive statistics are reported as M (SD).",
        "significance":"—","p_value":None,"p_display":"—",
        "data_quality": {
            "original_rows": original_rows,
            "rows_after_cleaning": len(df),
            "rows_removed": rows_removed,
            "percent_removed": round((rows_removed / original_rows) * 100, 2) if original_rows > 0 else 0
        },
        "engine_version": ENGINE_VERSION,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHI-SQUARE
# ══════════════════════════════════════════════════════════════════════════════
@with_timeout(MAX_RUNTIME_SECONDS)
def chi_square_test(df, var1, var2):
    # ── Input Validation ──────────────────────────────────────────────────
    validate_categorical_column(df, var1, "for chi-square test")
    validate_categorical_column(df, var2, "for chi-square test")
    
    original_rows = len(df)
    ct=pd.crosstab(df[var1],df[var2])
    
    if ct.empty or ct.sum().sum() == 0:
        return {"error": ERROR_CODES["ERR_002"], "message": "Contingency table is empty or has zero totals."}
    
    # ── FIXED: Fisher Exact Test Fallback ──────────────────────────────────
    if ct.shape == (2, 2):
        try:
            chi2, p, dof, exp = stats.chi2_contingency(ct)
            if (exp < 5).any():
                odds_ratio, p_fisher = stats.fisher_exact(ct)
                logger.info(f"Fisher's Exact Test used for 2x2 table with small expected frequencies")
                result = _chi_square_result(ct, None, p_fisher, None, "Fisher's Exact Test", 
                                           {"odds_ratio": round(float(odds_ratio), 4)}, original_rows, len(df))
                result["data_quality"] = {
                    "original_rows": original_rows,
                    "rows_after_cleaning": len(df),
                    "rows_removed": original_rows - len(df),
                    "percent_removed": round(((original_rows - len(df)) / original_rows) * 100, 2) if original_rows > 0 else 0
                }
                result["engine_version"] = ENGINE_VERSION
                result["analysis_timestamp"] = datetime.now().isoformat()
                return result
        except:
            pass
    
    try:
        chi2,p,dof,exp=stats.chi2_contingency(ct)
    except Exception as e:
        return {"error": ERROR_CODES["ERR_002"], "message": f"Chi-square test failed: {str(e)}"}
    
    result = _chi_square_result(ct, chi2, p, dof, "Chi-Square Test of Independence", {}, original_rows, len(df))
    result["data_quality"] = {
        "original_rows": original_rows,
        "rows_after_cleaning": len(df),
        "rows_removed": original_rows - len(df),
        "percent_removed": round(((original_rows - len(df)) / original_rows) * 100, 2) if original_rows > 0 else 0
    }
    result["engine_version"] = ENGINE_VERSION
    result["analysis_timestamp"] = datetime.now().isoformat()
    return result

def _chi_square_result(ct, chi2, p, dof, test_name, extra, original_rows, cleaned_rows):
    n=ct.values.sum(); min_dim=min(ct.shape)-1
    v=np.sqrt(chi2/(n*max(min_dim,1))) if chi2 else 0
    ef="negligible" if v<.1 else "small" if v<.3 else "moderate" if v<.5 else "large"
    vrd,h0=verdict(p); charts=[]

    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5)); fig.patch.set_facecolor(P["white"])
    ct.div(ct.sum(axis=1),axis=0).mul(100).plot(kind="bar",stacked=True,ax=a1,color=PAL[:len(ct.columns)],edgecolor="white",alpha=.88)
    a1.set_title(f"Stacked Bar: {ct.index.name} × {ct.columns.name}",fontweight="bold",color=P["navy"]); a1.set_xlabel(ct.index.name); a1.set_ylabel("Percentage (%)"); a1.tick_params(axis="x",rotation=30); a1.legend(title=ct.columns.name,bbox_to_anchor=(1.02,1),fontsize=8)
    sns.heatmap(ct,annot=True,fmt="d",cmap="Blues",ax=a2,linewidths=.5,cbar_kws={"shrink":.7})
    a2.set_title("Observed Frequencies",fontweight="bold",color=P["navy"])
    _style([a1]); fig.suptitle(f"{test_name}: {ct.index.name} × {ct.columns.name} | χ²({dof})={chi2:.3f}, {fp(p)}" if chi2 else f"{test_name}: {ct.index.name} × {ct.columns.name}",fontsize=11,fontweight="bold",color=P["navy"],y=1.01)
    plt.tight_layout(); charts.append({"title":"Chi-Square Visualisation","img":_store_chart(fig, "chisquare_viz", 110)})

    if chi2:
        exp_df=pd.DataFrame(exp,index=ct.index,columns=ct.columns); resid=(ct-exp_df)/np.sqrt(exp_df)
        fig2,ax=plt.subplots(figsize=(8,5)); fig2.patch.set_facecolor(P["white"])
        sns.heatmap(resid,annot=True,fmt=".2f",cmap="RdBu_r",center=0,ax=ax,linewidths=.5,cbar_kws={"shrink":.7})
        ax.set_title("Standardised Residuals (|value|>2 = significant cell)",fontweight="bold",color=P["navy"])
        charts.append({"title":"Standardised Residuals","img":_store_chart(fig2, "chisquare_residuals", 110)})

    ct_out=ct.copy(); ct_out["Row Total"]=ct_out.sum(axis=1); ct_out.loc["Col Total"]=ct_out.sum()
    return {"test":test_name,"chi2":round(float(chi2),4) if chi2 else None,"p_value":round(float(p),4),
            "p_display":fp(p),"dof":int(dof) if dof else None,"significance":sp(p),"cramers_v":round(float(v),4) if v else None,
            "effect_size":ef,"n":int(n),"contingency_table":ct_out.to_dict(),"charts":charts,
            "apa_citation":f"χ²({dof}, N={n})={chi2:.2f}, {fp(p)}, Cramér's V={v:.2f}" if chi2 else f"Fisher's Exact Test: p={fp(p)}",
            "interpretation":f"A {test_name} examined the relationship between {ct.index.name} and {ct.columns.name}. "
                f"The association was {vrd}, {fp(p)}. Effect size (Cramér's V={v:.2f}) is {ef}. "
                f"We therefore {h0} the null hypothesis of independence."}


# ══════════════════════════════════════════════════════════════════════════════
# 3. INDEPENDENT T-TEST
# ══════════════════════════════════════════════════════════════════════════════
@with_timeout(MAX_RUNTIME_SECONDS)
def independent_ttest(df, numeric_var, group_var, alpha=.05):
    # ── Input Validation ──────────────────────────────────────────────────
    validate_numeric_column(df, numeric_var, "for t-test")
    validate_categorical_column(df, group_var, "for t-test")
    validate_row_count(df, 3, "for t-test")

    original_rows = len(df)
    gs=df[group_var].dropna().unique()
    if len(gs)!=2: return {"error": ERROR_CODES["VAL_004"], "message": f"Need exactly 2 groups; found {len(gs)}: {list(gs)}"}
    g1=pd.to_numeric(df.loc[df[group_var]==gs[0],numeric_var],errors="coerce").dropna()
    g2=pd.to_numeric(df.loc[df[group_var]==gs[1],numeric_var],errors="coerce").dropna()
    t_s,t_p=stats.ttest_ind(g1,g2); lev_s,lev_p=stats.levene(g1,g2)

    # ── FIXED: Proper Welch correction with appropriate effect size ──────
    if lev_p < 0.05:
        t_s_welch, t_p_welch = stats.ttest_ind(g1, g2, equal_var=False)
        t_s, t_p = t_s_welch, t_p_welch
        d = (g1.mean() - g2.mean()) / g2.std() if g2.std() > 0 else 0
        d_label = "Glass's Δ"
    else:
        pool = np.sqrt(((len(g1)-1)*g1.std()**2+(len(g2)-1)*g2.std()**2)/(len(g1)+len(g2)-2))
        d = (g1.mean()-g2.mean())/pool if pool>0 else 0
        d_label = "Cohen's d"
    
    ef="negligible" if abs(d)<.2 else "small" if abs(d)<.5 else "medium" if abs(d)<.8 else "large"
    se=np.sqrt(g1.var()/len(g1)+g2.var()/len(g2))
    df_w=se**4/((g1.var()/len(g1))**2/(len(g1)-1)+(g2.var()/len(g2))**2/(len(g2)-1))
    tc=stats.t.ppf(1-alpha/2,df_w); md=g1.mean()-g2.mean()
    ci=[md-tc*se,md+tc*se]; vrd,h0=verdict(t_p); charts=[]

    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5)); fig.patch.set_facecolor(P["white"])
    pdf=pd.DataFrame({numeric_var:pd.concat([g1,g2]),group_var:[str(gs[0])]*len(g1)+[str(gs[1])]*len(g2)})
    parts=a1.violinplot([g1.values,g2.values],positions=[1,2],showmedians=True)
    for i,pc in enumerate(parts["bodies"]): pc.set_facecolor(PAL[i]); pc.set_alpha(.7)
    parts["cmedians"].set_color(P["navy"]); a1.set_xticks([1,2]); a1.set_xticklabels([str(gs[0]),str(gs[1])])
    a1.set_title(f"Violin: {numeric_var}",fontweight="bold",color=P["navy"]); a1.set_ylabel(numeric_var)
    sns.boxplot(data=pdf,x=group_var,y=numeric_var,palette=PAL[:2],ax=a2,width=.5)
    sns.stripplot(data=pdf,x=group_var,y=numeric_var,color=P["navy"],alpha=.4,size=4,jitter=True,ax=a2)
    a2.set_title(f"Box + Strip: {numeric_var}",fontweight="bold",color=P["navy"])
    _style([a1,a2]); fig.suptitle(f"T-Test: {numeric_var} by {group_var} | t({df_w:.1f})={t_s:.3f}, {fp(t_p)}",fontsize=11,fontweight="bold",color=P["navy"],y=1.02)
    plt.tight_layout(); charts.append({"title":"Group Comparison","img":_store_chart(fig, "ttest_violin", 110)})

    fig2,ax=plt.subplots(figsize=(7,5)); fig2.patch.set_facecolor(P["white"])
    for i,(lbl,gd) in enumerate([(str(gs[0]),g1),(str(gs[1]),g2)]):
        ax.errorbar(i,gd.mean(),yerr=gd.sem()*1.96,fmt="o",color=PAL[i],ms=12,capsize=8,lw=2,label=f"{lbl}: M={gd.mean():.2f}")
    ax.set_xticks([0,1]); ax.set_xticklabels([str(gs[0]),str(gs[1])],fontsize=11)
    ax.set_ylabel(numeric_var); ax.set_title("Means with 95% CI",fontweight="bold",color=P["navy"]); ax.legend()
    _style(ax); plt.tight_layout(); charts.append({"title":"Means with 95% CI","img":_store_chart(fig2, "ttest_means", 110)})

    stbl=[{"Group":str(gs[i]),"N":int(len(g)),"Mean":round(float(g.mean()),4),"Std Dev":round(float(g.std()),4),"Std Error":round(float(g.sem()),4)} for i,g in [(0,g1),(1,g2)]]
    
    rows_removed = original_rows - len(df)
    return {
        "test":"Independent Samples T-Test",
        "t_statistic":round(float(t_s),4),
        "p_value":round(float(t_p),4),
        "p_display":fp(t_p),
        "df":round(float(df_w),2),
        "significance":sp(t_p),
        "mean_difference":round(float(md),4),
        "ci_95":[round(float(ci[0]),4),round(float(ci[1]),4)],
        "effect_size":round(float(d),4),
        "effect_label":d_label,
        "levene_p":round(float(lev_p),4),
        "summary_table":stbl,
        "charts":charts,
        "apa_citation":f"t({df_w:.2f})={t_s:.2f}, {fp(t_p)}, {d_label}={d:.2f}, 95% CI [{ci[0]:.2f},{ci[1]:.2f}]",
        "interpretation":f"Independent samples t-test comparing {numeric_var} between {gs[0]} (M={g1.mean():.2f},SD={g1.std():.2f}) "
            f"and {gs[1]} (M={g2.mean():.2f},SD={g2.std():.2f}). Difference was {vrd}, t({df_w:.2f})={t_s:.2f}, {fp(t_p)}, {d_label}={d:.2f} ({ef} effect). "
            f"95% CI=[{ci[0]:.2f},{ci[1]:.2f}]. We {h0} the null hypothesis."
            +(f" Levene's test: {fp(lev_p)} — Welch correction applied." if lev_p<.05 else ""),
        "data_quality": {
            "original_rows": original_rows,
            "rows_after_cleaning": len(df),
            "rows_removed": rows_removed,
            "percent_removed": round((rows_removed / original_rows) * 100, 2) if original_rows > 0 else 0
        },
        "engine_version": ENGINE_VERSION,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. ONE-WAY ANOVA
# ══════════════════════════════════════════════════════════════════════════════
@with_timeout(MAX_RUNTIME_SECONDS)
def one_way_anova(df, numeric_var, group_var):
    # ── Input Validation ──────────────────────────────────────────────────
    validate_numeric_column(df, numeric_var, "for ANOVA")
    validate_categorical_column(df, group_var, "for ANOVA")
    validate_row_count(df, 3, "for ANOVA")

    original_rows = len(df)
    gs=df[group_var].dropna().unique()
    gdata=[]
    for g in gs:
        group_data = pd.to_numeric(df.loc[df[group_var]==g,numeric_var],errors="coerce").dropna().values
        if len(group_data) < 2:
            return {"error": ERROR_CODES["VAL_004"], "message": f"Group '{g}' has only {len(group_data)} observation(s). ANOVA requires at least 2 observations per group."}
        gdata.append(group_data)
    F,p=stats.f_oneway(*gdata); gm=np.concatenate(gdata).mean()
    ss_b=sum(len(g)*(g.mean()-gm)**2 for g in gdata); ss_t=sum((v-gm)**2 for g in gdata for v in g)
    eta=ss_b/ss_t if ss_t>0 else 0; ef="small" if eta<.06 else "medium" if eta<.14 else "large"
    n_t=sum(len(g) for g in gdata); k=len(gs); vrd,h0=verdict(p); charts=[]

    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5)); fig.patch.set_facecolor(P["white"])
    pdf=pd.DataFrame({numeric_var:np.concatenate(gdata),group_var:np.repeat([str(g) for g in gs],[len(g) for g in gdata])})
    sns.boxplot(data=pdf,x=group_var,y=numeric_var,palette=PAL[:k],ax=a1,width=.5)
    sns.stripplot(data=pdf,x=group_var,y=numeric_var,color=P["navy"],alpha=.35,size=4,jitter=True,ax=a1)
    a1.set_title(f"Group Distributions: {numeric_var}",fontweight="bold",color=P["navy"]); a1.tick_params(axis="x",rotation=30)
    means=[g.mean() for g in gdata]; sems=[stats.sem(g) for g in gdata]
    bars=a2.bar(range(k),means,color=PAL[:k],alpha=.85,edgecolor="white",lw=.5)
    a2.errorbar(range(k),means,yerr=[1.96*s for s in sems],fmt="none",color=P["navy"],capsize=6,lw=1.5)
    a2.set_xticks(range(k)); a2.set_xticklabels([str(g) for g in gs],rotation=30,ha="right")
    a2.set_ylabel(numeric_var); a2.set_title("Group Means with 95% CI",fontweight="bold",color=P["navy"])
    _style([a1,a2]); fig.suptitle(f"ANOVA: {numeric_var} by {group_var} | F({k-1},{n_t-k})={F:.3f}, {fp(p)}, η²={eta:.3f}",fontsize=11,fontweight="bold",color=P["navy"],y=1.02)
    plt.tight_layout(); charts.append({"title":"ANOVA Group Plots","img":_store_chart(fig, "anova_plots", 110)})

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    posthoc = []
    if p < 0.05 and k > 2:
        try:
            tukey = pairwise_tukeyhsd(endog=pdf[numeric_var], groups=pdf[group_var], alpha=0.05)
            summary_data = tukey.summary().data[1:]
            for row in summary_data:
                posthoc.append({
                    "Group 1": str(row[0]),
                    "Group 2": str(row[1]),
                    "Mean Diff": round(float(row[2]), 4),
                    "p-value": round(float(row[3]), 4),
                    "Significant": "Yes" if row[6] else "No"
                })
        except Exception as e:
            logger.warning(f"Tukey HSD failed: {str(e)}. Falling back to Bonferroni.")
            n_pairs = len(list(combinations(gs,2)))
            for (i,g_a),(j,g_b) in combinations(enumerate(gs),2):
                _,tp=stats.ttest_ind(gdata[i],gdata[j]); tadj=min(tp*n_pairs,1.0)
                posthoc.append({"Group 1":str(g_a),"Group 2":str(g_b),"Mean Diff":round(float(gdata[i].mean()-gdata[j].mean()),4),"p (Bonferroni)":round(float(tadj),4),"Significant":"Yes" if tadj<.05 else "No"})

    stbl=[{"Group":str(g),"N":int(len(gd)),"Mean":round(float(gd.mean()),4),"Std Dev":round(float(gd.std()),4),"Std Error":round(float(stats.sem(gd)),4),"Min":round(float(gd.min()),4),"Max":round(float(gd.max()),4)} for g,gd in zip(gs,gdata)]
    
    rows_removed = original_rows - len(df)
    return {
        "test":"One-Way ANOVA",
        "f_statistic":round(float(F),4),
        "p_value":round(float(p),4),
        "p_display":fp(p),
        "df_between":int(k-1),
        "df_within":int(n_t-k),
        "significance":sp(p),
        "eta_squared":round(float(eta),4),
        "effect_size":ef,
        "summary_table":stbl,
        "posthoc_table":posthoc,
        "charts":charts,
        "apa_citation":f"F({k-1},{n_t-k})={F:.2f}, {fp(p)}, η²={eta:.3f}",
        "interpretation":f"One-way ANOVA compared {numeric_var} across {k} groups of {group_var}. Result was {vrd}, F({k-1},{n_t-k})={F:.2f}, {fp(p)}, η²={eta:.3f} ({ef} effect). "
            f"We {h0} the null hypothesis of equal means."+(" Tukey HSD post-hoc comparisons provided." if p<.05 and len(posthoc)>0 else ""),
        "data_quality": {
            "original_rows": original_rows,
            "rows_after_cleaning": len(df),
            "rows_removed": rows_removed,
            "percent_removed": round((rows_removed / original_rows) * 100, 2) if original_rows > 0 else 0
        },
        "engine_version": ENGINE_VERSION,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
@with_timeout(MAX_RUNTIME_SECONDS)
def correlation_analysis(df, columns, method="pearson"):
    # ── Input Validation ──────────────────────────────────────────────────
    num=[c for c in columns if pd.to_numeric(df[c],errors="coerce").notna().sum()>len(df)*.4]
    if len(num)<2: return {"error": ERROR_CODES["VAL_001"], "message": "Need ≥2 numeric columns."}
    for col in num:
        validate_numeric_column(df, col, "for correlation")
    
    original_rows = len(df)
    sub=df[num].apply(pd.to_numeric,errors="coerce").dropna(); charts=[]

    constant_cols = [c for c in sub.columns if sub[c].nunique() <= 1]
    if constant_cols:
        warnings.warn(f"Dropped constant columns from correlation: {constant_cols}")
        sub = sub.drop(columns=constant_cols)
        num = [c for c in num if c not in constant_cols]
        if len(num) < 2:
            return {"error": ERROR_CODES["VAL_009"], "message": "After removing constant columns, fewer than 2 numeric columns remain."}

    if method=="spearman": corr_m=sub.corr(method="spearman")
    else: corr_m=sub.corr(method="pearson")

    pairs=[]
    n_comparisons = 0
    for c1,c2 in combinations(num,2):
        s2=sub[[c1,c2]].dropna()
        r,p=(stats.spearmanr(s2[c1],s2[c2]) if method=="spearman" else stats.pearsonr(s2[c1],s2[c2]))
        st="negligible" if abs(r)<.1 else "weak" if abs(r)<.3 else "moderate" if abs(r)<.5 else "strong" if abs(r)<.7 else "very strong"
        pairs.append({"Variable 1":c1,"Variable 2":c2,"r":round(float(r),4),"p-value":round(float(p),4),"p_display":fp(p),"Significance":sp(p),"Strength":f"{'positive' if r>0 else 'negative'} {st}"})
        n_comparisons += 1
    
    # ── FIXED: Add multiple testing correction ──────────────────────────
    if n_comparisons > 0:
        from statsmodels.stats.multitest import multipletests
        p_values = [p_["p-value"] for p_ in pairs]
        rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        for i, pair in enumerate(pairs):
            pair["p_corrected"] = round(float(pvals_corrected[i]), 4)
            pair["Significant_corrected"] = "Yes" if rejected[i] else "No"

    fig,ax=plt.subplots(figsize=(max(6,len(num)*1.3+2),max(5,len(num)*1.2))); fig.patch.set_facecolor(P["white"])
    mask=np.triu(np.ones_like(corr_m,dtype=bool))
    sns.heatmap(corr_m,annot=True,fmt=".3f",cmap="coolwarm",vmin=-1,vmax=1,center=0,ax=ax,mask=mask,square=True,linewidths=.5,annot_kws={"size":10,"weight":"bold"},cbar_kws={"shrink":.8})
    ax.set_title(f"{method.title()} Correlation Matrix",fontsize=13,fontweight="bold",color=P["navy"])
    plt.tight_layout(); charts.append({"title":"Correlation Heatmap","img":_store_chart(fig, "correlation_heatmap", 110)})

    if 2 <= len(num) <= MAX_PAIRPLOT_VARS:
        sample_size = min(MAX_PAIRPLOT_SAMPLE, len(sub))
        if len(sub) > sample_size:
            pairplot_df = sub.sample(sample_size, random_state=RANDOM_SEED)
        else:
            pairplot_df = sub
        g=sns.pairplot(pairplot_df,diag_kind="kde",plot_kws={"alpha":.5,"color":P["blue"],"s":20},diag_kws={"color":P["teal"],"fill":True})
        g.figure.suptitle("Scatter Matrix",y=1.02,fontsize=13,fontweight="bold",color=P["navy"])
        g.figure.patch.set_facecolor(P["white"])
        charts.append({"title":"Scatter Matrix","img":_store_chart(g.figure, "correlation_scatter", 110)})
    elif len(num) > MAX_PAIRPLOT_VARS:
        limited_num = num[:MAX_PAIRPLOT_VARS]
        sample_size = min(MAX_PAIRPLOT_SAMPLE, len(sub))
        if len(sub) > sample_size:
            pairplot_df = sub.iloc[:sample_size]
        else:
            pairplot_df = sub
        g=sns.pairplot(pairplot_df[limited_num],diag_kind="kde",plot_kws={"alpha":.5,"color":P["blue"],"s":20},diag_kws={"color":P["teal"],"fill":True})
        g.figure.suptitle("Scatter Matrix (Limited to 4 Variables)",y=1.02,fontsize=13,fontweight="bold",color=P["navy"])
        g.figure.patch.set_facecolor(P["white"])
        charts.append({"title":"Scatter Matrix (Limited to 4 Variables)","img":_store_chart(g.figure, "correlation_scatter_limited", 110)})

    if pairs:
        fig3,ax3=plt.subplots(figsize=(10,5)); fig3.patch.set_facecolor(P["white"])
        x_pos=range(len(pairs)); rs=[abs(p_["r"]) for p_ in pairs]; cols_=[P["ok"] if p_["r"]>0 else P["err"] for p_ in pairs]
        sc=ax3.scatter(x_pos,[p_["r"] for p_ in pairs],s=[r*500+30 for r in rs],c=cols_,alpha=.75,edgecolors="white",lw=1.5)
        ax3.axhline(0,color=P["slate"],lw=1,ls="--"); ax3.axhline(.3,color=P["warn"],lw=.8,ls=":"); ax3.axhline(-.3,color=P["warn"],lw=.8,ls=":")
        ax3.set_xticks(list(x_pos)); ax3.set_xticklabels([f"{p_['Variable 1'][:8]}×{p_['Variable 2'][:8]}" for p_ in pairs],rotation=35,ha="right",fontsize=8)
        ax3.set_ylabel("r value"); ax3.set_title("Correlation Bubble Chart (size = |r|, green=positive, red=negative)",fontweight="bold",color=P["navy"])
        _style(ax3); plt.tight_layout(); charts.append({"title":"Correlation Bubble Chart","img":_store_chart(fig3, "correlation_bubble", 110)})

    rows_removed = original_rows - len(df)
    return {
        "test":f"{method.title()} Correlation",
        "method":method,
        "n":int(len(sub)),
        "variables":num,
        "pairs_table":pairs,
        "charts":charts,
        "p_value":min([p_["p-value"] for p_ in pairs]) if pairs else 1,
        "p_display":"see table",
        "significance":"see table",
        "apa_citation":f"{method.title()} correlation analysis conducted on {len(num)} variables (N={len(sub)}).",
        "interpretation":f"**{method.title()} correlation** was conducted on {len(pairs)} variable pair(s).\n\n"+"".join(
            f"- **{p_['Variable 1']} & {p_['Variable 2']}**: r={p_['r']}, {p_['p_display']} ({sp(p_['p-value'])}) — {p_['Strength']}.\n" for p_ in pairs),
        "data_quality": {
            "original_rows": original_rows,
            "rows_after_cleaning": len(df),
            "rows_removed": rows_removed,
            "percent_removed": round((rows_removed / original_rows) * 100, 2) if original_rows > 0 else 0
        },
        "engine_version": ENGINE_VERSION,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
@with_timeout(MAX_RUNTIME_SECONDS)
def linear_regression(df, dependent, predictors):
    # ── Performance Guard ──────────────────────────────────────────────────
    check_analysis_limits(df, "regression", {"predictors": predictors})
    
    # ── Input Validation ──────────────────────────────────────────────────
    validate_numeric_column(df, dependent, "as dependent variable for regression")
    # ── FIXED: Correct parameter order ────────────────────────────────────
    validate_predictor_count(predictors, context="for regression")
    for pred in predictors:
        validate_numeric_column(df, pred, f"as predictor for regression")

    original_rows = len(df)
    sub=df[[dependent]+predictors].apply(pd.to_numeric,errors="coerce").dropna()
    y=sub[dependent].values; Xr=sub[predictors].values; charts=[]

    constant_preds = [p for p in predictors if sub[p].nunique() <= 1]
    if constant_preds:
        warnings.warn(f"Dropped constant predictors: {constant_preds}")
        predictors = [p for p in predictors if p not in constant_preds]
        sub = sub[[dependent] + predictors]
        if len(predictors) == 0:
            return {"error": ERROR_CODES["VAL_009"], "message": "All predictors are constant. Cannot perform regression."}

    min_n = max(20, len(predictors) * 10)
    n_obs = len(sub)
    
    if n_obs < min_n:
        return {"error": ERROR_CODES["VAL_010"], "message": f"Regression requires at least {min_n} valid observations. You have {n_obs}."}
    
    if n_obs <= len(predictors) + 1:
        return {"error": ERROR_CODES["VAL_010"], "message": f"Sample size ({n_obs}) must be greater than number of predictors ({len(predictors)}) + 1. Minimum required: {len(predictors) + 2}."}

    import statsmodels.api as sm
    X = sm.add_constant(Xr)
    
    # ── FIXED: Check for singular matrix ──────────────────────────────────
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise SingularMatrixError(ERROR_CODES["ERR_003"])
    
    model = sm.OLS(y, X).fit()
    
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    F_stat = model.fvalue
    p_f = model.f_pvalue
    
    coef = []
    coef.append({
        "Predictor": "(Intercept)",
        "B": round(float(model.params[0]), 4),
        "Std Error": round(float(model.bse[0]), 4),
        "t": round(float(model.tvalues[0]), 4),
        "p": round(float(model.pvalues[0]), 4),
        "Significance": sp(model.pvalues[0])
    })
    for i, pred in enumerate(predictors):
        coef.append({
            "Predictor": pred,
            "B": round(float(model.params[i+1]), 4),
            "Std Error": round(float(model.bse[i+1]), 4),
            "t": round(float(model.tvalues[i+1]), 4),
            "p": round(float(model.pvalues[i+1]), 4),
            "Significance": sp(model.pvalues[i+1])
        })
    
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = []
    for i in range(X.shape[1]):
        if i == 0:
            continue
        vif = variance_inflation_factor(X, i)
        vif_data.append({
            "Variable": predictors[i-1],
            "VIF": round(float(vif), 3)
        })
    
    conf_int = model.conf_int()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor(P["white"])
    
    axes[0, 0].scatter(model.fittedvalues, y, alpha=0.6, color=P["blue"], s=40)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, color=P["err"])
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")
    axes[0, 0].set_title("Actual vs Predicted", fontweight="bold", color=P["navy"])
    
    axes[0, 1].scatter(model.fittedvalues, model.resid, alpha=0.6, color=P["teal"], s=40)
    axes[0, 1].axhline(0, color=P["err"], linestyle="--", lw=1.5)
    axes[0, 1].set_xlabel("Fitted Values")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residuals vs Fitted", fontweight="bold", color=P["navy"])
    
    stats.probplot(model.resid, dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set(color=P["blue"], alpha=0.7, ms=5)
    axes[1, 0].get_lines()[1].set(color=P["err"], lw=1.5)
    axes[1, 0].set_title("Q-Q Plot of Residuals", fontweight="bold", color=P["navy"])
    
    axes[1, 1].stem(range(len(cooks_d)), cooks_d, linefmt=P["blue"], markerfmt='o', basefmt=" ")
    axes[1, 1].axhline(0.5, color=P["err"], linestyle="--", lw=1.5, label="Influential threshold (0.5)")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].set_title("Cook's Distance", fontweight="bold", color=P["navy"])
    axes[1, 1].legend()
    
    _style(axes.flatten())
    plt.tight_layout()
    charts.append({"title":"Regression Diagnostics","img":_store_chart(fig, "regression_diagnostics", 110)})
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor(P["white"])
    cv = [c["B"] for c in coef[1:]]
    ci_errors = [c["Std Error"] * 1.96 for c in coef[1:]]
    colors_ = [P["ok"] if v > 0 else P["err"] for v in cv]
    ax2.barh(range(len(cv)), cv, xerr=ci_errors, color=colors_, alpha=0.8, edgecolor="white", capsize=4, error_kw={"lw": 1.5, "color": P["navy"]})
    ax2.axvline(0, color=P["navy"], lw=1.5, ls="--")
    ax2.set_yticks(range(len(cv)))
    ax2.set_yticklabels(predictors, fontsize=10)
    ax2.set_xlabel("Coefficient (B)")
    ax2.set_title("Coefficient Plot with 95% CI", fontweight="bold", color=P["navy"])
    _style(ax2)
    plt.tight_layout()
    charts.append({"title":"Coefficient Plot","img":_store_chart(fig2, "regression_coefficients", 110)})

    eq_parts = [f"{coef[0]['B']}"]
    for row in coef[1:]:
        b_val = row['B']
        if b_val >= 0:
            eq_parts.append(f"+ {b_val}×{row['Predictor']}")
        else:
            eq_parts.append(f"- {abs(b_val)}×{row['Predictor']}")
    eq = "ŷ = " + " ".join(eq_parts)
    
    vrd, h0 = verdict(p_f)
    label = "Simple" if len(predictors) == 1 else "Multiple"
    
    rows_removed = original_rows - len(df)
    return {
        "test": f"{label} Linear Regression",
        "dependent": dependent,
        "predictors": predictors,
        "r_squared": round(float(r2), 4),
        "adj_r_squared": round(float(adj_r2), 4),
        "f_statistic": round(float(F_stat), 4),
        "p_value": round(float(p_f), 4),
        "p_display": fp(p_f),
        "significance": sp(p_f),
        "equation": eq,
        "coef_table": coef,
        "vif_table": vif_data,
        "cooks_distance": round(float(cooks_d.max()), 4),
        "n": int(len(y)),
        "charts": charts,
        "apa_citation": f"R²={r2:.3f}, Adj.R²={adj_r2:.3f}, F({len(predictors)},{len(y)-len(predictors)-1})={F_stat:.2f}, {fp(p_f)}",
        "interpretation": f"{label} linear regression predicted {dependent} from {', '.join(predictors)}. "
            f"Model explained {r2*100:.1f}% of variance (R²={r2:.3f}, Adj.R²={adj_r2:.3f}). "
            f"Overall model was {vrd} ({fp(p_f)}). Equation: {eq}.",
        "data_quality": {
            "original_rows": original_rows,
            "rows_after_cleaning": len(df),
            "rows_removed": rows_removed,
            "percent_removed": round((rows_removed / original_rows) * 100, 2) if original_rows > 0 else 0
        },
        "engine_version": ENGINE_VERSION,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. MANN-WHITNEY U
# ══════════════════════════════════════════════════════════════════════════════
@with_timeout(MAX_RUNTIME_SECONDS)
def mann_whitney(df, numeric_var, group_var):
    # ── Input Validation ──────────────────────────────────────────────────
    validate_numeric_column(df, numeric_var, "for Mann-Whitney U test")
    validate_categorical_column(df, group_var, "for Mann-Whitney U test")

    original_rows = len(df)
    gs=df[group_var].dropna().unique()
    if len(gs)!=2: return {"error": ERROR_CODES["VAL_004"], "message": "Requires exactly 2 groups."}
    g1=pd.to_numeric(df.loc[df[group_var]==gs[0],numeric_var],errors="coerce").dropna()
    g2=pd.to_numeric(df.loc[df[group_var]==gs[1],numeric_var],errors="coerce").dropna()
    
    from scipy.stats import mannwhitneyu
    res = mannwhitneyu(g1, g2, alternative='two-sided', method='exact' if min(len(g1), len(g2)) < 50 else 'auto')
    U = res.statistic
    p = res.pvalue
    
    n1, n2 = len(g1), len(g2)
    mu = n1 * n2 / 2
    
    combined = np.concatenate([g1, g2])
    from scipy.stats import rankdata
    ranks = rankdata(combined)
    
    unique, counts = np.unique(combined, return_counts=True)
    ties = counts[counts > 1]
    
    if len(ties) > 0:
        tie_correction = np.sum([t**3 - t for t in ties]) / 12
        sigma = np.sqrt((n1 * n2 / (n1 + n2)) * ((n1 + n2 + 1) - tie_correction / ((n1 + n2) * (n1 + n2 - 1))))
    else:
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    
    z = (U - mu) / sigma if sigma > 0 else 0
    r = abs(z) / np.sqrt(n1 + n2)
    
    vrd,h0=verdict(p); charts=[]
    fig,ax=plt.subplots(figsize=(8,5)); fig.patch.set_facecolor(P["white"])
    pdf=pd.DataFrame({numeric_var:pd.concat([g1,g2]),group_var:[str(gs[0])]*len(g1)+[str(gs[1])]*len(g2)})
    sns.violinplot(data=pdf,x=group_var,y=numeric_var,palette=PAL[:2],ax=ax,inner="box",linewidth=1.5)
    ax.set_title(f"Mann-Whitney: {numeric_var} by {group_var} | U={U:.0f}, {fp(p)}",fontweight="bold",color=P["navy"])
    _style(ax); plt.tight_layout(); charts.append({"title":"Mann-Whitney Violin","img":_store_chart(fig, "mannwhitney_violin", 110)})
    
    rows_removed = original_rows - len(df)
    return {
        "test":"Mann-Whitney U Test",
        "u_statistic":round(float(U),2),
        "p_value":round(float(p),4),
        "p_display":fp(p),
        "significance":sp(p),
        "effect_r":round(float(r),4),
        "medians":{str(gs[0]):round(float(g1.median()),4),str(gs[1]):round(float(g2.median()),4)},
        "charts":charts,
        "apa_citation":f"U={U:.0f}, {fp(p)}, r={r:.3f}",
        "interpretation":f"Mann-Whitney U compared {numeric_var} between {gs[0]} (Mdn={g1.median():.2f}) and {gs[1]} (Mdn={g2.median():.2f}). Difference was {vrd}, U={U:.0f}, {fp(p)}, r={r:.3f}. We {h0} H₀.",
        "data_quality": {
            "original_rows": original_rows,
            "rows_after_cleaning": len(df),
            "rows_removed": rows_removed,
            "percent_removed": round((rows_removed / original_rows) * 100, 2) if original_rows > 0 else 0
        },
        "engine_version": ENGINE_VERSION,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. KRUSKAL-WALLIS
# ══════════════════════════════════════════════════════════════════════════════
@with_timeout(MAX_RUNTIME_SECONDS)
def kruskal_wallis(df, numeric_var, group_var):
    # ── Input Validation ──────────────────────────────────────────────────
    validate_numeric_column(df, numeric_var, "for Kruskal-Wallis test")
    validate_categorical_column(df, group_var, "for Kruskal-Wallis test")

    original_rows = len(df)
    gs=df[group_var].dropna().unique()
    
    gd = []
    for g in gs:
        group_data = pd.to_numeric(df.loc[df[group_var]==g,numeric_var],errors="coerce").dropna().values
        if len(group_data) < 2:
            return {"error": ERROR_CODES["VAL_004"], "message": f"Group '{g}' has only {len(group_data)} observation(s). Kruskal-Wallis requires at least 2 observations per group."}
        gd.append(group_data)
    
    H,p=stats.kruskal(*gd); n_t=sum(len(g) for g in gd)
    eta=max((H-len(gs)+1)/(n_t-len(gs)),0); vrd,h0=verdict(p); charts=[]
    fig,ax=plt.subplots(figsize=(max(7,len(gs)*1.5),5)); fig.patch.set_facecolor(P["white"])
    pdf=pd.DataFrame({numeric_var:np.concatenate(gd),group_var:np.repeat([str(g) for g in gs],[len(g) for g in gd])})
    sns.boxplot(data=pdf,x=group_var,y=numeric_var,palette=PAL[:len(gs)],ax=ax,width=.5)
    sns.stripplot(data=pdf,x=group_var,y=numeric_var,color=P["navy"],alpha=.35,size=4,jitter=True,ax=ax)
    ax.set_title(f"Kruskal-Wallis: {numeric_var} by {group_var} | H({len(gs)-1})={H:.3f}, {fp(p)}",fontweight="bold",color=P["navy"]); ax.tick_params(axis="x",rotation=30)
    _style(ax); plt.tight_layout(); charts.append({"title":"Kruskal-Wallis Box Plot","img":_store_chart(fig, "kruskal_boxplot", 110)})
    
    rows_removed = original_rows - len(df)
    return {
        "test":"Kruskal-Wallis H Test",
        "h_statistic":round(float(H),4),
        "p_value":round(float(p),4),
        "p_display":fp(p),
        "df":int(len(gs)-1),
        "significance":sp(p),
        "eta_squared_h":round(float(eta),4),
        "group_medians":{str(g):round(float(np.median(gd[i])),4) for i,g in enumerate(gs)},
        "charts":charts,
        "apa_citation":f"H({len(gs)-1})={H:.2f}, {fp(p)}",
        "interpretation":f"Kruskal-Wallis H test compared {numeric_var} across {len(gs)} groups. Result was {vrd}, H({len(gs)-1})={H:.2f}, {fp(p)}. We {h0} H₀.",
        "data_quality": {
            "original_rows": original_rows,
            "rows_after_cleaning": len(df),
            "rows_removed": rows_removed,
            "percent_removed": round((rows_removed / original_rows) * 100, 2) if original_rows > 0 else 0
        },
        "engine_version": ENGINE_VERSION,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════
ANALYSIS_MAP = {
    "descriptive":descriptive_statistics,"chi_square":chi_square_test,
    "t_test":independent_ttest,"anova":one_way_anova,"correlation":correlation_analysis,
    "regression":linear_regression,"mann_whitney":mann_whitney,"kruskal_wallis":kruskal_wallis,
}
ANALYSIS_LABELS = {
    "descriptive":"Descriptive Statistics","chi_square":"Chi-Square Test",
    "t_test":"Independent T-Test","anova":"One-Way ANOVA","correlation":"Correlation Analysis",
    "regression":"Linear Regression","mann_whitney":"Mann-Whitney U Test","kruskal_wallis":"Kruskal-Wallis Test",
}

def run_analysis(df, analysis_type, params, client_ip=None):
    """Main entry point for analysis execution."""
    logger.info(f"Starting analysis: {analysis_type}")
    logger.info(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # ── FIXED: Rate Limiting ──────────────────────────────────────────────
    if client_ip:
        try:
            check_rate_limit(client_ip)
        except RateLimitError as e:
            return {
                "error": "rate_limit_exceeded",
                "error_code": "LIM_004",
                "message": str(e),
                "message_detail": ERROR_CODES["LIM_004"],
                "cta_text": "⏳ Please wait and try again in 60 seconds",
                "cta_url": "https://www.eduxellence.org/#contact"
            }
    
    if analysis_type not in ANALYSIS_MAP: 
        return {"error": "unknown_analysis", "error_code": "VAL_001", "message": f"Unknown analysis type: {analysis_type}"}
    
    try:
        start_time = time.time()
        
        # ── FIXED: Validate and prepare dataset first ──────────────────
        df = validate_dataset(df)
        
        # ── FIXED: Memory check ──────────────────────────────────────────
        check_memory_usage(df)
        
        # ── FIXED: Log the analysis request ────────────────────────────
        logger.info(f"Analysis request: {analysis_type} with params: {params}")
        
        # Run the analysis
        result = ANALYSIS_MAP[analysis_type](df, **params)
        
        elapsed = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed:.2f}s")
        
        # Ensure version metadata is always present
        if "engine_version" not in result:
            result["engine_version"] = ENGINE_VERSION
        if "analysis_timestamp" not in result:
            result["analysis_timestamp"] = datetime.now().isoformat()
        
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        error_code = "VAL_001" if "not found" in str(e) else "VAL_002"
        return {
            "error": "validation_error",
            "error_code": error_code,
            "message": str(e),
            "cta_text": "📞 Need help with your data?",
            "cta_url": "https://www.eduxellence.org/#contact"
        }
        
    except AnalysisTooLargeError as e:
        logger.warning(f"Analysis too large: {str(e)}")
        return {
            "error": "analysis_too_large",
            "error_code": "LIM_003",
            "message": str(e),
            "cta_text": "📞 Book a Free Expert Consultation",
            "cta_url": "https://www.eduxellence.org/#contact"
        }
        
    except AnalysisTimeoutError as e:
        logger.error(f"Analysis timeout: {str(e)}")
        return {
            "error": "analysis_timeout",
            "error_code": "LIM_001",
            "message": "The analysis took too long to complete. Try reducing the dataset size.",
            "cta_text": "📞 Need help with large datasets?",
            "cta_url": "https://www.eduxellence.org/#contact"
        }
        
    except MemoryLimitError as e:
        logger.error(f"Memory limit exceeded: {str(e)}")
        return {
            "error": "memory_limit_exceeded",
            "error_code": "LIM_002",
            "message": str(e),
            "cta_text": "📞 Need help with large datasets?",
            "cta_url": "https://www.eduxellence.org/#contact"
        }
        
    except SingularMatrixError as e:
        logger.error(f"Singular matrix: {str(e)}")
        return {
            "error": "singular_matrix",
            "error_code": "ERR_003",
            "message": "Perfect collinearity detected. Remove redundant predictors or combine correlated ones.",
            "cta_text": "📞 Need help with regression?",
            "cta_url": "https://www.eduxellence.org/#contact"
        }
        
    except Exception as e:
        # ── FIXED: Log error internally, return generic message ──────────
        logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        return {
            "error": "internal_server_error",
            "error_code": "ERR_001",
            "message": "An unexpected error occurred while processing your analysis. Our team has been notified.",
            "reference_id": str(uuid.uuid4())[:8],
            "cta_text": "📞 Contact our support team",
            "cta_url": "https://www.eduxellence.org/#contact"
        }