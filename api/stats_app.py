"""
Eduxellence Statistical Analysis Engine v2.0
============================================
8 statistical tests · Auto assumption checking · Smart test recommender
APA-formatted output · Publication-ready charts via matplotlib/seaborn
Zero paid dependencies. Vercel free-tier compatible.
Supabase Storage integration for charts to prevent payload size limits.

by Eduxellence Analytics · https://eduxellence.org
"""

import io, base64, warnings, traceback, os, json, uuid, logging
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
from datetime import datetime

# ── Configure Logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Targeted Warnings Filtering (NOT global suppression) ──────────────────
# Only suppress specific, known harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
# Keep scipy and statsmodels warnings visible for debugging
# Do NOT suppress: ConvergenceWarning, RuntimeWarning, etc.

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

# ── Custom Exception for Performance Limits ──────────────────────────────────
class AnalysisTooLargeError(Exception):
    """Raised when analysis exceeds free tier limits."""
    pass

# ── Brand palette ──────────────────────────────────────────────────────────────
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
    
    # Estimate size
    size_mb = len(img_data) / (1024 * 1024)
    
    # If size > 1MB and Supabase is available, use storage
    if use_storage and size_mb > 1.0 and supabase_available and supabase:
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"charts/{timestamp}_{unique_id}_{chart_name.replace(' ', '_')}.png"
            
            # Upload to Supabase Storage
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                file=img_data,
                path=filename,
                file_options={"content-type": "image/png"}
            )
            
            # Get public URL
            public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
            
            plt.close(fig)
            return {"type": "url", "url": public_url, "size_mb": round(size_mb, 2)}
            
        except Exception as e:
            # Log error but don't expose to user
            logger.error(f"Supabase upload failed: {str(e)}. Falling back to base64.")
            result = base64.b64encode(img_data).decode()
            plt.close(fig)
            return {"type": "base64", "data": result, "size_mb": round(size_mb, 2)}
    
    # For small charts or when Supabase unavailable
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

# ── Helper for Performance Checks ──────────────────────────────────────────
def check_analysis_limits(df, analysis_type, params, max_rows=50000, max_descriptive_cols=15, max_pairplot_vars=4, max_predictors=10):
    """Raises AnalysisTooLargeError if dataset exceeds free tier limits."""
    rows = len(df)
    if rows > max_rows:
        raise AnalysisTooLargeError(
            f"Your dataset contains {rows:,} rows, which exceeds our free tool's limit of {max_rows:,} rows. "
            "For larger datasets, our expert consulting team can provide a full analysis."
        )

    if analysis_type == "descriptive":
        cols = params.get("columns", [])
        if len(cols) > max_descriptive_cols:
            raise AnalysisTooLargeError(
                f"You selected {len(cols)} columns for descriptive analysis, which exceeds our free tool's limit of {max_descriptive_cols} columns. "
                "For such a comprehensive analysis, our expert team can generate a full report with all charts and statistics."
            )

    if analysis_type == "regression":
        predictors = params.get("predictors", [])
        if len(predictors) > max_predictors:
            raise AnalysisTooLargeError(
                f"Your regression model includes {len(predictors)} predictors, which exceeds our free tool's limit of {max_predictors} predictors. "
                "Our expert consulting team can build and interpret more complex models for you."
            )

# ── Security: CSV Bomb Protection ──────────────────────────────────────────
def validate_dataset(df, max_rows=50000, max_cols=100):
    """Validate dataset to prevent CSV bombs and malicious data."""
    if len(df) > max_rows:
        raise AnalysisTooLargeError(
            f"Dataset has {len(df):,} rows. Maximum allowed is {max_rows:,}. "
            "For larger datasets, our expert consulting team can provide a full analysis."
        )
    if len(df.columns) > max_cols:
        raise AnalysisTooLargeError(
            f"Dataset has {len(df.columns)} columns. Maximum allowed is {max_cols}. "
            "For such comprehensive data, our expert consulting team can provide a full analysis."
        )
    
    # ── FIXED: Protect against NaN, inf, -inf breaking calculations ──────
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Check for constant columns and drop with warning
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        warnings.warn(f"Dropped constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    
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

        # 2. Normality - FIXED: Use Anderson-Darling for large n
        if n >= 3:
            if n < 5000:
                sw_stat, sw_p = stats.shapiro(series)
                normal = sw_p > 0.05
                chk("Normality (Shapiro-Wilk)", normal,
                    f"W = {sw_stat:.3f}, {fp(sw_p)}. {'Distribution appears normal.' if normal else 'Distribution is not normal.'}",
                    "" if normal else "Consider Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups) as non-parametric alternatives.",
                    "log_transform" if not normal else None)
            else:
                # Use Anderson-Darling for large datasets
                anderson_result = stats.anderson(series)
                critical_value = anderson_result.critical_values[2]  # 5% significance level
                normal = anderson_result.statistic < critical_value
                chk("Normality (Anderson-Darling)", normal,
                    f"A² = {anderson_result.statistic:.3f}. {'Distribution appears normal.' if normal else 'Distribution is not normal.'}",
                    "" if normal else "Consider Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups) as non-parametric alternatives.",
                    "log_transform" if not normal else None)

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
            # FIXED: Check for empty contingency table
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
                y  = sub[num_var].values
                Xm = np.column_stack([np.ones(len(sub)), sub[predictors].values])
                coeffs,_,_,_ = np.linalg.lstsq(Xm, y, rcond=None)
                y_pred = Xm @ coeffs; resids = y - y_pred

                # Normality of residuals - FIXED: Use Anderson-Darling for large n
                if len(resids) < 5000:
                    sw_s, sw_p2 = stats.shapiro(resids)
                    chk("Normality of residuals", sw_p2>.05,
                        f"Shapiro-Wilk on residuals: W = {sw_s:.3f}, {fp(sw_p2)}",
                        "Non-normal residuals suggest the linear model may not be the best fit." if sw_p2<=.05 else "")
                else:
                    anderson_result = stats.anderson(resids)
                    critical_value = anderson_result.critical_values[2]
                    normal = anderson_result.statistic < critical_value
                    chk("Normality of residuals", normal,
                        f"Anderson-Darling A² = {anderson_result.statistic:.3f}. {'Normal.' if normal else 'Non-normal.'}",
                        "Non-normal residuals suggest the linear model may not be the best fit." if not normal else "")

                # Homoscedasticity (Breusch-Pagan approximation via correlation)
                corr_bp, _ = stats.pearsonr(y_pred, np.abs(resids))
                homo = abs(corr_bp) < 0.3
                chk("Homoscedasticity", homo,
                    f"Fitted vs |residual| correlation = {corr_bp:.3f}. {'No major heteroscedasticity.' if homo else 'Possible heteroscedasticity detected.'}",
                    "Consider robust standard errors or a data transformation." if not homo else "")

                # ── FIXED: VIF for multicollinearity ──────────────────────
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
                        # Fallback to correlation check if VIF fails
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
def descriptive_statistics(df, columns):
    # ── Performance Guard ──────────────────────────────────────────────────
    check_analysis_limits(df, "descriptive", {"columns": columns})

    num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c],errors="coerce"))]
    num_cols = [c for c in columns if pd.to_numeric(df[c],errors="coerce").notna().sum() > len(df)*0.5]
    cat_cols = [c for c in columns if c not in num_cols]
    charts=[]; numeric_summary=[]; categorical_summary=[]

    if num_cols:
        for c in num_cols:
            s=pd.to_numeric(df[c],errors="coerce").dropna()
            if len(s)==0: continue
            # ── FIXED: Correct missing value count (no double counting) ──
            missing_mask = (
                df[c].isna() |
                df[c].astype(str).str.strip().isin(["", "N/A"])
            )
            missing_count = missing_mask.sum()
            
            # Shapiro-Wilk with Anderson-Darling fallback for large n
            if len(s) >= 3:
                if len(s) < 5000:
                    sw_s,sw_p=stats.shapiro(s)
                else:
                    anderson_result = stats.anderson(s)
                    sw_s, sw_p = anderson_result.statistic, anderson_result.critical_values[2]
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

        # Histogram grid
        nc=min(3, num_charts); nr=(num_charts+nc-1)//nc
        if num_charts > 0:
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

            # Boxplot grid
            if num_charts > 1:
                fig2,ax2=plt.subplots(figsize=(max(9, num_charts*1.6),5)); fig2.patch.set_facecolor(P["white"])
                data=[pd.to_numeric(df[c],errors="coerce").dropna().values for c in selected_num_cols]
                bp=ax2.boxplot(data,patch_artist=True,medianprops=dict(color=P["navy"],lw=2),flierprops=dict(marker="o",ms=4,alpha=.5))
                for patch,col in zip(bp["boxes"],PAL): patch.set_facecolor(col); patch.set_alpha(.75)
                ax2.set_xticklabels(selected_num_cols,rotation=30,ha="right",fontsize=9)
                ax2.set_title("Comparative Boxplots (First 15 Variables)",fontweight="bold",color=P["navy"])
                _style(ax2); plt.tight_layout(); charts.append({"title":"Boxplots (First 15)","img":_store_chart(fig2, "boxplots", 110)})

            # Q-Q plots
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

    if cat_cols:
        # ── Limit Categorical Charts to Top 10 ────────────────────────────
        selected_cat_cols = cat_cols[:10]
        for c in selected_cat_cols:
            vc=df[c].value_counts(); pct=df[c].value_counts(normalize=True)*100
            categorical_summary.append({"variable":c,"table":[{"Category":str(k),"Count":int(v),"Percent (%)":round(float(pct[k]),2)} for k,v in vc.items()]})
            fig4,ax4=plt.subplots(figsize=(max(6,len(vc)*.9+2),5)); fig4.patch.set_facecolor(P["white"])
            bars=ax4.bar(vc.index.astype(str),vc.values,color=PAL[:len(vc)],alpha=.88,edgecolor="white",lw=.5)
            for b,v in zip(bars,vc.values):
                ax4.text(b.get_x()+b.get_width()/2,b.get_height()+.3,f"{v}\n({v/len(df)*100:.1f}%)",ha="center",va="bottom",fontsize=9,color=P["navy"])
            ax4.set_title(f"Frequency: {c}",fontweight="bold",color=P["navy"]); ax4.tick_params(axis="x",rotation=30)
            _style(ax4); plt.tight_layout(); charts.append({"title":f"Bar Chart: {c}","img":_store_chart(fig4, f"barchart_{c}", 110)})

            # Pie chart
            fig5,ax5=plt.subplots(figsize=(6,5)); fig5.patch.set_facecolor(P["white"])
            ax5.pie(vc.values,labels=vc.index.astype(str),colors=PAL[:len(vc)],autopct="%1.1f%%",startangle=140,pctdistance=.82,
                    wedgeprops=dict(edgecolor="white",linewidth=1.5))
            ax5.set_title(f"Proportion: {c}",fontweight="bold",color=P["navy"])
            charts.append({"title":f"Pie Chart: {c}","img":_store_chart(fig5, f"piechart_{c}", 110)})

    interp=[]
    for r in numeric_summary:
        sk="approximately symmetric" if abs(r["Skewness"])<.5 else ("positively skewed" if r["Skewness"]>0 else "negatively skewed")
        interp.append(f"**{r['Variable']}**: N={r['N']}, M={r['Mean']}, SD={r['Std Dev']}, Median={r['Median']}. Range=[{r['Min']},{r['Max']}]. Distribution is {sk} (skewness={r['Skewness']}).")

    return {"test":"Descriptive Statistics","numeric_summary":numeric_summary,"categorical_summary":categorical_summary,
            "charts":charts,"interpretation":"\n\n".join(interp) if interp else "Selected columns analysed.",
            "apa_citation":"Descriptive statistics are reported as M (SD).",
            "significance":"—","p_value":None,"p_display":"—"}


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHI-SQUARE
# ══════════════════════════════════════════════════════════════════════════════
def chi_square_test(df, var1, var2):
    ct=pd.crosstab(df[var1],df[var2])
    
    # ── FIXED: Check for empty contingency table ──────────────────────────
    if ct.empty or ct.sum().sum() == 0:
        return {"error": "Contingency table is empty or has zero totals. Cannot perform chi-square test."}
    
    try:
        chi2,p,dof,exp=stats.chi2_contingency(ct)
    except Exception as e:
        return {"error": f"Chi-square test failed: {str(e)}. Check for invalid values in categorical variables."}
    
    n=ct.values.sum(); min_dim=min(ct.shape)-1
    v=np.sqrt(chi2/(n*max(min_dim,1))); ef="negligible" if v<.1 else "small" if v<.3 else "moderate" if v<.5 else "large"
    vrd,h0=verdict(p); charts=[]

    # Stacked bar + heatmap
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5)); fig.patch.set_facecolor(P["white"])
    ct.div(ct.sum(axis=1),axis=0).mul(100).plot(kind="bar",stacked=True,ax=a1,color=PAL[:len(ct.columns)],edgecolor="white",alpha=.88)
    a1.set_title(f"Stacked Bar: {var1} × {var2}",fontweight="bold",color=P["navy"]); a1.set_xlabel(var1); a1.set_ylabel("Percentage (%)"); a1.tick_params(axis="x",rotation=30); a1.legend(title=var2,bbox_to_anchor=(1.02,1),fontsize=8)
    sns.heatmap(ct,annot=True,fmt="d",cmap="Blues",ax=a2,linewidths=.5,cbar_kws={"shrink":.7})
    a2.set_title("Observed Frequencies",fontweight="bold",color=P["navy"])
    _style([a1]); fig.suptitle(f"Chi-Square: {var1} × {var2} | χ²({dof})={chi2:.3f}, {fp(p)}",fontsize=11,fontweight="bold",color=P["navy"],y=1.01)
    plt.tight_layout(); charts.append({"title":"Chi-Square Visualisation","img":_store_chart(fig, "chisquare_viz", 110)})

    # Standardised residuals
    exp_df=pd.DataFrame(exp,index=ct.index,columns=ct.columns); resid=(ct-exp_df)/np.sqrt(exp_df)
    fig2,ax=plt.subplots(figsize=(8,5)); fig2.patch.set_facecolor(P["white"])
    sns.heatmap(resid,annot=True,fmt=".2f",cmap="RdBu_r",center=0,ax=ax,linewidths=.5,cbar_kws={"shrink":.7})
    ax.set_title("Standardised Residuals (|value|>2 = significant cell)",fontweight="bold",color=P["navy"])
    charts.append({"title":"Standardised Residuals","img":_store_chart(fig2, "chisquare_residuals", 110)})

    ct_out=ct.copy(); ct_out["Row Total"]=ct_out.sum(axis=1); ct_out.loc["Col Total"]=ct_out.sum()
    return {"test":"Chi-Square Test of Independence","chi2":round(float(chi2),4),"p_value":round(float(p),4),
            "p_display":fp(p),"dof":int(dof),"significance":sp(p),"cramers_v":round(float(v),4),"effect_size":ef,"n":int(n),
            "contingency_table":ct_out.to_dict(),"expected_table":exp_df.round(2).to_dict(),"charts":charts,
            "apa_citation":f"χ²({dof}, N={n})={chi2:.2f}, {fp(p)}, Cramér's V={v:.2f}",
            "interpretation":f"A chi-square test of independence examined the relationship between {var1} and {var2}. "
                f"The association was {vrd}, χ²({dof}, N={n})={chi2:.2f}, {fp(p)}. Effect size (Cramér's V={v:.2f}) is {ef}. "
                f"We therefore {h0} the null hypothesis of independence."}


# ══════════════════════════════════════════════════════════════════════════════
# 3. INDEPENDENT T-TEST
# ══════════════════════════════════════════════════════════════════════════════
def independent_ttest(df, numeric_var, group_var, alpha=.05):
    # ── Performance Guard ──────────────────────────────────────────────────
    check_analysis_limits(df, "t_test", {})

    gs=df[group_var].dropna().unique()
    if len(gs)!=2: return {"error":f"Need exactly 2 groups; found {len(gs)}: {list(gs)}"}
    g1=pd.to_numeric(df.loc[df[group_var]==gs[0],numeric_var],errors="coerce").dropna()
    g2=pd.to_numeric(df.loc[df[group_var]==gs[1],numeric_var],errors="coerce").dropna()
    t_s,t_p=stats.ttest_ind(g1,g2); lev_s,lev_p=stats.levene(g1,g2)

    # ── FIXED: Proper Welch correction with appropriate effect size ──────
    if lev_p < 0.05:
        t_s_welch, t_p_welch = stats.ttest_ind(g1, g2, equal_var=False)
        t_s, t_p = t_s_welch, t_p_welch
        # Use Glass Delta for Welch case
        d = (g1.mean() - g2.mean()) / g2.std() if g2.std() > 0 else 0
        d_label = "Glass's Δ"
    else:
        # Pooled SD for equal variances (Cohen's d)
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

    # Means ± CI
    fig2,ax=plt.subplots(figsize=(7,5)); fig2.patch.set_facecolor(P["white"])
    for i,(lbl,gd) in enumerate([(str(gs[0]),g1),(str(gs[1]),g2)]):
        ax.errorbar(i,gd.mean(),yerr=gd.sem()*1.96,fmt="o",color=PAL[i],ms=12,capsize=8,lw=2,label=f"{lbl}: M={gd.mean():.2f}")
    ax.set_xticks([0,1]); ax.set_xticklabels([str(gs[0]),str(gs[1])],fontsize=11)
    ax.set_ylabel(numeric_var); ax.set_title("Means with 95% CI",fontweight="bold",color=P["navy"]); ax.legend()
    _style(ax); plt.tight_layout(); charts.append({"title":"Means with 95% CI","img":_store_chart(fig2, "ttest_means", 110)})

    stbl=[{"Group":str(gs[i]),"N":int(len(g)),"Mean":round(float(g.mean()),4),"Std Dev":round(float(g.std()),4),"Std Error":round(float(g.sem()),4)} for i,g in [(0,g1),(1,g2)]]
    return {"test":"Independent Samples T-Test","t_statistic":round(float(t_s),4),"p_value":round(float(t_p),4),
        "p_display":fp(t_p),"df":round(float(df_w),2),"significance":sp(t_p),"mean_difference":round(float(md),4),
        "ci_95":[round(float(ci[0]),4),round(float(ci[1]),4)],"cohens_d":round(float(d),4),"effect_size":ef,"effect_label":d_label,
        "levene_p":round(float(lev_p),4),"summary_table":stbl,"charts":charts,
        "apa_citation":f"t({df_w:.2f})={t_s:.2f}, {fp(t_p)}, {d_label}={d:.2f}, 95% CI [{ci[0]:.2f},{ci[1]:.2f}]",
        "interpretation":f"Independent samples t-test comparing {numeric_var} between {gs[0]} (M={g1.mean():.2f},SD={g1.std():.2f}) "
            f"and {gs[1]} (M={g2.mean():.2f},SD={g2.std():.2f}). Difference was {vrd}, t({df_w:.2f})={t_s:.2f}, {fp(t_p)}, {d_label}={d:.2f} ({ef} effect). "
            f"95% CI=[{ci[0]:.2f},{ci[1]:.2f}]. We {h0} the null hypothesis."
            +(f" Levene's test: {fp(lev_p)} — Welch correction applied." if lev_p<.05 else "")}


# ══════════════════════════════════════════════════════════════════════════════
# 4. ONE-WAY ANOVA
# ══════════════════════════════════════════════════════════════════════════════
def one_way_anova(df, numeric_var, group_var):
    # ── Performance Guard ──────────────────────────────────────────────────
    check_analysis_limits(df, "anova", {})

    gs=df[group_var].dropna().unique()
    # ── Fix: Check for small groups ──────────────────────────────────────
    gdata=[]
    for g in gs:
        group_data = pd.to_numeric(df.loc[df[group_var]==g,numeric_var],errors="coerce").dropna().values
        if len(group_data) < 2:
            return {"error": f"Group '{g}' has only {len(group_data)} observation(s). ANOVA requires at least 2 observations per group."}
        gdata.append(group_data)
    F,p=stats.f_oneway(*gdata); gm=np.concatenate(gdata).mean()
    ss_b=sum(len(g)*(g.mean()-gm)**2 for g in gdata); ss_t=sum((v-gm)**2 for g in gdata for v in g)
    eta=ss_b/ss_t if ss_t>0 else 0; ef="small" if eta<.06 else "medium" if eta<.14 else "large"
    n_t=sum(len(g) for g in gdata); k=len(gs); vrd,h0=verdict(p); charts=[]

    # Boxplot + Means bar
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

    # ── FIXED: Proper Tukey HSD post-hoc using summary data ──────────────
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    posthoc = []
    if p < 0.05 and k > 2:
        try:
            tukey = pairwise_tukeyhsd(endog=pdf[numeric_var], groups=pdf[group_var], alpha=0.05)
            # FIXED: Use summary data properly - gets ALL pairwise comparisons
            summary_data = tukey.summary().data[1:]  # Skip header row
            for row in summary_data:
                posthoc.append({
                    "Group 1": str(row[0]),
                    "Group 2": str(row[1]),
                    "Mean Diff": round(float(row[2]), 4),
                    "p-value": round(float(row[3]), 4),
                    "Significant": "Yes" if row[6] else "No"
                })
        except Exception as e:
            # Log error but continue
            logger.warning(f"Tukey HSD failed: {str(e)}. Falling back to Bonferroni.")
            # Fallback to Bonferroni if Tukey fails
            n_pairs = len(list(combinations(gs,2)))
            for (i,g_a),(j,g_b) in combinations(enumerate(gs),2):
                _,tp=stats.ttest_ind(gdata[i],gdata[j]); tadj=min(tp*n_pairs,1.0)
                posthoc.append({"Group 1":str(g_a),"Group 2":str(g_b),"Mean Diff":round(float(gdata[i].mean()-gdata[j].mean()),4),"p (Bonferroni)":round(float(tadj),4),"Significant":"Yes" if tadj<.05 else "No"})

    stbl=[{"Group":str(g),"N":int(len(gd)),"Mean":round(float(gd.mean()),4),"Std Dev":round(float(gd.std()),4),"Std Error":round(float(stats.sem(gd)),4),"Min":round(float(gd.min()),4),"Max":round(float(gd.max()),4)} for g,gd in zip(gs,gdata)]
    return {"test":"One-Way ANOVA","f_statistic":round(float(F),4),"p_value":round(float(p),4),"p_display":fp(p),
        "df_between":int(k-1),"df_within":int(n_t-k),"significance":sp(p),"eta_squared":round(float(eta),4),
        "effect_size":ef,"summary_table":stbl,"posthoc_table":posthoc,"charts":charts,
        "apa_citation":f"F({k-1},{n_t-k})={F:.2f}, {fp(p)}, η²={eta:.3f}",
        "interpretation":f"One-way ANOVA compared {numeric_var} across {k} groups of {group_var}. Result was {vrd}, F({k-1},{n_t-k})={F:.2f}, {fp(p)}, η²={eta:.3f} ({ef} effect). "
            f"We {h0} the null hypothesis of equal means."+(" Tukey HSD post-hoc comparisons provided." if p<.05 and len(posthoc)>0 else "")}


# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
def correlation_analysis(df, columns, method="pearson"):
    # ── Performance Guard ──────────────────────────────────────────────────
    check_analysis_limits(df, "correlation", {})

    num=[c for c in columns if pd.to_numeric(df[c],errors="coerce").notna().sum()>len(df)*.4]
    if len(num)<2: return {"error":"Need ≥2 numeric columns."}
    sub=df[num].apply(pd.to_numeric,errors="coerce").dropna(); charts=[]

    # ── Fix: Check for constant columns ──────────────────────────────────
    constant_cols = [c for c in sub.columns if sub[c].nunique() <= 1]
    if constant_cols:
        return {"error": f"The following column(s) are constant and cannot be used in correlation: {', '.join(constant_cols)}."}

    if method=="spearman": corr_m=sub.corr(method="spearman")
    else: corr_m=sub.corr(method="pearson")

    pairs=[]
    for c1,c2 in combinations(num,2):
        s2=sub[[c1,c2]].dropna()
        r,p=(stats.spearmanr(s2[c1],s2[c2]) if method=="spearman" else stats.pearsonr(s2[c1],s2[c2]))
        st="negligible" if abs(r)<.1 else "weak" if abs(r)<.3 else "moderate" if abs(r)<.5 else "strong" if abs(r)<.7 else "very strong"
        pairs.append({"Variable 1":c1,"Variable 2":c2,"r":round(float(r),4),"p-value":round(float(p),4),"p_display":fp(p),"Significance":sp(p),"Strength":f"{'positive' if r>0 else 'negative'} {st}"})

    # Heatmap
    fig,ax=plt.subplots(figsize=(max(6,len(num)*1.3+2),max(5,len(num)*1.2))); fig.patch.set_facecolor(P["white"])
    mask=np.triu(np.ones_like(corr_m,dtype=bool))
    sns.heatmap(corr_m,annot=True,fmt=".3f",cmap="coolwarm",vmin=-1,vmax=1,center=0,ax=ax,mask=mask,square=True,linewidths=.5,annot_kws={"size":10,"weight":"bold"},cbar_kws={"shrink":.8})
    ax.set_title(f"{method.title()} Correlation Matrix",fontsize=13,fontweight="bold",color=P["navy"])
    plt.tight_layout(); charts.append({"title":"Correlation Heatmap","img":_store_chart(fig, "correlation_heatmap", 110)})

    # ── FIXED: Limit Pairplot to Max 4 Variables and sample data ────────────
    MAX_PAIRPLOT_VARS = 4
    if 2 <= len(num) <= MAX_PAIRPLOT_VARS:
        # ── FIXED: Sample data for pairplot to prevent memory explosion ──
        sample_size = min(3000, len(sub))
        if len(sub) > sample_size:
            pairplot_df = sub.sample(sample_size)
        else:
            pairplot_df = sub
        g=sns.pairplot(pairplot_df,diag_kind="kde",plot_kws={"alpha":.5,"color":P["blue"],"s":20},diag_kws={"color":P["teal"],"fill":True})
        g.figure.suptitle("Scatter Matrix",y=1.02,fontsize=13,fontweight="bold",color=P["navy"])
        g.figure.patch.set_facecolor(P["white"])
        charts.append({"title":"Scatter Matrix","img":_store_chart(g.figure, "correlation_scatter", 110)})
    elif len(num) > MAX_PAIRPLOT_VARS:
        # Instead of pairplot, we generate a few key scatter plots
        # Pick first 4 variables for scatter matrix to avoid timeout
        limited_num = num[:MAX_PAIRPLOT_VARS]
        sample_size = min(3000, len(sub))
        if len(sub) > sample_size:
            pairplot_df = sub.iloc[:sample_size]
        else:
            pairplot_df = sub
        g=sns.pairplot(pairplot_df[limited_num],diag_kind="kde",plot_kws={"alpha":.5,"color":P["blue"],"s":20},diag_kws={"color":P["teal"],"fill":True})
        g.figure.suptitle("Scatter Matrix (Limited to 4 Variables)",y=1.02,fontsize=13,fontweight="bold",color=P["navy"])
        g.figure.patch.set_facecolor(P["white"])
        charts.append({"title":"Scatter Matrix (Limited to 4 Variables)","img":_store_chart(g.figure, "correlation_scatter_limited", 110)})

    # Bubble chart for pairs
    if pairs:
        fig3,ax3=plt.subplots(figsize=(10,5)); fig3.patch.set_facecolor(P["white"])
        x_pos=range(len(pairs)); rs=[abs(p_["r"]) for p_ in pairs]; cols_=[P["ok"] if p_["r"]>0 else P["err"] for p_ in pairs]
        sc=ax3.scatter(x_pos,[p_["r"] for p_ in pairs],s=[r*500+30 for r in rs],c=cols_,alpha=.75,edgecolors="white",lw=1.5)
        ax3.axhline(0,color=P["slate"],lw=1,ls="--"); ax3.axhline(.3,color=P["warn"],lw=.8,ls=":"); ax3.axhline(-.3,color=P["warn"],lw=.8,ls=":")
        ax3.set_xticks(list(x_pos)); ax3.set_xticklabels([f"{p_['Variable 1'][:8]}×{p_['Variable 2'][:8]}" for p_ in pairs],rotation=35,ha="right",fontsize=8)
        ax3.set_ylabel("r value"); ax3.set_title("Correlation Bubble Chart (size = |r|, green=positive, red=negative)",fontweight="bold",color=P["navy"])
        _style(ax3); plt.tight_layout(); charts.append({"title":"Correlation Bubble Chart","img":_store_chart(fig3, "correlation_bubble", 110)})

    return {"test":f"{method.title()} Correlation","method":method,"n":int(len(sub)),"variables":num,"pairs_table":pairs,"charts":charts,
        "p_value":min([p_["p-value"] for p_ in pairs]) if pairs else 1,"p_display":"see table","significance":"see table",
        "apa_citation":f"{method.title()} correlation analysis conducted on {len(num)} variables (N={len(sub)}).",
        "interpretation":f"**{method.title()} correlation** was conducted on {len(pairs)} variable pair(s).\n\n"+"".join(
            f"- **{p_['Variable 1']} & {p_['Variable 2']}**: r={p_['r']}, {p_['p_display']} ({sp(p_['p-value'])}) — {p_['Strength']}.\n" for p_ in pairs)}


# ══════════════════════════════════════════════════════════════════════════════
# 6. LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
def linear_regression(df, dependent, predictors):
    # ── Performance Guard ──────────────────────────────────────────────────
    check_analysis_limits(df, "regression", {"predictors": predictors})

    sub=df[[dependent]+predictors].apply(pd.to_numeric,errors="coerce").dropna()
    y=sub[dependent].values; Xr=sub[predictors].values; charts=[]

    # ── FIXED: Check for constant predictors ──────────────────────────────
    constant_preds = [p for p in predictors if sub[p].nunique() <= 1]
    if constant_preds:
        return {"error": f"The following predictor(s) are constant and cannot be used in regression: {', '.join(constant_preds)}."}

    # ── FIXED: Minimum sample size guard ──────────────────────────────────
    min_n = max(20, len(predictors) * 10)  # Rule of thumb: 10 per predictor
    n_obs = len(sub)
    
    if n_obs < min_n:
        return {"error": f"Regression requires at least {min_n} valid observations. You have {n_obs}."}
    
    if n_obs <= len(predictors) + 1:
        return {"error": f"Sample size ({n_obs}) must be greater than number of predictors ({len(predictors)}) + 1. "
                f"Minimum required: {len(predictors) + 2}."}

    # ── FIXED: Use statsmodels OLS for proper diagnostics ──────────────────
    import statsmodels.api as sm
    X = sm.add_constant(Xr)
    model = sm.OLS(y, X).fit()
    
    # Extract statistics
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    F_stat = model.fvalue
    p_f = model.f_pvalue
    
    # Coefficient table
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
    
    # ── FIXED: Added Cook's Distance ──────────────────────────────────────
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    
    # ── FIXED: Added VIF ──────────────────────────────────────────────────
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = []
    for i in range(X.shape[1]):
        if i == 0:  # Skip intercept
            continue
        vif = variance_inflation_factor(X, i)
        vif_data.append({
            "Variable": predictors[i-1],
            "VIF": round(float(vif), 3)
        })
    
    # Regression diagnostics plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor(P["white"])
    
    # Actual vs Predicted
    axes[0, 0].scatter(model.fittedvalues, y, alpha=0.6, color=P["blue"], s=40)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, color=P["err"])
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")
    axes[0, 0].set_title("Actual vs Predicted", fontweight="bold", color=P["navy"])
    
    # Residuals vs Fitted
    axes[0, 1].scatter(model.fittedvalues, model.resid, alpha=0.6, color=P["teal"], s=40)
    axes[0, 1].axhline(0, color=P["err"], linestyle="--", lw=1.5)
    axes[0, 1].set_xlabel("Fitted Values")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residuals vs Fitted", fontweight="bold", color=P["navy"])
    
    # Q-Q Plot
    stats.probplot(model.resid, dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set(color=P["blue"], alpha=0.7, ms=5)
    axes[1, 0].get_lines()[1].set(color=P["err"], lw=1.5)
    axes[1, 0].set_title("Q-Q Plot of Residuals", fontweight="bold", color=P["navy"])
    
    # Cook's Distance
    axes[1, 1].stem(range(len(cooks_d)), cooks_d, linefmt=P["blue"], markerfmt='o', basefmt=" ")
    axes[1, 1].axhline(0.5, color=P["err"], linestyle="--", lw=1.5, label="Influential threshold (0.5)")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].set_title("Cook's Distance", fontweight="bold", color=P["navy"])
    axes[1, 1].legend()
    
    _style(axes.flatten())
    plt.tight_layout()
    charts.append({"title":"Regression Diagnostics","img":_store_chart(fig, "regression_diagnostics", 110)})
    
    # Coefficient plot
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

    # ── FIXED: Proper equation formatting with signs ──────────────────────
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
            f"Overall model was {vrd} ({fp(p_f)}). Equation: {eq}."
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. MANN-WHITNEY U
# ══════════════════════════════════════════════════════════════════════════════
def mann_whitney(df, numeric_var, group_var):
    # ── Performance Guard ──────────────────────────────────────────────────
    check_analysis_limits(df, "mann_whitney", {})

    gs=df[group_var].dropna().unique()
    if len(gs)!=2: return {"error":"Requires exactly 2 groups."}
    g1=pd.to_numeric(df.loc[df[group_var]==gs[0],numeric_var],errors="coerce").dropna()
    g2=pd.to_numeric(df.loc[df[group_var]==gs[1],numeric_var],errors="coerce").dropna()
    
    # ── FIXED: Proper tie correction for z-statistic ──────────────────────
    from scipy.stats import mannwhitneyu
    # Get both U statistic and p-value
    res = mannwhitneyu(g1, g2, alternative='two-sided', method='exact' if min(len(g1), len(g2)) < 50 else 'auto')
    U = res.statistic
    p = res.pvalue
    
    # Calculate z with tie correction
    n1, n2 = len(g1), len(g2)
    mu = n1 * n2 / 2
    
    # Calculate tied ranks for tie correction
    combined = np.concatenate([g1, g2])
    from scipy.stats import rankdata
    ranks = rankdata(combined)
    
    # Count ties
    unique, counts = np.unique(combined, return_counts=True)
    ties = counts[counts > 1]
    
    if len(ties) > 0:
        # Tie correction formula
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
    return {"test":"Mann-Whitney U Test","u_statistic":round(float(U),2),"p_value":round(float(p),4),"p_display":fp(p),
        "significance":sp(p),"effect_r":round(float(r),4),"medians":{str(gs[0]):round(float(g1.median()),4),str(gs[1]):round(float(g2.median()),4)},
        "charts":charts,"apa_citation":f"U={U:.0f}, {fp(p)}, r={r:.3f}",
        "interpretation":f"Mann-Whitney U compared {numeric_var} between {gs[0]} (Mdn={g1.median():.2f}) and {gs[1]} (Mdn={g2.median():.2f}). Difference was {vrd}, U={U:.0f}, {fp(p)}, r={r:.3f}. We {h0} H₀."}


# ══════════════════════════════════════════════════════════════════════════════
# 8. KRUSKAL-WALLIS
# ══════════════════════════════════════════════════════════════════════════════
def kruskal_wallis(df, numeric_var, group_var):
    # ── Performance Guard ──────────────────────────────────────────────────
    check_analysis_limits(df, "kruskal_wallis", {})

    gs=df[group_var].dropna().unique()
    
    # ── FIXED: Validate group sizes ──────────────────────────────────────
    gd = []
    for g in gs:
        group_data = pd.to_numeric(df.loc[df[group_var]==g,numeric_var],errors="coerce").dropna().values
        if len(group_data) < 2:
            return {"error": f"Group '{g}' has only {len(group_data)} observation(s). Kruskal-Wallis requires at least 2 observations per group."}
        gd.append(group_data)
    
    H,p=stats.kruskal(*gd); n_t=sum(len(g) for g in gd)
    eta=max((H-len(gs)+1)/(n_t-len(gs)),0); vrd,h0=verdict(p); charts=[]
    fig,ax=plt.subplots(figsize=(max(7,len(gs)*1.5),5)); fig.patch.set_facecolor(P["white"])
    pdf=pd.DataFrame({numeric_var:np.concatenate(gd),group_var:np.repeat([str(g) for g in gs],[len(g) for g in gd])})
    sns.boxplot(data=pdf,x=group_var,y=numeric_var,palette=PAL[:len(gs)],ax=ax,width=.5)
    sns.stripplot(data=pdf,x=group_var,y=numeric_var,color=P["navy"],alpha=.35,size=4,jitter=True,ax=ax)
    ax.set_title(f"Kruskal-Wallis: {numeric_var} by {group_var} | H({len(gs)-1})={H:.3f}, {fp(p)}",fontweight="bold",color=P["navy"]); ax.tick_params(axis="x",rotation=30)
    _style(ax); plt.tight_layout(); charts.append({"title":"Kruskal-Wallis Box Plot","img":_store_chart(fig, "kruskal_boxplot", 110)})
    return {"test":"Kruskal-Wallis H Test","h_statistic":round(float(H),4),"p_value":round(float(p),4),"p_display":fp(p),
        "df":int(len(gs)-1),"significance":sp(p),"eta_squared_h":round(float(eta),4),
        "group_medians":{str(g):round(float(np.median(gd[i])),4) for i,g in enumerate(gs)},"charts":charts,
        "apa_citation":f"H({len(gs)-1})={H:.2f}, {fp(p)}",
        "interpretation":f"Kruskal-Wallis H test compared {numeric_var} across {len(gs)} groups. Result was {vrd}, H({len(gs)-1})={H:.2f}, {fp(p)}. We {h0} H₀."}


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

def run_analysis(df, analysis_type, params):
    if analysis_type not in ANALYSIS_MAP: 
        return {"error": f"Unknown analysis type: {analysis_type}"}
    try:
        # Validate dataset first
        df = validate_dataset(df)
        return ANALYSIS_MAP[analysis_type](df, **params)
    except AnalysisTooLargeError as e:
        return {
            "error": "analysis_too_large",
            "message": str(e),
            "cta_text": "📞 Book a Free Expert Consultation",
            "cta_url": "https://www.eduxellence.org/#contact"
        }
    except Exception as e:
        # ── FIXED: Log error internally, return generic message ──────────
        logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        return {
            "error": "internal_server_error",
            "message": "An unexpected error occurred while processing your analysis. Our team has been notified.",
            "reference_id": str(uuid.uuid4())[:8]
        }
