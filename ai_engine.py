"""
Eduxellence AI Interpretation Engine — Phase 2
================================================
Generates intelligent, context-aware plain-English interpretations
of statistical results using Google Gemini 1.5 Flash (free tier).

Architecture:
  • Primary:  Gemini 1.5 Flash via REST API (urllib — no pip install)
  • Fallback: Rich template engine (works when API key missing / quota hit)
  • Modes:    simple (student) | academic (researcher) | executive (business)

Free tier limits (Gemini 1.5 Flash):
  15 requests/minute · 1 million tokens/minute · 1,500 requests/day
  Cost: $0.00 for these limits

by Eduxellence Analytics · https://eduxellence.org
"""

import os, json, re, ssl, urllib.request, urllib.error
from typing import Optional

# ── Gemini config ──────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-1.5-flash"
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT = 12   # seconds — stay well inside Vercel's 30s limit
MAX_TOKENS     = 600  # enough for a full paragraph, not wasteful


# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def generate_ai_interpretation(
    results: dict,
    mode: str = "simple",
    api_key: Optional[str] = None,
) -> dict:
    """
    Generate an AI interpretation of statistical results.

    Args:
        results:  The full results dict from stats_engine.run_analysis()
        mode:     'simple'   — plain English for students/general users
                  'academic' — formal academic language for researchers
                  'executive'— brief business-oriented summary
        api_key:  Gemini API key. Falls back to GEMINI_API_KEY env var.

    Returns:
        {
          "ai_interpretation": str,   # The generated text
          "mode": str,
          "source": "gemini" | "template",
          "model": str,
          "tokens_used": int | None,
          "error": str | None         # present only if something failed
        }
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")

    if key:
        try:
            return _call_gemini(results, mode, key)
        except Exception as exc:
            # Graceful fallback — never break the user's flow
            return {
                **_template_interpretation(results, mode),
                "error": f"Gemini unavailable ({exc}), using template.",
            }
    else:
        return {
            **_template_interpretation(results, mode),
            "error": None,
        }


# ══════════════════════════════════════════════════════════════════════════
#  GEMINI REST CALL (urllib — zero extra dependencies)
# ══════════════════════════════════════════════════════════════════════════

def _call_gemini(results: dict, mode: str, api_key: str) -> dict:
    prompt  = _build_prompt(results, mode)
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": MAX_TOKENS,
            "temperature":     0.4,
            "topP":            0.85,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }).encode("utf-8")

    url = f"{GEMINI_ENDPOINT}?key={api_key}"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=GEMINI_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {error_body[:200]}")

    # Parse response
    try:
        text        = body["candidates"][0]["content"]["parts"][0]["text"].strip()
        tokens_used = body.get("usageMetadata", {}).get("totalTokenCount")
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {exc}")

    return {
        "ai_interpretation": text,
        "mode":              mode,
        "source":            "gemini",
        "model":             GEMINI_MODEL,
        "tokens_used":       tokens_used,
        "error":             None,
    }


# ══════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════

_MODE_INSTRUCTION = {
    "simple": (
        "Write in plain, friendly English that a first-year university student "
        "can understand. Avoid jargon. Explain what the result means in real life. "
        "Keep it to 3–4 sentences."
    ),
    "academic": (
        "Write in formal academic language suitable for a peer-reviewed journal "
        "Results section. Use APA 7th Edition conventions. Include the key statistic, "
        "effect size, and a brief interpretive statement. 4–6 sentences."
    ),
    "executive": (
        "Write a concise business-oriented summary for a non-technical executive. "
        "Focus on what the finding means for decisions. "
        "Avoid statistical jargon entirely. 2–3 sentences maximum."
    ),
}

def _build_prompt(results: dict, mode: str) -> str:
    test   = results.get("test", "Statistical Analysis")
    p_disp = results.get("p_display", "—")
    p_val  = results.get("p_value")
    sig    = results.get("significance", "")
    apa    = results.get("apa_citation", "")
    effect = _extract_effect(results)
    tmpl   = results.get("interpretation", "")

    # Pull key variables from params where possible
    dep    = results.get("dependent", "")
    preds  = results.get("predictors", [])
    r2     = results.get("r_squared")
    chi2   = results.get("chi2")
    cramv  = results.get("cramers_v")
    cohensd= results.get("cohens_d")
    eta    = results.get("eta_squared")

    sig_text = "statistically significant" if (p_val is not None and p_val < 0.05) \
               else "not statistically significant"

    details = []
    if apa:             details.append(f"APA result: {apa}")
    if effect:          details.append(f"Effect size: {effect}")
    if r2 is not None:  details.append(f"R² = {r2} (variance explained)")
    if chi2:            details.append(f"Chi-square = {chi2}, Cramér's V = {cramv}")
    if cohensd:         details.append(f"Cohen's d = {cohensd}")
    if eta:             details.append(f"Eta-squared = {eta}")
    if dep and preds:   details.append(f"Predicting '{dep}' from: {', '.join(preds)}")

    instruction = _MODE_INSTRUCTION.get(mode, _MODE_INSTRUCTION["simple"])

    prompt = f"""You are an expert statistician at Eduxellence Analytics (eduxellence.org).
A user has just run a {test} and needs help understanding the results.

Statistical Summary:
- Test: {test}
- Result is: {sig_text} ({p_disp}, {sig})
{chr(10).join(f'- {d}' for d in details)}

Template interpretation for context (do NOT copy this verbatim):
{tmpl[:400] if tmpl else 'Not available.'}

Your task:
{instruction}

Important rules:
- Do NOT start with "The results show" or "This analysis"
- Do NOT mention "Eduxellence" in the interpretation text itself
- Do NOT add disclaimers or suggest consulting a statistician
- Write ONE cohesive paragraph only — no bullet points, no headings
- If the result is NOT significant, still explain what that means clearly
- Be specific about variable names if they are available in the summary above
"""
    return prompt


# ══════════════════════════════════════════════════════════════════════════
#  TEMPLATE FALLBACK ENGINE
# ══════════════════════════════════════════════════════════════════════════

def _template_interpretation(results: dict, mode: str) -> dict:
    """
    Rich template-based interpretation. Used when Gemini key is absent
    or quota is exceeded. Produces a high-quality canned response
    that is still specific to the test and result.
    """
    test  = results.get("test", "")
    p_val = results.get("p_value")
    sig   = p_val is not None and p_val < 0.05
    p_str = results.get("p_display", "p = —")

    text  = _dispatch_template(test, results, sig, p_str, mode)

    return {
        "ai_interpretation": text,
        "mode":              mode,
        "source":            "template",
        "model":             None,
        "tokens_used":       None,
        "error":             None,
    }


def _dispatch_template(test, r, sig, p_str, mode):
    t = test.lower()
    if "descriptive" in t:     return _tmpl_descriptive(r, mode)
    if "chi"         in t:     return _tmpl_chi(r, sig, p_str, mode)
    if "t-test"      in t or "t_test" in t: return _tmpl_ttest(r, sig, p_str, mode)
    if "anova"       in t:     return _tmpl_anova(r, sig, p_str, mode)
    if "correlation" in t:     return _tmpl_correlation(r, mode)
    if "regression"  in t:     return _tmpl_regression(r, sig, p_str, mode)
    if "mann"        in t:     return _tmpl_mann(r, sig, p_str, mode)
    if "kruskal"     in t:     return _tmpl_kruskal(r, sig, p_str, mode)
    return r.get("interpretation", "No interpretation available.")


def _sig_phrase(sig, mode):
    if mode == "executive":
        return ("a meaningful difference was found" if sig
                else "no meaningful difference was detected")
    if mode == "academic":
        return ("statistically significant" if sig else "not statistically significant")
    return ("a statistically significant relationship" if sig
            else "no statistically significant relationship")


def _tmpl_descriptive(r, mode):
    rows = r.get("numeric_summary", [])
    if not rows:
        return "Descriptive statistics were computed for the selected variables."
    highlights = []
    for row in rows[:3]:
        var  = row.get("Variable","?")
        mean = row.get("Mean","?")
        sd   = row.get("Std Dev","?")
        skew = row.get("Skewness", 0)
        skew_label = ("approximately normal" if abs(float(skew)) < 0.5
                      else ("positively skewed" if float(skew) > 0 else "negatively skewed"))
        highlights.append(f"{var} (M = {mean}, SD = {sd}, {skew_label})")
    if mode == "executive":
        return (f"A summary of your data shows: {'; '.join(highlights)}. "
                f"Review the charts for distribution patterns and potential outliers.")
    if mode == "academic":
        return (f"Descriptive statistics were computed for all variables. "
                f"Key findings include: {'; '.join(highlights)}. "
                f"Distribution normality was assessed via skewness coefficients and Q-Q plots.")
    return (f"Here is a summary of your data: {'; '.join(highlights)}. "
            f"Check the histograms and boxplots below to see how each variable "
            f"is distributed and whether any unusual values are present.")


def _tmpl_chi(r, sig, p_str, mode):
    v1  = r.get("contingency_table") and "the variables" or "the two variables"
    v   = r.get("cramers_v", "")
    ef  = r.get("effect_size", "")
    apa = r.get("apa_citation","")
    if mode == "executive":
        return (f"{'A significant association' if sig else 'No significant association'} "
                f"was found between the two categorical variables ({p_str}). "
                + (f"The association strength is {ef} (Cramér's V = {v})." if sig else ""))
    if mode == "academic":
        return (f"A chi-square test of independence revealed that the association between "
                f"the variables was {_sig_phrase(sig,'academic')} ({apa}). "
                + (f"The effect size was {ef} (Cramér's V = {v}), indicating a "
                   f"practically {ef} association." if sig else
                   f"These findings suggest the variables operate independently in this sample."))
    return (f"{'There is a statistically significant relationship' if sig else 'There is no significant relationship'} "
            f"between your two categorical variables ({p_str}). "
            + (f"The strength of this association is {ef} (Cramér's V = {v}), "
               f"meaning the link between the categories is {ef} in practice. "
               f"Check the heatmap to see which specific category combinations are driving this result."
               if sig else
               f"This means knowing someone's value on one variable does not help predict "
               f"their value on the other. The two categories appear to be independent."))


def _tmpl_ttest(r, sig, p_str, mode):
    stbl = r.get("summary_table", [])
    g1   = stbl[0] if len(stbl) > 0 else {}
    g2   = stbl[1] if len(stbl) > 1 else {}
    d    = r.get("cohens_d","")
    ef   = r.get("effect_size","")
    ci   = r.get("ci_95","")
    md   = r.get("mean_difference","")
    apa  = r.get("apa_citation","")
    g1n  = g1.get("Group","Group 1"); g1m = g1.get("Mean","")
    g2n  = g2.get("Group","Group 2"); g2m = g2.get("Mean","")
    if mode == "executive":
        return (f"{'A significant difference' if sig else 'No significant difference'} "
                f"was found between {g1n} (M={g1m}) and {g2n} (M={g2m}) ({p_str}). "
                + (f"The effect size is {ef} (d={d})." if sig else ""))
    if mode == "academic":
        return (f"An independent samples t-test indicated that the difference between "
                f"{g1n} (M={g1m}) and {g2n} (M={g2m}) was {_sig_phrase(sig,'academic')} "
                f"({apa}). "
                + (f"The effect size was {ef} (Cohen's d = {d}), and the 95% CI for the "
                   f"mean difference was {ci}, suggesting a practically {ef} and "
                   f"statistically reliable difference." if sig else
                   f"These results suggest the two groups perform similarly on this measure."))
    return (f"{'Your two groups are significantly different' if sig else 'Your two groups are not significantly different'} "
            f"({p_str}). {g1n} scored an average of {g1m} while {g2n} averaged {g2m}. "
            + (f"The effect size is {ef} (Cohen's d = {d}), meaning this is a {ef} "
               f"real-world difference — not just a statistical one. "
               f"The 95% confidence interval for the gap is {ci}."
               if sig else
               f"The difference in averages ({md}) is small enough that it could easily "
               f"be due to random chance in your sample."))


def _tmpl_anova(r, sig, p_str, mode):
    k    = len(r.get("summary_table",[]))
    eta  = r.get("eta_squared","")
    ef   = r.get("effect_size","")
    apa  = r.get("apa_citation","")
    if mode == "executive":
        return (f"{'Significant differences' if sig else 'No significant differences'} "
                f"were found across the {k} groups ({p_str}). "
                + (f"Group membership explains {float(eta)*100:.1f}% of variance (η² = {eta})." if sig and eta else ""))
    if mode == "academic":
        return (f"A one-way ANOVA revealed that group differences were "
                f"{_sig_phrase(sig,'academic')} ({apa}). "
                + (f"The effect size (η² = {eta}) indicates that group membership accounts "
                   f"for a {ef} proportion of variance in the outcome. "
                   f"Bonferroni-corrected post-hoc comparisons identify which specific "
                   f"group pairs differ significantly." if sig else
                   f"These results indicate that the groups do not differ significantly "
                   f"on this measure in the present sample."))
    return (f"{'There are significant differences' if sig else 'There are no significant differences'} "
            f"between your {k} groups ({p_str}). "
            + (f"Group membership explains {float(eta)*100:.1f}% of the variation in scores "
               f"(η² = {eta}), which is a {ef} effect. "
               f"Check the post-hoc table to see exactly which groups are different from each other."
               if sig else
               f"All {k} groups perform similarly — any differences you see in the chart "
               f"are likely due to random sampling variation rather than real group differences."))


def _tmpl_correlation(r, mode):
    pairs = r.get("pairs_table", [])
    if not pairs:
        return "Correlation analysis was completed. Review the heatmap for patterns."
    top   = max(pairs, key=lambda x: abs(x.get("r",0)))
    v1    = top.get("Variable 1","Var1")
    v2    = top.get("Variable 2","Var2")
    rv    = top.get("r","")
    pv    = top.get("p_display","")
    st    = top.get("Strength","")
    meth  = r.get("method","Pearson").title()
    if mode == "executive":
        return (f"The strongest relationship in your data is between {v1} and {v2} "
                f"(r = {rv}, {pv}): a {st} correlation. "
                f"See the heatmap for all variable relationships.")
    if mode == "academic":
        return (f"{meth} correlation analysis was conducted across {len(pairs)} variable pair(s). "
                f"The strongest association was between {v1} and {v2} "
                f"(r = {rv}, {pv}), indicating a {st} relationship. "
                f"The correlation matrix and scatter plots are provided for full inspection.")
    return (f"Your data shows the strongest connection between {v1} and {v2}: "
            f"r = {rv} ({pv}), which is a {st} correlation. "
            + ("This means as one goes up, the other tends to go up too. "
               if rv and float(str(rv)) > 0
               else "This means as one goes up, the other tends to go down. ")
            + f"The heatmap shows all relationships at once — the darker the colour, "
              f"the stronger the link.")


def _tmpl_regression(r, sig, p_str, mode):
    r2   = r.get("r_squared","")
    ar2  = r.get("adj_r_squared","")
    dep  = r.get("dependent","the outcome")
    preds= r.get("predictors",[])
    eq   = r.get("equation","")
    apa  = r.get("apa_citation","")
    label= r.get("test","Regression")
    if mode == "executive":
        return (f"Your model {'successfully' if sig else 'does not'} predicts {dep} "
                f"from {', '.join(preds)} ({p_str}). "
                + (f"It explains {float(r2)*100:.1f}% of variation in {dep} (R² = {r2})."
                   if sig else ""))
    if mode == "academic":
        return (f"A {label.lower()} was conducted to predict {dep} from "
                f"{', '.join(preds)}. The model was {_sig_phrase(sig,'academic')} "
                f"({apa}), explaining {float(r2)*100:.1f}% of variance "
                f"(R² = {r2}, Adj. R² = {ar2}). "
                + (f"The regression equation is: {eq}. "
                   f"Coefficient significance is reported in the table below."
                   if sig else
                   f"These results suggest the selected predictors do not significantly "
                   f"account for variance in {dep} in this sample."))
    return (f"Your model {'can' if sig else 'cannot'} significantly predict {dep} "
            f"using {', '.join(preds)} ({p_str}). "
            + (f"It explains {float(r2)*100:.1f}% of the variation in {dep} — "
               f"meaning {float(r2)*100:.1f}% of what makes {dep} go up or down "
               f"is captured by your predictors. "
               f"The equation is: {eq}. Check the coefficient table to see which "
               f"predictors matter most."
               if sig else
               f"Only {float(r2)*100:.1f}% of the variation in {dep} is explained, "
               f"which suggests your predictors may not be the main drivers. "
               f"Consider adding other variables or checking the residuals plot."))


def _tmpl_mann(r, sig, p_str, mode):
    meds = r.get("medians",{})
    U    = r.get("u_statistic","")
    rv   = r.get("effect_r","")
    apa  = r.get("apa_citation","")
    groups = list(meds.keys())
    g1n  = groups[0] if groups else "Group 1"
    g2n  = groups[1] if len(groups)>1 else "Group 2"
    g1m  = meds.get(g1n,""); g2m = meds.get(g2n,"")
    if mode == "executive":
        return (f"{'A significant difference' if sig else 'No significant difference'} "
                f"was found between {g1n} (Mdn={g1m}) and {g2n} (Mdn={g2m}) ({p_str}).")
    if mode == "academic":
        return (f"A Mann-Whitney U test indicated that the difference between "
                f"{g1n} (Mdn = {g1m}) and {g2n} (Mdn = {g2m}) was "
                f"{_sig_phrase(sig,'academic')} ({apa}, r = {rv}). "
                + ("This non-parametric test was selected due to violations of normality."
                   if sig else ""))
    return (f"Using a non-parametric test (which works better when data is not normally distributed), "
            f"{'a significant difference was found' if sig else 'no significant difference was found'} "
            f"between {g1n} (median = {g1m}) and {g2n} (median = {g2m}) ({p_str}). "
            + (f"The effect size r = {rv} tells us how large this difference is in practice."
               if sig else
               f"The two groups appear to have similar distributions."))


def _tmpl_kruskal(r, sig, p_str, mode):
    k    = len(r.get("group_medians",{}))
    H    = r.get("h_statistic","")
    apa  = r.get("apa_citation","")
    if mode == "executive":
        return (f"{'Significant differences' if sig else 'No significant differences'} "
                f"detected across {k} groups ({p_str}, H = {H}).")
    if mode == "academic":
        return (f"A Kruskal-Wallis H test indicated that differences across {k} groups "
                f"were {_sig_phrase(sig,'academic')} ({apa}). "
                "This non-parametric test was appropriate given the distributional properties of the data.")
    return (f"A Kruskal-Wallis test (a non-parametric version of ANOVA) found "
            f"{'significant differences' if sig else 'no significant differences'} "
            f"across your {k} groups ({p_str}). "
            + (f"At least one group is meaningfully different from the others. "
               f"Check the boxplot to see which groups stand apart."
               if sig else
               f"All {k} groups have similar distributions — any visible differences "
               f"in the chart are within normal sampling variation."))


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _extract_effect(results: dict) -> str:
    for key, label in [
        ("cohens_d",   "Cohen's d"),
        ("cramers_v",  "Cramér's V"),
        ("eta_squared","η²"),
        ("r_squared",  "R²"),
        ("effect_r",   "r"),
    ]:
        val = results.get(key)
        if val is not None:
            return f"{label} = {val} ({results.get('effect_size','')})"
    return results.get("effect_size","")


# ══════════════════════════════════════════════════════════════════════════
#  BATCH: interpret ALL test results in one call
# ══════════════════════════════════════════════════════════════════════════

def interpret_all_modes(results: dict, api_key: Optional[str] = None) -> dict:
    """
    Generate interpretations in all three modes at once.
    Returns { 'simple': ..., 'academic': ..., 'executive': ... }
    Each value is a generate_ai_interpretation() response dict.
    """
    out = {}
    for mode in ["simple", "academic", "executive"]:
        out[mode] = generate_ai_interpretation(results, mode=mode, api_key=api_key)
    return out
