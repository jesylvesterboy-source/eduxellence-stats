"""
Eduxellence Pricing & Currency Engine — Phase 5
=================================================
Defines credit costs per analysis type and converts USD prices to
local currency for display, using a live free exchange-rate API
with an in-memory cache and a hardcoded safety-net fallback rate
(so pricing NEVER breaks even if the rate API is unreachable).

by Eduxellence Analytics · https://eduxellence.org
"""

import json, ssl, urllib.request, urllib.error, time
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════
#  CREDIT PRICING TABLE  (1 credit = $1 USD)
# ══════════════════════════════════════════════════════════════════════════

CREDIT_COSTS = {
    # Free forever
    "cleaning":          0,
    "descriptive":       0,
    # Paid — inferential tests
    "chi_square":         1,
    "correlation":        1,
    "t_test":             1,
    "mann_whitney":       1,   # non-parametric t-test, same tier
    "anova":               2,
    "kruskal_wallis":       2,   # non-parametric anova, same tier
    "regression":           3,   # simple regression
    "multiple_regression":  4,   # regression with 2+ predictors (auto-detected)
    "factor_analysis":      5,   # reserved for future test
    "reliability":           3,   # Cronbach's alpha — reserved for future test
    # Bundles
    "thesis_package":       12,   # mid-point of $10-15 range
    "ai_interpretation":      1,
    "export_docx":             1,
    "export_pdf":              1,
    "export_xlsx":              0,   # excel stays free — drives adoption
}

# Human-readable labels for the pricing page
CREDIT_LABELS = {
    "cleaning":            "Data Cleaning",
    "descriptive":         "Descriptive Statistics",
    "chi_square":          "Chi-Square Test",
    "correlation":         "Correlation Analysis",
    "t_test":              "T-Test",
    "mann_whitney":        "Mann-Whitney U Test",
    "anova":               "ANOVA",
    "kruskal_wallis":      "Kruskal-Wallis Test",
    "regression":          "Linear Regression",
    "multiple_regression": "Multiple Regression",
    "factor_analysis":     "Factor Analysis",
    "reliability":         "Reliability Test (Cronbach's Alpha)",
    "thesis_package":      "Full Research / Thesis Package",
    "ai_interpretation":   "AI-Powered Interpretation",
    "export_docx":         "APA Word Export",
    "export_pdf":          "PDF Export",
    "export_xlsx":         "Excel Export",
}

# Free tier dataset row limit (Feature 7 in the freemium plan)
FREE_TIER_ROW_LIMIT = 500

# ══════════════════════════════════════════════════════════════════════════
#  SUBSCRIPTION PLANS
# ══════════════════════════════════════════════════════════════════════════

SUBSCRIPTION_PLANS = {
    "student": {
        "name": "Student Plan",
        "price_usd": 5,
        "analyses_per_month": 20,
        "features": ["AI interpretations", "All 8 statistical tests", "Excel/Word/PDF export"],
    },
    "researcher": {
        "name": "Researcher Plan",
        "price_usd": 15,
        "analyses_per_month": 100,
        "features": ["Everything in Student", "Advanced exports", "Priority support"],
    },
    "professional": {
        "name": "Professional Plan",
        "price_usd": 30,
        "analyses_per_month": -1,  # -1 = unlimited (fair use)
        "features": ["Everything in Researcher", "Unlimited analyses (fair use)",
                     "Priority processing", "API access"],
    },
}


# ══════════════════════════════════════════════════════════════════════════
#  CREDIT COST RESOLUTION
# ══════════════════════════════════════════════════════════════════════════

def get_credit_cost(analysis_type: str, params: Optional[dict] = None) -> int:
    """
    Resolve the credit cost for a given analysis. Regression is special-cased:
    1 predictor = 'regression' tier ($3), 2+ predictors = 'multiple_regression' ($4).
    """
    if analysis_type == "regression" and params:
        predictors = params.get("predictors", [])
        if isinstance(predictors, list) and len(predictors) > 1:
            return CREDIT_COSTS["multiple_regression"]
        return CREDIT_COSTS["regression"]
    return CREDIT_COSTS.get(analysis_type, 0)


def is_free_analysis(analysis_type: str) -> bool:
    return CREDIT_COSTS.get(analysis_type, 1) == 0


def get_full_pricing_table() -> list:
    """Returns the full pricing table for the public pricing page."""
    rows = []
    seen = set()
    order = ["cleaning", "descriptive", "chi_square", "correlation", "t_test",
             "mann_whitney", "anova", "kruskal_wallis", "regression",
             "multiple_regression", "factor_analysis", "reliability",
             "ai_interpretation", "export_docx", "export_pdf", "export_xlsx",
             "thesis_package"]
    for key in order:
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "key": key,
            "label": CREDIT_LABELS.get(key, key),
            "credits": CREDIT_COSTS[key],
            "usd": CREDIT_COSTS[key],
            "free": CREDIT_COSTS[key] == 0,
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════
#  CURRENCY CONVERSION — live rate + cache + hardcoded fallback
# ══════════════════════════════════════════════════════════════════════════

# Safety-net fallback rates (updated periodically; used if the live API
# is unreachable so pricing NEVER breaks for the user).
FALLBACK_RATES = {
    "NGN": 1530.0,
    "GBP": 0.79,
    "EUR": 0.92,
    "KES": 129.0,
    "GHS": 15.5,
    "ZAR": 18.2,
    "USD": 1.0,
}

SUPPORTED_CURRENCIES = list(FALLBACK_RATES.keys())

_RATE_CACHE = {"rates": None, "fetched_at": 0}
_CACHE_TTL_SECONDS = 6 * 60 * 60  # refresh at most every 6 hours


def _fetch_live_rates() -> Optional[dict]:
    """Fetch live USD-base exchange rates from the free exchangerate-api.com tier."""
    ctx = ssl.create_default_context()
    urls = [
        "https://open.er-api.com/v6/latest/USD",
        "https://api.exchangerate-api.com/v4/latest/USD",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 EduxellenceBot/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rates = data.get("rates")
                if rates:
                    return rates
        except Exception:
            continue
    return None


def get_exchange_rates(force_refresh: bool = False) -> dict:
    """
    Returns a dict of {currency: rate_from_usd}. Uses a 6-hour in-memory
    cache to stay within free API limits. Falls back to hardcoded rates
    if the live API is unreachable.
    """
    now = time.time()
    if (not force_refresh and _RATE_CACHE["rates"] is not None
            and (now - _RATE_CACHE["fetched_at"]) < _CACHE_TTL_SECONDS):
        return _RATE_CACHE["rates"]

    live = _fetch_live_rates()
    if live:
        merged = {cur: live.get(cur, FALLBACK_RATES[cur]) for cur in SUPPORTED_CURRENCIES}
        _RATE_CACHE["rates"] = merged
        _RATE_CACHE["fetched_at"] = now
        return merged

    # Live fetch failed — use fallback, but don't cache the failure
    # (so we retry live fetch sooner next time)
    return dict(FALLBACK_RATES)


def convert_usd(amount_usd: float, target_currency: str) -> dict:
    """Convert a USD amount to the target currency."""
    target_currency = target_currency.upper()
    rates = get_exchange_rates()
    rate = rates.get(target_currency, FALLBACK_RATES.get(target_currency, 1.0))
    converted = round(amount_usd * rate, 2 if target_currency != "NGN" else 0)
    is_live = _RATE_CACHE["rates"] is not None and (time.time() - _RATE_CACHE["fetched_at"]) < _CACHE_TTL_SECONDS
    return {
        "usd": amount_usd,
        "currency": target_currency,
        "rate": rate,
        "converted": converted,
        "source": "live" if is_live else "fallback",
    }


CURRENCY_SYMBOLS = {"USD": "$", "NGN": "₦", "GBP": "£", "EUR": "€",
                    "KES": "KSh", "GHS": "GH₵", "ZAR": "R"}

def format_price(amount_usd: float, currency: str) -> str:
    """Format a USD amount as a localised display string, e.g. '₦1,530 (~$1)'."""
    conv = convert_usd(amount_usd, currency)
    symbol = CURRENCY_SYMBOLS.get(currency.upper(), currency.upper() + " ")
    if currency.upper() == "USD":
        return f"${amount_usd:,.2f}".rstrip("0").rstrip(".") if amount_usd != int(amount_usd) else f"${int(amount_usd)}"
    local_str = f"{symbol}{conv['converted']:,.0f}" if currency.upper() == "NGN" else f"{symbol}{conv['converted']:,.2f}"
    usd_str = f"${int(amount_usd)}" if amount_usd == int(amount_usd) else f"${amount_usd:,.2f}"
    return f"{local_str} (~{usd_str})"


# ══════════════════════════════════════════════════════════════════════════
#  COUNTRY → CURRENCY/GATEWAY MAPPING
# ══════════════════════════════════════════════════════════════════════════

# ISO country code -> preferred currency + gateway
# Gateway routing follows the confirmed rule: Nigeria -> Paystack, else -> Lemon Squeezy
COUNTRY_CONFIG = {
    "NG": {"currency": "NGN", "gateway": "paystack"},
    "GH": {"currency": "GHS", "gateway": "lemonsqueezy"},
    "KE": {"currency": "KES", "gateway": "lemonsqueezy"},
    "ZA": {"currency": "ZAR", "gateway": "lemonsqueezy"},
    "GB": {"currency": "GBP", "gateway": "lemonsqueezy"},
    "US": {"currency": "USD", "gateway": "lemonsqueezy"},
}
DEFAULT_CONFIG = {"currency": "USD", "gateway": "lemonsqueezy"}


def resolve_country_config(country_code: Optional[str]) -> dict:
    """Given an ISO-2 country code (e.g. from CF-IPCountry or Accept-Language), return currency+gateway."""
    if not country_code:
        return dict(DEFAULT_CONFIG)
    return dict(COUNTRY_CONFIG.get(country_code.upper(), DEFAULT_CONFIG))


def gateway_for_country(country_code: Optional[str]) -> str:
    """if country == Nigeria → Paystack; else → Lemon Squeezy (per the agreed flow)."""
    if country_code and country_code.upper() == "NG":
        return "paystack"
    return "lemonsqueezy"
