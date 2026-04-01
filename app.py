"""
Millenniallity Volatility Mismatch Scanner — Streamlit MVP.
Air-gapped: no broker links, no position uploads.
API keys: load from environment — optional `.env` in this folder (see `.env.example`).
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional, Union

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:
    from dotenv import load_dotenv

    _ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_ENV_PATH)
except ImportError:
    pass

# Optional third-party (import guarded where used)
try:
    from sec_api import ExtractorApi
except ImportError:
    ExtractorApi = None  # type: ignore

# -----------------------------------------------------------------------------
# Branding & page
# -----------------------------------------------------------------------------
PAGE_TITLE = "Millenniallity Volatility Mismatch Scanner"
MILLENNIALITY_CSS = """
<style>
    /* Dark Millenniallity theme */
    .stApp { background: linear-gradient(165deg, #0d0618 0%, #12091f 45%, #0a1628 100%); }
    [data-testid="stSidebar"] { background: #100818 !important; border-right: 1px solid #2d1f4a; }
    h1, h2, h3 { color: #f4e9ff !important; font-family: 'Segoe UI', system-ui, sans-serif; letter-spacing: -0.02em; }
    .millenniallity-badge {
        display: inline-block;
        background: linear-gradient(90deg, #a855f7, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.1rem;
    }
    .stMetric { background: rgba(30, 20, 50, 0.6); border-radius: 12px; padding: 8px; border: 1px solid #3d2a5c; }
    div[data-testid="stExpander"] { background: rgba(20, 12, 35, 0.85); border: 1px solid #3d2a5c; border-radius: 14px; }
    footer { visibility: visible; }
    .mill-footer { text-align: center; color: #8b7aa8; font-size: 0.85rem; margin-top: 2rem; padding: 1rem; border-top: 1px solid #2d1f4a; }
</style>
"""

# Default universe (high-vol sectors you trade)
DEFAULT_TICKERS_BY_SECTOR: dict[str, list[str]] = {
    "Autos/EV": ["CVNA", "RIVN"],
    "AI/Tech": ["SMCI", "MU", "TTD"],
    "Consumer": ["UAA"],
    "Energy/Shipping": ["CNQ", "ZIM"],
}

DEFAULT_TICKERS: list[str] = sorted({t for ts in DEFAULT_TICKERS_BY_SECTOR.values() for t in ts})

# Rough sector P/E anchors when peers are thin or outlier-heavy (blend with peer median in logic).
SECTOR_PE_BENCHMARK: dict[str, float] = {
    "Autos/EV": 18.0,
    "AI/Tech": 28.0,
    "Consumer": 22.0,
    "Energy/Shipping": 14.0,
    "Custom": 20.0,
}

FMP_BASE = "https://financialmodelingprep.com"
SCAN_CACHE_TTL_SEC = 3600
HIGH_SCORE_EXPAND_THRESHOLD = 65

REQUEST_TIMEOUT = 25


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
API_KEY_WIDGETS: list[tuple[str, str, bool]] = [
    ("FMP_API_KEY", "FMP_API_KEY (Financial Modeling Prep)", False),
    ("SEC_API_KEY", "SEC_API_KEY (sec-api.io Extractor)", False),
    ("UNUSUAL_WHALES_API_KEY", "UNUSUAL_WHALES_API_KEY", True),
    ("XAI_API_KEY", "XAI_API_KEY (xAI Grok — 10-K narrative LLM)", True),
    ("POLYGON_API_KEY", "POLYGON_API_KEY (fallback data)", True),
]


def _ensure_api_keys_in_session() -> None:
    for key, _, _ in API_KEY_WIDGETS:
        if key not in st.session_state:
            st.session_state[key] = os.getenv(key, "")


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------
def _get_json(url: str) -> Any:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _safe_get_json(url: str) -> tuple[Any, Optional[str]]:
    try:
        data = _get_json(url)
        return data, None
    except Exception as e:
        return None, str(e)


# xAI (Grok) — OpenAI-compatible chat endpoint; no `openai` SDK required.
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
LLM_TIMEOUT = 120


def xai_chat_completion(api_key: str, user_content: str) -> str:
    model = os.getenv("XAI_MODEL", "grok-2-latest")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content[:28000]}],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    r = requests.post(
        XAI_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("xAI response missing choices")
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


# -----------------------------------------------------------------------------
# Data: yfinance
# -----------------------------------------------------------------------------
@dataclass
class VolatilitySnapshot:
    beta: Optional[float]
    hist_vol_annual_pct: Optional[float]
    last_close: Optional[float]
    currency: Optional[str]
    source_note: str = ""


def fetch_yf_volatility(symbol: str) -> VolatilitySnapshot:
    note = "yfinance"
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        beta = info.get("beta") or info.get("beta3Year")
        hist = t.history(period="3mo")
        hv = None
        if hist is not None and len(hist) > 10 and "Close" in hist.columns:
            rets = hist["Close"].pct_change().dropna()
            if len(rets) > 5:
                hv = float(rets.std() * (252**0.5) * 100)
        last = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
        cur = info.get("currency")
        return VolatilitySnapshot(
            beta=float(beta) if beta is not None else None,
            hist_vol_annual_pct=hv,
            last_close=last,
            currency=str(cur) if cur else None,
            source_note=note,
        )
    except Exception as e:
        return VolatilitySnapshot(
            beta=None, hist_vol_annual_pct=None, last_close=None, currency=None, source_note=f"yfinance error: {e}"
        )


def passes_volatility_filter(
    beta: Optional[float],
    hist_vol_pct: Optional[float],
    iv_rank_or_pct: Optional[float],
) -> tuple[bool, str]:
    """Must pass: beta > 1.5 OR IV > 40% OR historical ann. vol > 40%."""
    if beta is not None and beta > 1.5:
        return True, f"beta {beta:.2f} > 1.5"
    if iv_rank_or_pct is not None and iv_rank_or_pct > 40:
        return True, f"IV / IV proxy {iv_rank_or_pct:.1f}% > 40%"
    if hist_vol_pct is not None and hist_vol_pct > 40:
        return True, f"hist vol {hist_vol_pct:.1f}% ann. > 40%"
    return False, "failed volatility filter (need beta>1.5 or IV/hist>40%)"


# -----------------------------------------------------------------------------
# FMP: peers, ratios, IV placeholder, 10-K link
# -----------------------------------------------------------------------------
def _yf_last_price(sym: str) -> Optional[float]:
    try:
        h = yf.Ticker(sym).history(period="5d")
        if h is not None and not h.empty and "Close" in h.columns:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None


def fmp_analyst_forward_eps(sym: str, api_key: str) -> Optional[float]:
    """Next forward-looking consensus EPS (annual estimates), if available."""
    urls = [
        f"{FMP_BASE}/api/v3/analyst-estimates/{sym}?period=annual&apikey={api_key}",
        f"{FMP_BASE}/stable/analyst-estimates?symbol={sym}&period=annual&apikey={api_key}",
    ]
    today = date.today()
    for url in urls:
        data, _ = _safe_get_json(url)
        if not data:
            continue
        rows = data if isinstance(data, list) else data.get("data") or []
        if not rows:
            continue
        candidates: list[tuple[date, float]] = []
        for est in rows:
            if not isinstance(est, dict):
                continue
            d_raw = est.get("date") or est.get("filingDate") or est.get("year")
            eps = est.get("estimatedEpsAvg") or est.get("epsAvg") or est.get("estimatedEps")
            if eps is None:
                continue
            try:
                eps_f = float(eps)
            except (TypeError, ValueError):
                continue
            if eps_f <= 0:
                continue
            dt = None
            if isinstance(d_raw, str) and len(d_raw) >= 10:
                try:
                    dt = datetime.strptime(d_raw[:10], "%Y-%m-%d").date()
                except ValueError:
                    dt = None
            elif isinstance(d_raw, (int, float)):
                try:
                    dt = date(int(d_raw), 12, 31)
                except ValueError:
                    dt = None
            if dt is not None:
                candidates.append((dt, eps_f))
        future = [(d, e) for d, e in candidates if d >= today]
        pick = min(future, key=lambda x: x[0]) if future else None
        if pick:
            return pick[1]
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
    return None


def resolve_forward_pe(
    sym: str,
    api_key: str,
    price: Optional[float],
    *,
    allow_analyst_implied: bool = True,
) -> tuple[Optional[float], str]:
    """
    Prefer true forward P/E: ratios-ttm forwardPE → key-metrics-ttm → analyst-implied (price/next EPS).
    Falls back to trailing P/E fields only when needed (tagged in source string).
    """
    url = f"{FMP_BASE}/api/v3/ratios-ttm/{sym}?apikey={api_key}"
    data, _ = _safe_get_json(url)
    row: dict = data[0] if data and isinstance(data, list) and isinstance(data[0], dict) else {}

    v = row.get("forwardPE")
    if v is not None and isinstance(v, (int, float)) and v > 0:
        return float(v), "FMP ratios-ttm forwardPE"

    km_url = f"{FMP_BASE}/api/v3/key-metrics-ttm/{sym}?apikey={api_key}"
    km, _ = _safe_get_json(km_url)
    km_row: dict = km[0] if km and isinstance(km, list) and isinstance(km[0], dict) else {}
    for k in ("forwardPE", "peRatio", "priceEarningsRatio"):
        val = km_row.get(k)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            return float(val), f"FMP key-metrics-ttm {k}"

    # TTM / trailing from ratios (label clearly)
    for k in ("priceToEarningsRatioTTM", "peRatioTTM", "peRatio", "priceToEarningsRatio"):
        val = row.get(k)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            return float(val), f"FMP ratios-ttm {k} (TTM / trailing)"

    if allow_analyst_implied and price is not None and price > 0:
        eps = fmp_analyst_forward_eps(sym, api_key)
        if eps and eps > 0:
            return float(price / eps), "FMP analyst-estimates implied (price ÷ next EPS)"

    return None, "n/a"


def blended_peer_benchmark(median_peers: Optional[float], sector: str) -> tuple[float, str]:
    """Combine peer median with sector benchmark when peer coverage is weak."""
    bench = SECTOR_PE_BENCHMARK.get(sector) or SECTOR_PE_BENCHMARK["Custom"]
    if median_peers is not None and median_peers > 0:
        blended = 0.65 * median_peers + 0.35 * bench
        return blended, "0.65×peer_median + 0.35×sector_bench"
    return bench, "sector_bench_only"


def max_pe_deviation_vs_benchmarks(
    target_pe: Optional[float],
    peer_values: list[float],
    sector: str,
) -> tuple[Optional[float], bool, Optional[float], Optional[float]]:
    """
    Returns (max_deviation_pct, mismatch_ge_30, peer_median, blended_benchmark).
    """
    med = statistics.median(peer_values) if len(peer_values) >= 1 else None
    blended, _ = blended_peer_benchmark(med, sector)
    devs: list[float] = []
    if target_pe is not None and med is not None and med != 0:
        devs.append(abs(target_pe - med) / abs(med) * 100.0)
    if target_pe is not None and blended != 0:
        devs.append(abs(target_pe - blended) / abs(blended) * 100.0)
    bench_only = SECTOR_PE_BENCHMARK.get(sector) or SECTOR_PE_BENCHMARK["Custom"]
    if target_pe is not None and bench_only != 0:
        devs.append(abs(target_pe - bench_only) / abs(bench_only) * 100.0)
    if not devs:
        return None, False, med, blended
    mx = max(devs)
    return mx, mx >= 30.0, med, blended


def fmp_forward_pe_and_peer_avg(
    symbol: str,
    peers: list[str],
    api_key: str,
    sector: str,
    spot_price: Optional[float],
) -> tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    dict[str, Optional[float]],
    str,
    bool,
    Optional[float],
]:
    """
    Forward / best-available P/E for symbol + peers, robust peer median and blended benchmark.
    Returns: fwd_pe, peer_median, blended_benchmark, sector_benchmark, pe_map, fwd_pe_source, pe_mismatch_30
    """
    pe_map: dict[str, Optional[float]] = {}
    price_main = spot_price if spot_price and spot_price > 0 else _yf_last_price(symbol)
    main_pe, main_src = resolve_forward_pe(symbol, api_key, price_main, allow_analyst_implied=True)
    pe_map[symbol] = main_pe

    peer_vals: list[float] = []
    for p in peers:
        if p == symbol:
            continue
        px = _yf_last_price(p)
        ppe, _ = resolve_forward_pe(p, api_key, px, allow_analyst_implied=False)
        pe_map[p] = ppe
        if ppe is not None and ppe > 0:
            peer_vals.append(float(ppe))

    sector_bench = SECTOR_PE_BENCHMARK.get(sector) or SECTOR_PE_BENCHMARK["Custom"]
    med = statistics.median(peer_vals) if peer_vals else None
    blended, _ = blended_peer_benchmark(med, sector)
    dev_pct, mismatch, _, _ = max_pe_deviation_vs_benchmarks(main_pe, peer_vals, sector)

    return main_pe, med, blended, sector_bench, pe_map, main_src, bool(mismatch), dev_pct


def fmp_stock_peers(symbol: str, api_key: str) -> list[str]:
    """FMP stable stock-peers."""
    url = f"{FMP_BASE}/stable/stock-peers?symbol={symbol}&apikey={api_key}"
    data, err = _safe_get_json(url)
    if err or data is None:
        return []
    # stable may return list of symbols or wrapped object
    if isinstance(data, list) and data and isinstance(data[0], str):
        return [s.upper() for s in data if s][:15]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        row = data[0]
        peers = row.get("peers") or row.get("stockPeertickerList") or row.get("symbols")
        if isinstance(peers, list):
            return [str(p).upper() for p in peers if p][:15]
    if isinstance(data, dict):
        peers = data.get("peers") or data.get("stockPeertickerList")
        if isinstance(peers, list):
            return [str(p).upper() for p in peers if p][:15]
    return []


def fmp_try_iv_metric(symbol: str, api_key: str) -> tuple[Optional[float], str]:
    """
    Best-effort implied-vol / IV rank from FMP (endpoint availability varies by plan).
    Returns a percentage-like number for comparison to 40%, or None.
    """
    candidates = [
        f"{FMP_BASE}/api/v4/implied-volatility/{symbol}?apikey={api_key}",
        f"{FMP_BASE}/stable/implied-volatility?symbol={symbol}&apikey={api_key}",
    ]
    for url in candidates:
        data, err = _safe_get_json(url)
        if err or not data:
            continue
        # Heuristic extraction
        if isinstance(data, list) and data and isinstance(data[0], dict):
            row = data[0]
            for k in ("impliedVolatility", "iv", "avgIV", "oneMonthIV", "impliedMovement"):
                v = row.get(k)
                if isinstance(v, (int, float)):
                    return float(v), "FMP IV endpoint"
        if isinstance(data, dict):
            for k in ("impliedVolatility", "iv", "avgIV"):
                v = data.get(k)
                if isinstance(v, (int, float)):
                    return float(v), "FMP IV endpoint"
    return None, "FMP IV n/a"


def fmp_latest_10k_filing_url(symbol: str, api_key: str) -> Optional[str]:
    url = f"{FMP_BASE}/api/v3/sec_filings/{symbol}?type=10-K&page=0&apikey={api_key}"
    data, _ = _safe_get_json(url)
    if not data or not isinstance(data, list):
        # alternate path used in some FMP versions
        url2 = f"{FMP_BASE}/api/v3/sec-filings/{symbol}?type=10-k&apikey={api_key}"
        data, _ = _safe_get_json(url2)
    if not data or not isinstance(data, list):
        return None
    row = data[0]
    link = row.get("finalLink") or row.get("link") or row.get("linkToFilingDetails")
    return str(link) if link else None


def fmp_high_beta_universe(api_key: str, limit: int = 100) -> list[str]:
    """FMP stock screener — liquid, high-beta common stocks for a wider volatile sleeve."""
    q = (
        f"{FMP_BASE}/api/v3/stock-screener?"
        f"marketCapMoreThan=30000000&betaMoreThan=1.35&volumeMoreThan=400000"
        f"&isEtf=false&limit={limit}&apikey={api_key}"
    )
    data, err = _safe_get_json(q)
    if err or not isinstance(data, list):
        return []
    out: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        s = row.get("symbol")
        if s:
            out.append(str(s).upper())
    return out[:limit]


# -----------------------------------------------------------------------------
# Polygon (free tier) fallback price
# -----------------------------------------------------------------------------
def polygon_prev_close(symbol: str, api_key: str) -> tuple[Optional[float], str]:
    if not api_key:
        return None, "no Polygon key"
    sym = symbol.upper()
    url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev?adjusted=true&apikey={api_key}"
    data, err = _safe_get_json(url)
    if err or not data:
        return None, err or "polygon empty"
    try:
        r = data.get("results") or []
        if not r:
            return None, "polygon no results"
        c = r[0].get("c")
        return (float(c), "Polygon") if c is not None else (None, "polygon no close")
    except Exception as e:
        return None, str(e)


def parse_housing_tracker_csv(source: Union[str, bytes]) -> tuple[Optional[float], str]:
    """Load housing tracker; returns (approx MoM % change of inventory-like series, summary text)."""
    try:
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(io.BytesIO(source))
    except Exception as e:
        return None, f"Housing CSV error: {e}"

    inv_col = None
    for c in df.columns:
        if "invent" in str(c).lower():
            inv_col = c
            break
    if inv_col is None:
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        inv_col = num[0] if num else None
    if inv_col is None:
        return None, "Housing tracker: no usable numeric column."

    series = pd.to_numeric(df[inv_col], errors="coerce").dropna()
    if len(series) < 3:
        return None, "Housing tracker: need at least 3 data points."

    last = float(series.iloc[-1])
    prev = float(series.iloc[-2])
    mom = (last - prev) / abs(prev) if prev else 0.0
    summary = f"Series `{inv_col}` latest={last:.4g}, prior={prev:.4g}, MoM ~{100.0 * mom:+.1f}%"
    return mom, summary


def housing_overlay_note(ticker: str, sector: str, mom: Optional[float], base_summary: str) -> str:
    if mom is None:
        return base_summary
    t = ticker.upper()
    if t in ("CVNA", "RIVN") or sector == "Autos/EV":
        if mom > 0.04:
            return (
                base_summary
                + " **Macro overlay:** inventories trending up — stress-test used-auto / EV demand vs your P/E mismatch read."
            )
        if mom < -0.04:
            return (
                base_summary
                + " **Macro overlay:** inventories easing — can align with recovery-style bullish 10-K tone on names like CVNA/RIVN."
            )
    return base_summary + " (housing series loaded; no autos-specific flag)"


# -----------------------------------------------------------------------------
# SEC 10-K narrative + LLM
# -----------------------------------------------------------------------------
def extract_10k_sections(filing_url: str, sec_api_key: str) -> tuple[str, str, str]:
    if ExtractorApi is None:
        return "", "", "sec-api package not installed (`pip install sec-api`)"
    try:
        ex = ExtractorApi(sec_api_key)
        item1 = ex.get_section(filing_url, "1", "text") or ""
        item7 = ex.get_section(filing_url, "7", "text") or ""
        meta = "sec-api ExtractorApi"
        return item1[:12000], item7[:12000], meta
    except Exception as e:
        return "", "", f"ExtractorApi error: {e}"


def run_llm_valuation_mismatch(
    ticker: str,
    fwd_pe: Optional[float],
    peer_avg_pe: Optional[float],
    item1: str,
    item7: str,
    *,
    xai_key: str,
) -> tuple[str, str, str]:
    """
    Returns (full_text, directional_flag, provider).
    directional_flag in {Bullish, Bearish, Neutral, Unknown}
    """
    prompt = f"""Stock: {ticker}. Forward P/E (if known): {fwd_pe}. Blended peer/sector benchmark P/E (if known): {peer_avg_pe}.

Does management tone/catalysts contradict the current P/E valuation? Summarize in 100 words + bullish/bearish flag.

End with exactly these two lines:
Flag: Bullish OR Bearish OR Neutral
Contradiction: Yes OR No OR Unclear"""

    content = prompt + "\n\n--- Item 1 (Business) ---\n" + item1 + "\n\n--- Item 7 (MD&A) ---\n" + item7

    if not xai_key:
        return "LLM unavailable (set XAI_API_KEY in .env or sidebar)", "Unknown", "none"

    try:
        text = xai_chat_completion(xai_key, content)
        return text, _parse_llm_flag(text), "xAI"
    except Exception as e:
        return f"LLM error (xAI): {e}", "Unknown", "xAI-error"


def _parse_llm_flag(text: str) -> str:
    u = text.upper()
    if "FLAG: BEARISH" in u or re.search(r"\bBEARISH\b", u):
        return "Bearish"
    if "FLAG: BULLISH" in u or re.search(r"\bBULLISH\b", u):
        return "Bullish"
    if "FLAG: NEUTRAL" in u or re.search(r"\bNEUTRAL\b", u):
        return "Neutral"
    return "Unknown"


def pe_deviation_pct(target: Optional[float], peer_avg: Optional[float]) -> Optional[float]:
    if target is None or peer_avg is None or peer_avg == 0:
        return None
    return abs(target - peer_avg) / abs(peer_avg) * 100.0


# -----------------------------------------------------------------------------
# Scoring & recommendation
# -----------------------------------------------------------------------------
@dataclass
class ScanRow:
    ticker: str
    sector: str
    vol_ok: bool
    vol_reason: str
    beta: Optional[float]
    hist_vol_pct: Optional[float]
    iv_metric: Optional[float]
    fwd_pe: Optional[float]
    peer_avg_pe: Optional[float]
    pe_deviation_pct: Optional[float]
    pe_mismatch_30: bool
    peers_for_chart: list[str] = field(default_factory=list)
    pe_by_ticker: dict = field(default_factory=dict)
    llm_summary: str = ""
    llm_flag: str = "Unknown"
    llm_provider: str = ""
    narrative_contradiction: str = "Unclear"
    unusual_whales_note: str = ""
    peer_median_pe: Optional[float] = None
    sector_benchmark_pe: Optional[float] = None
    fwd_pe_source: str = ""
    insider_net_shares: float = 0.0
    insider_buy_shares: float = 0.0
    insider_sell_shares: float = 0.0
    insider_net_buyer: bool = False
    score_insider_bonus: float = 0.0
    housing_overlay: str = ""
    mismatch_score: float = 0.0
    recommendation: str = "Skip"
    option_suggestion: str = ""
    errors: list[str] = field(default_factory=list)


def score_row(
    vol_ok: bool,
    pe_dev: Optional[float],
    pe_mismatch: bool,
    llm_flag: str,
    narrative_text: str,
    *,
    insider_bonus: float = 0.0,
) -> float:
    if not vol_ok:
        return 0.0
    s = 15.0  # base for passing vol
    # P/E vs peers: up to 40
    if pe_mismatch and pe_dev is not None:
        s += min(40.0, 25.0 + max(0.0, pe_dev - 30.0) * 0.5)
    elif pe_dev is not None:
        s += min(20.0, pe_dev * 0.35)
    # Narrative: up to 35 if contradiction indicated
    u = narrative_text.upper()
    contradicts = "CONTRADICTION: YES" in u or "CONTRADICTION:YES" in u
    if contradicts:
        s += 35.0
    elif "CONTRADICTION: UNCLEAR" in u:
        s += 15.0
    else:
        s += 8.0
    # small bonus for known directional extreme
    if llm_flag in ("Bearish", "Bullish"):
        s += 5.0
    s += insider_bonus
    return max(0.0, min(100.0, s))


def recommend_trade(
    vol_ok: bool,
    score: float,
    llm_flag: str,
    pe_mismatch: bool,
    contradiction_yes: bool,
) -> str:
    if not vol_ok:
        return "Skip"
    if score < 40:
        return "Skip"
    if pe_mismatch and contradiction_yes:
        if llm_flag == "Bearish":
            return "Buy Puts"
        if llm_flag == "Bullish":
            return "Buy Calls (CVNA-style)"
    if score >= 65 and llm_flag == "Bullish" and pe_mismatch:
        return "Buy Calls (CVNA-style)"
    if score >= 65 and llm_flag == "Bearish" and pe_mismatch:
        return "Buy Puts"
    if score >= 55 and pe_mismatch:
        return "Buy Stock"
    return "Skip"


def suggest_options(symbol: str, last: Optional[float], *, weeks: tuple[int, ...] = (1, 2, 3, 4)) -> str:
    if last is None or last <= 0:
        return "No price — add price data; then use ~25–35 delta, monthly/weeklies per liquidity on Public.com."
    # Round strike to typical increments
    def strike(px: float) -> float:
        if px < 5:
            return round(px, 1)
        if px < 50:
            return round(px)
        if px < 200:
            return round(px / 2.5) * 2.5
        return round(px / 5) * 5

    atm = strike(last)
    otm_call = strike(last * 1.03)
    otm_put = strike(last * 0.97)
    today = date.today()
    exps = []
    for w in weeks:
        exps.append((today + timedelta(weeks=w)).strftime("%Y-%m-%d"))
    return (
        f"{symbol}: spot ~${last:.2f}. Suggested (1–4 week hold): "
        f"calls ${otm_call}C / puts ${otm_put}P; ATM ref ${atm}. "
        f"Example expirations ~ {', '.join(exps)} — verify chain & spreads on Public."
    )


def sector_for_ticker(ticker: str, extra_sector_map: dict[str, str]) -> str:
    for sec, tickers in DEFAULT_TICKERS_BY_SECTOR.items():
        if ticker in tickers:
            return sec
    return extra_sector_map.get(ticker, "Custom")


# -----------------------------------------------------------------------------
# Unusual Whales — insider flows (~400d window)
# -----------------------------------------------------------------------------
@dataclass
class InsiderFlowSummary:
    net_shares: float
    buy_shares: float
    sell_shares: float
    n_transactions: int
    summary_line: str
    net_buyer: bool


def _signed_insider_shares(row: dict) -> float:
    raw = row.get("transaction_shares") or row.get("shares") or row.get("share") or 0
    try:
        sh = float(raw)
    except (TypeError, ValueError):
        return 0.0
    t = str(
        row.get("transaction_type")
        or row.get("type")
        or row.get("acquisition_or_disposition")
        or row.get("acquisitionOrDisposition")
        or ""
    ).lower()
    if any(x in t for x in ("sale", "sell", "disposition", "dispose", "d ")) and "purchase" not in t:
        return -abs(sh)
    if any(x in t for x in ("purchase", "buy", "acquisition", "gift", "award", "grant", "a ")):
        return abs(sh)
    code = str(row.get("transaction_code") or row.get("code") or "").upper()
    if code == "S":
        return -abs(sh)
    if code in ("P", "A", "M", "G", "F"):
        return abs(sh)
    return 0.0


def _net_insider_amount(row: dict) -> float:
    """Prefer API `amount` (often signed); fallback to heuristics on share fields + transaction_code."""
    amt = row.get("amount")
    if amt is not None:
        try:
            return float(amt)
        except (TypeError, ValueError):
            pass
    return _signed_insider_shares(row)


def fetch_unusual_insider_4q(ticker: str, uw_key: str) -> InsiderFlowSummary:
    if not uw_key:
        return InsiderFlowSummary(0.0, 0.0, 0.0, 0, "Unusual Whales: no API key.", False)
    start = (datetime.now() - timedelta(days=int(365 * 1.1))).strftime("%Y-%m-%d")
    url = "https://api.unusualwhales.com/api/insider/transactions"
    params = {"ticker_symbol": ticker.upper(), "start_date": start, "limit": 500}
    headers = {"Authorization": f"Bearer {uw_key}", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 401:
            return InsiderFlowSummary(0.0, 0.0, 0.0, 0, "Unusual Whales: 401 unauthorized (check key).", False)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        return InsiderFlowSummary(0.0, 0.0, 0.0, 0, f"Unusual Whales request error: {e}", False)

    rows: list = []
    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict):
        rows = body.get("data") or body.get("transactions") or body.get("results") or []

    buy = sell = 0.0
    net = 0.0
    symu = ticker.upper()
    for item in rows:
        if not isinstance(item, dict):
            continue
        it = str(item.get("ticker") or "").upper()
        if it and it != symu:
            continue
        signed = _net_insider_amount(item)
        net += signed
        if signed >= 0:
            buy += abs(signed)
        else:
            sell += abs(signed)

    n = len(rows)
    nb = net > 0
    line = (
        f"Insider ~400d: net **{net:,.0f}** (API amount units / signed flow; buys {buy:,.0f} / sells {sell:,.0f}), n={n} — "
        f"{'net buyer' if nb else 'net seller / flat'}."
    )
    return InsiderFlowSummary(net, buy, sell, n, line, nb)


# -----------------------------------------------------------------------------
# Scan pipeline
# -----------------------------------------------------------------------------
def _scan_ticker_impl(
    ticker: str,
    sector: str,
    *,
    fmp_key: str,
    sec_key: str,
    xai_key: str,
    polygon_key: str,
    uw_key: str,
    housing_mom: Optional[float],
    housing_tracker_summary: str,
) -> ScanRow:
    errors: list[str] = []
    t = ticker.upper().strip()
    row = ScanRow(
        ticker=t,
        sector=sector,
        vol_ok=False,
        vol_reason="",
        beta=None,
        hist_vol_pct=None,
        iv_metric=None,
        fwd_pe=None,
        peer_avg_pe=None,
        pe_deviation_pct=None,
        pe_mismatch_30=False,
    )

    yv = fetch_yf_volatility(t)
    row.beta = yv.beta
    row.hist_vol_pct = yv.hist_vol_annual_pct
    last = yv.last_close

    iv_guess, iv_note = fmp_try_iv_metric(t, fmp_key)
    row.iv_metric = iv_guess
    if iv_guess is None and polygon_key:
        pc, _ = polygon_prev_close(t, polygon_key)
        if pc and last is None:
            last = pc

    vol_ok, reason = passes_volatility_filter(row.beta, row.hist_vol_pct, row.iv_metric)
    row.vol_ok = vol_ok
    row.vol_reason = reason + (f"; {iv_note}" if iv_note else "")

    peers = fmp_stock_peers(t, fmp_key) if fmp_key else []
    peers = [p for p in peers if p != t][:12]
    row.peers_for_chart = [t] + peers[:8]
    if not peers:
        errors.append("No peer list from FMP (check key or symbol).")

    if fmp_key:
        (
            fwd,
            peer_med,
            blended,
            sec_bench,
            pe_map,
            main_src,
            mismatch,
            dev_pct,
        ) = fmp_forward_pe_and_peer_avg(t, peers or [t], fmp_key, sector, last)
        row.fwd_pe = fwd
        row.peer_median_pe = peer_med
        row.peer_avg_pe = blended
        row.sector_benchmark_pe = sec_bench
        row.pe_by_ticker = pe_map
        row.fwd_pe_source = main_src
        row.pe_deviation_pct = dev_pct
        row.pe_mismatch_30 = mismatch
    else:
        row.fwd_pe_source = "n/a (no FMP key)"

    peer_for_llm = row.peer_avg_pe

    filing_url = fmp_latest_10k_filing_url(t, fmp_key) if fmp_key else None
    item1, item7, sec_meta = ("", "", "")
    if not filing_url:
        errors.append("Could not resolve 10-K filing URL via FMP.")
        row.llm_summary = "No 10-K URL — skipping narrative (FMP sec_filings)."
        row.llm_flag = "Unknown"
        row.llm_provider = "n/a"
    else:
        item1, item7, sec_meta = extract_10k_sections(filing_url, sec_key)
        if not item1 and not item7:
            errors.append(sec_meta)
            row.llm_summary = sec_meta
        elif not xai_key:
            row.llm_summary = "10-K text retrieved — add XAI_API_KEY in .env or sidebar for LLM summary."
            row.llm_provider = "none"
        else:
            text, flag, prov = run_llm_valuation_mismatch(
                t, row.fwd_pe, peer_for_llm, item1, item7, xai_key=xai_key
            )
            row.llm_summary = text
            row.llm_flag = flag
            row.llm_provider = prov

    u = row.llm_summary.upper()
    if "CONTRADICTION: YES" in u:
        row.narrative_contradiction = "Yes"
    elif "CONTRADICTION: NO" in u:
        row.narrative_contradiction = "No"
    else:
        row.narrative_contradiction = "Unclear"

    ins = fetch_unusual_insider_4q(t, uw_key)
    row.unusual_whales_note = ins.summary_line
    row.insider_net_shares = ins.net_shares
    row.insider_buy_shares = ins.buy_shares
    row.insider_sell_shares = ins.sell_shares
    row.insider_net_buyer = ins.net_buyer

    base_housing = housing_tracker_summary or "No housing tracker loaded."
    row.housing_overlay = housing_overlay_note(t, sector, housing_mom, base_housing)

    insider_bonus = 0.0
    if ins.net_buyer and row.pe_mismatch_30:
        insider_bonus = 10.0
    row.score_insider_bonus = insider_bonus

    row.mismatch_score = score_row(
        row.vol_ok,
        row.pe_deviation_pct,
        row.pe_mismatch_30,
        row.llm_flag,
        row.llm_summary,
        insider_bonus=insider_bonus,
    )
    row.recommendation = recommend_trade(
        row.vol_ok,
        row.mismatch_score,
        row.llm_flag,
        row.pe_mismatch_30,
        row.narrative_contradiction == "Yes",
    )
    row.option_suggestion = suggest_options(t, last or yv.last_close)
    row.errors = errors
    return row


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner=False)
def scan_ticker_cached(
    ticker: str,
    sector: str,
    fmp_key: str,
    sec_key: str,
    xai_key: str,
    polygon_key: str,
    uw_key: str,
    housing_mom: Optional[float],
    housing_digest: str,
    housing_tracker_summary: str,
) -> ScanRow:
    """Cached per-parameter set — ttl 3600s; `housing_digest` busts cache when CSV/URL changes."""
    return _scan_ticker_impl(
        ticker,
        sector,
        fmp_key=fmp_key,
        sec_key=sec_key,
        xai_key=xai_key,
        polygon_key=polygon_key,
        uw_key=uw_key,
        housing_mom=housing_mom,
        housing_tracker_summary=housing_tracker_summary,
    )


# -----------------------------------------------------------------------------
# Plotly: P/E vs peers
# -----------------------------------------------------------------------------
def pe_bar_figure(row: ScanRow) -> Optional[go.Figure]:
    labels: list[str] = []
    values: list[float] = []
    for sym in row.peers_for_chart:
        pe = row.pe_by_ticker.get(sym)
        if pe is not None and pe > 0:
            labels.append(sym)
            values.append(float(pe))
    if not labels:
        return None
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=["#a855f7" if lab == row.ticker else "#06b6d4" for lab in labels],
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,12,35,0.6)",
        font_color="#e9d8ff",
        title="P/E vs peers (forward / best-effort via FMP)",
        margin=dict(l=8, r=8, t=40, b=8),
        yaxis_title="Forward P/E",
        height=320,
    )
    return fig


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main() -> None:
    _ensure_api_keys_in_session()
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="📊")
    st.markdown(MILLENNIALITY_CSS, unsafe_allow_html=True)
    st.markdown('<span class="millenniallity-badge">@millenniallity</span>', unsafe_allow_html=True)
    st.title("Millenniallity Volatility Mismatch Scanner")
    st.markdown(
        "Highly volatile names where **forward P/E disagrees with earnings reality** — air-gapped opportunity generator for Public.com."
    )

    with st.sidebar:
        st.markdown("### API keys (session only)")
        st.caption(
            "Stored in `st.session_state` for this session — **prefilled from** shell env or **`.env`** in the app folder (see `.env.example`)."
        )
        _ensure_api_keys_in_session()
        for key, label, optional in API_KEY_WIDGETS:
            opt = " — optional" if optional else ""
            st.text_input(f"{label}{opt}", type="password", key=key)
        st.divider()
        st.markdown("### Filters (apply after scan)")
        sector_options = list(DEFAULT_TICKERS_BY_SECTOR.keys()) + ["Custom"]
        selected_sectors = st.multiselect("Sectors", sector_options, default=sector_options)
        min_score = st.slider("Minimum mismatch score", 0, 100, 35)

    # Main inputs
    universe_mode = st.radio(
        "Universe",
        ("My sectors + adds", "Top 100 volatile (FMP screener) + adds"),
        horizontal=True,
        help="Wider screen pulls high-beta liquid names from FMP (merges your comma-separated adds).",
    )
    col_a, col_b = st.columns([2, 1])
    with col_a:
        user_add = st.text_input("Add tickers (comma-separated)", "")
    with col_b:
        st.caption("Defaults = your high-vol sectors when using “My sectors”.")

    st.markdown("**Optional — housing tracker (CSV upload or URL)** for CVNA/RIVN / Autos macro overlay.")
    hu_col, hurl_col = st.columns(2)
    with hu_col:
        housing_upload = st.file_uploader("Housing CSV", type=["csv"], key="housing_csv_upload")
    with hurl_col:
        housing_url = st.text_input("Or CSV URL (https://…)", "", key="housing_csv_url")

    extra = [x.strip().upper() for x in user_add.replace(" ", "").split(",") if x.strip()]
    extra_map = {x: "Custom" for x in extra}

    run = st.button("Run Full Market Scanner", type="primary", use_container_width=True)

    st.info(
        f"**Daily refresh:** re-run before the open. Last app date: **{date.today().isoformat()}** "
        "(data freshness depends on vendors; not real-time advice). "
        f"Ticker-level scan cache TTL: **{SCAN_CACHE_TTL_SEC // 60} min** (same inputs/API keys)."
    )

    if not run:
        st.stop()

    fmp_key = st.session_state.get("FMP_API_KEY", "")
    sec_key = st.session_state.get("SEC_API_KEY", "")
    xai_key = st.session_state.get("XAI_API_KEY", "")
    uw_key = st.session_state.get("UNUSUAL_WHALES_API_KEY", "")
    polygon_key = st.session_state.get("POLYGON_API_KEY", "")

    if not fmp_key:
        st.error("FMP_API_KEY is required for peers, ratios, and filing links.")
        st.stop()
    if not sec_key:
        st.warning("SEC_API_KEY missing — 10-K extraction will fail until provided.")
    if not xai_key:
        st.warning("Add XAI_API_KEY (`.env` or sidebar) for 10-K narrative scoring.")

    housing_mom: Optional[float] = None
    housing_digest = ""
    housing_tracker_summary = ""
    if housing_upload is not None:
        raw = housing_upload.getvalue()
        housing_digest = hashlib.md5(raw).hexdigest()[:16]
        housing_mom, housing_tracker_summary = parse_housing_tracker_csv(raw)
    elif housing_url.strip():
        housing_digest = hashlib.md5(housing_url.strip().encode()).hexdigest()[:16]
        housing_mom, housing_tracker_summary = parse_housing_tracker_csv(housing_url.strip())

    if universe_mode.startswith("Top 100"):
        wide = fmp_high_beta_universe(fmp_key, 100)
        if not wide:
            st.warning("FMP screener returned no symbols — falling back to default watchlist.")
            wide = list(DEFAULT_TICKERS)
        universe = sorted(set(wide + extra))
    else:
        universe = sorted(set(DEFAULT_TICKERS + extra))

    st.session_state["__last_universe"] = universe

    progress = st.progress(0.0, text="Starting scan…")
    results: list[ScanRow] = []

    for i, tick in enumerate(universe):
        progress.progress((i + 1) / max(1, len(universe)), text=f"Scanning {tick}…")
        sec = sector_for_ticker(tick, extra_map)
        try:
            results.append(
                scan_ticker_cached(
                    tick,
                    sec,
                    fmp_key,
                    sec_key,
                    xai_key,
                    polygon_key,
                    uw_key,
                    housing_mom,
                    housing_digest,
                    housing_tracker_summary,
                )
            )
        except Exception as e:
            results.append(
                ScanRow(
                    ticker=tick.upper(),
                    sector=sector_for_ticker(tick, extra_map),
                    vol_ok=False,
                    vol_reason="error",
                    beta=None,
                    hist_vol_pct=None,
                    iv_metric=None,
                    fwd_pe=None,
                    peer_avg_pe=None,
                    pe_deviation_pct=None,
                    pe_mismatch_30=False,
                    llm_summary=f"Fatal scan error: {e}",
                    mismatch_score=0.0,
                    recommendation="Skip",
                    errors=[str(e)],
                )
            )

    progress.empty()

    # Filter by sector + score
    def sector_ok(r: ScanRow) -> bool:
        if r.sector == "Custom":
            return "Custom" in selected_sectors
        return r.sector in selected_sectors

    filtered = [r for r in results if sector_ok(r) and r.mismatch_score >= min_score]
    filtered.sort(key=lambda r: r.mismatch_score, reverse=True)

    summary_records = []
    for r in results:
        summary_records.append(
            {
                "Ticker": r.ticker,
                "Sector": r.sector,
                "MismatchScore": round(r.mismatch_score, 1),
                "Recommendation": r.recommendation,
                "VolOK": r.vol_ok,
                "VolDetail": r.vol_reason,
                "Beta": r.beta,
                "HistVolPct": r.hist_vol_pct,
                "IVMetric": r.iv_metric,
                "FwdPE": r.fwd_pe,
                "FwdPESource": r.fwd_pe_source,
                "PeerMedianPE": r.peer_median_pe,
                "BlendedBenchmarkPE": r.peer_avg_pe,
                "SectorBenchPE": r.sector_benchmark_pe,
                "PEDeviationPct": r.pe_deviation_pct,
                "PEMismatch30Pct": r.pe_mismatch_30,
                "InsiderNetSh": r.insider_net_shares,
                "InsiderBonus": r.score_insider_bonus,
                "LLMFlag": r.llm_flag,
                "NarrativeContradiction": r.narrative_contradiction,
                "LLMProvider": r.llm_provider,
                "UnusualWhales": r.unusual_whales_note,
                "HousingOverlay": r.housing_overlay,
            }
        )
    sum_df = pd.DataFrame(summary_records).sort_values("MismatchScore", ascending=False)

    st.subheader("Summary table")
    st.dataframe(sum_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Export CSV (filtered top opportunities)",
        data=filtered_to_csv(filtered),
        file_name=f"millenniallity_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    st.subheader("Opportunity cards")
    if not filtered:
        st.warning("No rows match sector + min score — lower the slider or widen sectors.")
    for r in filtered:
        title = f"{r.ticker}  ·  Score {r.mismatch_score:.0f}  ·  {r.recommendation}  ·  {r.sector}"
        auto_expand = r.mismatch_score >= HIGH_SCORE_EXPAND_THRESHOLD or (
            r.score_insider_bonus > 0 and r.pe_mismatch_30
        )
        with st.expander(title, expanded=auto_expand):
            c1, c2 = st.columns([1, 1])
            with c1:
                fig = pe_bar_figure(r)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Insufficient P/E peer data for chart.")
                st.metric("P/E vs blended benchmark", f"{r.fwd_pe} vs {r.peer_avg_pe}")
                st.caption(
                    f"Peer median: **{r.peer_median_pe}** | Sector bench: **{r.sector_benchmark_pe}** | "
                    f"Source: _{r.fwd_pe_source}_"
                )
                st.caption(f"Volatility: {r.vol_reason}")
            with c2:
                st.markdown("**LLM (10-K vs valuation)**")
                st.write(r.llm_summary or "—")
                st.caption(f"Provider: {r.llm_provider} | Flag: **{r.llm_flag}** | Contradiction: **{r.narrative_contradiction}**")
                st.markdown(f"**Trade rec:** `{r.recommendation}`")
                if r.score_insider_bonus:
                    st.success(f"Insider alignment bonus applied: **+{r.score_insider_bonus:.0f}** (net buying + P/E mismatch).")
                st.markdown("**Options (heuristic)**")
                st.write(r.option_suggestion)
                st.markdown("**Unusual Whales (insider ~400d)**")
                st.markdown(r.unusual_whales_note or "—")
                st.caption(
                    f"Net flow **{r.insider_net_shares:,.0f}** · aggressive buys {r.insider_buy_shares:,.0f} · sells {r.insider_sell_shares:,.0f} "
                    "(UW `amount` signed sum; verify vs filings if sizing trades)."
                )
                st.markdown("**Housing / macro**")
                st.caption(r.housing_overlay or "—")
            if r.errors:
                st.error("Warnings: " + "; ".join(r.errors))

    st.markdown(
        '<div class="mill-footer">Built for Public.com options edge — volatile P/E mismatches only</div>',
        unsafe_allow_html=True,
    )


def filtered_to_csv(rows: list[ScanRow]) -> str:
    buf = io.StringIO()
    out = []
    for r in rows:
        out.append(
            {
                "Ticker": r.ticker,
                "Sector": r.sector,
                "MismatchScore": round(r.mismatch_score, 1),
                "Recommendation": r.recommendation,
                "FwdPE": r.fwd_pe,
                "FwdPESource": r.fwd_pe_source,
                "BlendedBenchmarkPE": r.peer_avg_pe,
                "InsiderNetSh": r.insider_net_shares,
                "InsiderBonus": r.score_insider_bonus,
                "LLMFlag": r.llm_flag,
                "HousingOverlay": r.housing_overlay,
                "OptionIdea": r.option_suggestion,
            }
        )
    pd.DataFrame(out).to_csv(buf, index=False)
    return buf.getvalue()


if __name__ == "__main__":
    main()
