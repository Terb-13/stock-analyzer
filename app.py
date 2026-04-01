"""
Millenniallity Volatility Mismatch Scanner — auto-discovery for volatile P/E mismatches.
Air-gapped: no broker links, no position uploads.
API keys: `.env` / environment variables only (never shown in UI).
"""
from __future__ import annotations

import io
import os
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

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

try:
    from sec_api import ExtractorApi
except ImportError:
    ExtractorApi = None  # type: ignore

# -----------------------------------------------------------------------------
# Branding & config
# -----------------------------------------------------------------------------
PAGE_TITLE = "Millenniallity Volatility Mismatch Scanner"
MILLENNIALITY_CSS = """
<style>
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

DEFAULT_TICKERS_BY_SECTOR: dict[str, list[str]] = {
    "Autos/EV": ["CVNA", "RIVN"],
    "AI/Tech": ["SMCI", "MU", "TTD"],
    "Consumer": ["UAA"],
    "Energy/Shipping": ["CNQ", "ZIM"],
}

DEFAULT_TICKERS: list[str] = sorted({t for ts in DEFAULT_TICKERS_BY_SECTOR.values() for t in ts})

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
MIN_DISPLAY_SCORE = 45

REQUEST_TIMEOUT = 25


def _env_keys() -> dict[str, str]:
    return {
        "fmp": os.getenv("FMP_API_KEY", "").strip(),
        "sec": os.getenv("SEC_API_KEY", "").strip(),
        "xai": os.getenv("XAI_API_KEY", "").strip(),
        "uw": os.getenv("UNUSUAL_WHALES_API_KEY", "").strip(),
        "polygon": os.getenv("POLYGON_API_KEY", "").strip(),
    }


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
# yfinance
# -----------------------------------------------------------------------------
@dataclass
class VolatilitySnapshot:
    beta: Optional[float]
    hist_vol_annual_pct: Optional[float]
    last_close: Optional[float]
    currency: Optional[str]
    source_note: str = ""


def fetch_yf_volatility(symbol: str) -> VolatilitySnapshot:
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
            source_note="yfinance",
        )
    except Exception as e:
        return VolatilitySnapshot(
            beta=None, hist_vol_annual_pct=None, last_close=None, currency=None, source_note=f"yfinance error: {e}"
        )


def passes_volatility_filter(beta: Optional[float], hist_vol_pct: Optional[float]) -> tuple[bool, str]:
    """Must pass: beta > 1.5 OR annualized hist vol > 40%."""
    if beta is not None and beta > 1.5:
        return True, f"beta {beta:.2f} > 1.5"
    if hist_vol_pct is not None and hist_vol_pct > 40:
        return True, f"hist vol {hist_vol_pct:.1f}% ann. > 40%"
    return False, "failed vol filter (need beta>1.5 or hist vol>40% ann.)"


# -----------------------------------------------------------------------------
# FMP
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
    url = f"{FMP_BASE}/stable/stock-peers?symbol={symbol}&apikey={api_key}"
    data, err = _safe_get_json(url)
    if err or data is None:
        return []
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
    candidates = [
        f"{FMP_BASE}/api/v4/implied-volatility/{symbol}?apikey={api_key}",
        f"{FMP_BASE}/stable/implied-volatility?symbol={symbol}&apikey={api_key}",
    ]
    for url in candidates:
        data, err = _safe_get_json(url)
        if err or not data:
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict):
            row = data[0]
            for k in ("impliedVolatility", "iv", "avgIV", "oneMonthIV", "impliedMovement"):
                v = row.get(k)
                if isinstance(v, (int, float)):
                    return float(v), "FMP IV"
        if isinstance(data, dict):
            for k in ("impliedVolatility", "iv", "avgIV"):
                v = data.get(k)
                if isinstance(v, (int, float)):
                    return float(v), "FMP IV"
    return None, ""


def fmp_latest_10k_filing_url(symbol: str, api_key: str) -> Optional[str]:
    url = f"{FMP_BASE}/api/v3/sec_filings/{symbol}?type=10-K&page=0&apikey={api_key}"
    data, _ = _safe_get_json(url)
    if not data or not isinstance(data, list):
        url2 = f"{FMP_BASE}/api/v3/sec-filings/{symbol}?type=10-k&apikey={api_key}"
        data, _ = _safe_get_json(url2)
    if not data or not isinstance(data, list):
        return None
    row = data[0]
    link = row.get("finalLink") or row.get("link") or row.get("linkToFilingDetails")
    return str(link) if link else None


def fmp_high_beta_universe(api_key: str, limit: int = 100) -> list[str]:
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
# Polygon
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


# -----------------------------------------------------------------------------
# SEC + xAI
# -----------------------------------------------------------------------------
def extract_10k_sections(filing_url: str, sec_api_key: str) -> tuple[str, str, str]:
    if ExtractorApi is None:
        return "", "", "sec-api package not installed (`pip install sec-api`)"
    try:
        ex = ExtractorApi(sec_api_key)
        item1 = ex.get_section(filing_url, "1", "text") or ""
        item7 = ex.get_section(filing_url, "7", "text") or ""
        return item1[:12000], item7[:12000], "sec-api ExtractorApi"
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
    prompt = f"""Stock: {ticker}. Forward P/E (if known): {fwd_pe}. Blended peer/sector benchmark P/E (if known): {peer_avg_pe}.

Does management tone/catalysts contradict the current P/E valuation? Summarize in 100 words + bullish/bearish flag.

End with exactly these two lines:
Flag: Bullish OR Bearish OR Neutral
Contradiction: Yes OR No OR Unclear"""

    content = prompt + "\n\n--- Item 1 (Business) ---\n" + item1 + "\n\n--- Item 7 (MD&A) ---\n" + item7

    if not xai_key:
        return "LLM unavailable — set XAI_API_KEY in `.env`", "Unknown", "none"

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


# -----------------------------------------------------------------------------
# Scoring & recommendations
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
    s = 15.0
    if pe_mismatch and pe_dev is not None:
        s += min(40.0, 25.0 + max(0.0, pe_dev - 30.0) * 0.5)
    elif pe_dev is not None:
        s += min(20.0, pe_dev * 0.35)
    u = narrative_text.upper()
    contradicts = "CONTRADICTION: YES" in u or "CONTRADICTION:YES" in u
    if contradicts:
        s += 35.0
    elif "CONTRADICTION: UNCLEAR" in u:
        s += 15.0
    else:
        s += 8.0
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
        return "No spot — check data; on Public.com use ~25–35 delta, 1–4 week expiries per liquidity."
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
    exps = [(today + timedelta(weeks=w)).strftime("%Y-%m-%d") for w in weeks]
    return (
        f"{symbol}: ~${last:.2f}. 1–4 week hold (Public.com): ~${otm_call}C / ~${otm_put}P; ATM ~${atm}. "
        f"Sample expiries {', '.join(exps)} — verify chain & spreads."
    )


def sector_for_ticker(ticker: str, extra_sector_map: dict[str, str]) -> str:
    for sec, tickers in DEFAULT_TICKERS_BY_SECTOR.items():
        if ticker in tickers:
            return sec
    return extra_sector_map.get(ticker, "Custom")


# -----------------------------------------------------------------------------
# Unusual Whales
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
    amt = row.get("amount")
    if amt is not None:
        try:
            return float(amt)
        except (TypeError, ValueError):
            pass
    return _signed_insider_shares(row)


def fetch_unusual_insider_4q(ticker: str, uw_key: str) -> InsiderFlowSummary:
    if not uw_key:
        return InsiderFlowSummary(0.0, 0.0, 0.0, 0, "Unusual Whales: no API key in environment.", False)
    start = (datetime.now() - timedelta(days=int(365 * 1.1))).strftime("%Y-%m-%d")
    url = "https://api.unusualwhales.com/api/insider/transactions"
    params = {"ticker_symbol": ticker.upper(), "start_date": start, "limit": 500}
    headers = {"Authorization": f"Bearer {uw_key}", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 401:
            return InsiderFlowSummary(0.0, 0.0, 0.0, 0, "Unusual Whales: 401 unauthorized.", False)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        return InsiderFlowSummary(0.0, 0.0, 0.0, 0, f"Unusual Whales error: {e}", False)

    rows: list = body if isinstance(body, list) else (body.get("data") or body.get("transactions") or [])
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
        f"~400d insider: net **{net:,.0f}** (buys {buy:,.0f} / sells {sell:,.0f}), n={n} — "
        f"**{'net buyer' if nb else 'net seller / flat'}**."
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

    iv_guess, _iv_note = fmp_try_iv_metric(t, fmp_key)
    row.iv_metric = iv_guess
    if iv_guess is None and polygon_key:
        pc, _ = polygon_prev_close(t, polygon_key)
        if pc and last is None:
            last = pc

    vol_ok, reason = passes_volatility_filter(row.beta, row.hist_vol_pct)
    row.vol_ok = vol_ok
    extra_iv = f" | IV/read-through: {iv_guess:.1f}" if iv_guess is not None else ""
    row.vol_reason = reason + extra_iv

    peers = fmp_stock_peers(t, fmp_key) if fmp_key else []
    peers = [p for p in peers if p != t][:12]
    row.peers_for_chart = [t] + peers[:8]
    if not peers:
        errors.append("No peer list from FMP.")

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
        row.fwd_pe_source = "n/a"

    peer_for_llm = row.peer_avg_pe
    filing_url = fmp_latest_10k_filing_url(t, fmp_key) if fmp_key else None

    if not filing_url:
        errors.append("No 10-K filing URL from FMP.")
        row.llm_summary = "No 10-K URL."
        row.llm_flag = "Unknown"
        row.llm_provider = "n/a"
    else:
        item1, item7, sec_meta = extract_10k_sections(filing_url, sec_key)
        if not item1 and not item7:
            errors.append(sec_meta)
            row.llm_summary = sec_meta
        elif not xai_key:
            row.llm_summary = "10-K retrieved — add XAI_API_KEY to `.env` for narrative."
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

    insider_bonus = 10.0 if ins.net_buyer and row.pe_mismatch_30 else 0.0
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
) -> ScanRow:
    return _scan_ticker_impl(
        ticker,
        sector,
        fmp_key=fmp_key,
        sec_key=sec_key,
        xai_key=xai_key,
        polygon_key=polygon_key,
        uw_key=uw_key,
    )


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
        title="P/E vs peers (forward / best-effort)",
        margin=dict(l=8, r=8, t=40, b=8),
        yaxis_title="P/E",
        height=300,
    )
    return fig


def filtered_to_csv(rows: list[ScanRow]) -> str:
    buf = io.StringIO()
    out = []
    for r in rows:
        out.append(
            {
                "Ticker": r.ticker,
                "MismatchScore": round(r.mismatch_score, 1),
                "Recommendation": r.recommendation,
                "FwdPE": r.fwd_pe,
                "VolReason": r.vol_reason,
                "LLMFlag": r.llm_flag,
                "InsiderBonus": r.score_insider_bonus,
                "OptionIdea": r.option_suggestion,
            }
        )
    pd.DataFrame(out).to_csv(buf, index=False)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="📊")
    st.markdown(MILLENNIALITY_CSS, unsafe_allow_html=True)

    keys = _env_keys()
    if not keys["fmp"]:
        st.error("**FMP_API_KEY** is missing. Add it to `.env` or your environment and restart.")
        st.stop()

    with st.sidebar:
        st.caption("API keys loaded from **`.env`** / environment (**FMP** required). Nothing is entered here.")
        st.divider()
        st.caption(
            f"Scan cache: **{SCAN_CACHE_TTL_SEC // 60} min** TTL · "
            f"Display cutoff: score ≥ **{MIN_DISPLAY_SCORE}**"
        )

    st.markdown('<span class="millenniallity-badge">@millenniallity</span>', unsafe_allow_html=True)
    st.title("Millenniallity Volatility Mismatch Scanner")
    st.markdown(
        "Auto-discovers **highly volatile** names where **forward P/E disagrees with earnings reality** — "
        "air-gapped for **Public.com** (stock + options)."
    )

    user_add = st.text_input(
        "Add specific tickers (optional, comma-separated)",
        "",
        help="Merged with the auto high-beta universe from FMP.",
    )
    extra = [x.strip().upper() for x in user_add.replace(" ", "").split(",") if x.strip()]
    extra_map = {x: "Custom" for x in extra}

    run = st.button("🚀 Run Auto Scanner — Find Volatile P/E Mismatches", type="primary", use_container_width=True)

    st.info(
        f"**Tip:** run before the open. Data as of app date **{date.today().isoformat()}** — not investment advice."
    )

    if not run:
        st.stop()

    fmp_key = keys["fmp"]
    sec_key = keys["sec"]
    xai_key = keys["xai"]
    uw_key = keys["uw"]
    polygon_key = keys["polygon"]

    if not sec_key:
        st.warning("**SEC_API_KEY** missing — 10-K extraction will fail until set in `.env`.")
    if not xai_key:
        st.warning("**XAI_API_KEY** missing — narrative scoring disabled until set in `.env`.")

    wide = fmp_high_beta_universe(fmp_key, 100)
    if not wide:
        st.warning("FMP screener returned no symbols — using default watchlist fallback.")
        wide = list(DEFAULT_TICKERS)
    universe = sorted(set(wide + extra))

    progress = st.progress(0.0, text="Scanning universe…")
    results: list[ScanRow] = []

    for i, tick in enumerate(universe):
        progress.progress((i + 1) / max(1, len(universe)), text=f"{tick}…")
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
                    llm_summary=f"Fatal error: {e}",
                    mismatch_score=0.0,
                    recommendation="Skip",
                    errors=[str(e)],
                )
            )

    progress.empty()

    filtered = [r for r in results if r.mismatch_score >= MIN_DISPLAY_SCORE]
    filtered.sort(key=lambda r: r.mismatch_score, reverse=True)

    summary_records = [
        {
            "Ticker": r.ticker,
            "Mismatch Score": round(r.mismatch_score, 1),
            "Recommendation": r.recommendation,
            "Fwd P/E": r.fwd_pe,
            "Vol Reason": r.vol_reason,
            "LLM Flag": r.llm_flag,
        }
        for r in results
    ]
    sum_df = pd.DataFrame(summary_records).sort_values("Mismatch Score", ascending=False)

    st.subheader("Results overview")
    st.dataframe(sum_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download CSV (score-filtered list)",
        data=filtered_to_csv(filtered),
        file_name=f"millenniallity_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    st.subheader(f"Detail — score ≥ {MIN_DISPLAY_SCORE} ({'no rows' if not filtered else len(filtered)} names)")
    if not filtered:
        st.warning(f"No names met score ≥ {MIN_DISPLAY_SCORE}. Re-run later or add tickers.")

    for r in filtered:
        title = f"{r.ticker} · {r.mismatch_score:.0f} · {r.recommendation}"
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
                    st.caption("Not enough peer P/E data to chart.")
                st.metric("Stock P/E vs benchmark", f"{r.fwd_pe} vs {r.peer_avg_pe}")
                st.caption(
                    f"Peer median **{r.peer_median_pe}** · Sector bench **{r.sector_benchmark_pe}** · _{r.fwd_pe_source}_"
                )
            with c2:
                st.markdown("**LLM (10-K vs valuation)**")
                st.write(r.llm_summary or "—")
                st.caption(f"Flag: **{r.llm_flag}** · Contradiction: **{r.narrative_contradiction}**")
                st.markdown(f"### Trade: **{r.recommendation}**")
                if r.score_insider_bonus:
                    st.success(f"Insider + P/E mismatch bonus: **+{r.score_insider_bonus:.0f}** pts.")
                st.markdown("**Options (1–4 week Public.com)**")
                st.write(r.option_suggestion)
                st.markdown("**Unusual Whales**")
                st.markdown(r.unusual_whales_note or "—")
                st.caption(
                    f"Flow: net {r.insider_net_shares:,.0f} · buys {r.insider_buy_shares:,.0f} · sells {r.insider_sell_shares:,.0f}"
                )
            if r.errors:
                st.error("Notes: " + "; ".join(r.errors))

    st.markdown(
        '<div class="mill-footer">Built for Public.com options edge — volatile P/E mismatches only</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
