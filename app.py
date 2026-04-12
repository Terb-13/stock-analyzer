"""
Millenniallity Volatility Mismatch Scanner — auto-discovery for volatile P/E mismatches.
Air-gapped: no broker links, no position uploads.
API keys: `.env` / environment variables only (never shown in UI).
"""
from __future__ import annotations

import io
import json
import os
import re
import statistics
import urllib.parse
from collections.abc import Mapping
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

import streamlit_authenticator as stauth

try:
    from streamlit_authenticator.utilities.hasher import Hasher
except ImportError:
    from streamlit_authenticator import Hasher


def _hash_password_for_storage(plain: str) -> str:
    if hasattr(Hasher, "hash_list"):
        out = Hasher.hash_list([plain])  # type: ignore[attr-defined]
        if isinstance(out, list) and out:
            return str(out[0])
        return str(out)
    if hasattr(Hasher, "hash"):
        return str(Hasher.hash(plain))  # type: ignore[attr-defined]
    gen = Hasher([plain]).generate()  # type: ignore[call-arg]
    if isinstance(gen, list) and gen:
        return str(gen[0])
    return str(gen)
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

SECTOR_PE_BENCHMARK: dict[str, float] = {
    "Autos/EV": 18.0,
    "AI/Tech": 28.0,
    "Consumer": 22.0,
    "Energy/Shipping": 14.0,
    "Custom": 20.0,
}

# yfinance "industry" text (lowercased) substring → comparison tickers when FMP & Yahoo fail
INDUSTRY_PEER_FALLBACK: list[tuple[str, tuple[str, ...]]] = [
    ("semiconductor", ("AMD", "INTC", "AVGO", "QCOM", "MU", "AMAT", "LRCX", "ON")),
    ("consumer electronics", ("AAPL", "SONY", "DELL", "HPQ")),
    ("software", ("MSFT", "ORCL", "ADBE", "NOW", "CRM", "INTU", "PANW")),
    ("internet content", ("GOOGL", "META", "SNAP", "PINS", "RDDT")),
    ("internet", ("GOOGL", "META", "SNAP", "EQIX")),
    ("oil", ("XOM", "CVX", "COP", "SLB", "EOG", "MPC")),
    ("bank", ("JPM", "BAC", "WFC", "C", "GS", "MS")),
    ("insurance", ("PGR", "ALL", "TRV", "MET")),
    ("reit", ("AMT", "PLD", "EQIX", "SPG", "O", "DLR")),
    ("drug", ("JNJ", "PFE", "LLY", "MRK", "ABBV", "BMY")),
    ("biotechnology", ("AMGN", "GILD", "VRTX", "REGN", "BIIB")),
    ("medical", ("ABT", "TMO", "DHR", "SYK", "MDT")),
    ("retail", ("WMT", "TGT", "COST", "DG", "DLTR")),
    ("specialty retail", ("HD", "LOW", "NKE", "TJX")),
    ("airlines", ("DAL", "UAL", "AAL", "LUV")),
    ("aerospace", ("BA", "LMT", "RTX", "NOC", "GD")),
    ("auto", ("F", "GM", "STLA", "RIVN")),
    ("utilities", ("NEE", "DUK", "SO", "AEP", "XEL")),
    ("entertainment", ("DIS", "NFLX", "WBD", "SONY")),
    ("restaurants", ("MCD", "SBUX", "YUM", "DPZ")),
    ("beverages", ("KO", "PEP", "MNST", "KDP")),
    ("marine shipping", ("ZIM", "DAC", "GSL", "MATX")),
    ("capital markets", ("GS", "MS", "SCHW", "BLK")),
    ("credit services", ("V", "MA", "AXP")),
    ("household", ("PG", "CL", "KMB", "EL")),
]

FMP_BASE = "https://financialmodelingprep.com"
SCAN_CACHE_TTL_SEC = 3600
HIGH_SCORE_EXPAND_THRESHOLD = 65
MIN_DISPLAY_SCORE = 45
VOLATILE_UNIVERSE_LIMIT = 120
BURRY_UNIVERSE_LIMIT = 150

REQUEST_TIMEOUT = 25

# Loaded once per process from sec.gov (ticker → zero-padded 10-digit CIK)
_SEC_TICKER_CIK_MAP: Optional[dict[str, str]] = None


def _sec_user_agent() -> str:
    return os.getenv("SEC_EDGAR_USER_AGENT", "MillenniallityScanner contact@example.com")


def _sec_ticker_to_cik_map() -> dict[str, str]:
    """SEC company_tickers.json — public, no key."""
    global _SEC_TICKER_CIK_MAP
    if _SEC_TICKER_CIK_MAP is not None:
        return _SEC_TICKER_CIK_MAP
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": _sec_user_agent()},
            timeout=45,
        )
        r.raise_for_status()
        raw = r.json()
    except Exception:
        return {}
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for v in raw.values():
            if not isinstance(v, dict):
                continue
            tick = v.get("ticker")
            cik = v.get("cik_str")
            if tick is None or cik is None:
                continue
            cik_int = int(cik) if not isinstance(cik, int) else cik
            out[str(tick).upper()] = str(cik_int).zfill(10)
    _SEC_TICKER_CIK_MAP = out
    return out


def sec_submissions_latest_10k_primary_url(ticker: str) -> Optional[str]:
    """Latest Form 10-K primary document via data.sec.gov submissions JSON (no FMP)."""
    sym = ticker.upper().strip()
    if not sym:
        return None
    cik10 = _sec_ticker_to_cik_map().get(sym)
    if not cik10:
        return None
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    try:
        r = requests.get(url, headers={"User-Agent": _sec_user_agent()}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    if not forms:
        return None
    n = min(len(forms), len(dates), len(accs), len(docs))
    best_i = -1
    best_date = ""
    for i in range(n):
        if forms[i] != "10-K":
            continue
        fd = str(dates[i]) if i < len(dates) else ""
        if fd >= best_date:
            best_date = fd
            best_i = i
    if best_i < 0:
        return None
    acc = accs[best_i]
    doc = docs[best_i]
    acc_nodash = acc.replace("-", "")
    cik_numeric = str(int(cik10))
    return f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{acc_nodash}/{doc}"


def _env_or_secret(key: str) -> str:
    """Prefer `os.environ` (`.env` locally); fall back to `st.secrets` (Streamlit Community Cloud)."""
    v = os.getenv(key, "").strip()
    if v:
        return v
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return ""


def _env_keys() -> dict[str, str]:
    return {
        "fmp": _env_or_secret("FMP_API_KEY"),
        "sec": _env_or_secret("SEC_API_KEY"),
        "xai": _env_or_secret("XAI_API_KEY"),
        "uw": _env_or_secret("UNUSUAL_WHALES_API_KEY"),
        "polygon": _env_or_secret("POLYGON_API_KEY"),
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


def _xai_message_content_to_str(content: Any) -> str:
    """xAI/OpenAI-style content is usually str; reasoning models may return list segments."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
                else:
                    nested = p.get("content")
                    if isinstance(nested, str):
                        parts.append(nested)
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts).strip()
    return str(content).strip()


def _xai_error_detail(response: requests.Response) -> str:
    try:
        j = response.json()
    except Exception:
        return (response.text or "")[:600]
    err = j.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err)[:600]
    if err:
        return str(err)[:600]
    return str(j)[:600]


def xai_chat_completion(api_key: str, user_content: str) -> str:
    # Override with `XAI_MODEL` in `.env` if your key uses a different id (console.x.ai).
    model = os.getenv("XAI_MODEL", "grok-4.20-0309-reasoning").strip() or "grok-4.20-0309-reasoning"
    text_in = (user_content or "")[:28000]
    if not text_in.strip():
        raise ValueError("xAI: empty prompt (no 10-K text to analyze)")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text_in}],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    r = requests.post(
        XAI_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT,
    )
    if r.status_code >= 400:
        detail = _xai_error_detail(r)
        raise RuntimeError(f"{r.status_code} {XAI_CHAT_URL} — {detail}")
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"xAI: invalid JSON ({e.message})") from e
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("xAI response missing choices")
    msg = choices[0].get("message") or {}
    return _xai_message_content_to_str(msg.get("content"))


def xai_chat_messages(
    api_key: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.35,
) -> str:
    """OpenAI-style multi-turn chat for follow-ups (roles: system, user, assistant)."""
    model = os.getenv("XAI_MODEL", "grok-4.20-0309-reasoning").strip() or "grok-4.20-0309-reasoning"
    safe: list[dict[str, str]] = []
    total = 0
    for m in messages:
        role = (m.get("role") or "user").strip()
        content = (m.get("content") or "").strip()
        if role not in ("system", "user", "assistant") or not content:
            continue
        chunk = content[:24000]
        total += len(chunk)
        safe.append({"role": role, "content": chunk})
    if not safe:
        raise ValueError("xAI: no valid messages")
    while total > 100000 and len(safe) > 2:
        removed = safe.pop(1)
        total -= len(removed.get("content", ""))
    payload = {
        "model": model,
        "messages": safe,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        XAI_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT,
    )
    if r.status_code >= 400:
        detail = _xai_error_detail(r)
        raise RuntimeError(f"{r.status_code} {XAI_CHAT_URL} — {detail}")
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"xAI: invalid JSON ({e.message})") from e
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("xAI response missing choices")
    msg = choices[0].get("message") or {}
    return _xai_message_content_to_str(msg.get("content"))


def _grok_scan_context_system_message(row: ScanRow) -> str:
    summ = (row.llm_summary or "").strip()
    if len(summ) > 6000:
        summ = summ[:6000] + "…"
    return (
        f"You are **Grok** on the Millenniallity Volatility Scanner. The user is discussing **{row.ticker}** from a "
        f"**saved scan snapshot** (not live prices). Be direct, nuanced, and educational — not personalized "
        f"investment advice.\n\n"
        f"**Snapshot:**\n"
        f"- Trade bias: {row.recommendation}\n"
        f"- Valuation vs peers: {row.valuation_skew} | Fwd P/E {row.fwd_pe} vs bench ~{row.peer_avg_pe}\n"
        f"- Vol filter: {row.vol_reason}\n"
        f"- LLM (10-K vs valuation): {summ or '—'}\n"
        f"- Insider (Unusual Whales note): {(row.unusual_whales_note or '—')[:800]}\n"
        f"- Options blurb: {row.option_suggestion or '—'}\n"
        f"If they ask for data you do not have from this snapshot, say so and reason qualitatively."
    )


def render_grok_followup_chat(ticker: str, row: ScanRow, xai_key: str) -> None:
    """Per-ticker Grok chat (form-based; multiple tickers per page)."""
    safe_t = re.sub(r"[^a-zA-Z0-9_]", "_", ticker.upper())
    sess_hist = f"grok_chat_hist_{safe_t}"
    if sess_hist not in st.session_state:
        st.session_state[sess_hist] = []
    hist_len = len(st.session_state[sess_hist])

    if not xai_key:
        with st.expander("Grok — deeper dive & chat", expanded=False):
            st.caption("_Add **XAI_API_KEY** to `.env` to chat with Grok here._")
        return

    with st.expander("Grok — deeper dive & chat", expanded=bool(hist_len)):
        st.caption(
            "Chat with **Grok** using this scan as context. Replies are **not** financial advice; verify anything material."
        )
        q1, q2, q3 = st.columns(3)
        with q1:
            if st.button("Expand analysis", key=f"grok_ex_{safe_t}", help="Deepen the thesis"):
                st.session_state[sess_hist].append(
                    {
                        "role": "user",
                        "content": (
                            "Expand on this scan: bull vs bear, key risks, and what would invalidate the "
                            f"{row.recommendation} bias. ~3 short paragraphs."
                        ),
                    }
                )
                try:
                    with st.spinner("Grok…"):
                        messages = [{"role": "system", "content": _grok_scan_context_system_message(row)}]
                        messages.extend(st.session_state[sess_hist])
                        reply = xai_chat_messages(xai_key, messages, max_tokens=2048, temperature=0.35)
                    st.session_state[sess_hist].append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state[sess_hist].pop()
                    st.error(str(e))
                st.rerun()
        with q2:
            if st.button("Key risks", key=f"grok_risk_{safe_t}"):
                st.session_state[sess_hist].append(
                    {"role": "user", "content": f"What are the top 5 risks for {row.ticker} here, given this scan?"}
                )
                try:
                    with st.spinner("Grok…"):
                        messages = [{"role": "system", "content": _grok_scan_context_system_message(row)}]
                        messages.extend(st.session_state[sess_hist])
                        reply = xai_chat_messages(xai_key, messages, max_tokens=1536, temperature=0.35)
                    st.session_state[sess_hist].append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state[sess_hist].pop()
                    st.error(str(e))
                st.rerun()
        with q3:
            if st.button("Clear chat", key=f"grok_clr_{safe_t}"):
                st.session_state[sess_hist] = []
                st.rerun()

        for msg in st.session_state[sess_hist]:
            role_raw = (msg.get("role") or "user").strip().lower()
            label = "You" if role_raw == "user" else "Grok"
            body = msg.get("content")
            text = body if isinstance(body, str) else _xai_message_content_to_str(body)
            with st.container(border=True):
                st.caption(label)
                st.markdown(text or "_—_")

        with st.form(f"grok_chat_form_{safe_t}", clear_on_submit=True):
            prompt = st.text_area(
                "Message Grok",
                height=88,
                placeholder="Ask anything about this name given the scan context…",
                key=f"grok_ta_{safe_t}",
                label_visibility="collapsed",
            )
            sent = st.form_submit_button("Send")
        if sent and prompt and prompt.strip():
            st.session_state[sess_hist].append({"role": "user", "content": prompt.strip()})
            try:
                with st.spinner("Grok…"):
                    messages = [{"role": "system", "content": _grok_scan_context_system_message(row)}]
                    messages.extend(st.session_state[sess_hist])
                    reply = xai_chat_messages(xai_key, messages, max_tokens=2048, temperature=0.35)
                st.session_state[sess_hist].append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state[sess_hist].pop()
                st.error(str(e))
            st.rerun()


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


def _fmp_first_fundamental_row(data: Any) -> dict:
    """Normalize stable / v3 list-or-{data:[]} FMP payloads into one ratio row."""
    if not data:
        return {}
    if isinstance(data, dict):
        if data.get("Error Message") or data.get("error"):
            return {}
        inner = data.get("data")
        if isinstance(inner, list) and inner and isinstance(inner[0], dict):
            return inner[0]
        if any(
            data.get(k) is not None
            for k in ("forwardPE", "peRatioTTM", "peRatio", "priceToEarningsRatioTTM", "grossProfitMarginTTM")
        ):
            return data
        return {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


def resolve_forward_pe(
    sym: str,
    api_key: str,
    price: Optional[float],
    *,
    allow_analyst_implied: bool = True,
) -> tuple[Optional[float], str]:
    ratios_row: dict = {}
    if api_key:
        for path in (
            f"{FMP_BASE}/stable/ratios-ttm?symbol={sym}&apikey={api_key}",
            f"{FMP_BASE}/api/v3/ratios-ttm/{sym}?apikey={api_key}",
        ):
            raw, _ = _safe_get_json(path)
            row = _fmp_first_fundamental_row(raw)
            if row:
                ratios_row = row
            v = row.get("forwardPE")
            if v is not None and isinstance(v, (int, float)) and v > 0:
                tag = "FMP stable ratios-ttm forwardPE" if "stable" in path else "FMP ratios-ttm forwardPE"
                return float(v), tag

        km_url = f"{FMP_BASE}/stable/key-metrics-ttm?symbol={sym}&apikey={api_key}"
        km, _ = _safe_get_json(km_url)
        km_row = _fmp_first_fundamental_row(km)
        if not km_row:
            km_url = f"{FMP_BASE}/api/v3/key-metrics-ttm/{sym}?apikey={api_key}"
            km, _ = _safe_get_json(km_url)
            km_row = _fmp_first_fundamental_row(km)
        for k in ("forwardPE", "peRatio", "priceEarningsRatio"):
            val = km_row.get(k)
            if val is not None and isinstance(val, (int, float)) and val > 0:
                return float(val), f"FMP key-metrics-ttm {k}"

        for k in ("priceToEarningsRatioTTM", "peRatioTTM", "peRatio", "priceToEarningsRatio"):
            val = ratios_row.get(k)
            if val is not None and isinstance(val, (int, float)) and val > 0:
                return float(val), f"FMP ratios-ttm {k} (TTM / trailing)"

        if allow_analyst_implied and price is not None and price > 0:
            eps = fmp_analyst_forward_eps(sym, api_key)
            if eps and eps > 0:
                return float(price / eps), "FMP analyst-estimates implied (price ÷ next EPS)"

    try:
        yinfo = yf.Ticker(sym).info or {}
        for k in ("forwardPE", "trailingPE"):
            val = yinfo.get(k)
            if val is not None and isinstance(val, (int, float)) and val > 0:
                return float(val), f"yfinance {k}"
    except Exception:
        pass

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
        ppe, _ = resolve_forward_pe(p, api_key, px, allow_analyst_implied=True)
        pe_map[p] = ppe
        if ppe is not None and ppe > 0:
            peer_vals.append(float(ppe))

    sector_bench = SECTOR_PE_BENCHMARK.get(sector) or SECTOR_PE_BENCHMARK["Custom"]
    med = statistics.median(peer_vals) if peer_vals else None
    blended, _ = blended_peer_benchmark(med, sector)
    dev_pct, mismatch, _, _ = max_pe_deviation_vs_benchmarks(main_pe, peer_vals, sector)

    return main_pe, med, blended, sector_bench, pe_map, main_src, bool(mismatch), dev_pct


def fmp_stock_peers(symbol: str, api_key: str) -> list[str]:
    if not api_key:
        return []
    urls = [
        f"{FMP_BASE}/stable/stock-peers?symbol={symbol}&apikey={api_key}",
        f"{FMP_BASE}/api/v4/stock_peers?symbol={symbol}&apikey={api_key}",
        f"{FMP_BASE}/api/v3/stock_peers?symbol={symbol}&apikey={api_key}",
    ]
    data: Any = None
    err: Optional[str] = None
    for url in urls:
        data, err = _safe_get_json(url)
        if err or data is None:
            continue
        if isinstance(data, dict) and (data.get("Error Message") or data.get("error")):
            data = None
            continue
        break
    if data is None:
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


def _yf_industry_sector_blob(symbol: str) -> str:
    try:
        inf = yf.Ticker(symbol).info or {}
        ind = str(inf.get("industry") or "")
        sec = str(inf.get("sector") or "")
        return re.sub(r"[^a-z0-9]+", " ", f"{ind} {sec}".lower()).strip()
    except Exception:
        return ""


def yfinance_industry_peer_fallback(symbol: str, limit: int = 10) -> list[str]:
    sym = symbol.upper().strip()
    blob = _yf_industry_sector_blob(symbol)
    if not blob:
        return []
    for key, tickers in INDUSTRY_PEER_FALLBACK:
        if key in blob:
            return [t for t in tickers if t.upper() != sym][:limit]
    return []


def yahoo_recommended_peer_symbols(symbol: str, limit: int = 10) -> list[str]:
    """
    Yahoo `recommendedSymbols` when FMP stock-peers is unavailable (rate limit / plan).
    Not a formal comp sheet — good enough for P/E context bars.
    """
    sym = symbol.upper().strip()
    if not sym:
        return []
    url = f"https://query1.finance.yahoo.com/v6/finance/recommendationsbysymbol/{sym}"
    headers = {"User-Agent": YAHOO_SCREENER_UA}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception:
        return []
    fin = (j.get("finance") or {}).get("result") or []
    if not fin:
        return []
    rec = fin[0].get("recommendedSymbols") or []
    out: list[str] = []
    for item in rec:
        if not isinstance(item, dict):
            continue
        s = item.get("symbol")
        if not s:
            continue
        u = str(s).upper().strip()
        if u != sym:
            out.append(u)
        if len(out) >= limit:
            break
    return out


def discover_stock_peers(symbol: str, fmp_key: str) -> tuple[list[str], str]:
    fmp_list = fmp_stock_peers(symbol, fmp_key) if fmp_key else []
    fmp_list = [p for p in fmp_list if p.upper() != symbol.upper()]
    if fmp_list:
        return fmp_list[:12], "FMP"
    y = yahoo_recommended_peer_symbols(symbol, 12)
    y = [p for p in y if p.upper() != symbol.upper()][:10]
    if y:
        return y, "Yahoo"
    basket = yfinance_industry_peer_fallback(symbol, 10)
    if basket:
        return basket, "yfinance-industry"
    return [], "none"


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


def sec_edgar_latest_10k_index_url(ticker: str) -> Optional[str]:
    """Latest 10-K filing index page from SEC EDGAR Atom (no API key; set SEC_EDGAR_USER_AGENT in `.env`)."""
    sym = ticker.upper().strip()
    if not sym:
        return None
    q = urllib.parse.urlencode(
        {"action": "getcompany", "ticker": sym, "type": "10-K", "owner": "exclude", "count": "1", "output": "atom"}
    )
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?{q}"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": _sec_user_agent(), "Accept": "application/atom+xml"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        text = r.text
    except Exception:
        return None
    m = re.search(
        r"<filing-href>(https://www\.sec\.gov/Archives/edgar/data/[^<]+\.htm)</filing-href>",
        text,
    )
    return m.group(1) if m else None


def resolve_latest_10k_filing_url(symbol: str, api_key: str) -> tuple[Optional[str], str]:
    """Prefer SEC.gov (free, reliable); FMP last so a dead quota does not block the scan."""
    sym = symbol.upper().strip()
    prim = sec_submissions_latest_10k_primary_url(sym)
    if prim:
        return prim, "SEC submissions"
    sec = sec_edgar_latest_10k_index_url(sym)
    if sec:
        return sec, "SEC atom"
    if api_key:
        for tpl in (
            f"{FMP_BASE}/api/v3/sec_filings/{sym}?type=10-K&page=0&apikey={api_key}",
            f"{FMP_BASE}/api/v3/sec-filings/{sym}?type=10-k&apikey={api_key}",
        ):
            data, err = _safe_get_json(tpl)
            if err or not data:
                continue
            if isinstance(data, dict) and (data.get("Error Message") or data.get("error")):
                continue
            if not isinstance(data, list) or not data:
                continue
            row = data[0]
            if not isinstance(row, dict):
                continue
            link = row.get("finalLink") or row.get("link") or row.get("linkToFilingDetails")
            if link:
                return str(link), "FMP"
    return None, "none"


def fmp_latest_10k_filing_url(symbol: str, api_key: str) -> Optional[str]:
    u, _ = resolve_latest_10k_filing_url(symbol, api_key)
    return u


def _fmp_response_to_symbols(data: Any) -> list[str]:
    """Normalize FMP screener / movers JSON into uppercase tickers."""
    if data is None:
        return []
    if isinstance(data, dict):
        if data.get("Error Message") or data.get("error"):
            return []
        data = data.get("data") or data.get("results") or data.get("stockList") or []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for row in data:
        if isinstance(row, str):
            if row.strip():
                out.append(row.strip().upper())
            continue
        if not isinstance(row, dict):
            continue
        s = row.get("symbol") or row.get("ticker") or row.get("Symbol") or row.get("tickerSymbol")
        if s and str(s).strip():
            out.append(str(s).strip().upper())
    return out


def fmp_high_beta_universe(api_key: str, limit: int = 100) -> tuple[list[str], str, str]:
    """
    Build a volatile discovery list: try **stable company-screener**, then legacy **v3 stock-screener**,
    then **most actives / gainers** (usually allowed on free tier if screener is paywalled or params reject).

    Returns `(symbols, source_label, fmp_debug_text)` — the third string logs HTTP status for each FMP
    attempt (for the Debug panel when the universe is empty or misbehaving).
    """
    debug_lines: list[str] = []
    if not api_key:
        return [], "none", "FMP: no API key — skipped high-beta universe calls."

    attempts: list[tuple[str, dict[str, str]]] = [
        (
            f"{FMP_BASE}/stable/company-screener",
            {
                "apikey": api_key,
                "limit": str(limit),
                "isActivelyTrading": "true",
                "isEtf": "false",
                "betaMoreThan": "1.25",
                "volumeMoreThan": "100000",
                "marketCapMoreThan": "10000000",
            },
        ),
        (
            f"{FMP_BASE}/stable/company-screener",
            {
                "apikey": api_key,
                "limit": str(limit),
                "isActivelyTrading": "true",
                "betaMoreThan": "1.15",
                "marketCapMoreThan": "5000000",
            },
        ),
        (
            f"{FMP_BASE}/stable/company-screener",
            {
                "apikey": api_key,
                "limit": str(limit),
                "isActivelyTrading": "true",
                "marketCapMoreThan": "30000000",
                "betaMoreThan": "1.1",
            },
        ),
        (
            f"{FMP_BASE}/api/v3/stock-screener",
            {
                "apikey": api_key,
                "limit": str(limit),
                "isActivelyTrading": "true",
                "isEtf": "false",
                "betaMoreThan": "1.25",
                "volumeMoreThan": "100000",
                "marketCapMoreThan": "10000000",
            },
        ),
        (
            f"{FMP_BASE}/api/v3/stock-screener",
            {
                "apikey": api_key,
                "limit": str(limit),
                "marketCapMoreThan": "1000000",
                "betaMoreThan": "1.05",
            },
        ),
    ]

    primary_logged = False
    for base, params in attempts:
        url = f"{base}?{urllib.parse.urlencode(params)}"
        short = "stable/company-screener" if "stable" in base else "v3/stock-screener"
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if not primary_logged and "stable" in base and "betaMoreThan" in params:
                debug_lines.append(
                    f"Primary high-beta universe call ({short}, strict filters): HTTP {r.status_code}"
                )
                primary_logged = True
            if r.status_code != 200:
                debug_lines.append(f"{short}: HTTP {r.status_code}")
                continue
            data = r.json()
        except Exception as e:
            debug_lines.append(f"{short}: {e}")
            continue
        syms = _fmp_response_to_symbols(data)
        if syms:
            label = "stable/company-screener" if "stable" in base else "v3/stock-screener"
            return syms[:limit], label, "\n".join(debug_lines)

    for tail in ("stock_market/actives", "stock_market/gainers", "stock_market/losers"):
        url = f"{FMP_BASE}/api/v3/{tail}?apikey={api_key}"
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            debug_lines.append(f"v3/{tail}: HTTP {r.status_code}")
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception as e:
            debug_lines.append(f"v3/{tail}: {e}")
            continue
        syms = _fmp_response_to_symbols(data)
        if syms:
            return syms[:limit], f"v3/{tail}", "\n".join(debug_lines)

    return [], "none", "\n".join(debug_lines) if debug_lines else "FMP: no HTTP lines recorded."


YAHOO_PREDEFINED_SCREENER_IDS = ("most_actives", "day_gainers", "day_losers")
YAHOO_SCREENER_UA = "Mozilla/5.0 (compatible; MillenniallityScanner/1.0; +https://github.com)"


def _safe_get_json_with_headers(url: str, headers: dict[str, str]) -> tuple[Any, Optional[str]]:
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def yahoo_screener_universe(limit: int) -> tuple[list[str], str]:
    """Liquid movers from Yahoo predefined screeners (no API key). Used when FMP list endpoints fail."""
    if limit <= 0:
        return [], "yahoo/none"
    headers = {"User-Agent": YAHOO_SCREENER_UA}
    per_call = max(25, min(limit, 100))
    seen: list[str] = []
    used: list[str] = []
    for scr_id in YAHOO_PREDEFINED_SCREENER_IDS:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            f"?count={per_call}&scrIds={scr_id}&formatted=false"
        )
        data, err = _safe_get_json_with_headers(url, headers)
        if err or not data:
            continue
        fin = data.get("finance") if isinstance(data, dict) else None
        result = (fin or {}).get("result") or []
        if not result:
            continue
        quotes = result[0].get("quotes") or []
        if quotes:
            used.append(scr_id)
        for q in quotes:
            if not isinstance(q, dict):
                continue
            sym = q.get("symbol")
            if sym and str(sym).strip():
                u = str(sym).strip().upper()
                if u not in seen:
                    seen.append(u)
            if len(seen) >= limit:
                return seen[:limit], "yahoo/" + "+".join(used) if used else "yahoo/empty"
    return seen[:limit], "yahoo/" + "+".join(used) if used else "yahoo/none"


def build_discovery_universe(
    fmp_api_key: str, limit: int,
) -> tuple[list[str], str, str, bool, int]:
    """
    FMP volatile universe first; if empty (402/403 paywalls), Yahoo predefined screeners.

    Returns `(symbols, source_label, fmp_debug_text, fmp_returned_any_symbols, fmp_symbol_count)`.
    """
    syms, src, fmp_dbg = fmp_high_beta_universe(fmp_api_key, limit)
    fmp_n = len(syms)
    fmp_ok = fmp_n > 0
    if syms:
        return syms, src, fmp_dbg, fmp_ok, fmp_n
    y_syms, y_src = yahoo_screener_universe(limit)
    if y_syms:
        return (
            y_syms,
            y_src,
            fmp_dbg + "\nUniverse fallback: Yahoo predefined screeners (FMP list empty).",
            fmp_ok,
            fmp_n,
        )
    return (
        [],
        "none",
        fmp_dbg + "\nUniverse fallback: Yahoo also returned no symbols.",
        fmp_ok,
        fmp_n,
    )


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


def valuation_skew_label(
    fwd: Optional[float],
    blended: Optional[float],
    peer_median: Optional[float],
    *,
    rich_ratio: float = 1.22,
    cheap_ratio: float = 0.78,
) -> str:
    """rich = fwd P/E materially *above* peers/sector; cheap = materially below."""
    if fwd is None or fwd <= 0:
        return "unknown"
    ratios: list[float] = []
    if blended and blended > 0:
        ratios.append(fwd / blended)
    if peer_median and peer_median > 0:
        ratios.append(fwd / peer_median)
    if not ratios:
        return "unknown"
    hi = max(ratios)
    lo = min(ratios)
    if hi >= rich_ratio:
        return "rich"
    if lo <= cheap_ratio:
        return "cheap"
    return "inline"


def burry_fifty_pct_above_blended(fwd: Optional[float], blended: Optional[float]) -> bool:
    """True when forward P/E is ≥50% above blended peer/sector benchmark (Burry overvaluation bar)."""
    if fwd is None or blended is None or blended <= 0:
        return False
    return fwd >= blended * 1.5


def burry_large_insider_extraction(
    insider_net_seller: bool,
    insider_buy_shares: float,
    insider_sell_shares: float,
    insider_net_shares: float,
) -> bool:
    """Heuristic for owner / insider cash-out scale (large sells vs buys)."""
    if not insider_net_seller:
        return False
    if insider_sell_shares >= max(insider_buy_shares * 2.5, 1.0) and insider_sell_shares >= 50_000.0:
        return True
    if abs(insider_net_shares) >= 250_000.0:
        return True
    return False


def run_llm_valuation_mismatch(
    ticker: str,
    fwd_pe: Optional[float],
    peer_avg_pe: Optional[float],
    item1: str,
    item7: str,
    *,
    xai_key: str,
    valuation_skew: str = "unknown",
) -> tuple[str, str, str]:
    prompt = f"""Stock: {ticker}. Forward P/E (if known): {fwd_pe}. Blended peer/sector benchmark P/E (if known): {peer_avg_pe}.
Structural valuation vs peers/sector (from data): **{valuation_skew}** — rich means multiples above peers (often stretched / short-bias), cheap means below (often recovery / long-bias).

Does management tone/catalysts contradict the current P/E valuation? Summarize in 100 words + bullish/bearish flag.
If the stock is **rich** vs peers but the story is rosy, call out why the multiple may still be unjustified. If **cheap**, note turnaround vs value-trap risk.

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
    peer_source: str = ""
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
    insider_net_seller: bool = False
    score_insider_bonus: float = 0.0
    valuation_skew: str = "unknown"
    mismatch_score: float = 0.0
    recommendation: str = "Skip"
    option_suggestion: str = ""
    errors: list[str] = field(default_factory=list)
    burry_fit: str = ""


def score_row(
    vol_ok: bool,
    pe_dev: Optional[float],
    pe_mismatch: bool,
    llm_flag: str,
    narrative_text: str,
    *,
    insider_bonus: float = 0.0,
    burry_mode: bool = False,
    burry_extreme_rich: bool = False,
    burry_insider_bearish_combo: bool = False,
    burry_large_extraction: bool = False,
    valuation_skew: str = "unknown",
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
    if burry_mode:
        if burry_extreme_rich:
            s += 30.0
        if burry_insider_bearish_combo:
            s += 18.0
            if burry_large_extraction:
                s += 10.0
        if valuation_skew == "cheap" and llm_flag == "Bullish":
            s += 8.0
    return max(0.0, min(100.0, s))


def recommend_trade(
    vol_ok: bool,
    score: float,
    llm_flag: str,
    pe_mismatch: bool,
    contradiction_yes: bool,
    valuation_skew: str,
    insider_net_seller: bool,
    *,
    burry_mode: bool = False,
    burry_extreme_rich: bool = False,
    burry_large_extraction: bool = False,
) -> str:
    """
    Discovery bias: rich forward P/E vs peers/sector → default **Buy Puts** (e.g. multiple ~2× peers + insider selling).
    Cheap mismatch → **Buy Calls** / **Buy Stock** for recovery-style mispricing.
    Burry mode: relabel strongest stretched-multiple + insider / bearish setups as **Burry Put Candidate**.
    """
    if not vol_ok:
        return "Skip"
    if not pe_mismatch and score < 48:
        return "Skip"

    base = "Skip"

    if pe_mismatch and valuation_skew == "unknown":
        if insider_net_seller or llm_flag == "Bearish":
            base = "Buy Puts"
        elif llm_flag == "Bullish":
            base = "Buy Calls (CVNA-style)"
        elif score >= 56:
            base = "Buy Stock"

    # Stretched multiple vs industry → puts-first (e.g. ~46x vs ~18–20x peers; insider selling reinforces)
    if base == "Skip" and pe_mismatch and valuation_skew == "rich":
        base = "Buy Puts"

    if base == "Skip" and pe_mismatch and valuation_skew == "cheap":
        if insider_net_seller and (llm_flag == "Bearish" or contradiction_yes):
            base = "Buy Puts"
        elif llm_flag == "Bullish":
            base = "Buy Calls (CVNA-style)"
        elif score >= 55:
            base = "Buy Stock"
        elif score >= 48:
            base = "Buy Stock"

    # Large % deviation but ratio inside inline band — use narrative + score
    if base == "Skip" and pe_mismatch and valuation_skew == "inline":
        if llm_flag == "Bearish" or insider_net_seller:
            base = "Buy Puts"
        elif llm_flag == "Bullish":
            base = "Buy Calls (CVNA-style)"
        elif contradiction_yes:
            base = "Buy Puts"
        elif score >= 58:
            base = "Buy Stock"

    if base == "Skip" and pe_mismatch and contradiction_yes:
        if llm_flag == "Bearish":
            base = "Buy Puts"
        elif llm_flag == "Bullish":
            base = "Buy Calls (CVNA-style)"

    if base == "Skip" and score >= 62 and pe_mismatch:
        if llm_flag == "Bearish":
            base = "Buy Puts"
        elif llm_flag == "Bullish":
            base = "Buy Calls (CVNA-style)"
    if base == "Skip" and score >= 52 and pe_mismatch:
        base = "Buy Stock"

    if burry_mode and base == "Buy Puts":
        if burry_extreme_rich and (
            llm_flag == "Bearish" or (insider_net_seller and burry_large_extraction)
        ):
            return "Burry Put Candidate"
        return "Buy Puts"
    return base


def burry_fit_label(
    burry_mode: bool,
    burry_extreme_rich: bool,
    insider_net_seller: bool,
    llm_flag: str,
    large_extraction: bool,
    valuation_skew: str,
    fwd: Optional[float],
    blended: Optional[float],
    recommendation: str,
) -> str:
    """Short label for table/cards when Burry Mode is on."""
    if not burry_mode:
        return ""
    if valuation_skew == "cheap" and llm_flag == "Bullish":
        return "Deep value long (Burry)"
    if recommendation == "Burry Put Candidate":
        return (
            "Burry Put — extreme vs bench + seller/Bearish (e.g. CVNA ~46× vs ~18–20× peers + owner cash-out)"
        )
    if burry_extreme_rich and (llm_flag == "Bearish" or insider_net_seller):
        pe_hint = ""
        if fwd is not None and blended is not None and blended > 0:
            pe_hint = f" ~{fwd:.0f}× vs ~{blended:.0f}× bench"
        ex = " + large insider extraction" if large_extraction else ""
        return f"Puts bias — stretched multiple{pe_hint}{ex}"
    if burry_extreme_rich:
        return "Extreme vs blended benchmark (≥50%)"
    if insider_net_seller and llm_flag == "Bearish":
        return "Insider selling + Bearish LLM"
    return "—"


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


def _parse_uw_iso_time(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    t = t.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(t)
    except ValueError:
        try:
            return datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            try:
                return datetime.strptime(t[:10], "%Y-%m-%d")
            except ValueError:
                return None


def _uw_rows_to_daily_series(rows: list[dict]) -> tuple[list[datetime], list[float]]:
    """
    Unusual Whales 1d+1Y often returns multiple rows per calendar `date` (pre / post / regular).
    For each day, keep the bar with best session priority (regular `r` wins).
    """
    pri_map = {"r": 3, "po": 2, "pr": 2}
    by_day: dict[str, tuple[datetime, float, int]] = {}

    for item in rows:
        if not isinstance(item, dict):
            continue
        ds = item.get("date")
        if isinstance(ds, str) and len(ds) >= 10:
            day = ds[:10]
            try:
                dt = datetime.strptime(day, "%Y-%m-%d")
            except ValueError:
                continue
        else:
            pt = _parse_uw_iso_time(item.get("end_time") or item.get("start_time") or ds)
            if pt is None:
                continue
            day = pt.strftime("%Y-%m-%d")
            dt = datetime.strptime(day, "%Y-%m-%d")
        c = item.get("close")
        try:
            cf = float(c) if c is not None else None
        except (TypeError, ValueError):
            cf = None
        if cf is None or cf <= 0:
            continue
        mt = str(item.get("market_time") or "").lower()
        pri = pri_map.get(mt, 1)
        prev = by_day.get(day)
        if prev is None or pri > prev[2] or (pri == prev[2] and dt >= prev[0]):
            by_day[day] = (dt, cf, pri)

    if len(by_day) < 2:
        return [], []

    ordered = sorted(by_day.items(), key=lambda x: x[0])
    return [p[1][0] for p in ordered], [p[1][1] for p in ordered]


def fetch_uw_stock_ohlc_1y(ticker: str, uw_key: str) -> tuple[list[datetime], list[float], str]:
    """Daily OHLC from Unusual Whales (~12 months via timeframe=1Y)."""
    if not uw_key:
        return [], [], "no Unusual Whales API key"
    sym = ticker.upper().strip()
    url = f"https://api.unusualwhales.com/api/stock/{sym}/ohlc/1d"
    params = {"timeframe": "1Y"}
    headers = {"Authorization": f"Bearer {uw_key}", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 401:
            return [], [], "Unusual Whales: 401 unauthorized"
        if r.status_code == 402:
            return [], [], "Unusual Whales: OHLC requires a paid tier (402)"
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        return [], [], f"Unusual Whales OHLC error: {e}"

    rows_raw = body.get("data") if isinstance(body, dict) else body
    if not isinstance(rows_raw, list) or not rows_raw:
        return [], [], "Unusual Whales: no OHLC rows"
    rows = [x for x in rows_raw if isinstance(x, dict)]
    times_o, closes_o = _uw_rows_to_daily_series(rows)
    if len(times_o) < 2:
        return [], [], f"Unusual Whales: parsed {len(times_o)} daily points (need ≥2)"
    return times_o, closes_o, ""


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
    burry_mode: bool = False,
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

    peers, peer_src = discover_stock_peers(t, fmp_key)
    peers = [p for p in peers if p.upper() != t][:12]
    row.peer_source = peer_src
    row.peers_for_chart = [t] + peers[:8]
    if not peers:
        errors.append(
            "No peer list — FMP, Yahoo recommendations, and yfinance industry basket all empty "
            "(verify ticker; check network)."
        )

    fmp_for_ratios = fmp_key or ""
    (
        fwd,
        peer_med,
        blended,
        sec_bench,
        pe_map,
        main_src,
        mismatch,
        dev_pct,
    ) = fmp_forward_pe_and_peer_avg(t, peers or [t], fmp_for_ratios, sector, last)
    row.fwd_pe = fwd
    row.peer_median_pe = peer_med
    row.peer_avg_pe = blended
    row.sector_benchmark_pe = sec_bench
    row.pe_by_ticker = pe_map
    row.fwd_pe_source = main_src if fmp_key else (main_src or "yfinance / no FMP key")
    row.pe_deviation_pct = dev_pct
    row.pe_mismatch_30 = mismatch

    row.valuation_skew = valuation_skew_label(row.fwd_pe, row.peer_avg_pe, row.peer_median_pe)

    peer_for_llm = row.peer_avg_pe
    filing_url, _filing_src = resolve_latest_10k_filing_url(t, fmp_key or "")

    if not filing_url:
        errors.append(
            "No 10-K URL — FMP limited/unavailable; SEC `data.sec.gov` submissions + atom also failed. "
            "Set **SEC_EDGAR_USER_AGENT** in `.env` to `YourApp your@email` (required by sec.gov)."
        )
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
                t,
                row.fwd_pe,
                peer_for_llm,
                item1,
                item7,
                xai_key=xai_key,
                valuation_skew=row.valuation_skew,
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
    row.insider_net_seller = ins.net_shares < 0

    insider_bonus = 0.0
    if row.pe_mismatch_30:
        if row.valuation_skew == "rich" and row.insider_net_seller:
            insider_bonus = 12.0
        elif row.valuation_skew == "cheap" and ins.net_buyer:
            insider_bonus = 10.0
    burry_extreme_rich = burry_mode and burry_fifty_pct_above_blended(row.fwd_pe, row.peer_avg_pe)
    burry_large_ex = burry_large_insider_extraction(
        row.insider_net_seller,
        row.insider_buy_shares,
        row.insider_sell_shares,
        row.insider_net_shares,
    )
    burry_insider_bearish_combo = burry_mode and row.insider_net_seller and row.llm_flag == "Bearish"
    if burry_mode and row.pe_mismatch_30 and burry_extreme_rich and row.insider_net_seller:
        insider_bonus = max(insider_bonus, 12.0) + 6.0
    row.score_insider_bonus = insider_bonus

    row.mismatch_score = score_row(
        row.vol_ok,
        row.pe_deviation_pct,
        row.pe_mismatch_30,
        row.llm_flag,
        row.llm_summary,
        insider_bonus=insider_bonus,
        burry_mode=burry_mode,
        burry_extreme_rich=burry_extreme_rich,
        burry_insider_bearish_combo=burry_insider_bearish_combo,
        burry_large_extraction=burry_large_ex,
        valuation_skew=row.valuation_skew,
    )
    row.recommendation = recommend_trade(
        row.vol_ok,
        row.mismatch_score,
        row.llm_flag,
        row.pe_mismatch_30,
        row.narrative_contradiction == "Yes",
        row.valuation_skew,
        row.insider_net_seller,
        burry_mode=burry_mode,
        burry_extreme_rich=burry_extreme_rich,
        burry_large_extraction=burry_large_ex,
    )
    row.burry_fit = burry_fit_label(
        burry_mode,
        burry_extreme_rich,
        row.insider_net_seller,
        row.llm_flag,
        burry_large_ex,
        row.valuation_skew,
        row.fwd_pe,
        row.peer_avg_pe,
        row.recommendation,
    )
    row.option_suggestion = suggest_options(t, last or yv.last_close)
    row.errors = errors
    return row


def is_actionable_recommendation(row: ScanRow) -> bool:
    return row.recommendation != "Skip"


def _burry_results_sort_key(row: ScanRow, burry_mode: bool) -> tuple:
    if not burry_mode:
        return (-row.mismatch_score,)
    pri = {
        "Burry Put Candidate": 0,
        "Buy Puts": 1,
        "Buy Calls (CVNA-style)": 2,
        "Buy Stock": 3,
    }.get(row.recommendation, 4)
    return (pri, -row.mismatch_score)


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner=False)
def scan_ticker_cached(
    ticker: str,
    sector: str,
    fmp_key: str,
    sec_key: str,
    xai_key: str,
    polygon_key: str,
    uw_key: str,
    burry_mode: bool = False,
) -> ScanRow:
    return _scan_ticker_impl(
        ticker,
        sector,
        fmp_key=fmp_key,
        sec_key=sec_key,
        xai_key=xai_key,
        polygon_key=polygon_key,
        uw_key=uw_key,
        burry_mode=burry_mode,
    )


def _pe_map_lookup(pe_by_ticker: dict, sym: str) -> Optional[float]:
    v = pe_by_ticker.get(sym)
    if v is not None:
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError):
            pass
    su = sym.upper()
    for k, val in (pe_by_ticker or {}).items():
        if str(k).upper() == su:
            try:
                f = float(val)
                return f if f > 0 else None
            except (TypeError, ValueError):
                return None
    return None


def render_peer_ticker_labels(row: ScanRow) -> None:
    """Show competitor tickers and which ones contributed P/E to the median / chart."""
    t = row.ticker.upper()
    comps = [str(p).upper().strip() for p in row.peers_for_chart if str(p).upper().strip() != t]
    if not comps:
        st.caption(
            "_No peer set — comparison leans on sector benchmark only._"
        )
        return
    src = (row.peer_source or "").strip()
    if src == "Yahoo":
        label = "Related symbols (Yahoo — used when FMP peers unavailable)"
    elif src == "FMP":
        label = "Competitors (FMP stock peers)"
    elif src == "yfinance-industry":
        label = "Industry basket (yfinance — when FMP & Yahoo peers unavailable)"
    else:
        label = "Comparison set"
    st.markdown("**" + label + ":** " + " · ".join(f"`{p}`" for p in comps))
    used = [p for p in comps if _pe_map_lookup(row.pe_by_ticker, p) is not None]
    if not used:
        st.caption("_No usable P/E for peers this run — chart may show only your ticker._")
        return
    missing = [p for p in comps if p not in used]
    if missing:
        st.caption(
            "**P/E in median & bars:** "
            + ", ".join(f"`{x}`" for x in used)
            + f" ({len(used)}). _No multiple:_ "
            + ", ".join(f"`{x}`" for x in missing)
        )
    else:
        st.caption(f"**P/E in median & bars:** all **{len(used)}** peers above (and **{t}** in chart).")


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
    if row.peer_avg_pe is not None and row.peer_avg_pe > 0:
        fig.add_hline(
            y=float(row.peer_avg_pe),
            line_dash="dash",
            line_color="rgba(196, 181, 253, 0.65)",
            annotation_text="blended benchmark",
            annotation_position="top right",
        )
    return fig


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner=False)
def uw_price_chart_figure_cached(ticker: str, uw_key: str) -> tuple[Optional[go.Figure], str]:
    t = ticker.upper().strip()
    dates, closes, err = fetch_uw_stock_ohlc_1y(t, uw_key)
    if err or not dates:
        return None, err or "no price data"
    fig = go.Figure(
        data=[
            go.Scatter(
                x=dates,
                y=closes,
                mode="lines",
                line=dict(color="#4ade80", width=2),
                name="Close",
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,12,35,0.6)",
        font_color="#e9d8ff",
        title="Unusual Whales — price (1Y daily)",
        margin=dict(l=8, r=8, t=36, b=8),
        yaxis_title="Close",
        height=280,
        showlegend=False,
    )
    return fig, ""


def filtered_to_csv(rows: list[ScanRow], *, burry_mode: bool = False) -> str:
    buf = io.StringIO()
    out: list[dict[str, Any]] = []
    for r in rows:
        rec: dict[str, Any] = {
            "Ticker": r.ticker,
            "MismatchScore": round(r.mismatch_score, 1),
            "Recommendation": r.recommendation,
            "ValuationSkew": r.valuation_skew,
            "FwdPE": r.fwd_pe,
            "VolReason": r.vol_reason,
            "LLMFlag": r.llm_flag,
            "InsiderBonus": r.score_insider_bonus,
            "OptionIdea": r.option_suggestion,
        }
        if burry_mode:
            rec["BurryFit"] = r.burry_fit
        out.append(rec)
    pd.DataFrame(out).to_csv(buf, index=False)
    return buf.getvalue()


def _llm_summary_count_ok(rows: list[ScanRow]) -> int:
    """Count tickers where xAI returned a narrative (not placeholder / error-only)."""
    return sum(1 for r in rows if (r.llm_provider or "").strip() == "xAI")


def _collect_per_ticker_error_lines(rows: list[ScanRow]) -> list[str]:
    lines: list[str] = []
    for r in rows:
        for e in r.errors or []:
            es = str(e).strip()
            if es:
                lines.append(f"**{r.ticker}:** {es}")
        if r.vol_reason == "error" and r.llm_summary and str(r.llm_summary).startswith("Fatal error"):
            lines.append(f"**{r.ticker}:** {r.llm_summary}")
    return lines


def _running_scan_counters(rows: list[ScanRow], *, burry_mode: bool) -> tuple[int, int, int, int]:
    """Live cumulative stats: vol pass, forward P/E OK, xAI summaries, current final-result count."""
    n_vol = sum(1 for r in rows if r.vol_ok)
    n_pe = sum(1 for r in rows if r.fwd_pe is not None)
    n_llm = _llm_summary_count_ok(rows)
    actionable = [r for r in rows if is_actionable_recommendation(r)]
    n_final = len([r for r in actionable if r.mismatch_score >= MIN_DISPLAY_SCORE])
    return n_vol, n_pe, n_llm, n_final


# -----------------------------------------------------------------------------
# Auth (streamlit-authenticator — self-contained in this file)
# -----------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_USERS_JSON = os.path.join(_APP_DIR, "auth_users.json")
ADMIN_USERNAME = "terb13"


def _secrets_get(key: str, default: Any = None) -> Any:
    try:
        if hasattr(st, "secrets"):
            sec = st.secrets
            if key in sec:
                return sec[key]
            if hasattr(sec, "get"):
                v = sec.get(key)
                if v is not None:
                    return v
    except Exception:
        pass
    return default


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Streamlit `Secrets` sections are often `Mapping` but not `dict` — normalize for `.get(...)`."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, Mapping):
        try:
            return dict(obj)
        except Exception:
            pass
    try:
        if hasattr(obj, "to_dict"):
            return dict(obj.to_dict())  # type: ignore[call-arg]
    except Exception:
        pass
    try:
        return {str(k): obj[k] for k in obj}  # type: ignore[operator]
    except Exception:
        return {}


def _load_file_users_only() -> dict[str, Any]:
    """Users persisted by admin (auth_users.json) — not the secrets.toml bootstrap users."""
    if not os.path.isfile(AUTH_USERS_JSON):
        return {}
    try:
        with open(AUTH_USERS_JSON, encoding="utf-8") as f:
            extra = json.load(f)
        if isinstance(extra, dict) and isinstance(extra.get("usernames"), dict):
            return {str(k): dict(v) for k, v in extra["usernames"].items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def _load_credentials_dict() -> dict[str, Any]:
    """Merge `[credentials]` from secrets.toml with optional `auth_users.json` (admin-added users)."""
    raw = _to_plain_dict(_secrets_get("credentials"))
    creds: dict[str, Any] = {"usernames": {}}
    un = _to_plain_dict(raw.get("usernames"))
    for username, udata in un.items():
        ukey = str(username)
        row = _to_plain_dict(udata)
        if row:
            creds["usernames"][ukey] = {str(k): row[k] for k in row}
    for u, row in _load_file_users_only().items():
        creds["usernames"][u] = dict(row)
    return creds


def _save_auth_users_json(usernames: dict[str, Any]) -> Optional[str]:
    try:
        with open(AUTH_USERS_JSON, "w", encoding="utf-8") as f:
            json.dump({"usernames": usernames}, f, indent=2)
        return None
    except Exception as e:
        return str(e)


def _auth_logged_in() -> bool:
    return st.session_state.get("authentication_status") is True


def _is_admin() -> bool:
    if not _auth_logged_in():
        return False
    un = str(st.session_state.get("username") or "").strip().lower()
    return un == ADMIN_USERNAME.lower()


def _auth_display_name() -> str:
    if _auth_logged_in():
        return str(st.session_state.get("name") or st.session_state.get("username") or "User")
    return "User"


def _build_authenticator() -> tuple[dict[str, Any], Any]:
    creds = _load_credentials_dict()
    if not creds.get("usernames"):
        st.error(
            "No **`[credentials.usernames.*]`** in secrets. "
            "Locally: **`.streamlit/secrets.toml`**. "
            "**Streamlit Community Cloud:** paste the same TOML into **App settings → Secrets**. "
            f"Include at least **`{ADMIN_USERNAME}`** with a **bcrypt** `password` hash "
            "(see **`.streamlit/secrets.toml.example`**)."
        )
        st.stop()
    cookie = _to_plain_dict(_secrets_get("cookie"))
    if not cookie or not (cookie.get("name") and cookie.get("key")):
        st.error(
            "Missing or invalid **`[cookie]`** in secrets (local **`.streamlit/secrets.toml`** or **Cloud → Secrets**). "
            "Add `name`, `key` (≥16 chars), and `expiry_days` under **`[cookie]`** — see **`.streamlit/secrets.toml.example`**."
        )
        st.stop()
    cname = str(cookie.get("name") or "mill_auth_cookie").strip()
    ckey = str(cookie.get("key") or "").strip()
    if len(ckey) < 16:
        st.error("**`cookie.key`** must be a long random string (≥16 characters) in secrets.toml.")
        st.stop()
    try:
        exp = int(cookie.get("expiry_days", 30))
    except (TypeError, ValueError):
        exp = 30
    authenticator = stauth.Authenticate(creds, cname, ckey, exp)
    return creds, authenticator


def render_login_screen(authenticator: Any) -> None:
    st.markdown('<span class="millenniallity-badge">@millenniallity</span>', unsafe_allow_html=True)
    st.title("Millenniallity Volatility Mismatch Scanner")
    st.markdown(
        "**Private app** — sign in to run the scanner. Air-gapped: no broker integration. "
        "Data API keys: **`.env`** locally or **Streamlit Cloud → Secrets** as environment-style entries."
    )
    st.divider()
    st.subheader("Login")
    authenticator.login(location="main")


def render_user_management_sidebar() -> None:
    if not _is_admin():
        return
    with st.expander("User Management"):
        st.caption(
            "Adds users to **`auth_users.json`** (bcrypt hash). Merges on load. "
            "**Streamlit Cloud:** the filesystem is often ephemeral — new users may disappear after restart; "
            "for persistent accounts, add **`[credentials.usernames.*]`** in **Cloud → Secrets** (or commit `auth_users.json` privately)."
        )
        with st.form("add_local_user"):
            nu = st.text_input("Username", key="um_username")
            ne = st.text_input("Email", key="um_email")
            np1 = st.text_input("Password", type="password", key="um_pw")
            np2 = st.text_input("Confirm password", type="password", key="um_pw2")
            sub = st.form_submit_button("Add user")
            if sub:
                u = (nu or "").strip()
                em = (ne or "").strip()
                if not u or not np1:
                    st.error("Username and password are required.")
                elif len(np1) < 10:
                    st.error("Password must be at least **10** characters.")
                elif np1 != np2:
                    st.error("Passwords do not match.")
                elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", em):
                    st.error("Enter a valid **email**.")
                else:
                    merged = _load_credentials_dict()
                    if u.lower() in {k.lower() for k in (merged.get("usernames") or {})}:
                        st.error("That username already exists.")
                    else:
                        try:
                            hp = _hash_password_for_storage(np1)
                        except Exception as e:
                            st.error(f"Hash error: {e}")
                            return
                        file_users = _load_file_users_only()
                        file_users[u] = {
                            "email": em,
                            "name": u,
                            "password": hp,
                        }
                        err = _save_auth_users_json(file_users)
                        if err:
                            st.error(f"Could not save: {err}")
                        else:
                            st.success(f"Saved **{u}** — they can sign in with username/password.")
                            st.rerun()


def render_auth_sidebar(authenticator: Any) -> None:
    st.caption(f"Signed in: **{_auth_display_name()}**")
    if _auth_logged_in():
        authenticator.logout("Log out", "sidebar")
    st.divider()


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="📊")
    st.markdown(MILLENNIALITY_CSS, unsafe_allow_html=True)

    _, authenticator = _build_authenticator()

    if not _auth_logged_in():
        render_login_screen(authenticator)
        st.stop()

    keys = _env_keys()
    if not keys["fmp"]:
        st.warning(
            "**FMP_API_KEY** is missing — discovery uses **Yahoo** screeners only; P/E uses **yfinance** + SEC fallbacks. "
            "Add FMP back if your quota returns."
        )

    with st.sidebar:
        render_auth_sidebar(authenticator)
        render_user_management_sidebar()
        st.caption(
            "API keys: **`.env`** locally, or top-level keys in **Streamlit Cloud → Secrets** (e.g. `FMP_API_KEY = \"...\"`). "
            "Nothing is entered in the sidebar."
        )
        st.divider()
        st.caption(
            f"Scan cache: **{SCAN_CACHE_TTL_SEC // 60} min** TTL · "
            f"Shown names: **actionable** (not Skip) and score ≥ **{MIN_DISPLAY_SCORE}**"
        )

    st.markdown('<span class="millenniallity-badge">@millenniallity</span>', unsafe_allow_html=True)
    st.title("Millenniallity Volatility Mismatch Scanner")
    st.markdown(
        "**Pure discovery:** each run pulls a fresh **liquid** universe from **FMP** (or **Yahoo** if FMP list APIs "
        "are unavailable on your plan) and scores what to **investigate next** — not a watchlist you maintain here. "
        "**Rich** P/E vs peers → **Buy Puts** bias; **cheap** mismatch → calls / stock. Air-gapped for **Public.com**."
    )

    burry_mode = st.checkbox(
        "Burry Mode",
        value=False,
        help=(
            "Uses the Top 150 high-beta discovery universe and strengthens short-thesis signals "
            "(extreme vs blended P/E, insider distribution, Bearish LLM)."
        ),
    )
    st.caption(
        "**Burry Mode:** Strengthens signals for extreme overvalued shorts and deep value longs per Michael Burry's "
        "Cassandra Unchained approach."
    )

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

    if burry_mode:
        st.success(
            f"**Burry Mode on** — scanning **Top {BURRY_UNIVERSE_LIMIT}** high-beta discovery names (no manual ticker list)."
        )

    uni_limit = BURRY_UNIVERSE_LIMIT if burry_mode else VOLATILE_UNIVERSE_LIMIT
    wide, universe_source, fmp_universe_debug, fmp_universe_had_symbols, fmp_universe_n = (
        build_discovery_universe(fmp_key, uni_limit)
    )
    if not wide:
        st.error(
            "**No discovery symbols.** FMP **stable company-screener** is often **402** (paid) on lower tiers; "
            "**v3** actives/gainers can return **403** (“Legacy Endpoint”) for keys issued after Aug 2025 — so the "
            "symbol list can be empty even with a working key. We also tried **Yahoo Finance** predefined screeners "
            "(check network if still empty). Per-ticker FMP routes like **stable/ratios** may still work; upgrade "
            "FMP or use a grandfathered key for full list APIs. "
            "[Pricing](https://site.financialmodelingprep.com/developer/docs/pricing)"
        )
        st.stop()
    if universe_source.startswith("yahoo"):
        st.info(
            f"**Universe source:** `{universe_source}` — FMP did not return a symbol list (screener paywall or "
            "legacy list block). Using **Yahoo** liquid movers for discovery; **FMP** still powers ratios, peers, "
            "and filings where your key allows."
        )
    elif "stock_market" in universe_source:
        st.info(
            f"**Universe source:** `{universe_source}` — company **screener** returned no rows for your key/filters; "
            "using **high-liquidity movers** instead (still fine for discovery). "
            "For strict high-beta lists, check your FMP plan or API playground."
        )
    universe = sorted(set(wide))
    n_universe = len(universe)

    progress = st.progress(0.0, text="Starting scan…")
    live_stats = st.empty()
    results: list[ScanRow] = []

    for i, tick in enumerate(universe):
        sec = sector_for_ticker(tick, {})
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
                    burry_mode=burry_mode,
                )
            )
        except Exception as e:
            results.append(
                ScanRow(
                    ticker=tick.upper(),
                    sector=sector_for_ticker(tick, {}),
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
                    burry_fit="",
                    errors=[str(e)],
                )
            )

        x_v, y_pe, z_llm, w_cur = _running_scan_counters(results, burry_mode=burry_mode)
        progress.progress(
            (i + 1) / max(1, n_universe),
            text=(
                f"Scanning {i + 1}/{n_universe} — {tick} | "
                f"Passed so far: {x_v}"
            ),
        )
        live_stats.caption(
            f"Passed volatility: **{x_v}** | P/E lookup OK: **{y_pe}** | LLM done: **{z_llm}** | "
            f"Current results: **{w_cur}** (meets score ≥ {MIN_DISPLAY_SCORE})"
        )

    progress.empty()
    live_stats.empty()

    actionable_pool = [r for r in results if is_actionable_recommendation(r)]
    filtered = [r for r in actionable_pool if r.mismatch_score >= MIN_DISPLAY_SCORE]
    filtered.sort(key=lambda r: _burry_results_sort_key(r, burry_mode))

    total_scanned = len(results)
    n_vol_pass = sum(1 for r in results if r.vol_ok)
    n_pe_ok = sum(1 for r in results if r.fwd_pe is not None)
    n_llm_ok = _llm_summary_count_ok(results)
    n_final = len(filtered)
    n_fail_vol = sum(1 for r in results if not r.vol_ok)
    n_fail_pe = sum(1 for r in results if r.fwd_pe is None)
    n_skip_all = sum(1 for r in results if r.recommendation == "Skip")
    n_actionable = len(actionable_pool)
    n_below_score = max(0, n_actionable - n_final)
    err_lines = _collect_per_ticker_error_lines(results)

    with st.container(border=True):
        st.markdown("### Scan Complete")
        st.caption(
            "Final counts for this run (live counters above were cleared). "
            "See **Debug / Errors** for FMP traces and per-ticker notes."
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total tickers scanned", total_scanned)
        m2.metric("Passed volatility filter", n_vol_pass)
        m3.metric("Successful forward P/E lookups", n_pe_ok)
        m4.metric("LLM summaries completed (xAI)", n_llm_ok)
        m5.metric("Final results displayed", n_final)
        if not filtered:
            st.warning(
                f"**No stocks passed all filters this run** (minimum score **{MIN_DISPLAY_SCORE}** after actionable filter). "
                f"Counts: **{total_scanned}** scanned · **{n_vol_pass}** passed volatility · **{n_pe_ok}** forward P/E OK · "
                f"**{n_llm_ok}** LLM summaries · **0** final results."
            )
            st.info(
                "**Suggestions:** Try **Burry Mode OFF** (narrower universe, different names), **Burry Mode ON** "
                f"(Top {BURRY_UNIVERSE_LIMIT}), lower **`MIN_DISPLAY_SCORE`** in code, or verify your **FMP** key "
                "(discovery + ratios), **SEC** / **xAI** keys, and the **Debug / Errors** section below."
            )

    summary_records = []
    for r in filtered:
        rec = {
            "Ticker": r.ticker,
            "Mismatch Score": round(r.mismatch_score, 1),
            "Recommendation": r.recommendation,
            "Vs Peers": r.valuation_skew,
            "Fwd P/E": r.fwd_pe,
            "Vol Reason": r.vol_reason,
            "LLM Flag": r.llm_flag,
        }
        if burry_mode:
            rec["Burry Fit"] = r.burry_fit
        summary_records.append(rec)
    sum_df = pd.DataFrame(summary_records) if summary_records else pd.DataFrame()

    st.subheader("Results overview (actionable only)")
    if sum_df.empty:
        st.caption("No rows to show — see note below.")
    else:
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download CSV (actionable & score-filtered)",
        data=filtered_to_csv(filtered, burry_mode=burry_mode),
        file_name=f"millenniallity_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    st.subheader(
        f"Detail — actionable, score ≥ {MIN_DISPLAY_SCORE} "
        f"({'no rows' if not filtered else len(filtered)} names)"
    )
    if not filtered:
        if not results:
            st.warning("Scan produced no rows.")
        elif not actionable_pool:
            st.warning(
                "**No actionable ideas** — every name was rated **Skip** (volatility / P&E / narrative gates). "
                "Re-run later or adjust scoring in code if you want more signals."
            )
        else:
            st.warning(
                f"**{len(actionable_pool)}** non-Skip name(s) below score **{MIN_DISPLAY_SCORE}**; "
                "raising the cutoff in code would show them (not recommended unless you widen mismatch rules)."
            )

    for r in filtered:
        title = f"{r.ticker} · {r.mismatch_score:.0f} · {r.recommendation}"
        if burry_mode and r.recommendation == "Burry Put Candidate":
            title = "🎯 " + title
        auto_expand = r.mismatch_score >= HIGH_SCORE_EXPAND_THRESHOLD or (
            r.score_insider_bonus > 0 and r.pe_mismatch_30
        )
        if burry_mode and r.recommendation == "Burry Put Candidate":
            auto_expand = True
        with st.expander(title, expanded=auto_expand):
            if burry_mode and r.burry_fit and r.burry_fit.strip() not in ("", "—"):
                st.markdown(f"**Burry Fit:** {r.burry_fit}")
            if burry_mode and r.recommendation == "Burry Put Candidate":
                st.warning(
                    "**Burry Put Candidate** — classic setup: **very rich** forward P/E vs blended peer/sector bench "
                    "(e.g. **Carvana ~46×** vs **~18–20×** peer context) **plus** insider distribution / owner cash-out "
                    "and a **Bearish** narrative flag."
                )
            c1, c2 = st.columns([1, 1])
            with c1:
                fig = pe_bar_figure(r)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Not enough peer P/E data to chart.")
                render_peer_ticker_labels(r)
                uw_fig, uw_err = uw_price_chart_figure_cached(r.ticker, uw_key)
                if uw_fig:
                    st.plotly_chart(uw_fig, use_container_width=True)
                elif uw_err:
                    st.caption(f"_Unusual Whales 1Y price: {uw_err}_")
                st.metric("Stock P/E vs benchmark", f"{r.fwd_pe} vs {r.peer_avg_pe}")
                st.caption(
                    f"Peer median **{r.peer_median_pe}** · Sector bench **{r.sector_benchmark_pe}** · _{r.fwd_pe_source}_"
                )
                if r.valuation_skew == "rich":
                    st.warning("**Vs peers:** **rich** multiple — default trade bias is **long puts** / short-vol on euphoric pricing.")
                elif r.valuation_skew == "cheap":
                    st.info("**Vs peers:** **cheap** — explore **calls** or stock if the story holds (watch value traps).")
            with c2:
                st.markdown("**LLM (10-K vs valuation)**")
                st.write(r.llm_summary or "—")
                st.caption(
                    f"Vs peers: **{r.valuation_skew}** · Flag: **{r.llm_flag}** · Contradiction: **{r.narrative_contradiction}**"
                )
                st.markdown(f"### Trade: **{r.recommendation}**")
                if r.score_insider_bonus:
                    st.success(f"Insider / valuation alignment bonus: **+{r.score_insider_bonus:.0f}** pts.")
                st.markdown("**Options (1–4 week Public.com)**")
                st.write(r.option_suggestion)
                st.markdown("**Unusual Whales**")
                st.markdown(r.unusual_whales_note or "—")
                st.caption(
                    f"Flow: net {r.insider_net_shares:,.0f} · buys {r.insider_buy_shares:,.0f} · sells {r.insider_sell_shares:,.0f}"
                )
            if r.errors:
                st.error("Notes: " + "; ".join(r.errors))
            render_grok_followup_chat(r.ticker, r, xai_key)

    with st.expander("Debug / Errors", expanded=(n_final == 0)):
        st.markdown("**Discovery: `fmp_high_beta_universe`**")
        st.markdown(
            f"- **Returned any symbols:** **{'Yes' if fmp_universe_had_symbols else 'No'}** "
            f"(count from FMP before Yahoo fallback: **{fmp_universe_n}**)\n"
            f"- **Universe source used for scan:** `{universe_source}` · **unique tickers scanned:** {total_scanned}"
        )
        st.markdown("**FMP HTTP trace (high-beta / movers attempts)**")
        st.code(fmp_universe_debug or "(empty)", language=None)
        st.markdown("**Stocks failing each major step** (this run)")
        st.markdown(
            f"- **Volatility filter failed:** {n_fail_vol} / {total_scanned}\n"
            f"- **Forward P/E missing:** {n_fail_pe} / {total_scanned}\n"
            f"- **Recommendation = Skip:** {n_skip_all} / {total_scanned}\n"
            f"- **Actionable but below score cutoff ({MIN_DISPLAY_SCORE}):** {n_below_score} "
            f"(non-Skip actionable: **{n_actionable}**)"
        )
        if err_lines:
            st.markdown("**Per-ticker errors / notes**")
            for line in err_lines:
                st.markdown(f"- {line}")
        else:
            st.info(
                "No per-ticker error strings were recorded. If results are empty, the pipeline may have "
                "skipped names for scoring rules (not API errors)."
            )

    st.markdown(
        '<div class="mill-footer">Built for Public.com options edge — volatile P/E mismatches only</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
