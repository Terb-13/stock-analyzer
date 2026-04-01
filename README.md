# Millenniallity Volatility Mismatch Scanner

Streamlit app: volatile equities where forward P/E diverges from peers, sector context, and 10-K narrative (xAI). See `app.py` for details.

## Setup

```bash
cp .env.example .env
# Edit .env with your API keys (never commit .env)

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Required keys: **FMP**, **SEC** (sec-api), **xAI**; optional **Unusual Whales**, **Polygon**.
