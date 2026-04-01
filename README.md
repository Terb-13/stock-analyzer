# Millenniallity Volatility Mismatch Scanner

Streamlit app: auto-discovers volatile equities where forward P/E diverges from peers / sector and runs a 10-K narrative check via **xAI**. Keys are read **only** from `.env` / environment (nothing is typed in the UI).

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
