# Stock Direction Forecast Tracker

A self-updating forecasting pipeline that ingests daily stock prices, trains a
next-day price-direction model, and honestly tracks its own prediction
accuracy over time — live, on a public dashboard.

**Live dashboard:** _add your Streamlit Cloud link here once deployed_

## Why this project

Most portfolio ML projects train a model once on a static CSV and report a
single accuracy number. This project instead builds a small **production-style
pipeline**: data keeps arriving automatically, the model retrains on a
schedule, and every prediction is logged *before* the outcome is known —
so the accuracy shown is a genuine track record, not a backtest that's easy
to accidentally overfit.

Predicting next-day stock direction is genuinely hard (markets are close to
efficient) — the goal here isn't to "beat the market," it's to demonstrate
a correctly-built forecasting pipeline with honest evaluation.

## Architecture

```
GitHub Actions (daily cron)
        │
        ▼
   ingest.py ──────► Postgres (Neon) ──────► train.py (weekly retrain)
   [yfinance]           │  prices                │  predictions logged
                         │  predictions           ▼
                         └──────────────► app.py (Streamlit dashboard)
```

## Stack

- **Data source:** [yfinance](https://github.com/ranaroussi/yfinance) (free, no API key)
- **Database:** Postgres, hosted free on [Neon](https://neon.tech)
- **Modeling:** scikit-learn (baseline) + XGBoost (main model)
- **Automation:** GitHub Actions (scheduled cron)
- **Dashboard:** Streamlit + Plotly

## Setup

1. **Create a free Postgres database** at [neon.tech](https://neon.tech) → new project → copy the connection string.
2. **Run the schema:** paste the contents of `schema.sql` into Neon's SQL editor and run it.
3. **Clone this repo and install dependencies:**
   ```bash
   git clone <your-repo-url>
   cd stock-forecast-pipeline
   pip install -r requirements.txt
   ```
4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # edit .env with your real DATABASE_URL and desired TICKERS
   ```
5. **Backfill historical data (one-time):**
   ```bash
   python ingest.py --full-history
   ```
6. **Train the first model and log a prediction:**
   ```bash
   python train.py
   ```
7. **Run the dashboard locally:**
   ```bash
   streamlit run app.py
   ```

## Automating it (GitHub Actions)

1. Push this repo to GitHub.
2. Go to repo Settings → Secrets and variables → Actions:
   - Add secret `DATABASE_URL` = your Neon connection string
   - Add variable `TICKERS` = e.g. `AAPL,MSFT,GOOGL,AMZN,NVDA`
3. The workflow in `.github/workflows/pipeline.yml` runs automatically on
   weekdays after market close. You can also trigger it manually from the
   Actions tab ("Run workflow").

## Deploying the dashboard

1. Push to GitHub (if not already).
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → New app → point at this repo, `app.py`.
3. In the app's Settings → Secrets, add:
   ```toml
   DATABASE_URL = "postgresql://..."
   ```

## Results

_Fill this in once you have a few weeks of tracked predictions:_

- Model: XGBoost, next-day direction classification
- Test accuracy vs. naive baseline: `__%` vs `__%`
- Live tracked accuracy (cumulative, as of `<date>`): `__%` over `N` predictions

## Limitations & honest caveats

- Daily stock direction is close to a coin flip in efficient markets;
  small edges over 50% are the realistic target, not 70-80%.
- Feature set is intentionally simple (price/volume technicals only) — no
  fundamentals, news sentiment, or macro data.
- Small ticker universe and short live-tracking window mean the accuracy
  numbers above will have wide uncertainty until more predictions accumulate.
