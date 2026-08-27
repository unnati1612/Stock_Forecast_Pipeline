-- =============================================================
-- Schema for the Stock/Crypto Forecasting Pipeline project
-- Run this once against your Neon Postgres database to set up
-- the tables the ingestion script and dashboard depend on.
-- =============================================================

CREATE TABLE IF NOT EXISTS prices (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(10)     NOT NULL,
    date        DATE            NOT NULL,
    open        NUMERIC(12, 4),
    high        NUMERIC(12, 4),
    low         NUMERIC(12, 4),
    close       NUMERIC(12, 4)  NOT NULL,
    volume      BIGINT,
    inserted_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, date)  -- prevents duplicate rows if the daily job reruns
);

CREATE TABLE IF NOT EXISTS predictions (
    id                SERIAL PRIMARY KEY,
    ticker            VARCHAR(10)  NOT NULL,
    prediction_date   DATE         NOT NULL,   -- the day the prediction was made
    target_date       DATE         NOT NULL,   -- the day being predicted (usually prediction_date + 1)
    predicted_direction SMALLINT,              -- 1 = up, 0 = down
    predicted_prob    NUMERIC(5, 4),           -- model's confidence, e.g. 0.63
    actual_direction  SMALLINT,                -- filled in once target_date's data arrives
    correct           BOOLEAN,                 -- filled in once actual_direction is known
    model_version     VARCHAR(50),             -- e.g. 'xgboost_v1_2026-08-10'
    created_at        TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, target_date, model_version)
);

-- Indexes to keep the dashboard's queries fast as data accumulates
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices (ticker, date);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date ON predictions (ticker, target_date);
