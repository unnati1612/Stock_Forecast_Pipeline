"""
train.py
========
Pulls accumulated price history from Postgres, builds features, trains a
baseline (logistic regression) and a stronger model (XGBoost) to predict
next-day price direction, evaluates both with a proper time-based
train/test split (NOT random — that would leak future info into training),
and logs the newest prediction into the 'predictions' table for tracking.

Usage:
    python train.py
"""

import os
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from dotenv import load_dotenv

from features import build_features, FEATURE_COLUMNS

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
TICKERS = os.environ.get("TICKERS", "AAPL,MSFT,GOOGL,AMZN,NVDA").split(",")

MODEL_VERSION = f"xgboost_v1_{datetime.utcnow().date().isoformat()}"


def load_prices(engine, ticker: str) -> pd.DataFrame:
    query = text("""
        SELECT date, open, high, low, close, volume
        FROM prices
        WHERE ticker = :ticker
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Split chronologically, not randomly — the last test_frac of dates
    become the test set. Random splitting on time series data leaks
    future information into training and gives falsely optimistic results.
    """
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_for_ticker(engine, ticker: str):
    print(f"\n=== {ticker} ===")
    raw = load_prices(engine, ticker)
    if len(raw) < 60:
        print(f"  [skip] only {len(raw)} rows available, need 60+ for reliable features/training")
        return None

    feat = build_features(raw)
    model_df = feat.dropna(subset=FEATURE_COLUMNS + ["target_direction"])

    train_df, test_df = time_based_split(model_df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_direction"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_direction"]

    # Baseline: naive "always predict majority class" for comparison
    majority_class = y_train.mode()[0]
    baseline_acc = (y_test == majority_class).mean()

    # Baseline model: logistic regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_acc = accuracy_score(y_test, logreg.predict(X_test))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # Main model: XGBoost
    xgb = XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        min_child_weight=5,      # requires more samples per leaf before splitting further
        subsample=0.7,           # each tree trains on a random 70% of rows
        colsample_bytree=0.7,    # each tree considers a random 70% of features
        reg_alpha=0.1,           # L1 regularization, encourages simpler trees
        reg_lambda=1.0,          # L2 regularization, shrinks leaf weights
        eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

    print(f"  Naive baseline accuracy:     {baseline_acc:.3f}")
    print(f"  Logistic regression accuracy:{logreg_acc:.3f}")
    print(f"  Random Forest accuracy:      {rf_acc:.3f}")
    print(f"  XGBoost accuracy:            {xgb_acc:.3f}")

    metrics = {
        "Logistic Regression": logreg_acc,
        "Random Forest": rf_acc,
        "XGBoost": xgb_acc
    }

    log_model_metrics(engine, ticker, metrics)

    # Refit XGBoost on ALL available data (not just train split) for the
    # actual live prediction, since we want to use every data point we have
    xgb_full = XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42,
    )
    xgb_full.fit(model_df[FEATURE_COLUMNS], model_df["target_direction"])

    # Predict for the MOST RECENT row that has valid features but whose
    # target_direction is NaN (i.e. tomorrow hasn't happened yet)
    latest_row = feat.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    pred_prob = xgb_full.predict_proba(latest_row[FEATURE_COLUMNS])[0][1]
    pred_direction = int(pred_prob > 0.5)
    prediction_date = latest_row["date"].iloc[0]
    target_date = (
        pd.Timestamp(prediction_date) + BDay(1)
    ).date()

    return {
        "ticker": ticker,
        "prediction_date": prediction_date,
        "target_date": target_date,
        "predicted_direction": pred_direction,
        "predicted_prob": round(float(pred_prob), 4),
        "model_version": MODEL_VERSION,
        "test_accuracy": round(float(xgb_acc), 4),
        "baseline_accuracy": round(float(baseline_acc), 4),
    }


def log_prediction(engine, pred: dict):
    insert_sql = text("""
        INSERT INTO predictions
            (ticker, prediction_date, target_date, predicted_direction, predicted_prob, model_version)
        VALUES
            (:ticker, :prediction_date, :target_date, :predicted_direction, :predicted_prob, :model_version)
        ON CONFLICT (ticker, target_date, model_version) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(insert_sql, {k: pred[k] for k in
                     ["ticker", "prediction_date", "target_date", "predicted_direction",
                      "predicted_prob", "model_version"]})


def backfill_actuals(engine):
    """
    For past predictions whose target_date has now happened, look up the
    real outcome and fill in actual_direction / correct. Run this before
    making new predictions so the tracking table stays up to date.
    """
    update_sql = text("""
        UPDATE predictions p
        SET
            actual_direction = CASE WHEN price_change.direction > 0 THEN 1 ELSE 0 END,
            correct = (p.predicted_direction = CASE WHEN price_change.direction > 0 THEN 1 ELSE 0 END)
        FROM (
            SELECT
                curr.ticker,
                curr.date AS target_date,
                curr.close - prev.close AS direction
            FROM prices curr
            JOIN prices prev
                ON curr.ticker = prev.ticker
                AND prev.date = (
                    SELECT MAX(date) FROM prices p2
                    WHERE p2.ticker = curr.ticker AND p2.date < curr.date
                )
        ) price_change
        WHERE p.ticker = price_change.ticker
          AND p.target_date = price_change.target_date
          AND p.actual_direction IS NULL
    """)
    with engine.begin() as conn:
        result = conn.execute(update_sql)
    print(f"Backfilled {result.rowcount} prediction outcome(s).")

def log_model_metrics(engine, ticker, metrics):
    insert_sql = text("""
        INSERT INTO model_metrics
            (ticker, model_name, accuracy, model_version)
        VALUES
            (:ticker, :model_name, :accuracy, :model_version)
    """)

    with engine.begin() as conn:
        for model_name, accuracy in metrics.items():
            conn.execute(
                insert_sql,
                {
                    "ticker": ticker,
                    "model_name": model_name,
                    "accuracy": float(accuracy),
                    "model_version": MODEL_VERSION
                }
            )

def main():
    if not DATABASE_URL:
        raise SystemExit("ERROR: DATABASE_URL not set. Copy .env.example to .env and fill it in.")

    engine = create_engine(DATABASE_URL)

    print("Backfilling outcomes for past predictions...")
    backfill_actuals(engine)

    print("\nTraining models and generating new predictions...")
    for ticker in TICKERS:
        ticker = ticker.strip()
        result = train_for_ticker(engine, ticker)
        if result:
            log_prediction(engine, result)
            print(f"  -> logged prediction for {result['target_date']}: "
                  f"{'UP' if result['predicted_direction'] else 'DOWN'} "
                  f"(confidence {result['predicted_prob']:.2%})")


if __name__ == "__main__":
    main()
