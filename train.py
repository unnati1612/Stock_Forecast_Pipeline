"""
train.py - Stock direction forecasting pipeline.

Trains Logistic Regression, Random Forest, and XGBoost using a chronological
split, logs their test accuracies, selects the best model, refits it on all
labelled data, and creates one prediction per ticker/target trading day.
"""

import os
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import BDay
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from dotenv import load_dotenv
from features import build_features, FEATURE_COLUMNS

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
TICKERS = [
    x.strip() for x in os.environ.get(
        "TICKERS", "AAPL,MSFT,GOOGL,AMZN,NVDA"
    ).split(",") if x.strip()
]
RUN_DATE = datetime.utcnow().date().isoformat()


def load_prices(engine, ticker):
    query = text("""
        SELECT date, open, high, low, close, volume
        FROM prices
        WHERE ticker = :ticker
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"ticker": ticker})


def time_based_split(df, test_frac=0.20):
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def make_logreg():
    return LogisticRegression(max_iter=1000, random_state=42)


def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_split=10,
        min_samples_leaf=5, max_features="sqrt",
        random_state=42, n_jobs=-1
    )


def make_xgb():
    return XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42
    )


def log_model_metrics(engine, ticker, metrics):
    sql = text("""
        INSERT INTO model_metrics
            (ticker, model_name, accuracy, model_version)
        VALUES
            (:ticker, :model_name, :accuracy, :model_version)
    """)
    with engine.begin() as conn:
        for name, accuracy in metrics.items():
            version = f"{name.lower().replace(' ', '_')}_v1_{RUN_DATE}"
            conn.execute(sql, {
                "ticker": ticker,
                "model_name": name,
                "accuracy": float(accuracy),
                "model_version": version
            })


def train_for_ticker(engine, ticker):
    print(f"\n=== {ticker} ===")

    raw = load_prices(engine, ticker)

    if len(raw) < 60:
        print(f"[skip] only {len(raw)} rows; need at least 60")
        return None

    feat = build_features(raw)

    model_df = feat.dropna(
        subset=FEATURE_COLUMNS + ["target_direction"]
    ).copy()

    if len(model_df) < 50:
        print(f"[skip] only {len(model_df)} usable training rows")
        return None

    train_df, test_df = time_based_split(model_df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target_direction"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target_direction"]

    majority_class = y_train.mode()[0]
    baseline_acc = accuracy_score(
        y_test, [majority_class] * len(y_test)
    )

    models = {
        "Logistic Regression": make_logreg(),
        "Random Forest": make_rf(),
        "XGBoost": make_xgb()
    }

    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = float(accuracy_score(y_test, pred))

    print(f"Naive baseline:       {baseline_acc:.3f}")
    for name, acc in metrics.items():
        print(f"{name:<22} {acc:.3f}")

    best_model_name = max(metrics, key=metrics.get)
    best_accuracy = metrics[best_model_name]

    print(f"BEST MODEL:           {best_model_name}")
    print(f"BEST TEST ACCURACY:   {best_accuracy:.3%}")

    log_model_metrics(engine, ticker, metrics)

    # Refit ONLY the best-performing model on all labelled history.
    factories = {
        "Logistic Regression": make_logreg,
        "Random Forest": make_rf,
        "XGBoost": make_xgb
    }

    best_model = factories[best_model_name]()
    best_model.fit(
        model_df[FEATURE_COLUMNS],
        model_df["target_direction"]
    )

    # Latest row with valid features is the row used to predict the future.
    latest_candidates = feat.dropna(subset=FEATURE_COLUMNS)

    if latest_candidates.empty:
        print("[skip] no valid feature row for prediction")
        return None

    latest_row = latest_candidates.iloc[[-1]]

    prediction_date = pd.Timestamp(
        latest_row["date"].iloc[0]
    ).date()

    # Next BUSINESS day, not simply prediction_date + 1 calendar day.
    target_date = (
        pd.Timestamp(prediction_date) + BDay(1)
    ).date()

    pred_prob = best_model.predict_proba(
        latest_row[FEATURE_COLUMNS]
    )[0][1]

    pred_direction = int(pred_prob >= 0.5)

    model_version = (
        f"{best_model_name.lower().replace(' ', '_')}"
        f"_v1_{RUN_DATE}"
    )

    print(f"Prediction model:     {best_model_name}")
    print(f"Prediction date:      {prediction_date}")
    print(f"Target date:          {target_date}")
    print(f"Prediction:           {'UP' if pred_direction else 'DOWN'}")
    print(f"Confidence:            {pred_prob:.2%}")

    return {
        "ticker": ticker,
        "prediction_date": prediction_date,
        "target_date": target_date,
        "predicted_direction": pred_direction,
        "predicted_prob": round(float(pred_prob), 4),
        "model_name": best_model_name,
        "model_version": model_version
    }


def log_prediction(engine, pred):
    # First check prevents the duplicate rows you created while testing.
    check_sql = text("""
        SELECT id
        FROM predictions
        WHERE ticker = :ticker
          AND target_date = :target_date
        LIMIT 1
    """)

    with engine.begin() as conn:
        existing = conn.execute(check_sql, {
            "ticker": pred["ticker"],
            "target_date": pred["target_date"]
        }).fetchone()

        if existing:
            print(
                f"[skip] prediction already exists for "
                f"{pred['ticker']} -> {pred['target_date']} "
                f"(id={existing[0]})"
            )
            return False

        # Your current schema may not yet have model_name.
        has_model_name = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'predictions'
                  AND column_name = 'model_name'
            )
        """)).scalar()

        if has_model_name:
            sql = text("""
                INSERT INTO predictions
                    (ticker, prediction_date, target_date, model_name,
                     predicted_direction, predicted_prob, model_version)
                VALUES
                    (:ticker, :prediction_date, :target_date, :model_name,
                     :predicted_direction, :predicted_prob, :model_version)
            """)
            conn.execute(sql, pred)
        else:
            sql = text("""
                INSERT INTO predictions
                    (ticker, prediction_date, target_date,
                     predicted_direction, predicted_prob, model_version)
                VALUES
                    (:ticker, :prediction_date, :target_date,
                     :predicted_direction, :predicted_prob, :model_version)
            """)
            conn.execute(sql, {
                "ticker": pred["ticker"],
                "prediction_date": pred["prediction_date"],
                "target_date": pred["target_date"],
                "predicted_direction": pred["predicted_direction"],
                "predicted_prob": pred["predicted_prob"],
                "model_version": pred["model_version"]
            })

    print(f"Logged prediction for {pred['ticker']} -> {pred['target_date']}")
    return True


def backfill_actuals(engine):
    sql = text("""
        UPDATE predictions p
        SET
            actual_direction = CASE
                WHEN price_change.direction > 0 THEN 1 ELSE 0
            END,
            correct = (
                p.predicted_direction =
                CASE
                    WHEN price_change.direction > 0 THEN 1 ELSE 0
                END
            )
        FROM (
            SELECT
                curr.ticker,
                curr.date AS target_date,
                curr.close - prev.close AS direction
            FROM prices curr
            JOIN prices prev
              ON curr.ticker = prev.ticker
             AND prev.date = (
                SELECT MAX(p2.date)
                FROM prices p2
                WHERE p2.ticker = curr.ticker
                  AND p2.date < curr.date
             )
        ) price_change
        WHERE p.ticker = price_change.ticker
          AND p.target_date = price_change.target_date
          AND p.actual_direction IS NULL
    """)

    with engine.begin() as conn:
        result = conn.execute(sql)

    print(f"Backfilled {result.rowcount} prediction outcome(s).")


def main():
    if not DATABASE_URL:
        raise SystemExit(
            "ERROR: DATABASE_URL not set. Add it to your .env file."
        )

    engine = create_engine(DATABASE_URL)

    print("=" * 60)
    print("STOCK FORECAST PIPELINE")
    print("=" * 60)

    print("\n1. Resolving previous predictions...")
    backfill_actuals(engine)

    print("\n2. Training models and generating predictions...")

    for ticker in TICKERS:
        result = train_for_ticker(engine, ticker)

        if result:
            log_prediction(engine, result)

    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()
