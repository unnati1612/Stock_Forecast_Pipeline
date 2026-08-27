"""
app.py
======
Streamlit dashboard for the live stock direction forecasting project.

Shows:
- Latest closing price
- Current next-day prediction
- Running prediction accuracy
- Historical price chart
- Cumulative accuracy over time
- Prediction log
- Logistic Regression vs Random Forest vs XGBoost comparison

Run locally with:
    streamlit run app.py
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL") or st.secrets.get(
    "DATABASE_URL", ""
)

TICKERS = [
    ticker.strip()
    for ticker in os.environ.get(
        "TICKERS",
        "AAPL,MSFT,GOOGL,AMZN,NVDA"
    ).split(",")
]

st.set_page_config(
    page_title="Stock Direction Forecast Tracker",
    layout="wide"
)


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

@st.cache_resource
def get_engine():
    if not DATABASE_URL:
        st.error(
            "DATABASE_URL not configured. "
            "Set it in .env locally or Streamlit secrets when deployed."
        )
        st.stop()

    return create_engine(DATABASE_URL)


# --------------------------------------------------
# DATA LOADING FUNCTIONS
# --------------------------------------------------

@st.cache_data(ttl=3600)
def load_prices(ticker: str) -> pd.DataFrame:
    engine = get_engine()

    query = text("""
        SELECT date, close
        FROM prices
        WHERE ticker = :ticker
        ORDER BY date ASC
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"ticker": ticker}
        )


@st.cache_data(ttl=3600)
def load_predictions(ticker: str) -> pd.DataFrame:
    engine = get_engine()

    query = text("""
        SELECT
            prediction_date,
            target_date,
            predicted_direction,
            predicted_prob,
            actual_direction,
            correct
        FROM predictions
        WHERE ticker = :ticker
        ORDER BY target_date ASC
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"ticker": ticker}
        )


@st.cache_data(ttl=3600)
def load_model_metrics(ticker: str) -> pd.DataFrame:
    engine = get_engine()

    query = text("""
        SELECT
            model_name,
            accuracy,
            evaluated_at,
            model_version
        FROM model_metrics
        WHERE ticker = :ticker
        ORDER BY evaluated_at DESC, accuracy DESC
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"ticker": ticker}
        )


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("📈 Stock Direction Forecast Tracker")

st.caption(
    "A live, self-updating pipeline: daily data ingestion, periodic model "
    "retraining, and an honest running accuracy log — predictions are logged "
    "before outcomes are known."
)


# --------------------------------------------------
# TICKER SELECTION
# --------------------------------------------------

ticker = st.selectbox(
    "Select ticker",
    TICKERS
)

prices = load_prices(ticker)
preds = load_predictions(ticker)
metrics_df = load_model_metrics(ticker)


# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------

st.subheader(f"{ticker} Overview")

col1, col2, col3, col4 = st.columns(4)

if not prices.empty:
    latest_price = prices.iloc[-1]["close"]
    latest_date = prices.iloc[-1]["date"]

    col1.metric(
        "Latest closing price",
        f"${latest_price:,.2f}",
        f"As of {latest_date}"
    )
else:
    col1.metric("Latest closing price", "—")


if not preds.empty:
    latest_pred = preds.iloc[-1]

    direction_label = (
        "📈 UP"
        if latest_pred["predicted_direction"] == 1
        else "📉 DOWN"
    )

    probability = latest_pred["predicted_prob"]

    col2.metric(
        "Latest prediction",
        direction_label,
        f"{probability:.1%} confidence"
    )

    resolved = preds.dropna(subset=["correct"])

    if not resolved.empty:
        running_accuracy = resolved["correct"].astype(float).mean()

        col3.metric(
            "Running accuracy",
            f"{running_accuracy:.1%}",
            f"{len(resolved)} resolved predictions"
        )
    else:
        col3.metric(
            "Running accuracy",
            "—",
            "No resolved predictions yet"
        )

else:
    col2.metric("Latest prediction", "—")
    col3.metric("Running accuracy", "—")


col4.metric(
    "Baseline",
    "50.0%",
    "Random guessing"
)


# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

st.divider()

st.subheader(f"🤖 Model Performance — {ticker}")
st.caption("Test accuracy using the chronological holdout set.")

if not metrics_df.empty:

    # If train.py stores multiple training runs, keep only the
    # latest result for each model.
    if "evaluated_at" in metrics_df.columns:
        metrics_df = (
            metrics_df
            .sort_values("evaluated_at", ascending=False)
            .drop_duplicates(subset=["model_name"], keep="first")
        )

    # Convert accuracy from 0-1 to percentage.
    metrics_df["accuracy"] = (
        metrics_df["accuracy"].astype(float) * 100
    ).round(2)

    # Sort highest-performing model first.
    metrics_df = metrics_df.sort_values(
        "accuracy",
        ascending=False
    ).reset_index(drop=True)

    best_model = metrics_df.iloc[0]["model_name"]
    best_accuracy = metrics_df.iloc[0]["accuracy"]

    best_col1, best_col2 = st.columns(2)

    best_col1.metric(
        "🏆 Best Model",
        best_model
    )

    best_col2.metric(
        "Best Test Accuracy",
        f"{best_accuracy:.2f}%"
    )

    # Display only the main comparison columns.
    display_metrics = metrics_df[
        ["model_name", "accuracy"]
    ].copy()

    display_metrics.columns = [
        "Model",
        "Test Accuracy (%)"
    ]

    st.dataframe(
        display_metrics,
        use_container_width=True,
        hide_index=True
    )

    fig_model = go.Figure(
        data=[
            go.Bar(
                x=metrics_df["model_name"],
                y=metrics_df["accuracy"],
                text=metrics_df["accuracy"].apply(
                    lambda x: f"{x:.2f}%"
                ),
                textposition="outside"
            )
        ]
    )

    fig_model.update_layout(
        title="Model Test Accuracy Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy (%)",
        yaxis=dict(
            range=[0, 60],
            ticksuffix="%"
        ),
        height=400
    )

    fig_model.add_hline(
    y=50,
    line_dash="dash",
    annotation_text="50% baseline",
    annotation_position="bottom right"
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True
    )

else:
    st.info(
        f"No model metrics found for {ticker}. "
        "Run train.py after adding Logistic Regression, Random Forest, "
        "and XGBoost metric logging."
    )


# --------------------------------------------------
# HISTORICAL PRICE CHART
# --------------------------------------------------

st.divider()

st.subheader(f"📊 {ticker} Closing Price")

if not prices.empty:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prices["date"],
            y=prices["close"],
            mode="lines",
            name="Close Price"
        )
    )

    fig.update_layout(
        title=f"{ticker} — Historical Closing Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.warning(
        "No price data found. Run ingest.py first."
    )


# --------------------------------------------------
# RUNNING ACCURACY CHART
# --------------------------------------------------

st.divider()

st.subheader("🎯 Prediction Track Record")

if not preds.empty:

    resolved = preds.dropna(
        subset=["correct"]
    ).copy()

    if not resolved.empty:

        resolved["correct"] = (
            resolved["correct"].astype(float)
        )

        resolved["cumulative_accuracy"] = (
            resolved["correct"].expanding().mean()
        )

        fig_accuracy = go.Figure()

        fig_accuracy.add_trace(
            go.Scatter(
                x=resolved["target_date"],
                y=resolved["cumulative_accuracy"],
                mode="lines+markers",
                name="Cumulative Accuracy"
            )
        )

        fig_accuracy.add_hline(
            y=0.5,
            line_dash="dash",
            annotation_text="50% baseline"
        )

        fig_accuracy.update_layout(
            title="Cumulative Prediction Accuracy Over Time",
            xaxis_title="Target Date",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1],
            height=400
        )

        st.plotly_chart(
            fig_accuracy,
            use_container_width=True
        )

    else:
        st.info(
            "Predictions have been generated, but none have been "
            "resolved yet. Accuracy will appear once actual outcomes "
            "are available."
        )

else:
    st.info(
        "No predictions logged yet. Run train.py to generate predictions."
    )


# --------------------------------------------------
# PREDICTION LOG
# --------------------------------------------------

st.divider()

st.subheader("📋 Prediction Log")

if not preds.empty:

    prediction_log = preds.copy()

    # Make direction columns easier to read.
    prediction_log["predicted_direction"] = (
        prediction_log["predicted_direction"]
        .map({
            1: "UP",
            0: "DOWN"
        })
        .fillna(prediction_log["predicted_direction"])
    )

    prediction_log["actual_direction"] = (
        prediction_log["actual_direction"]
        .map({
            1: "UP",
            0: "DOWN"
        })
        .fillna(prediction_log["actual_direction"])
    )

    prediction_log["correct"] = (
        prediction_log["correct"]
        .map({
            True: "✓ Correct",
            False: "✗ Incorrect"
        })
        .fillna("Pending")
    )

    prediction_log["predicted_prob"] = (
        prediction_log["predicted_prob"]
        .astype(float)
        .map(lambda x: f"{x:.1%}")
    )

    prediction_log = prediction_log.rename(
        columns={
            "prediction_date": "Prediction Date",
            "target_date": "Target Date",
            "predicted_direction": "Predicted Direction",
            "predicted_prob": "Confidence",
            "actual_direction": "Actual Direction",
            "correct": "Result"
        }
    )

    st.dataframe(
        prediction_log.sort_values(
            "Target Date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No predictions available yet.")
