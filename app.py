import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_trading_engine import Config, run_engine


st.set_page_config(
    page_title="AI Trading Engine",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Trading Engine v11")
st.caption(
    "Purged Walk-Forward • Fold-local Feature Selection • "
    "Expected Return • Portfolio Risk"
)

with st.sidebar:
    st.header("Universe")

    tickers = st.multiselect(
        "Assets",
        [
            "BTC-USD", "ETH-USD", "SOL-USD",
            "QQQ", "SPY",
            "NVDA", "MSFT", "AAPL",
            "AMZN", "META",
        ],
        default=[
            "BTC-USD", "ETH-USD", "SOL-USD",
            "QQQ", "NVDA", "MSFT",
        ],
    )

    benchmark = st.text_input(
        "Benchmark",
        "^GSPC",
    )

    st.header("Backtest")

    start = st.date_input(
        "Start",
        pd.Timestamp("2018-01-01"),
    )

    end = st.date_input(
        "End",
        pd.Timestamp.today(),
    )

    initial_capital = st.number_input(
        "Initial capital",
        min_value=1_000.0,
        max_value=10_000_000.0,
        value=100_000.0,
        step=5_000.0,
    )

    st.header("Model")

    classifier_trees = st.slider(
        "Classifier trees",
        100, 800, 350, 25,
    )

    regressor_trees = st.slider(
        "Regressor trees",
        100, 800, 350, 25,
    )

    learning_rate = st.slider(
        "Learning rate",
        0.01, 0.15, 0.035, 0.005,
    )

    max_depth = st.slider(
        "Model depth",
        2, 8, 5,
    )

    min_leaf = st.slider(
        "Min samples / leaf",
        10, 100, 25, 5,
    )

    st.header("Walk-Forward")

    train_days = st.number_input(
        "Training days",
        126, 1500, 504, 21,
    )

    validation_days = st.number_input(
        "Validation days",
        21, 252, 63, 21,
    )

    step_days = st.number_input(
        "Step days",
        21, 252, 63, 21,
    )

    purge_days = st.number_input(
        "Purge days",
        0, 30, 5,
    )

    embargo_days = st.number_input(
        "Embargo days",
        0, 30, 2,
    )

    st.header("Signal")

    horizon_days = st.slider(
        "Label horizon",
        2, 20, 5,
    )

    min_probability = st.slider(
        "Min probability",
        0.50, 0.80, 0.55, 0.01,
    )

    min_expected_return = st.number_input(
        "Min expected return",
        -0.05, 0.10, 0.004,
        0.001,
        format="%.3f",
    )

    top_signals = st.slider(
        "Top signals / day",
        1, 10, 3,
    )

    st.header("Risk")

    risk_per_trade = st.slider(
        "Base risk / trade",
        0.002, 0.04, 0.012, 0.001,
    )

    min_risk = st.slider(
        "Min risk / trade",
        0.001, 0.02, 0.003, 0.001,
    )

    max_risk = st.slider(
        "Max risk / trade",
        0.005, 0.06, 0.025, 0.001,
    )

    kelly_fraction = st.slider(
        "Kelly fraction",
        0.0, 1.0, 0.35, 0.05,
    )

    max_exposure = st.slider(
        "Max total exposure",
        0.10, 1.00, 0.85, 0.05,
    )

    max_position = st.slider(
        "Max single position",
        0.05, 0.50, 0.25, 0.01,
    )

    max_positions = st.slider(
        "Max open positions",
        1, 15, 5,
    )

    st.header("Exits / Costs")

    atr_stop = st.slider(
        "ATR stop",
        0.5, 4.0, 1.8, 0.1,
    )

    atr_target = st.slider(
        "ATR target",
        0.5, 8.0, 3.0, 0.1,
    )

    max_hold = st.slider(
        "Max hold days",
        2, 60, 15,
    )

    commission_bps = st.number_input(
        "Commission (bps)",
        0.0, 50.0, 5.0, 0.5,
    )

    slippage_bps = st.number_input(
        "Slippage (bps)",
        0.0, 100.0, 5.0, 0.5,
    )

    run = st.button(
        "🚀 Run Research Backtest",
        type="primary",
        use_container_width=True,
    )


if not tickers:
    st.warning("Select at least one asset.")
    st.stop()


if run:
    cfg = Config(
        tickers=tickers,
        benchmark=benchmark,
        start=str(start),
        end=str(end),
        initial_capital=float(initial_capital),
        classifier_trees=int(classifier_trees),
        regressor_trees=int(regressor_trees),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_leaf),
        train_days=int(train_days),
        validation_days=int(validation_days),
        step_days=int(step_days),
        purge_days=int(purge_days),
        embargo_days=int(embargo_days),
        horizon_days=int(horizon_days),
        min_probability=float(min_probability),
        min_expected_return=float(min_expected_return),
        top_signals_per_day=int(top_signals),
        risk_per_trade=float(risk_per_trade),
        min_risk_per_trade=float(min_risk),
        max_risk_per_trade=float(max_risk),
        kelly_fraction=float(kelly_fraction),
        max_total_exposure=float(max_exposure),
        max_single_position=float(max_position),
        max_positions=int(max_positions),
        atr_stop=float(atr_stop),
        atr_target=float(atr_target),
        max_hold_days=int(max_hold),
        commission_bps=float(commission_bps),
        slippage_bps=float(slippage_bps),
    )

    with st.spinner(
        "Daten laden, Features bauen und "
        "purged Walk-Forward berechnen …"
    ):
        try:
            st.session_state["result"] = run_engine(cfg)
        except Exception as exc:
            st.error(
                f"Backtest fehlgeschlagen: {exc}"
            )
            st.stop()


if "result" not in st.session_state:
    st.info(
        "Parameter wählen und **Run Research Backtest** starten."
    )

    st.markdown(
        """
### Was diese Version verbessert

**1. Kein Same-Day-Lookahead**  
Das Modell erzeugt ein Signal am Schlusskurs von `t`.
Die Position wird erst am nächsten verfügbaren Open desselben Assets eröffnet.

**2. Purged Walk-Forward**  
Jeder Fold trainiert sein eigenes Modell.

**3. Fold-local Feature Selection**  
Die Feature-Auswahl wird ausschließlich innerhalb des jeweiligen
Trainingsfensters durchgeführt.

**4. Label-Interval-Purge**  
Trainingsbeobachtungen, deren Zukunftsfenster in den OOS-Bereich
hineinreicht, werden entfernt.

**5. Cross-Sectional Ranking**  
Die Assets werden pro Ausführungstag nach einem
risiko-adjustierten Score sortiert.

**6. Portfolio Risk Engine**  
Fractional Kelly, Drawdown-Anpassung, Positionslimits und
Gesamtexposure arbeiten zusammen.

**7. Realistischere Kosten**  
Kommission und Slippage werden berücksichtigt.
"""
    )

    st.stop()


result = st.session_state["result"]
metrics = result["metrics"]

cards = st.columns(6)

cards[0].metric(
    "Endkapital",
    f"{metrics.get('end_capital', np.nan):,.0f}",
)

cards[1].metric(
    "Total Return",
    f"{metrics.get('total_return', np.nan) * 100:.1f}%",
)

cards[2].metric(
    "CAGR",
    f"{metrics.get('cagr', np.nan) * 100:.1f}%",
)

cards[3].metric(
    "Max Drawdown",
    f"{metrics.get('max_drawdown', np.nan) * 100:.1f}%",
)

cards[4].metric(
    "Sharpe",
    f"{metrics.get('sharpe', np.nan):.2f}",
)

cards[5].metric(
    "Profit Factor",
    f"{metrics.get('profit_factor', np.nan):.2f}",
)


tab_equity, tab_trades, tab_wf, tab_signals, tab_data = st.tabs(
    [
        "Equity",
        "Trades",
        "Walk-Forward",
        "Signals",
        "Data",
    ]
)


with tab_equity:
    equity = result["equity"]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["equity"],
            name="AI Strategy",
        )
    )

    benchmarks = result["benchmarks"]

    for column in benchmarks.columns:
        curve = benchmarks[column].dropna()

        figure.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve.values,
                name=column,
            )
        )

    figure.update_layout(
        height=520,
        hovermode="x unified",
        yaxis_title="Equity",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    drawdown = (
        equity
        .set_index("date")["drawdown"]
    )

    dd_figure = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="Drawdown",
        )
    )

    dd_figure.update_layout(
        height=300,
        yaxis_title="Drawdown",
    )

    st.plotly_chart(
        dd_figure,
        use_container_width=True,
    )


with tab_trades:
    trades = result["trades"]

    if trades.empty:
        st.warning("Keine Trades.")
    else:
        st.dataframe(
            trades.sort_values(
                "exit_date",
                ascending=False,
            ),
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Trades CSV",
            trades.to_csv(
                index=False
            ).encode("utf-8"),
            "trades.csv",
            "text/csv",
        )


with tab_wf:
    folds = result["folds"]

    st.dataframe(
        folds,
        use_container_width=True,
    )

    if (
        not folds.empty
        and folds["auc"].notna().any()
    ):
        wf_figure = go.Figure()

        wf_figure.add_trace(
            go.Scatter(
                x=folds["fold"],
                y=folds["auc"],
                mode="lines+markers",
                name="OOS AUC",
            )
        )

        wf_figure.add_hline(
            y=0.5,
            line_dash="dash",
        )

        wf_figure.update_layout(
            height=380,
            yaxis_title="AUC",
        )

        st.plotly_chart(
            wf_figure,
            use_container_width=True,
        )

    st.download_button(
        "⬇️ Walk-Forward CSV",
        folds.to_csv(
            index=False
        ).encode("utf-8"),
        "walk_forward_folds.csv",
        "text/csv",
    )


with tab_signals:
    predictions = result["predictions"]

    display_columns = [
        "date",
        "ticker",
        "prob_up",
        "expected_return",
        "score",
        "exec_date",
        "fold",
    ]

    st.dataframe(
        predictions
        .sort_values(
            ["date", "score"],
            ascending=[False, False],
        )[display_columns]
        .head(200),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Predictions CSV",
        predictions.to_csv(
            index=False
        ).encode("utf-8"),
        "predictions.csv",
        "text/csv",
    )


with tab_data:
    st.write(
        "Assets:",
        ", ".join(result["raw"].keys()),
    )

    st.write(
        "Feature-panel rows:",
        f"{len(result['panel']):,}",
    )

    st.write(
        "Prediction rows:",
        f"{len(result['predictions']):,}",
    )

    st.json(result["config"])


st.divider()

st.caption(
    "Research/backtesting only. Historical performance does not "
    "guarantee future results. This application is not financial advice."
)
