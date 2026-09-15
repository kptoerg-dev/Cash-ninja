"""
AI Trading Engine v11

Research/backtesting engine with:
- strict temporal execution
- label-interval-aware purged walk-forward validation
- fold-local feature selection
- classifier + expected-return regression
- cross-sectional ranking
- volatility-adjusted scoring
- fractional Kelly sizing
- drawdown-adaptive risk
- portfolio exposure controls
- ATR stop/target/time exits
- transaction costs

Research only. Not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


@dataclass
class Config:
    tickers: List[str]
    benchmark: str = "^GSPC"
    start: str = "2018-01-01"
    end: str = "2026-01-01"

    initial_capital: float = 100_000.0

    classifier_trees: int = 350
    regressor_trees: int = 350
    learning_rate: float = 0.035
    max_depth: int = 5
    min_samples_leaf: int = 25

    train_days: int = 504
    validation_days: int = 63
    step_days: int = 63
    purge_days: int = 5
    embargo_days: int = 2

    horizon_days: int = 5
    target_atr_multiple: float = 1.25

    min_probability: float = 0.55
    min_expected_return: float = 0.004
    top_signals_per_day: int = 3

    risk_per_trade: float = 0.012
    min_risk_per_trade: float = 0.003
    max_risk_per_trade: float = 0.025
    kelly_fraction: float = 0.35

    max_total_exposure: float = 0.85
    max_single_position: float = 0.25
    max_positions: int = 5

    atr_stop: float = 1.8
    atr_target: float = 3.0
    max_hold_days: int = 15

    commission_bps: float = 5.0
    slippage_bps: float = 5.0

    random_state: int = 42


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={c: str(c).title() for c in df.columns})

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            if col == "Volume":
                df[col] = 0.0
            else:
                return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df.dropna(subset=["Open", "High", "Low", "Close"])


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(
    df: pd.DataFrame,
    benchmark: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    x = df.copy()
    close = x["Close"]
    returns = close.pct_change()

    for n in [1, 2, 3, 5, 10, 20, 60]:
        x[f"ret_{n}"] = close.pct_change(n)

    for n in [5, 10, 20, 60]:
        x[f"vol_{n}"] = returns.rolling(n).std()

    x["atr"] = calculate_atr(x)
    x["atr_pct"] = x["atr"] / close.replace(0, np.nan)

    for n in [5, 10, 20, 50, 100, 200]:
        x[f"dist_ma_{n}"] = close / close.rolling(n).mean() - 1

    for n in [7, 14, 28]:
        x[f"rsi_{n}"] = calculate_rsi(close, n)

    x["range_pct"] = (x["High"] - x["Low"]) / close
    x["body_pct"] = (x["Close"] - x["Open"]) / x["Open"]
    x["upper_wick"] = (
        x["High"] - x[["Open", "Close"]].max(axis=1)
    ) / close
    x["lower_wick"] = (
        x[["Open", "Close"]].min(axis=1) - x["Low"]
    ) / close

    volume = x["Volume"].replace(0, np.nan)
    x["volume_z20"] = (
        (volume - volume.rolling(20).mean())
        / volume.rolling(20).std()
    )
    x["volume_change"] = volume.pct_change()

    x["momentum_accel"] = x["ret_5"] - x["ret_20"]
    x["trend_strength"] = x["dist_ma_20"] - x["dist_ma_100"]

    x["dow_sin"] = np.sin(2 * np.pi * x.index.dayofweek / 5)
    x["dow_cos"] = np.cos(2 * np.pi * x.index.dayofweek / 5)
    x["month_sin"] = np.sin(2 * np.pi * x.index.month / 12)
    x["month_cos"] = np.cos(2 * np.pi * x.index.month / 12)

    if benchmark is not None and not benchmark.empty:
        benchmark_close = benchmark["Close"].reindex(x.index).ffill()
        benchmark_returns = benchmark_close.pct_change()

        x["bench_ret_5"] = benchmark_close.pct_change(5)
        x["bench_ret_20"] = benchmark_close.pct_change(20)
        x["relative_5"] = x["ret_5"] - x["bench_ret_5"]
        x["relative_20"] = x["ret_20"] - x["bench_ret_20"]

        x["beta_60"] = (
            returns.rolling(60).cov(benchmark_returns)
            / benchmark_returns.rolling(60).var().replace(0, np.nan)
        )
        x["corr_60"] = returns.rolling(60).corr(benchmark_returns)

    return x.replace([np.inf, -np.inf], np.nan)


EXCLUDED_COLUMNS = {
    "Open", "High", "Low", "Close", "Volume",
    "future_return", "label", "label_exit",
    "ticker", "exec_date",
}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def make_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Label is calculated only from observations after signal time t.

    Signal timestamp:
        close of t

    Execution:
        open of next available bar

    Future target:
        close after horizon_days following that entry bar

    label_exit:
        last timestamp used by the label, enabling interval-aware purging.
    """
    x = df.copy()
    dates = x.index

    entry_open = x["Open"].shift(-1)
    future_close = x["Close"].shift(-(cfg.horizon_days + 1))

    x["future_return"] = future_close / entry_open - 1.0

    threshold = (
        cfg.target_atr_multiple
        * x["atr"]
        / x["Close"].replace(0, np.nan)
    )

    x["label"] = (x["future_return"] > threshold).astype(float)

    exits = []
    for i in range(len(dates)):
        j = i + cfg.horizon_days + 1
        exits.append(dates[j] if j < len(dates) else pd.NaT)

    x["label_exit"] = pd.to_datetime(exits)

    x.loc[x["future_return"].isna(), "label"] = np.nan

    return x


def train_models(train: pd.DataFrame, cfg: Config) -> dict:
    train = train.dropna(
        subset=["label", "future_return"]
    ).copy()

    features = get_feature_columns(train)

    if len(train) < 150:
        raise ValueError("Insufficient training observations.")

    if train["label"].nunique() < 2:
        raise ValueError("Training window contains only one label class.")

    X = train[features]
    y = train["label"].astype(int)

    # Feature selection is fitted ONLY on this fold's training data.
    selector = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=0.25,
                    max_iter=3000,
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )

    selector.fit(X, y)

    weights = np.abs(
        selector.named_steps["model"].coef_
    ).ravel()

    threshold = np.median(weights)

    selected = [
        feature
        for feature, weight in zip(features, weights)
        if weight >= threshold and weight > 0
    ]

    if len(selected) < 5:
        order = np.argsort(weights)[::-1]
        selected = [
            features[i]
            for i in order[:min(12, len(features))]
        ]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train[selected])

    leaf_nodes = 2 ** min(cfg.max_depth, 8)

    classifier = HistGradientBoostingClassifier(
        max_iter=cfg.classifier_trees,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=0.2,
        random_state=cfg.random_state,
    )

    regressor = HistGradientBoostingRegressor(
        max_iter=cfg.regressor_trees,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=0.2,
        loss="huber",
        random_state=cfg.random_state,
    )

    classifier.fit(X_train, y)
    regressor.fit(
        X_train,
        train["future_return"].clip(-0.5, 0.5),
    )

    return {
        "classifier": classifier,
        "regressor": regressor,
        "imputer": imputer,
        "features": selected,
    }


def predict_models(bundle: dict, data: pd.DataFrame) -> pd.DataFrame:
    x = data.copy()

    X = bundle["imputer"].transform(
        x[bundle["features"]]
    )

    x["prob_up"] = bundle["classifier"].predict_proba(X)[:, 1]
    x["expected_return"] = bundle["regressor"].predict(X)

    volatility = (
        x["vol_20"]
        .clip(lower=0.003)
        .fillna(0.03)
    )

    # Cross-sectional ranking score:
    # expected edge × confidence / volatility.
    x["score"] = (
        x["expected_return"]
        * (0.5 + x["prob_up"])
        / volatility
    )

    return x


def walk_forward(
    panel: pd.DataFrame,
    cfg: Config,
):
    dates = pd.DatetimeIndex(
        sorted(panel.index.unique())
    )

    predictions = []
    fold_rows = []

    position = cfg.train_days
    fold_id = 0

    while position + cfg.validation_days <= len(dates):
        train_start = dates[
            max(0, position - cfg.train_days)
        ]
        test_start = dates[position]
        test_end = (
            dates[
                min(
                    position + cfg.validation_days,
                    len(dates),
                ) - 1
            ]
            + pd.Timedelta(days=1)
        )

        train = panel[
            (panel.index >= train_start)
            & (panel.index < test_start)
        ].copy()

        test = panel[
            (panel.index >= test_start)
            & (panel.index < test_end)
        ].copy()

        # Purge 1: remove samples whose label interval overlaps OOS.
        train = train[
            train["label_exit"] < test_start
        ]

        # Purge 2: additional calendar safety margin.
        train = train[
            train.index
            < test_start - pd.Timedelta(
                days=cfg.purge_days
            )
        ]

        if len(train) >= 150 and not test.empty:
            try:
                bundle = train_models(train, cfg)
                predicted = predict_models(bundle, test)

                predicted["fold"] = fold_id
                predicted["train_end"] = test_start

                predictions.append(
                    predicted[
                        [
                            "ticker",
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "atr",
                            "vol_20",
                            "prob_up",
                            "expected_return",
                            "score",
                            "fold",
                            "train_end",
                        ]
                    ]
                )

                valid = test["label"].notna()

                if (
                    valid.sum() > 0
                    and test.loc[valid, "label"].nunique() > 1
                ):
                    auc = roc_auc_score(
                        test.loc[valid, "label"],
                        predicted.loc[valid, "prob_up"],
                    )
                else:
                    auc = np.nan

                fold_rows.append(
                    {
                        "fold": fold_id,
                        "train_start": train_start,
                        "train_end": test_start,
                        "test_start": test_start,
                        "test_end": test_end,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "features": len(bundle["features"]),
                        "auc": auc,
                    }
                )

            except Exception as exc:
                fold_rows.append(
                    {
                        "fold": fold_id,
                        "train_start": train_start,
                        "train_end": test_start,
                        "test_start": test_start,
                        "test_end": test_end,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "features": 0,
                        "auc": np.nan,
                        "error": str(exc),
                    }
                )

        fold_id += 1
        position += cfg.step_days

    if not predictions:
        return pd.DataFrame(), pd.DataFrame(fold_rows)

    pred = (
        pd.concat(predictions)
        .reset_index()
        .rename(columns={"index": "date"})
    )

    pred["date"] = pd.to_datetime(pred["date"])

    # Critical execution rule:
    # signal at t -> next available bar for SAME asset.
    pred["exec_date"] = (
        pred.groupby("ticker")["date"].shift(-1)
    )

    return pred, pd.DataFrame(fold_rows)


def transaction_cost(
    notional: float,
    cfg: Config,
) -> float:
    return (
        abs(notional)
        * (cfg.commission_bps + cfg.slippage_bps)
        / 10_000
    )


def fractional_kelly(
    probability: float,
    average_win: float,
    average_loss: float,
) -> float:
    if average_win <= 0 or average_loss <= 0:
        return 0.0

    b = average_win / average_loss

    return float(
        np.clip(
            (
                probability * b
                - (1 - probability)
            ) / b,
            0,
            1,
        )
    )


def backtest(
    predictions: pd.DataFrame,
    raw: Dict[str, pd.DataFrame],
    cfg: Config,
):
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    predictions = predictions[
        (predictions["prob_up"] >= cfg.min_probability)
        & (
            predictions["expected_return"]
            >= cfg.min_expected_return
        )
        & predictions["exec_date"].notna()
    ].copy()

    market = raw
    positions = {}
    trades = []
    equity_rows = []

    capital = float(cfg.initial_capital)
    peak = capital

    recent_wins = []
    recent_losses = []

    for current_date in sorted(
        predictions["exec_date"].dropna().unique()
    ):
        current_date = pd.Timestamp(current_date)

        # Existing position management.
        for ticker, position in list(positions.items()):
            df = market.get(ticker)

            if (
                df is None
                or current_date not in df.index
            ):
                continue

            bar = df.loc[current_date]

            exit_price = None
            exit_reason = None

            low = float(bar["Low"])
            high = float(bar["High"])
            close = float(bar["Close"])

            if (
                low <= position["stop"]
                and high >= position["target"]
            ):
                # Conservative same-bar assumption.
                exit_price = position["stop"]
                exit_reason = (
                    "stop_and_target_stop_first"
                )
            elif low <= position["stop"]:
                exit_price = position["stop"]
                exit_reason = "stop"
            elif high >= position["target"]:
                exit_price = position["target"]
                exit_reason = "target"
            elif (
                current_date
                - position["entry_date"]
            ).days >= cfg.max_hold_days:
                exit_price = close
                exit_reason = "time"

            if exit_price is not None:
                gross = (
                    exit_price
                    - position["entry_price"]
                ) * position["shares"]

                costs = transaction_cost(
                    position["entry_price"]
                    * position["shares"]
                    + exit_price
                    * position["shares"],
                    cfg,
                )

                pnl = gross - costs
                capital += pnl

                recent_wins.append(max(pnl, 0))
                recent_losses.append(
                    abs(min(pnl, 0))
                )

                recent_wins = recent_wins[-50:]
                recent_losses = recent_losses[-50:]

                trades.append(
                    {
                        **position,
                        "exit_date": current_date,
                        "exit_price": exit_price,
                        "gross_pnl": gross,
                        "costs": costs,
                        "pnl": pnl,
                        "return_pct": (
                            exit_price
                            / position["entry_price"]
                            - 1
                        ),
                        "reason": exit_reason,
                    }
                )

                positions.pop(ticker)

        # Rank all eligible assets cross-sectionally.
        candidates = (
            predictions[
                predictions["exec_date"]
                == current_date
            ]
            .sort_values(
                "score",
                ascending=False,
            )
            .head(cfg.top_signals_per_day)
        )

        for _, signal in candidates.iterrows():
            ticker = signal["ticker"]

            if ticker in positions:
                continue

            if len(positions) >= cfg.max_positions:
                break

            df = market.get(ticker)

            if (
                df is None
                or current_date not in df.index
            ):
                continue

            entry_price = float(
                df.loc[current_date, "Open"]
            )

            atr_value = float(signal["atr"])

            if (
                not np.isfinite(entry_price)
                or not np.isfinite(atr_value)
                or atr_value <= 0
            ):
                continue

            probability = float(
                signal["prob_up"]
            )
            expected_return = float(
                signal["expected_return"]
            )

            average_win = (
                np.mean(recent_wins)
                if recent_wins
                else max(
                    expected_return,
                    0.01,
                ) * capital
            )

            average_loss = (
                np.mean(recent_losses)
                if recent_losses
                else cfg.atr_stop * atr_value
            )

            kelly = fractional_kelly(
                probability,
                average_win,
                average_loss,
            )

            risk_fraction = (
                cfg.risk_per_trade
                * (
                    0.5
                    + cfg.kelly_fraction
                    * kelly
                )
            )

            drawdown = max(
                0,
                1 - capital / max(peak, 1),
            )

            risk_fraction *= max(
                0.35,
                1 - 1.5 * drawdown,
            )

            risk_fraction = float(
                np.clip(
                    risk_fraction,
                    cfg.min_risk_per_trade,
                    cfg.max_risk_per_trade,
                )
            )

            stop_distance = (
                cfg.atr_stop * atr_value
            )

            target_distance = (
                cfg.atr_target * atr_value
            )

            risk_budget = (
                capital * risk_fraction
            )

            shares = (
                risk_budget
                / stop_distance
            )

            # Single-position cap.
            shares = min(
                shares,
                (
                    capital
                    * cfg.max_single_position
                    / entry_price
                ),
            )

            used_exposure = sum(
                p["capital_committed"]
                for p in positions.values()
            )

            available = max(
                0,
                capital
                * cfg.max_total_exposure
                - used_exposure,
            )

            shares = min(
                shares,
                available / entry_price,
            )

            notional = shares * entry_price

            if notional < capital * 0.002:
                continue

            capital -= transaction_cost(
                notional,
                cfg,
            )

            positions[ticker] = {
                "ticker": ticker,
                "entry_date": current_date,
                "entry_price": entry_price,
                "shares": shares,
                "stop": (
                    entry_price
                    - stop_distance
                ),
                "target": (
                    entry_price
                    + target_distance
                ),
                "risk_fraction": risk_fraction,
                "capital_committed": notional,
            }

        unrealized = 0.0

        for ticker, position in positions.items():
            df = market.get(ticker)

            if (
                df is not None
                and current_date in df.index
            ):
                close = float(
                    df.loc[
                        current_date,
                        "Close",
                    ]
                )

                unrealized += (
                    close
                    - position["entry_price"]
                ) * position["shares"]

        equity = capital + unrealized
        peak = max(peak, equity)

        equity_rows.append(
            {
                "date": current_date,
                "equity": equity,
                "drawdown": (
                    equity / peak - 1
                ),
                "open_positions": len(
                    positions
                ),
            }
        )

    # Force-close positions at their own final available bar.
    for ticker, position in list(
        positions.items()
    ):
        df = market[ticker]

        final_date = df.index.max()
        exit_price = float(
            df.loc[
                final_date,
                "Close",
            ]
        )

        gross = (
            exit_price
            - position["entry_price"]
        ) * position["shares"]

        costs = transaction_cost(
            position["entry_price"]
            * position["shares"]
            + exit_price
            * position["shares"],
            cfg,
        )

        pnl = gross - costs
        capital += pnl

        trades.append(
            {
                **position,
                "exit_date": final_date,
                "exit_price": exit_price,
                "gross_pnl": gross,
                "costs": costs,
                "pnl": pnl,
                "return_pct": (
                    exit_price
                    / position["entry_price"]
                    - 1
                ),
                "reason": "end_of_test",
            }
        )

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trades)

    return (
        equity,
        trades,
        performance_metrics(
            equity,
            trades,
        ),
    )


def performance_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict:
    if equity.empty:
        return {}

    series = (
        equity
        .set_index("date")["equity"]
        .sort_index()
    )

    total_return = (
        series.iloc[-1]
        / series.iloc[0]
        - 1
    )

    years = max(
        (
            series.index[-1]
            - series.index[0]
        ).days / 365.25,
        1 / 365.25,
    )

    cagr = (
        series.iloc[-1]
        / series.iloc[0]
    ) ** (1 / years) - 1

    daily = (
        series
        .resample("D")
        .last()
        .ffill()
        .pct_change()
        .dropna()
    )

    sharpe = (
        daily.mean()
        / daily.std()
        * np.sqrt(252)
        if daily.std() > 0
        else np.nan
    )

    max_drawdown = float(
        (series / series.cummax() - 1).min()
    )

    if trades.empty:
        profit_factor = np.nan
        win_rate = np.nan
        average_trade = np.nan
    else:
        wins = trades.loc[
            trades["pnl"] > 0,
            "pnl",
        ].sum()

        losses = trades.loc[
            trades["pnl"] < 0,
            "pnl",
        ].sum()

        profit_factor = (
            wins / abs(losses)
            if losses != 0
            else np.inf
        )

        win_rate = float(
            (trades["pnl"] > 0).mean()
        )

        average_trade = float(
            trades["pnl"].mean()
        )

    return {
        "end_capital": float(
            series.iloc[-1]
        ),
        "total_return": float(
            total_return
        ),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
        "sharpe": float(sharpe),
        "profit_factor": float(
            profit_factor
        ),
        "win_rate": win_rate,
        "trades": int(len(trades)),
        "avg_trade": average_trade,
    }


def build_benchmarks(
    raw: Dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    benchmark_name: str,
    initial_capital: float,
) -> pd.DataFrame:
    curves = {}

    if (
        benchmark_df is not None
        and not benchmark_df.empty
    ):
        close = benchmark_df["Close"]
        curves[
            f"Buy & Hold {benchmark_name}"
        ] = (
            initial_capital
            * close
            / close.iloc[0]
        )

    assets = []

    for ticker, df in raw.items():
        if df.empty:
            continue

        assets.append(
            df["Close"].rename(ticker)
        )

    if assets:
        prices = (
            pd.concat(assets, axis=1)
            .sort_index()
            .ffill()
        )

        normalized = (
            prices / prices.iloc[0]
        )

        curves[
            "Equal Weight Universe"
        ] = (
            initial_capital
            * normalized.mean(axis=1)
        )

    return pd.DataFrame(curves)


def run_engine(cfg: Config) -> dict:
    raw = {}

    for ticker in cfg.tickers:
        df = download_ohlcv(
            ticker,
            cfg.start,
            cfg.end,
        )

        if not df.empty:
            raw[ticker] = df

    if not raw:
        raise RuntimeError(
            "No asset data could be downloaded."
        )

    benchmark_df = download_ohlcv(
        cfg.benchmark,
        cfg.start,
        cfg.end,
    )

    feature_panels = []

    for ticker, df in raw.items():
        features = make_features(
            df,
            benchmark_df,
        )

        features["ticker"] = ticker

        feature_panels.append(
            make_labels(
                features,
                cfg,
            )
        )

    panel = (
        pd.concat(feature_panels)
        .sort_index()
    )

    panel.index.name = "date"

    predictions, folds = walk_forward(
        panel,
        cfg,
    )

    if predictions.empty:
        raise RuntimeError(
            "No walk-forward predictions were produced. "
            "Try a longer history or smaller training window."
        )

    equity, trades, metrics = backtest(
        predictions,
        raw,
        cfg,
    )

    benchmarks = build_benchmarks(
        raw,
        benchmark_df,
        cfg.benchmark,
        cfg.initial_capital,
    )

    return {
        "raw": raw,
        "panel": panel,
        "predictions": predictions,
        "folds": folds,
        "equity": equity,
        "trades": trades,
        "metrics": metrics,
        "benchmarks": benchmarks,
        "config": asdict(cfg),
    }
