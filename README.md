# AI Trading Engine — Streamlit

Research-grade AI trading and backtesting application.

## Main improvements

- Strict no-lookahead execution
- Signal at the close of `t`, execution at the next available bar open
- Purged walk-forward validation
- Label-interval-aware purge
- Fold-local feature selection
- Classifier + expected-return regression
- Cross-sectional asset ranking
- Volatility-adjusted signal scoring
- Fractional Kelly position sizing
- Drawdown-adaptive risk
- Portfolio and single-position exposure caps
- ATR stop / target / time exits
- Commission + slippage model
- Buy & Hold and Equal Weight benchmarks
- Streamlit dashboard
- CSV exports

## GitHub upload

Upload these files/folders to the root of your repository:

```text
app.py
ai_trading_engine.py
requirements.txt
README.md
.gitignore
.streamlit/
    config.toml
```

Then deploy the repository with Streamlit and select `app.py` as the main file.

## Local installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important

This is research/backtesting software, not live-trading software and not financial advice.

Historical performance can be materially different from future performance. Test on untouched out-of-sample periods and realistic transaction costs before considering any real-money use.
