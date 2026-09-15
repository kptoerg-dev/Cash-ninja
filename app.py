import streamlit as st
import pandas as pd
from ai_trading_engine import Config, run_engine

st.set_page_config(page_title="AI Alpha Trading Engine", page_icon="📈", layout="wide")
st.title("📈 AI Alpha Trading Engine")
st.caption("Purged Walk-Forward • Expected Return • Cross-Asset Ranking • Risk-Aware Portfolio")

with st.sidebar:
    st.header("Universe")
    assets = ["BTC-USD","ETH-USD","SOL-USD","QQQ","NVDA","MSFT"]
    tickers = st.multiselect("Assets", assets, default=assets)
    benchmark = st.text_input("Benchmark", "^GSPC")

    st.header("Capital & Risk")
    initial_cash = st.number_input("Startkapital", 1000.0, 1_000_000.0, 10000.0, 500.0)
    profile = st.selectbox("Risk-Profil", ["Defensiv","Balanced","Aggressiv","Alpha Max"], index=2)
    presets = {"Defensiv":(.40,.002,.025),"Balanced":(.55,.0025,.035),
               "Aggressiv":(.70,.0025,.045),"Alpha Max":(.85,.004,.060)}
    kelly, min_risk, max_risk = presets[profile]
    kelly = st.slider("Kelly-Faktor", .10, 1.0, kelly, .05)
    min_risk = st.slider("Min. Risiko / Trade", .001, .02, min_risk, .001)
    max_risk = st.slider("Max. Risiko / Trade", .005, .10, max_risk, .005)
    max_exposure = st.slider("Max. Exposure", .10, 1.50, 1.00, .05)
    max_single = st.slider("Max. Einzelposition", .05, .75, .45, .05)
    max_positions = st.slider("Max. Positionen", 1, 10, 4)

    st.header("Signal")
    min_prob = st.slider("Min. Wahrscheinlichkeit", .50, .80, .52, .01)
    min_edge = st.slider("Min. Expected Return", 0.0, .10, .015, .0025)
    top_n = st.slider("Top Signale / Tag", 1, 10, 3)

    st.header("Stops")
    atr_stop = st.slider("ATR Stop", .5, 4.0, 1.7, .1)
    atr_target = st.slider("ATR Target", 1.0, 10.0, 4.2, .1)
    hold = st.slider("Max. Haltedauer", 2, 30, 10)

    st.header("Walk Forward")
    folds = st.slider("Folds", 3, 12, 7)
    train_days = st.slider("Min. Trainingstage", 250, 1500, 600, 50)
    val_days = st.slider("Validierungstage", 30, 180, 90, 10)
    purge_days = st.slider("Purge-Tage", 1, 30, 10)

    st.header("LightGBM")
    trees = st.slider("Trees", 100, 1000, 450, 50)
    lr = st.slider("Learning Rate", .005, .10, .025, .005)

    st.header("Zeitraum")
    start = st.date_input("Start", pd.Timestamp("2018-01-01"))
    end = st.date_input("Ende", pd.Timestamp("2026-09-01"))
    run = st.button("🚀 Backtest starten", type="primary", use_container_width=True)

if not tickers:
    st.warning("Bitte mindestens ein Asset auswählen.")
    st.stop()

cfg = Config(
    tickers=tuple(tickers), benchmark=benchmark, start=str(start), end=str(end),
    initial_cash=initial_cash, kelly_fraction=kelly, min_risk=min_risk,
    max_risk=max_risk, max_exposure=max_exposure, max_single_position=max_single,
    max_positions=max_positions, min_probability=min_prob, min_edge=min_edge,
    top_n=top_n, atr_stop=atr_stop, atr_target=atr_target, max_hold_days=hold,
    n_folds=folds, min_train_days=train_days, validation_days=val_days,
    purge_days=purge_days, n_estimators=trees, learning_rate=lr
)

if run:
    bar = st.progress(0)
    status = st.empty()
    try:
        result = run_engine(cfg, progress=lambda done,total,msg: (
            bar.progress(min(100, int(done/max(total,1)*100))), status.info(msg)))
        st.session_state["result"] = result
        bar.progress(100); status.success("Backtest abgeschlossen.")
    except Exception as exc:
        st.error(f"Fehler: {exc}")
        st.exception(exc)

if "result" not in st.session_state:
    st.info("Parameter einstellen und Backtest starten.")
    st.markdown("""### Engine
- Purged Walk-Forward
- Fold-lokale Feature Selection
- Klassifikation + Expected-Return-Regressor
- Signal am Close → Ausführung am nächsten Open
- ATR Stop/Target
- Fractional Kelly
- Drawdown-adaptives Risiko
- Portfolio-Exposure-Limits
- Gebühren und Slippage
""")
    st.stop()

r = st.session_state["result"]
m = r["metrics"]
cols = st.columns(7)
for col,(label,key,fmt) in zip(cols,[
    ("Endkapital","End Capital","${:,.0f}"),("Return","Total Return","{:.1%}"),
    ("CAGR","CAGR","{:.1%}"),("Max DD","Max Drawdown","{:.1%}"),
    ("Sharpe","Sharpe","{:.2f}"),("Profit Factor","Profit Factor","{:.2f}"),
    ("Trades","Trades","{:,.0f}")]):
    col.metric(label, fmt.format(m.get(key,0)))

t1,t2,t3,t4 = st.tabs(["📈 Equity","💱 Trades","🧪 Walk Forward","🧠 Features"])
with t1:
    eq=r["equity"]
    if not eq.empty:
        eq["Date"]=pd.to_datetime(eq["Date"])
        st.line_chart(eq.set_index("Date")["Equity"])
        st.dataframe(eq.tail(100),use_container_width=True)
with t2:
    tr=r["trades"]
    st.dataframe(tr,use_container_width=True)
    if not tr.empty:
        st.download_button("⬇️ Trades CSV",tr.to_csv(index=False).encode(),"trades.csv","text/csv")
with t3:
    f=r["folds"]
    st.dataframe(f,use_container_width=True)
    if not f.empty:
        st.download_button("⬇️ Fold Report CSV",f.to_csv(index=False).encode(),"walk_forward.csv","text/csv")
with t4:
    st.dataframe(r["features"],use_container_width=True)

st.divider()
st.caption("⚠️ Research-/Backtesting-Software. Historische Ergebnisse garantieren keine zukünftigen Renditen.")
