from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings("ignore")

@dataclass
class Config:
    tickers: Tuple[str,...]=("BTC-USD","ETH-USD","SOL-USD","QQQ","NVDA","MSFT")
    benchmark: str="^GSPC"
    start: str="2018-01-01"; end: str="2026-09-01"
    initial_cash: float=10000.0
    fee_rate: float=.001; slippage_rate: float=.00025
    atr_stop: float=1.7; atr_target: float=4.2; max_hold_days: int=10
    min_probability: float=.52; min_edge: float=.015; top_n: int=3
    kelly_fraction: float=.70; min_risk: float=.0025; max_risk: float=.045
    max_exposure: float=1.0; max_single_position: float=.45; max_positions: int=4
    soft_drawdown: float=.10; hard_drawdown: float=.20
    soft_risk_multiplier: float=.70; hard_risk_multiplier: float=.35
    high_vol_multiplier: float=.65
    n_folds: int=7; min_train_days: int=600; validation_days: int=90; purge_days: int=10
    n_estimators: int=450; max_depth: int=5; num_leaves: int=31
    min_child_samples: int=40; learning_rate: float=.025
    subsample: float=.85; colsample_bytree: float=.85
    feature_threshold: str="median"; random_state: int=42

FEATURES=["ret_1","ret_3","ret_5","ret_10","ret_20","rsi_14","atr_pct",
"realized_vol_10","realized_vol_20","skew_20","kurt_20","ema_10_dist",
"ema_20_dist","ema_50_dist","ema_100_dist","ema_20_slope","ema_50_slope",
"trend_strength","momentum_accel","volume_ratio","range_pct","body_pct",
"upper_wick","lower_wick","spx_ret_5","spx_ret_20","beta_60","rel_strength_20",
"dow_sin","dow_cos","month_sin","month_cos"]

def download(ticker,start,end):
    x=yf.download(ticker,start=start,end=end,auto_adjust=True,progress=False,threads=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=[c[0] for c in x.columns]
    x=x.rename(columns=str.title)
    need=["Open","High","Low","Close","Volume"]
    if any(c not in x for c in need): raise ValueError(f"{ticker}: OHLCV fehlen")
    return x[need].dropna()

def fetch_market_data(cfg):
    market={t:download(t,cfg.start,cfg.end) for t in cfg.tickers}
    bench=download(cfg.benchmark,cfg.start,cfg.end)["Close"]
    return market,bench

def rsi(c,p=14):
    d=c.diff(); g=d.clip(lower=0).ewm(alpha=1/p,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(alpha=1/p,adjust=False).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def build_features(df,bench):
    x=df.copy(); c,h,l,o,v=x.Close,x.High,x.Low,x.Open,x.Volume
    pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.ewm(alpha=1/14,adjust=False).mean()
    x["ret_1"]=c.pct_change(); x["ret_3"]=c.pct_change(3); x["ret_5"]=c.pct_change(5)
    x["ret_10"]=c.pct_change(10); x["ret_20"]=c.pct_change(20); x["rsi_14"]=rsi(c)
    x["atr_pct"]=x.ATR/c; x["realized_vol_10"]=x.ret_1.rolling(10).std()*np.sqrt(252)
    x["realized_vol_20"]=x.ret_1.rolling(20).std()*np.sqrt(252)
    x["skew_20"]=x.ret_1.rolling(20).skew(); x["kurt_20"]=x.ret_1.rolling(20).kurt()
    for p in [10,20,50,100]:
        ema=c.ewm(span=p,adjust=False).mean(); x[f"ema_{p}_dist"]=c/ema-1
    x["ema_20_slope"]=c.ewm(span=20,adjust=False).mean().pct_change(5)
    x["ema_50_slope"]=c.ewm(span=50,adjust=False).mean().pct_change(10)
    x["trend_strength"]=(x.ema_20_dist+x.ema_50_dist).abs()
    x["momentum_accel"]=x.ret_5-x.ret_20/4
    x["volume_ratio"]=v/v.rolling(20).mean()
    rng=(h-l).replace(0,np.nan); x["range_pct"]=rng/c; x["body_pct"]=(c-o).abs()/rng
    x["upper_wick"]=(h-pd.concat([o,c],axis=1).max(axis=1))/rng
    x["lower_wick"]=(pd.concat([o,c],axis=1).min(axis=1)-l)/rng
    b=bench.reindex(x.index).ffill(); br=b.pct_change()
    x["spx_ret_5"]=br.rolling(5).sum(); x["spx_ret_20"]=br.rolling(20).sum()
    x["beta_60"]=x.ret_1.rolling(60).cov(br)/br.rolling(60).var().replace(0,np.nan)
    x["rel_strength_20"]=x.ret_20-b.pct_change(20)
    dow=x.index.dayofweek; mon=x.index.month
    x["dow_sin"]=np.sin(2*np.pi*dow/7); x["dow_cos"]=np.cos(2*np.pi*dow/7)
    x["month_sin"]=np.sin(2*np.pi*mon/12); x["month_cos"]=np.cos(2*np.pi*mon/12)
    return x

def label(df,cfg):
    x=df.copy(); rets=[]; targets=[]; exits=[]
    for i in range(len(x)):
        ei=i+1
        if ei>=len(x): rets.append(np.nan);targets.append(np.nan);exits.append(pd.NaT);continue
        entry=float(x.Open.iloc[ei]); atr=float(x.ATR.iloc[i])
        if not np.isfinite(entry) or not np.isfinite(atr) or atr<=0:
            rets.append(np.nan);targets.append(np.nan);exits.append(pd.NaT);continue
        stop=entry-cfg.atr_stop*atr; target=entry+cfg.atr_target*atr
        last=min(len(x)-1,ei+cfg.max_hold_days-1); hit=None; ex=last
        for j in range(ei,last+1):
            if x.Low.iloc[j]<=stop: hit="stop";ex=j;break
            if x.High.iloc[j]>=target: hit="target";ex=j;break
        ret=float(x.Close.iloc[ex])/entry-1
        if hit=="target": ret=cfg.atr_target*atr/entry
        elif hit=="stop": ret=-cfg.atr_stop*atr/entry
        rets.append(ret);targets.append(1 if hit=="target" else 0);exits.append(x.index[ex])
    x["future_return"]=rets;x["target"]=targets;x["label_exit"]=exits
    return x

def base_cls(c):
    return LGBMClassifier(n_estimators=c.n_estimators,max_depth=c.max_depth,
      num_leaves=c.num_leaves,min_child_samples=c.min_child_samples,
      learning_rate=c.learning_rate,subsample=c.subsample,
      colsample_bytree=c.colsample_bytree,reg_alpha=.05,reg_lambda=.20,
      random_state=c.random_state,verbosity=-1)

def base_reg(c):
    return LGBMRegressor(n_estimators=c.n_estimators,max_depth=c.max_depth,
      num_leaves=c.num_leaves,min_child_samples=c.min_child_samples,
      learning_rate=c.learning_rate,subsample=c.subsample,
      colsample_bytree=c.colsample_bytree,reg_alpha=.05,reg_lambda=.20,
      random_state=c.random_state,verbosity=-1)

def train(train,c):
    X=train[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0); yc=train.target.astype(int)
    sel_model=base_cls(c).fit(X,yc)
    selector=SelectFromModel(sel_model,threshold=c.feature_threshold,prefit=True)
    selected=[f for f,k in zip(FEATURES,selector.get_support()) if k] or FEATURES[:12]
    clf=base_cls(c).fit(X[selected],yc)
    reg=base_reg(c).fit(X[selected],train.future_return.clip(-.5,.5))
    return clf,reg,selected

def folds(n,c):
    if n<c.min_train_days+c.validation_days:return
    starts=np.linspace(c.min_train_days,n-c.validation_days,c.n_folds,dtype=int)
    for s in sorted(set(starts)):
        yield s,s+c.validation_days

def walk_forward(df,c):
    x=label(df,c).dropna(subset=FEATURES+["target","future_return"]).replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES)
    if len(x)<c.min_train_days+c.validation_days:return pd.DataFrame(),pd.DataFrame()
    ps=[]; rs=[]
    for no,(s,e) in enumerate(folds(len(x),c),1):
        test=x.iloc[s:e]; test_start=test.index[0]
        # Expanding training set with a time purge. No future observations are used.
        train=x.iloc[:s]
        train=train[train.index < test_start-pd.Timedelta(days=c.purge_days)]
        if len(train)<200:continue
        clf,reg,selected=train_model=train, None, None
        clf,reg,selected=train_models(train,c)
        Xt=test[selected].fillna(0)
        p=clf.predict_proba(Xt)[:,1]; er=reg.predict(Xt)
        out=test[["Open","High","Low","Close","ATR","realized_vol_20"]].copy()
        out["prob"]=p;out["expected_return"]=er;out["fold"]=no
        out["selected_features"]=",".join(selected)
        out["signal_score"]=er/(test.realized_vol_20.abs()+1e-6)
        ps.append(out)
        rs.append({"fold":no,"train_rows":len(train),"test_rows":len(test),
                   "test_start":str(test.index[0].date()),"test_end":str(test.index[-1].date()),
                   "selected_features":len(selected)})
    return (pd.concat(ps).sort_index() if ps else pd.DataFrame(),pd.DataFrame(rs))

def train_models(train,c):
    X=train[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0); yc=train.target.astype(int)
    sm=base_cls(c).fit(X,yc); selector=SelectFromModel(sm,threshold=c.feature_threshold,prefit=True)
    selected=[f for f,k in zip(FEATURES,selector.get_support()) if k] or FEATURES[:12]
    clf=base_cls(c).fit(X[selected],yc)
    reg=base_reg(c).fit(X[selected],train.future_return.astype(float).clip(-.5,.5))
    return clf,reg,selected

class Backtest:
    def __init__(self,c):
        self.c=c;self.cash=c.initial_cash;self.pos={};self.trades=[];self.eq=[];self.peak=c.initial_cash
    def risk_mult(self,e):
        dd=1-e/max(self.peak,1e-9)
        return self.c.hard_risk_multiplier if dd>=self.c.hard_drawdown else self.c.soft_risk_multiplier if dd>=self.c.soft_drawdown else 1
    def exit(self,t,d,price,reason):
        p=self.pos.pop(t); px=price*(1-self.c.slippage_rate); proceeds=p["qty"]*px
        self.cash+=proceeds-proceeds*self.c.fee_rate
        pnl=(px-p["entry"])*p["qty"]-p["entry"]*p["qty"]*self.c.fee_rate
        self.trades.append({"ticker":t,"entry_date":p["date"],"exit_date":d,"entry":p["entry"],
          "exit":px,"qty":p["qty"],"pnl":pnl,"return":px/p["entry"]-1,"reason":reason,"bars":p["bars"]})
    def step(self,d,market,signals):
        for t in list(self.pos):
            if d not in market[t].index:continue
            r=market[t].loc[d];p=self.pos[t];p["bars"]+=1
            if r.Low<=p["stop"]:self.exit(t,d,p["stop"],"stop");continue
            if r.High>=p["target"]:self.exit(t,d,p["target"],"target");continue
            p["high"]=max(p["high"],float(r.High))
            if p["high"]>=p["entry"]+self.c.atr_stop*p["atr"]*1.15:p["stop"]=max(p["stop"],p["entry"])
            if p["high"]>=p["entry"]+self.c.atr_stop*p["atr"]*1.80:p["stop"]=max(p["stop"],p["high"]-2.2*p["atr"])
            if p["bars"]>=self.c.max_hold_days:self.exit(t,d,float(r.Close),"time")
        today=signals[signals.exec_date==d].sort_values("signal_score",ascending=False) if not signals.empty else pd.DataFrame()
        for _,s in today.head(self.c.top_n).iterrows():
            t=s.ticker
            if t in self.pos or len(self.pos)>=self.c.max_positions or s.prob<self.c.min_probability or s.expected_return<self.c.min_edge:continue
            if d not in market[t].index:continue
            r=market[t].loc[d];entry=float(r.Open)*(1+self.c.slippage_rate);atr=float(s.ATR)
            equity=self.eq[-1]["Equity"] if self.eq else self.cash
            pwin=float(np.clip(s.prob,.01,.99));rr=self.c.atr_target/self.c.atr_stop
            k=max(0,(pwin*rr-(1-pwin))/rr);risk=np.clip(k*self.c.kelly_fraction,self.c.min_risk,self.c.max_risk)*self.risk_mult(equity)
            risk_cash=equity*risk;stop=entry-self.c.atr_stop*atr;target=entry+self.c.atr_target*atr
            qty=risk_cash/max(entry-stop,1e-9);qty=min(qty,equity*self.c.max_single_position/entry)
            exposure=sum(q["qty"]*q["entry"] for q in self.pos.values())
            qty=min(qty,max(0,(equity*self.c.max_exposure-exposure)/entry))
            notional=qty*entry;fee=notional*self.c.fee_rate
            if qty<=0 or self.cash<notional+fee:continue
            self.cash-=notional+fee
            self.pos[t]={"date":d,"entry":entry,"qty":qty,"stop":stop,"target":target,"atr":atr,"high":entry,"bars":0}
        prices={t:float(market[t].loc[d].Close) for t in market if d in market[t].index}
        value=self.cash+sum(p["qty"]*prices[t] for t,p in self.pos.items() if t in prices)
        self.peak=max(self.peak,value);self.eq.append({"Date":d,"Equity":value,"Cash":self.cash,"Positions":len(self.pos),"Drawdown":1-value/self.peak})
    def run(self,market,preds):
        rows=[]
        for t,p in preds.items():
            if p.empty:continue
            q=p.copy();q["ticker"]=t;q["exec_date"]=q.index.to_series().shift(-1).values
            q["high_vol"]=(q.realized_vol_20>q.realized_vol_20.rolling(60).quantile(.8)).astype(int)
            rows.append(q.reset_index(names="signal_date"))
        sig=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
        for d in sorted(set().union(*[x.index for x in market.values()])):self.step(pd.Timestamp(d),market,sig)
        return pd.DataFrame(self.eq),pd.DataFrame(self.trades)

def metrics(eq,tr,initial):
    if eq.empty:return {}
    e=eq.Equity;total=e.iloc[-1]/initial-1;days=max((eq.Date.iloc[-1]-eq.Date.iloc[0]).days,1)
    cagr=(e.iloc[-1]/initial)**(365/days)-1 if e.iloc[-1]>0 else -1
    dd=(e/e.cummax()-1).min();ret=e.pct_change().dropna()
    sharpe=np.sqrt(252)*ret.mean()/ret.std() if ret.std()>0 else 0
    if tr.empty:win=pf=avg=0
    else:
        win=(tr.pnl>0).mean();g=tr.loc[tr.pnl>0,"pnl"].sum();l=-tr.loc[tr.pnl<0,"pnl"].sum();pf=g/l if l else np.inf;avg=tr["return"].mean()
    return {"End Capital":float(e.iloc[-1]),"Total Return":float(total),"CAGR":float(cagr),
            "Max Drawdown":float(dd),"Sharpe":float(sharpe),"Profit Factor":float(pf),
            "Win Rate":float(win),"Trades":int(len(tr)),"Avg Trade":float(avg)}

def run_engine(c,progress=None):
    market,bench=fetch_market_data(c);preds={};reports=[];features=[]
    for i,(t,df) in enumerate(market.items(),1):
        if progress:progress(i-1,len(market),f"Training {t}...")
        p,f=walk_forward(build_features(df,bench),c);preds[t]=p
        if not f.empty:f.insert(0,"ticker",t);reports.append(f)
        if not p.empty:features.append({"ticker":t,"folds":p.fold.nunique(),"prediction_rows":len(p),
          "selected_features_latest":p.selected_features.iloc[-1]})
    bt=Backtest(c);eq,tr=bt.run(market,preds)
    return {"metrics":metrics(eq,tr,c.initial_cash),"equity":eq,"trades":tr,
            "folds":pd.concat(reports,ignore_index=True) if reports else pd.DataFrame(),
            "features":pd.DataFrame(features),"predictions":preds,"config":asdict(c)}
