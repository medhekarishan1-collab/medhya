"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SEPA STOCK SCREENER DASHBOARD — NSE India                           ║
║         Based on Stage Analysis + CANSLIM + Technical Indicators            ║
║         Author: Claude (Anthropic)  |  Version: 2.0 Production              ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install -r requirements.txt
    streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import traceback
from typing import Optional, Tuple, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit command
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEPA Stock Screener — NSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark, professional trading terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ───────────────────────────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  /* ── Header ─────────────────────────────────────────────────── */
  .main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e40af33;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
  }
  .main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .main-header p { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.9rem; }

  /* ── KPI Cards ───────────────────────────────────────────────── */
  .kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .kpi-card:hover { border-color: #3b82f6; }
  .kpi-value { font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .kpi-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
  .kpi-green  { color: #22c55e; }
  .kpi-blue   { color: #60a5fa; }
  .kpi-purple { color: #a78bfa; }
  .kpi-orange { color: #f97316; }

  /* ── Section Titles ──────────────────────────────────────────── */
  .section-title {
    font-size: 1.1rem; font-weight: 600; color: #cbd5e1;
    border-left: 3px solid #3b82f6;
    padding-left: 10px; margin: 20px 0 14px 0;
    letter-spacing: 0.3px;
  }

  /* ── Score Badge ─────────────────────────────────────────────── */
  .score-high   { color: #22c55e; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .score-medium { color: #f59e0b; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .score-low    { color: #ef4444; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

  /* ── Streamlit overrides ─────────────────────────────────────── */
  .stDataFrame { border-radius: 8px; overflow: hidden; }
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5);
    color: white; border: none; border-radius: 8px;
    padding: 8px 20px; font-weight: 600;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }
  div[data-testid="stMetric"] {
    background: #1e293b; border-radius: 8px;
    padding: 12px 16px; border: 1px solid #334155;
  }
  .stSelectbox label, .stSlider label, .stMultiSelect label {
    color: #94a3b8 !important; font-size: 0.85rem !important;
  }

  /* ── Signal pills ────────────────────────────────────────────── */
  .pill-green  { background:#14532d; color:#4ade80; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
  .pill-red    { background:#450a0a; color:#f87171; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
  .pill-yellow { background:#451a03; color:#fbbf24; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }

  /* ── Sidebar ─────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #1e293b;
  }
  [data-testid="stSidebar"] .css-1d391kg { background: transparent; }

  /* ── Spinner ─────────────────────────────────────────────────── */
  .stSpinner > div > div { border-top-color: #3b82f6 !important; }

  /* ── Footer ─────────────────────────────────────────────────── */
  .footer {
    text-align: center; color: #475569;
    font-size: 0.75rem; padding: 20px 0;
    border-top: 1px solid #1e293b; margin-top: 40px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NSE STOCK UNIVERSE  (NIFTY 500 + MidSmall + Emerging picks)
# ─────────────────────────────────────────────────────────────────────────────
# fmt: off
NSE_STOCKS = [
    # NIFTY 50
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","WIPRO",
    "ULTRACEMCO","TITAN","BAJFINANCE","NESTLEIND","SUNPHARMA","POWERGRID",
    "NTPC","ONGC","TATAMOTORS","TECHM","HCLTECH","BAJAJFINSV","DRREDDY",
    "INDUSINDBK","COALINDIA","ADANIENT","ADANIPORTS","DIVISLAB","EICHERMOT",
    "BRITANNIA","CIPLA","APOLLOHOSP","TATACONSUM","BPCL","HINDALCO","GRASIM",
    "JSWSTEEL","M&M","HEROMOTOCO","TATASTEEL","LTIM","SBILIFE","HDFCLIFE",
    "BAJAJ-AUTO","UPL",
    # NIFTY NEXT 50
    "ADANIPOWER","AMBUJACEM","BANKBARODA","BERGEPAINT","BOSCHLTD","CANBK",
    "CHOLAFIN","COLPAL","DABUR","DMART","GAIL","GODREJCP","HAVELLS","HDFCAMC",
    "ICICIGI","ICICIPRULI","INDIGO","IOC","IRCTC","LUPIN","MCDOWELL-N",
    "MUTHOOTFIN","PIDILITIND","PNB","RECLTD","SAIL","SHREECEM","SIEMENS",
    "TORNTPHARM","TRENT","VEDL","VOLTAS","ZOMATO","NAUKRI","PAYTM",
    # Midcap leaders
    "PERSISTENT","MPHASIS","COFORGE","LTTS","KPITTECH","TATAELXSI",
    "ASTRAL","SUPREMEIND","POLYCAB","DIXON","AMBER","KAYNES","SYRMA",
    "DEEPAKNTR","AARTIIND","PIIND","SUMICHEM","TATACHEM","GNFC",
    "HINDPETRO","MRPL","AEGISCHEM","ALKYLAMINE",
    "STAR","PVR","INOXLEISUR","ZEEL","SUNCLAYLTD",
    "BANKBARODA","FEDERALBNK","IDFCFIRSTB","RBLBANK","BANDHANBNK","AUBANK",
    "ABCAPITAL","MANAPPURAM","BAJAJHLDNG","M&MFIN","SHRIRAMFIN",
    "ICICIGI","STARHEALTH","NIACL","GICRE","SBICARD",
    "TORNTPOWER","TATAPOWER","ADANITRANS","CESC","NHPC","SJVN","IRCON",
    "CONCOR","TIINDIA","APLAPOLLO","RAMCOCEM","JKCEMENT","HEIDELBERG",
    "WHIRLPOOL","BLUESTARCO","SYMPHONY","BATAINDIA","VBL","JUBLFOOD",
    "WESTLIFE","DEVYANI","SAPPHIRE","BURGER","METRO","SHOPERSTOP",
    "PAGEIND","RAJESH","ABFRL","MANYAVAR","VEDANT","IDFC","MFSL",
    "SUNDARMFIN","L&TFH","CANFINHOME","LICHSGFIN","PNBHOUSING",
    "INDIGOPNTS","KANSAINER","AKZOINDIA","NOCIL","SRF","GARFIBRES",
    "WELSPUNIND","TRIDENT","VARDHMAN","ARVIND","RAYMOND",
    "JINDALSAW","WELCORP","RATNAMANI","MAHINDCIE","ENDURANCE",
    "BALKRISIND","APOLLOTYRE","MRF","CEATLTD","GOODYEAR",
    "SCHAEFFLER","SKF","TIMKEN","GREAVESCOT","ELGIEQUIP","THERMAX",
    "CUMMINSIND","GMMPFAUDLR","JYOTHYLAB","MARICO","EMAMILTD",
    "GODREJIND","TATACOMM","MTNL","RAILTEL","STLTECH","ROUTE",
    "AFFLE","INDIAMART","JUSTDIAL","POLICYBZR","CARTRADE",
    "IRFC","PFC","REC","HUDCO","NHAI","NABARD",
    "DALBHARAT","MCEM","BIRLACORPN","NUVOCO","PRISM","STARCEMENT",
    "ABBOTINDIA","ASTRAZEN","PFIZER","SANOFI","GLAXO","ALKEM",
    "TORNTPHARM","IPCALAB","AJANTPHARM","GLENMARK","NATCOPHARMA",
    "GRANULES","LAURUSLABS","SEQUENT","SOLARA","STRIDES","SUVEN",
    "FLUOROCHEM","NAVINFLUOR","CAMLIN","HOCL",
    "GESHIP","SCI","SEQUENT","MAHLOG","DELHIVERY","BLUE DART",
    "ZYDUSLIFE","CADILAHC","WOCKPHARMA","FDC","JBCHEPHARM",
    "HINDCOPPER","HINDZINC","NATIONALUM","NMDC","GMRINFRA","IRB",
    "ASHOKLEY","ESCORTS","SONACOMS","MOTHERSON","SUPRAJIT","MINDA",
    "BOSCHLTD","JBMA","CRAFTSMAN","KALYANKJIL","SENCO","TITAN",
    "PCJEWELLER","THANGAMAYL","RVNL","RITES","PNCINFRA","HG INFRA",
    "GPPL","ADANIGREEN","ADANITRANS","TATARENEW","GREENKO",
    "WABCOINDIA","GABRIEL","JAMNAAUTO","RASANDIK",
    "REDINGTON","RATEGAIN","ZENSAR","CYIENT","KFINTECH","CDSL","BSE",
    "MCX","ICRA","CRISIL","CARE",
]
# fmt: on

# Remove duplicates and add .NS suffix
NSE_STOCKS = list(dict.fromkeys(NSE_STOCKS))
NSE_TICKERS = [f"{s}.NS" for s in NSE_STOCKS]
NIFTY_TICKER = "^NSEI"

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR MAP  (symbol → sector)
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_MAP: Dict[str, str] = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "PERSISTENT": "IT", "MPHASIS": "IT", "COFORGE": "IT",
    "LTTS": "IT", "KPITTECH": "IT", "TATAELXSI": "IT", "ZENSAR": "IT",
    "CYIENT": "IT", "AFFLE": "IT",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BANKBARODA": "Banking", "PNB": "Banking", "CANBK": "Banking",
    "FEDERALBNK": "Banking", "IDFCFIRSTB": "Banking", "RBLBANK": "Banking",
    "BANDHANBNK": "Banking", "AUBANK": "Banking",
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
    "MUTHOOTFIN": "NBFC", "MANAPPURAM": "NBFC", "M&MFIN": "NBFC",
    "SHRIRAMFIN": "NBFC", "ABCAPITAL": "NBFC",
    "RELIANCE": "Oil & Gas", "ONGC": "Oil & Gas", "BPCL": "Oil & Gas",
    "IOC": "Oil & Gas", "HINDPETRO": "Oil & Gas", "GAIL": "Oil & Gas",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "LUPIN": "Pharma", "DIVISLAB": "Pharma", "ALKEM": "Pharma",
    "TORNTPHARM": "Pharma", "IPCALAB": "Pharma", "AJANTPHARM": "Pharma",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "COLPAL": "FMCG", "GODREJCP": "FMCG", "EMAMILTD": "FMCG",
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "M&M": "Auto",
    "HEROMOTOCO": "Auto", "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto",
    "ASHOKLEY": "Auto", "ESCORTS": "Auto",
    "LT": "Capital Goods", "SIEMENS": "Capital Goods", "ABB": "Capital Goods",
    "CUMMINSIND": "Capital Goods", "THERMAX": "Capital Goods",
    "ADANIENT": "Conglomerate", "ADANIPORTS": "Ports",
    "ADANIPOWER": "Power", "TATAPOWER": "Power", "NTPC": "Power",
    "POWERGRID": "Power", "NHPC": "Power",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "COALINDIA": "Mining",
    "ASIANPAINT": "Paints", "BERGEPAINT": "Paints",
    "HAVELLS": "Electricals", "POLYCAB": "Electricals", "DIXON": "Electronics",
    "ULTRACEMCO": "Cement", "SHREECEM": "Cement", "AMBUJACEM": "Cement",
    "TITAN": "Jewellery", "KALYANKJIL": "Jewellery",
    "APOLLOHOSP": "Healthcare", "TATACONSUM": "Consumer",
    "IRCTC": "Travel", "INDIGO": "Aviation",
    "DMART": "Retail", "TRENT": "Retail",
    "ZOMATO": "Food Tech", "NAUKRI": "Internet",
    "BHARTIARTL": "Telecom",
}

def get_sector(symbol: str) -> str:
    clean = symbol.replace(".NS", "")
    return SECTOR_MAP.get(clean, "Others")


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nifty_data(period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch NIFTY 50 index data for RSC computation."""
    try:
        df = yf.download(NIFTY_TICKER, period=period, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not fetch NIFTY data: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_batch(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    """
    Batch-download a list of tickers via yfinance.
    Returns {ticker: OHLCV_dataframe}
    """
    result: Dict[str, pd.DataFrame] = {}
    batch_size = 50  # yfinance handles ~50 tickers per call reliably

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                period=period,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if raw is None or raw.empty:
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                    else:
                        df = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()
                    df.index = pd.to_datetime(df.index)
                    df = df.dropna(subset=["Close"])
                    if len(df) >= 50:  # need at least 50 rows for indicators
                        result[ticker] = df
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"⚠️ Batch {i//batch_size+1} fetch error: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATOR CALCULATORS  (vectorized)
# ─────────────────────────────────────────────────────────────────────────────
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(series: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9
              ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast   = calc_ema(series, fast)
    ema_slow   = calc_ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_bollinger(series: pd.Series, period: int = 20,
                   std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma   = calc_sma(series, period)
    std   = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


# ─────────────────────────────────────────────────────────────────────────────
# SEPA STAGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def detect_stage(df: pd.DataFrame) -> int:
    """
    Return SEPA Stage (1-4):
      Stage 1 – Basing / flat (price near 200 DMA, DMA trending flat)
      Stage 2 – Advancing (price > 50 DMA > 200 DMA, both rising) ← IDEAL
      Stage 3 – Topping (price below 50 DMA, 50 starts turning down)
      Stage 4 – Declining
    """
    if len(df) < 200:
        return 0
    try:
        close = df["Close"]
        sma50  = calc_sma(close, 50)
        sma200 = calc_sma(close, 200)
        p  = close.iloc[-1]
        m50  = sma50.iloc[-1]
        m200 = sma200.iloc[-1]
        # 200 DMA slope over 20 days
        slope200 = (sma200.iloc[-1] - sma200.iloc[-21]) / sma200.iloc[-21] if len(sma200) > 21 else 0
        slope50  = (sma50.iloc[-1]  - sma50.iloc[-21])  / sma50.iloc[-21]  if len(sma50)  > 21 else 0

        if p > m50 > m200 and slope50 > 0 and slope200 > 0:
            return 2  # Stage 2 ✅
        elif p < m50 < m200 and slope50 < 0:
            return 4  # Stage 4
        elif p > m200 and slope200 < 0.005 and abs(slope200) < 0.005:
            return 1  # Stage 1
        else:
            return 3  # Stage 3
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# RSC  (Relative Strength vs NIFTY)
# ─────────────────────────────────────────────────────────────────────────────
def calc_rsc(stock_close: pd.Series, nifty_close: pd.Series) -> float:
    """
    Mansfield RSC style:
      RSC = (Stock % change 52w) - (NIFTY % change 52w)
    Returns float.  Positive = outperforming.
    """
    try:
        common = stock_close.index.intersection(nifty_close.index)
        if len(common) < 52:
            return 0.0
        sc = stock_close.reindex(common).dropna()
        nc = nifty_close.reindex(common).dropna()
        if sc.empty or nc.empty:
            return 0.0
        stock_ret = (sc.iloc[-1] / sc.iloc[0]) - 1
        nifty_ret = (nc.iloc[-1] / nc.iloc[0]) - 1
        return round(float(stock_ret - nifty_ret) * 100, 2)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# VCP / BREAKOUT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_breakout(df: pd.DataFrame) -> Dict[str, float]:
    """
    Returns dict with:
      near_52w_high : 0–1 (1 = within 5% of 52w high)
      volatility_contraction : 0–1
      breakout_strength : 0–1
    """
    try:
        close  = df["Close"]
        volume = df["Volume"]
        high52w = close.tail(252).max()
        curr    = close.iloc[-1]
        near_high = max(0.0, 1.0 - (high52w - curr) / high52w / 0.10)
        near_high = min(near_high, 1.0)

        # Volatility contraction: ATR decreasing over last 10 days
        if len(df) >= 30:
            atr = calc_atr(df["High"], df["Low"], close)
            atr_recent = atr.iloc[-10:].mean()
            atr_older  = atr.iloc[-30:-10].mean()
            vc = max(0.0, min(1.0, (atr_older - atr_recent) / (atr_older + 1e-9)))
        else:
            vc = 0.0

        # Breakout: today's close > 20-day high AND volume surge
        high20 = df["High"].iloc[-21:-1].max() if len(df) > 21 else df["High"].iloc[:-1].max()
        vol20  = volume.iloc[-21:-1].mean()
        curr_vol = volume.iloc[-1]
        breakout_above  = 1.0 if curr > high20 else 0.0
        volume_surge    = min(1.0, curr_vol / (vol20 + 1) / 1.5)
        bk_strength     = breakout_above * 0.6 + volume_surge * 0.4

        return {"near_high": near_high, "vc": vc, "breakout": bk_strength}
    except Exception:
        return {"near_high": 0.0, "vc": 0.0, "breakout": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION  (per stock)
# ─────────────────────────────────────────────────────────────────────────────
def score_stock(ticker: str,
                df: pd.DataFrame,
                nifty_df: Optional[pd.DataFrame]) -> Optional[Dict]:
    """
    Returns a dict of scores + metadata for a single stock.
    Weights:
      Trend             25%
      Relative Strength 25%
      Momentum          20%
      Volume            15%
      Breakout          15%
    """
    try:
        if df is None or len(df) < 60:
            return None

        close  = df["Close"]
        volume = df["Volume"]
        high   = df["High"]
        low    = df["Low"]

        # ── 1. TREND SCORE (0–1) ──────────────────────────────────────────
        sma50  = calc_sma(close, 50)
        sma200 = calc_sma(close, 200) if len(close) >= 200 else pd.Series(dtype=float)
        ema50  = calc_ema(close, 50)
        ema200 = calc_ema(close, 200) if len(close) >= 200 else pd.Series(dtype=float)

        curr_close = float(close.iloc[-1])
        m50  = float(sma50.iloc[-1])
        m200 = float(sma200.iloc[-1]) if not sma200.empty else 0.0
        e50  = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1]) if not ema200.empty else 0.0

        stage = detect_stage(df)

        trend_pts = 0.0
        if curr_close > m50:                          trend_pts += 0.25
        if curr_close > m200 and m200 > 0:            trend_pts += 0.25
        if m50 > m200 and m200 > 0:                   trend_pts += 0.20
        if stage == 2:                                 trend_pts += 0.30

        # ── 2. RELATIVE STRENGTH SCORE (0–1) ─────────────────────────────
        rsc_raw = 0.0
        if nifty_df is not None and not nifty_df.empty:
            nifty_close = nifty_df["Close"].squeeze()
            rsc_raw = calc_rsc(close.tail(252), nifty_close.tail(252))
        rsc_score = max(0.0, min(1.0, (rsc_raw + 50) / 100))  # normalise ≈ 0-1

        # ── 3. MOMENTUM SCORE (0–1) ───────────────────────────────────────
        rsi14 = float(calc_rsi(close, 14).iloc[-1])
        macd_line, macd_sig, macd_hist = calc_macd(close)

        mom_pts = 0.0
        if rsi14 > 55:    mom_pts += 0.35
        if rsi14 > 65:    mom_pts += 0.20
        if rsi14 < 80:    mom_pts += 0.10  # not overbought
        macd_cross = (float(macd_line.iloc[-1]) > float(macd_sig.iloc[-1]) and
                      float(macd_line.iloc[-2]) <= float(macd_sig.iloc[-2]))
        macd_above = float(macd_line.iloc[-1]) > float(macd_sig.iloc[-1])
        if macd_cross:    mom_pts += 0.25
        elif macd_above:  mom_pts += 0.10
        mom_pts = min(mom_pts, 1.0)

        # ── 4. VOLUME SCORE (0–1) ─────────────────────────────────────────
        vol20   = float(volume.tail(21).iloc[:-1].mean())
        vol_today = float(volume.iloc[-1])
        vol_ratio = vol_today / (vol20 + 1)
        vol_score = min(1.0, vol_ratio / 2.0)   # 2x avg volume → score 1.0

        # ── 5. BREAKOUT SCORE (0–1) ───────────────────────────────────────
        bk = detect_breakout(df)
        bk_score = bk["near_high"] * 0.45 + bk["vc"] * 0.25 + bk["breakout"] * 0.30

        # ── COMPOSITE SCORE ───────────────────────────────────────────────
        composite = (
            trend_pts  * 0.25 +
            rsc_score  * 0.25 +
            mom_pts    * 0.20 +
            vol_score  * 0.15 +
            bk_score   * 0.15
        ) * 100  # scale to 100

        # ── 52-week change ────────────────────────────────────────────────
        chg_52w = ((curr_close / float(close.iloc[0])) - 1) * 100 if len(close) > 1 else 0.0

        # ── ATR % ─────────────────────────────────────────────────────────
        atr_pct = 0.0
        if len(df) >= 15:
            atr_val = float(calc_atr(high, low, close).iloc[-1])
            atr_pct = (atr_val / curr_close) * 100

        return {
            "Ticker"          : ticker,
            "Symbol"          : ticker.replace(".NS", ""),
            "Sector"          : get_sector(ticker),
            "Price"           : round(curr_close, 2),
            "Score"           : round(composite, 1),
            "Trend_Score"     : round(trend_pts * 100, 1),
            "RS_Score"        : round(rsc_score * 100, 1),
            "Momentum_Score"  : round(mom_pts * 100, 1),
            "Volume_Score"    : round(vol_score * 100, 1),
            "Breakout_Score"  : round(bk_score * 100, 1),
            "RSI"             : round(rsi14, 1),
            "RSC"             : round(rsc_raw, 1),
            "Stage"           : stage,
            "EMA50"           : round(e50, 2),
            "EMA200"          : round(e200, 2),
            "Vol_Ratio"       : round(vol_ratio, 2),
            "Chg_52W_pct"     : round(chg_52w, 1),
            "ATR_pct"         : round(atr_pct, 2),
            "MACD_Bullish"    : macd_above,
            "Near_52W_High"   : round(bk["near_high"] * 100, 1),
        }

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STOCK DETAIL CHART  (Plotly, interactive)
# ─────────────────────────────────────────────────────────────────────────────
def build_detail_chart(ticker: str, df: pd.DataFrame) -> go.Figure:
    """
    Multi-panel Plotly chart:
      Panel 1: Candlestick + EMA50 + EMA200 + Bollinger Bands
      Panel 2: Volume bars
      Panel 3: RSI
      Panel 4: MACD
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    open_  = df["Open"]
    volume = df["Volume"]

    ema50  = calc_ema(close, 50)
    ema200 = calc_ema(close, 200)
    rsi    = calc_rsi(close, 14)
    macd, macd_sig, macd_hist = calc_macd(close)
    bb_up, bb_mid, bb_low = calc_bollinger(close)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.15, 0.17, 0.18],
        subplot_titles=[
            f"{ticker.replace('.NS','')} — Price & Indicators",
            "Volume",
            "RSI (14)",
            "MACD (12,26,9)",
        ],
    )

    # ── Candlestick ───────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=open_, high=high, low=low, close=close,
        name="OHLCV",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        showlegend=False,
    ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_up, name="BB Upper",
        line=dict(color="rgba(100,116,139,0.5)", width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_low, name="BB Lower",
        line=dict(color="rgba(100,116,139,0.5)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(100,116,139,0.04)",
    ), row=1, col=1)

    # ── EMA 50 ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=ema50, name="EMA 50",
        line=dict(color="#f59e0b", width=1.8),
    ), row=1, col=1)

    # ── EMA 200 ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=ema200, name="EMA 200",
        line=dict(color="#a78bfa", width=1.8),
    ), row=1, col=1)

    # ── Buy signal marker: close > EMA50, above EMA200 ───────────────────
    buy_mask = (close > ema50) & (close > ema200) & (close.shift(1) <= ema50.shift(1))
    buy_dates = df.index[buy_mask]
    if len(buy_dates) > 0:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=low[buy_mask] * 0.99,
            mode="markers", name="Buy Signal",
            marker=dict(symbol="triangle-up", color="#22c55e", size=10),
        ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────
    vol_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(close, open_)
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=volume, name="Volume",
        marker_color=vol_colors, opacity=0.85, showlegend=False,
    ), row=2, col=1)
    vol_ma20 = calc_sma(volume, 20)
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_ma20, name="Vol MA20",
        line=dict(color="#f59e0b", width=1.2),
    ), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI(14)",
        line=dict(color="#60a5fa", width=1.5),
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", line_width=1, row=3, col=1)
    fig.add_hline(y=55, line_dash="dot", line_color="#22c55e", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#f97316", line_width=1, row=3, col=1)
    fig.add_hrect(y0=55, y1=70, fillcolor="rgba(34,197,94,0.05)",
                  line_width=0, row=3, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=df.index, y=macd_hist, name="MACD Hist",
        marker_color=[
            "#22c55e" if v >= 0 else "#ef4444" for v in macd_hist
        ], opacity=0.75, showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd, name="MACD",
        line=dict(color="#60a5fa", width=1.5),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_sig, name="Signal",
        line=dict(color="#f97316", width=1.5),
    ), row=4, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="#334155",
                  line_width=1, row=4, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter, JetBrains Mono", color="#94a3b8", size=11),
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=10),
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="#334155",
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    for r in range(1, 5):
        fig.update_yaxes(
            gridcolor="#1e293b", showgrid=True,
            zerolinecolor="#334155",
            row=r, col=1,
        )
        fig.update_xaxes(
            gridcolor="#1e293b", showgrid=True,
            row=r, col=1,
        )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SCORE DISTRIBUTION CHART
# ─────────────────────────────────────────────────────────────────────────────
def build_score_chart(df_scores: pd.DataFrame) -> go.Figure:
    top20 = df_scores.head(20).sort_values("Score")
    colors = [
        "#22c55e" if s >= 70
        else "#f59e0b" if s >= 55
        else "#60a5fa"
        for s in top20["Score"]
    ]
    fig = go.Figure(go.Bar(
        x=top20["Score"],
        y=top20["Symbol"],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}" for s in top20["Score"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=10),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter", color="#94a3b8", size=11),
        height=420,
        margin=dict(l=10, r=60, t=30, b=10),
        xaxis=dict(
            title="Composite Score (0–100)",
            gridcolor="#1e293b", range=[0, 115],
        ),
        yaxis=dict(gridcolor="#1e293b"),
    )
    return fig


def build_sector_chart(df_scores: pd.DataFrame) -> go.Figure:
    sector_avg = (
        df_scores.groupby("Sector")["Score"]
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )
    fig = px.bar(
        sector_avg, x="Sector", y="Score",
        color="Score",
        color_continuous_scale=["#1e40af", "#3b82f6", "#22c55e"],
        text=sector_avg["Score"].round(1),
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter", color="#94a3b8", size=11),
        height=320,
        margin=dict(l=10, r=10, t=10, b=60),
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-35, gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", range=[0, 110]),
    )
    return fig


def build_rsc_scatter(df_scores: pd.DataFrame) -> go.Figure:
    df_plot = df_scores.head(40).copy()
    df_plot["color"] = df_plot["Stage"].map(
        {2: "#22c55e", 1: "#f59e0b", 3: "#f97316", 4: "#ef4444", 0: "#64748b"}
    ).fillna("#64748b")

    fig = go.Figure(go.Scatter(
        x=df_plot["RSC"],
        y=df_plot["RSI"],
        mode="markers+text",
        text=df_plot["Symbol"],
        textposition="top center",
        textfont=dict(size=9, color="#94a3b8"),
        marker=dict(
            size=df_plot["Score"] / 6,
            color=df_plot["color"],
            opacity=0.85,
            line=dict(width=1, color="#0f172a"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "RSC: %{x:.1f}<br>"
            "RSI: %{y:.1f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=55, line_dash="dash", line_color="#334155", line_width=1)
    fig.add_vline(x=0,  line_dash="dash", line_color="#334155", line_width=1)
    fig.add_annotation(
        x=20, y=90, text="⭐ Sweet Spot",
        showarrow=False, font=dict(color="#22c55e", size=11),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter", color="#94a3b8", size=11),
        height=360,
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis=dict(title="RSC vs NIFTY (%)", gridcolor="#1e293b"),
        yaxis=dict(title="RSI (14)", gridcolor="#1e293b"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: STAGE BADGE
# ─────────────────────────────────────────────────────────────────────────────
def stage_badge(stage: int) -> str:
    badges = {
        1: "<span class='pill-yellow'>Stage 1</span>",
        2: "<span class='pill-green'>Stage 2 ✓</span>",
        3: "<span class='pill-yellow'>Stage 3</span>",
        4: "<span class='pill-red'>Stage 4</span>",
        0: "<span class='pill-red'>N/A</span>",
    }
    return badges.get(stage, "<span class='pill-red'>N/A</span>")


def score_color_class(score: float) -> str:
    if score >= 70:   return "score-high"
    if score >= 50:   return "score-medium"
    return "score-low"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
if "scan_done"      not in st.session_state: st.session_state.scan_done      = False
if "scores_df"      not in st.session_state: st.session_state.scores_df      = None
if "stock_data"     not in st.session_state: st.session_state.stock_data     = {}
if "nifty_df"       not in st.session_state: st.session_state.nifty_df       = None
if "selected_stock" not in st.session_state: st.session_state.selected_stock = None
if "last_scan_time" not in st.session_state: st.session_state.last_scan_time = None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
      <div style='font-size:2.4rem;'>📈</div>
      <div style='font-size:1.1rem; font-weight:700; color:#60a5fa;'>SEPA Screener</div>
      <div style='font-size:0.75rem; color:#475569;'>NSE India — Stage Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Scan Settings")
    max_stocks = st.slider(
        "Stocks to scan", min_value=50, max_value=len(NSE_TICKERS),
        value=min(200, len(NSE_TICKERS)), step=50,
        help="Higher = more thorough but slower"
    )
    top_n = st.slider("Top N to display", 5, 30, 15)
    data_period = st.selectbox(
        "Historical data window",
        ["6mo", "1y", "2y"], index=1,
        help="Longer windows give more accurate 200-DMA"
    )

    st.markdown("---")
    st.markdown("### 🔍 Filters")

    min_rsi = st.slider("Min RSI", 30, 80, 50,
                        help="Filter stocks with RSI above this value")
    min_vol_ratio = st.slider("Min Volume Ratio", 0.5, 3.0, 0.8, 0.1,
                              help="Min(Today's Vol / 20-day avg vol)")
    min_score = st.slider("Min Composite Score", 0, 90, 40)

    all_sectors = sorted(set(SECTOR_MAP.values()))
    sector_filter = st.multiselect(
        "Sector filter", all_sectors,
        placeholder="All sectors",
        help="Leave blank to include all sectors"
    )
    stage2_only = st.checkbox("Stage 2 only (SEPA)", value=False)

    st.markdown("---")
    scan_button = st.button("🚀 Run Full Scan", use_container_width=True, type="primary")
    if st.session_state.last_scan_time:
        st.caption(f"Last scan: {st.session_state.last_scan_time}")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; line-height:1.6;'>
    <b>Methodology</b><br>
    • SEPA Stage Analysis (Minervini)<br>
    • RSC vs NIFTY 50<br>
    • RSI + MACD Momentum<br>
    • VCP / Darvas Breakout<br>
    • Weighted Composite Score<br><br>
    <b>Disclaimer:</b> Educational purposes only.
    Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>📈 SEPA Stock Screener Dashboard</h1>
  <p>NSE India · Stage Analysis + CANSLIM + Relative Strength · Powered by yfinance + Streamlit</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCAN LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def run_scan(tickers_to_scan: List[str], period: str):
    """Main scan pipeline — fetch → score → rank."""
    results = []

    # Step 1: NIFTY index data
    nifty_status = st.empty()
    nifty_status.info("📥 Fetching NIFTY 50 index data…")
    nifty_df = fetch_nifty_data(period)
    if nifty_df is None or nifty_df.empty:
        st.warning("⚠️ NIFTY data unavailable — RSC scores will be 0")
    nifty_status.empty()
    st.session_state.nifty_df = nifty_df

    # Step 2: stock price data
    prog_bar   = st.progress(0, text="📦 Downloading stock data…")
    batch_size = 50
    all_data: Dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers_to_scan), batch_size):
        batch = tickers_to_scan[i : i + batch_size]
        prog  = min((i + batch_size) / len(tickers_to_scan), 1.0)
        prog_bar.progress(prog, text=f"📦 Downloading batch {i//batch_size+1}…")
        batch_data = fetch_stock_batch(batch, period=period)
        all_data.update(batch_data)

    prog_bar.empty()
    st.session_state.stock_data = all_data

    # Step 3: score each stock
    score_bar = st.progress(0, text="🔢 Scoring stocks…")
    nifty_close_series = nifty_df["Close"].squeeze() if nifty_df is not None else None

    for idx, ticker in enumerate(tickers_to_scan):
        score_bar.progress((idx + 1) / len(tickers_to_scan),
                           text=f"🔢 Scoring {ticker} ({idx+1}/{len(tickers_to_scan)})…")
        df = all_data.get(ticker)
        if df is None:
            continue
        score_dict = score_stock(ticker, df,
                                 nifty_df if nifty_df is not None else None)
        if score_dict:
            results.append(score_dict)

    score_bar.empty()

    if not results:
        st.error("❌ No results returned. Check network / yfinance API.")
        return None

    df_scores = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    df_scores.index = df_scores.index + 1  # 1-based rank
    return df_scores


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER SCAN
# ─────────────────────────────────────────────────────────────────────────────
if scan_button:
    tickers_to_scan = NSE_TICKERS[:max_stocks]
    with st.spinner(""):
        df_result = run_scan(tickers_to_scan, data_period)
    if df_result is not None:
        st.session_state.scores_df      = df_result
        st.session_state.scan_done      = True
        st.session_state.last_scan_time = datetime.now().strftime("%d %b %Y %H:%M:%S")
        st.session_state.selected_stock = None
        st.success(
            f"✅ Scan complete — {len(df_result)} stocks scored | "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.scan_done or st.session_state.scores_df is None:
    # ── Welcome / placeholder ──────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='kpi-card'>
          <div class='kpi-value kpi-blue'>SEPA</div>
          <div class='kpi-label'>Stage Analysis Method</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='kpi-card'>
          <div class='kpi-value kpi-green'>5-Factor</div>
          <div class='kpi-label'>Scoring Model</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='kpi-card'>
          <div class='kpi-value kpi-purple'>NSE</div>
          <div class='kpi-label'>Indian Stock Market</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "👈 **Configure your scan settings** in the sidebar, then click "
        "**🚀 Run Full Scan** to start screening NSE stocks.\n\n"
        "The scanner will analyse each stock across **Trend, Relative Strength, "
        "Momentum, Volume, and Breakout** criteria and rank them by composite score."
    )

    st.markdown("<div class='section-title'>How the Scoring Works</div>", unsafe_allow_html=True)
    weight_data = {
        "Factor": ["Trend (SEPA)", "Relative Strength", "Momentum", "Volume", "Breakout"],
        "Weight": ["25%", "25%", "20%", "15%", "15%"],
        "Signals": [
            "Price > EMA50 > EMA200, Stage 2",
            "RSC vs NIFTY 50 (1-year return diff)",
            "RSI > 55, MACD bullish crossover",
            "Today's vol > 20-day avg",
            "Near 52w high, VCP, vol surge",
        ],
    }
    st.dataframe(
        pd.DataFrame(weight_data),
        use_container_width=True,
        hide_index=True,
    )

else:
    # ── Apply sidebar filters ──────────────────────────────────────────────
    df = st.session_state.scores_df.copy()

    if min_rsi:         df = df[df["RSI"]       >= min_rsi]
    if min_vol_ratio:   df = df[df["Vol_Ratio"]  >= min_vol_ratio]
    if min_score:       df = df[df["Score"]      >= min_score]
    if stage2_only:     df = df[df["Stage"]      == 2]
    if sector_filter:   df = df[df["Sector"].isin(sector_filter)]

    df_top = df.head(top_n).reset_index(drop=True)
    df_top.index = df_top.index + 1

    # ── KPI Row ───────────────────────────────────────────────────────────
    total_scanned  = len(st.session_state.scores_df)
    after_filter   = len(df)
    stage2_count   = len(df[df["Stage"] == 2])
    avg_rsi        = df_top["RSI"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-value kpi-blue'>{total_scanned}</div>
          <div class='kpi-label'>Stocks Scanned</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-value kpi-green'>{after_filter}</div>
          <div class='kpi-label'>After Filters</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-value kpi-purple'>{stage2_count}</div>
          <div class='kpi-label'>Stage 2 Stocks</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-value kpi-orange'>{avg_rsi:.1f}</div>
          <div class='kpi-label'>Avg RSI (Top {top_n})</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        top_score = df_top["Score"].iloc[0] if len(df_top) > 0 else 0
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-value kpi-green'>{top_score:.1f}</div>
          <div class='kpi-label'>Highest Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top Ranked Stocks",
        "📊 Analytics & Charts",
        "🔍 Stock Detail",
        "📋 Full Results",
    ])

    # ════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(
            f"<div class='section-title'>Top {top_n} Stocks by Composite Score</div>",
            unsafe_allow_html=True,
        )

        if df_top.empty:
            st.warning("⚠️ No stocks match the current filters. Try relaxing the criteria.")
        else:
            # Display table with colour-coded score column
            display_cols = [
                "Symbol", "Sector", "Price", "Score",
                "RSI", "RSC", "Stage", "Vol_Ratio",
                "Trend_Score", "RS_Score", "Momentum_Score",
                "Breakout_Score", "Chg_52W_pct",
            ]
            disp = df_top[display_cols].copy()
            disp.columns = [
                "Symbol", "Sector", "Price (₹)", "Score",
                "RSI", "RSC (%)", "Stage", "Vol Ratio",
                "Trend", "Rel.Str", "Momentum",
                "Breakout", "52W Chg%",
            ]

            def style_table(styler):
                styler.background_gradient(
                    subset=["Score"], cmap="RdYlGn",
                    vmin=30, vmax=100,
                )
                styler.background_gradient(
                    subset=["RSI"], cmap="Blues",
                    vmin=40, vmax=80,
                )
                styler.background_gradient(
                    subset=["RSC (%)"], cmap="RdYlGn",
                    vmin=-30, vmax=50,
                )
                styler.background_gradient(
                    subset=["52W Chg%"], cmap="RdYlGn",
                    vmin=-20, vmax=100,
                )
                styler.format({
                    "Price (₹)": "₹{:.2f}",
                    "Score":     "{:.1f}",
                    "RSI":       "{:.1f}",
                    "RSC (%)":   "{:+.1f}%",
                    "Vol Ratio": "{:.2f}x",
                    "Trend":     "{:.0f}",
                    "Rel.Str":   "{:.0f}",
                    "Momentum":  "{:.0f}",
                    "Breakout":  "{:.0f}",
                    "52W Chg%":  "{:+.1f}%",
                })
                return styler

            st.dataframe(
                disp.style.pipe(style_table),
                use_container_width=True,
                height=420,
            )

            # ── Quick score bar chart ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            fig_bar = build_score_chart(df_top)
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Card view for top 5 ───────────────────────────────────
            st.markdown(
                "<div class='section-title'>🥇 Top 5 — Spotlight</div>",
                unsafe_allow_html=True,
            )
            top5 = df_top.head(5)
            cols = st.columns(5)
            for i, (_, row) in enumerate(top5.iterrows()):
                with cols[i]:
                    sc_cls = score_color_class(row["Score"])
                    macd_txt = "✅ Bull" if row["MACD_Bullish"] else "⛔ Bear"
                    st.markdown(f"""
                    <div class='kpi-card' style='padding:14px;'>
                      <div style='font-size:1.0rem;font-weight:700;color:#e2e8f0;'>
                        {row["Symbol"]}
                      </div>
                      <div style='font-size:0.72rem;color:#64748b;margin-bottom:8px;'>
                        {row["Sector"]}
                      </div>
                      <div class='kpi-value {sc_cls}'>{row["Score"]:.1f}</div>
                      <div class='kpi-label'>Composite Score</div>
                      <div style='margin-top:8px;font-size:0.8rem;'>
                        ₹{row["Price"]:.2f}<br>
                        RSI: <b>{row["RSI"]:.0f}</b> &nbsp;
                        RSC: <b>{row["RSC"]:+.0f}%</b><br>
                        MACD: {macd_txt}
                      </div>
                      {stage_badge(int(row["Stage"]))}
                    </div>
                    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    with tab2:
        c_l, c_r = st.columns(2)

        with c_l:
            st.markdown(
                "<div class='section-title'>📊 Sector Avg Score (Top 12)</div>",
                unsafe_allow_html=True,
            )
            fig_sec = build_sector_chart(df)
            st.plotly_chart(fig_sec, use_container_width=True)

        with c_r:
            st.markdown(
                "<div class='section-title'>🎯 RSC vs RSI Scatter (Top 40)</div>",
                unsafe_allow_html=True,
            )
            st.caption("Bubble size = Score | Colour: 🟢 Stage 2 | 🟡 Stage 1 | 🔴 Stage 4")
            fig_scatter = build_rsc_scatter(df)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Score distribution histogram
        st.markdown(
            "<div class='section-title'>📈 Score Distribution (All Filtered Stocks)</div>",
            unsafe_allow_html=True,
        )
        fig_hist = px.histogram(
            df, x="Score", nbins=30,
            color_discrete_sequence=["#3b82f6"],
        )
        fig_hist.add_vline(
            x=df["Score"].mean(), line_dash="dash",
            line_color="#f59e0b", annotation_text="Mean",
            annotation_font_color="#f59e0b",
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter", color="#94a3b8"),
            height=280,
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Stage breakdown
        st.markdown(
            "<div class='section-title'>🏗️ Stage Analysis Breakdown</div>",
            unsafe_allow_html=True,
        )
        stage_counts = df["Stage"].value_counts().reset_index()
        stage_counts.columns = ["Stage", "Count"]
        stage_counts["Stage"] = stage_counts["Stage"].map(
            {1: "Stage 1 – Base", 2: "Stage 2 – Advance ✅",
             3: "Stage 3 – Top",  4: "Stage 4 – Decline", 0: "Unknown"}
        )
        fig_stage = px.pie(
            stage_counts, names="Stage", values="Count",
            color_discrete_sequence=["#f59e0b", "#22c55e", "#f97316", "#ef4444", "#64748b"],
            hole=0.45,
        )
        fig_stage.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            font=dict(family="Inter", color="#94a3b8"),
            height=280,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_stage, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            "<div class='section-title'>🔍 Individual Stock Analysis</div>",
            unsafe_allow_html=True,
        )

        # Symbol selector
        symbol_options = df_top["Symbol"].tolist() + \
                         [s for s in df["Symbol"].tolist() if s not in df_top["Symbol"].tolist()]
        selected_symbol = st.selectbox(
            "Select a stock to analyse",
            symbol_options,
            index=0 if symbol_options else 0,
        )

        if selected_symbol:
            ticker_sel = f"{selected_symbol}.NS"
            stock_df = st.session_state.stock_data.get(ticker_sel)
            row = df[df["Symbol"] == selected_symbol]

            if stock_df is None or stock_df.empty:
                st.error(f"❌ No price data for {selected_symbol}. Try re-running the scan.")
            elif row.empty:
                st.warning(f"⚠️ {selected_symbol} not in current filtered results.")
            else:
                row = row.iloc[0]

                # ── Metric strip ──────────────────────────────────────
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Price", f"₹{row['Price']:.2f}")
                m2.metric("Score", f"{row['Score']:.1f}")
                m3.metric("RSI", f"{row['RSI']:.1f}",
                          delta="Bullish" if row["RSI"] > 55 else "Caution")
                m4.metric("RSC", f"{row['RSC']:+.1f}%",
                          delta="vs NIFTY")
                m5.metric("Vol Ratio", f"{row['Vol_Ratio']:.2f}x")
                m6.metric("52W Chg", f"{row['Chg_52W_pct']:+.1f}%")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Candlestick chart ─────────────────────────────────
                with st.spinner(f"Building chart for {selected_symbol}…"):
                    fig_detail = build_detail_chart(ticker_sel, stock_df)
                st.plotly_chart(fig_detail, use_container_width=True)

                # ── Score breakdown radar ─────────────────────────────
                st.markdown(
                    "<div class='section-title'>🎯 Score Component Breakdown</div>",
                    unsafe_allow_html=True,
                )
                cat_names = ["Trend", "Rel. Strength", "Momentum", "Volume", "Breakout"]
                cat_vals  = [
                    row["Trend_Score"], row["RS_Score"],
                    row["Momentum_Score"], row["Volume_Score"], row["Breakout_Score"],
                ]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=cat_vals + [cat_vals[0]],
                    theta=cat_names + [cat_names[0]],
                    fill="toself",
                    fillcolor="rgba(59,130,246,0.15)",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=6, color="#60a5fa"),
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100],
                                        gridcolor="#1e293b", tickcolor="#475569",
                                        tickfont=dict(size=9)),
                        angularaxis=dict(gridcolor="#1e293b",
                                         tickfont=dict(size=10, color="#94a3b8")),
                        bgcolor="#0f172a",
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#0a0e1a",
                    font=dict(family="Inter", color="#94a3b8"),
                    height=340,
                    margin=dict(l=40, r=40, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # ── Key signals ───────────────────────────────────────
                st.markdown(
                    "<div class='section-title'>📋 Signal Summary</div>",
                    unsafe_allow_html=True,
                )
                curr_price = row["Price"]
                e50 = row["EMA50"]
                e200 = row["EMA200"]

                signals = []
                signals.append(("📈 Price > EMA 50",
                                 "✅ Yes" if curr_price > e50 > 0 else "❌ No",
                                 curr_price > e50 > 0))
                signals.append(("📈 Price > EMA 200",
                                 "✅ Yes" if curr_price > e200 > 0 else "❌ No",
                                 curr_price > e200 > 0))
                signals.append(("📐 EMA 50 > EMA 200",
                                 "✅ Yes" if e50 > e200 > 0 else "❌ No",
                                 e50 > e200 > 0))
                signals.append(("🎯 Stage 2 (SEPA)",
                                 "✅ Yes" if row["Stage"] == 2 else f"❌ Stage {row['Stage']}",
                                 row["Stage"] == 2))
                signals.append(("💪 RSI > 55",
                                 f"✅ {row['RSI']:.1f}" if row["RSI"] > 55 else f"❌ {row['RSI']:.1f}",
                                 row["RSI"] > 55))
                signals.append(("📊 MACD Bullish",
                                 "✅ Yes" if row["MACD_Bullish"] else "❌ No",
                                 row["MACD_Bullish"]))
                signals.append(("📦 Volume Surge",
                                 f"✅ {row['Vol_Ratio']:.2f}x" if row["Vol_Ratio"] > 1.0 else f"❌ {row['Vol_Ratio']:.2f}x",
                                 row["Vol_Ratio"] > 1.0))
                signals.append(("🎯 Near 52W High",
                                 f"✅ {row['Near_52W_High']:.0f}%" if row["Near_52W_High"] > 50 else f"⚠️ {row['Near_52W_High']:.0f}%",
                                 row["Near_52W_High"] > 50))
                signals.append(("🌍 Outperforms NIFTY",
                                 f"✅ +{row['RSC']:.1f}%" if row["RSC"] > 0 else f"❌ {row['RSC']:.1f}%",
                                 row["RSC"] > 0))

                sig_cols = st.columns(3)
                for i, (label, val, ok) in enumerate(signals):
                    with sig_cols[i % 3]:
                        color = "#22c55e" if ok else "#ef4444"
                        st.markdown(f"""
                        <div style='background:#1e293b;border:1px solid #334155;
                                    border-radius:8px;padding:10px 14px;margin-bottom:8px;
                                    border-left:3px solid {color};'>
                          <div style='font-size:0.8rem;color:#94a3b8;'>{label}</div>
                          <div style='font-size:0.95rem;font-weight:600;color:{color};'>{val}</div>
                        </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown(
            f"<div class='section-title'>📋 All Filtered Results ({len(df)} stocks)</div>",
            unsafe_allow_html=True,
        )

        # Download button
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"sepa_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        st.dataframe(
            df[[
                "Symbol", "Sector", "Price", "Score",
                "RSI", "RSC", "Stage", "Vol_Ratio",
                "Trend_Score", "RS_Score", "Momentum_Score",
                "Volume_Score", "Breakout_Score",
                "Chg_52W_pct", "ATR_pct", "MACD_Bullish",
            ]].style.format({
                "Price":           "₹{:.2f}",
                "Score":           "{:.1f}",
                "RSI":             "{:.1f}",
                "RSC":             "{:+.1f}%",
                "Vol_Ratio":       "{:.2f}x",
                "Trend_Score":     "{:.0f}",
                "RS_Score":        "{:.0f}",
                "Momentum_Score":  "{:.0f}",
                "Volume_Score":    "{:.0f}",
                "Breakout_Score":  "{:.0f}",
                "Chg_52W_pct":     "{:+.1f}%",
                "ATR_pct":         "{:.2f}%",
            }).background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True,
            height=500,
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  SEPA Stock Screener &nbsp;|&nbsp; NSE India &nbsp;|&nbsp;
  Built with Streamlit + yfinance + Plotly &nbsp;|&nbsp;
  <b>⚠️ For educational purposes only. Not financial advice.</b><br>
  Methodology: Stage Analysis (Minervini) · CANSLIM · RSC · MACD · VCP Breakout
</div>
""", unsafe_allow_html=True)