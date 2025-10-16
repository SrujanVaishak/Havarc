import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import pyotp
import math
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
import numpy as np

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True
EXPIRY_INFO_ONLY = False
EXPIRY_RELAX_FACTOR = 0.7
GAMMA_VOL_SPIKE_THRESHOLD = 1.8
DELTA_OI_RATIO = 1.5
MOMENTUM_VOL_AMPLIFIER = 1.3

# New strategy configurations
VCP_CONTRACTION_RATIO = 0.65
FAULTY_BASE_BREAK_THRESHOLD = 0.15
WYCKOFF_VOLUME_SPRING = 1.8
LIQUIDITY_SWEEP_DISTANCE = 0.003
PEAK_REJECTION_WICK_RATIO = 0.7
FVG_GAP_THRESHOLD = 0.0015
VOLUME_GAP_IMBALANCE = 2.0
OTE_RETRACEMENT_LEVELS = [0.618, 0.786]
DEMAND_SUPPLY_ZONE_LOOKBACK = 20

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
feedToken = client.getfeedToken()

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STARTED_SENT = False
STOP_SENT = False

def send_telegram(msg, reply_to=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except:
        return None

# --------- MARKET HOURS ---------
def is_market_open():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

def should_stop_trading():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return current_time_ist >= dtime(15,30)

# --------- STRIKE ROUNDING FOR ALL INDICES ---------
def round_strike(index, price):
    try:
        if price is None:
            return None
        if isinstance(price, float) and math.isnan(price):
            return None
        price = float(price)
        
        if index == "NIFTY": 
            return int(round(price / 50.0) * 50)
        elif index == "BANKNIFTY": 
            return int(round(price / 100.0) * 100)
        elif index == "SENSEX": 
            return int(round(price / 100.0) * 100)
        elif index == "FINNIFTY": 
            return int(round(price / 50.0) * 50)
        elif index == "MIDCPNIFTY": 
            return int(round(price / 25.0) * 25)
        elif index == "EICHERMOT": 
            return int(round(price / 50.0) * 50)
        elif index == "TRENT": 
            return int(round(price / 100.0) * 100)
        elif index == "RELIANCE": 
            return int(round(price / 10.0) * 10)
        else: 
            return int(round(price / 50.0) * 50)
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FETCH INDEX DATA FOR ALL INDICES ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS", 
        "EICHERMOT": "EICHERMOT.NS",
        "TRENT": "TRENT.NS",
        "RELIANCE": "RELIANCE.NS"
    }
    df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
    return None if df.empty else df

# --------- LOAD TOKEN MAP ---------
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        return df.set_index('symbol')['token'].to_dict()
    except:
        return {}

token_map=load_token_map()

# --------- SAFE LTP FETCH ---------
def fetch_option_price(symbol, retries=3, delay=3):
    token=token_map.get(symbol.upper())
    if not token:
        return None
    for _ in range(retries):
        try:
            exchange = "BFO" if "SENSEX" in symbol.upper() else "NFO"
            data=client.ltpData(exchange, symbol, token)
            return float(data['data']['ltp'])
        except:
            time.sleep(delay)
    return None

# --------- DETECT LIQUIDITY ZONE ---------
def detect_liquidity_zone(df, lookback=20):
    high_series = ensure_series(df['High']).dropna()
    low_series = ensure_series(df['Low']).dropna()
    try:
        if len(high_series) <= lookback:
            high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
        else:
            high_pool = float(high_series.rolling(lookback).max().iloc[-2])
    except Exception:
        high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
    try:
        if len(low_series) <= lookback:
            low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')
        else:
            low_pool = float(low_series.rolling(lookback).min().iloc[-2])
    except Exception:
        low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')

    if math.isnan(high_pool) and len(high_series)>0:
        high_pool = float(high_series.max())
    if math.isnan(low_pool) and len(low_series)>0:
        low_pool = float(low_series.min())

    return round(high_pool,0), round(low_pool,0)

# --------- INSTITUTIONAL LIQUIDITY HUNT ---------
def institutional_liquidity_hunt(index, df):
    prev_high = None
    prev_low = None
    try:
        prev_high_val = ensure_series(df['High']).iloc[-2]
        prev_low_val = ensure_series(df['Low']).iloc[-2]
        prev_high = float(prev_high_val) if not (isinstance(prev_high_val,float) and math.isnan(prev_high_val)) else None
        prev_low = float(prev_low_val) if not (isinstance(prev_low_val,float) and math.isnan(prev_low_val)) else None
    except Exception:
        prev_high = None
        prev_low = None

    high_zone, low_zone = detect_liquidity_zone(df, lookback=15)

    last_close_val = None
    try:
        lc = ensure_series(df['Close']).iloc[-1]
        if isinstance(lc, float) and math.isnan(lc):
            last_close_val = None
        else:
            last_close_val = float(lc)
    except Exception:
        last_close_val = None

    if last_close_val is None:
        highest_ce_oi_strike = None
        highest_pe_oi_strike = None
    else:
        highest_ce_oi_strike = round_strike(index, last_close_val + 50)
        highest_pe_oi_strike = round_strike(index, last_close_val - 50)

    bull_liquidity = []
    if prev_low is not None: bull_liquidity.append(prev_low)
    if low_zone is not None: bull_liquidity.append(low_zone)
    if highest_pe_oi_strike is not None: bull_liquidity.append(highest_pe_oi_strike)

    bear_liquidity = []
    if prev_high is not None: bear_liquidity.append(prev_high)
    if high_zone is not None: bear_liquidity.append(high_zone)
    if highest_ce_oi_strike is not None: bear_liquidity.append(highest_ce_oi_strike)

    return bull_liquidity, bear_liquidity

def liquidity_zone_entry_check(price, bull_liq, bear_liq):
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None

    for zone in bull_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:
                return "CE"
        except:
            continue
    for zone in bear_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:
                return "PE"
        except:
            continue

    valid_bear = [z for z in bear_liq if z is not None]
    valid_bull = [z for z in bull_liq if z is not None]
    if valid_bear and valid_bull:
        try:
            if price > max(valid_bear) or price < min(valid_bull):
                return "BOTH"
        except:
            return None
    return None

# 🚨 LAYER 1: OPENING-RANGE INSTITUTIONAL PLAY 🚨
def institutional_opening_play(index, df):
    try:
        prev_high = float(ensure_series(df['High']).iloc[-2])
        prev_low = float(ensure_series(df['Low']).iloc[-2])
        prev_close = float(ensure_series(df['Close']).iloc[-2])
        current_price = float(ensure_series(df['Close']).iloc[-1])
    except Exception:
        return None
    if current_price > prev_high + 10: return "CE"
    if current_price < prev_low - 10: return "PE"
    if current_price > prev_close + 20: return "CE"
    if current_price < prev_close - 20: return "PE"
    return None

# 🚨 LAYER 2: GAMMA SQUEEZE / EXPIRY LAYER 🚨
EXPIRIES = {
    "NIFTY": "20 OCT 2025",
    "BANKNIFTY": "16 OCT 2025",
    "SENSEX": "16 OCT 2025",
    "FINNIFTY": "16 OCT 2025",
    "MIDCPNIFTY": "16 OCT 2025",
    "EICHERMOT": "31 OCT 2025",
    "TRENT": "31 OCT 2025",
    "RELIANCE": "28 OCT 2025"
}

def is_expiry_day_for_index(index):
    try:
        ex = EXPIRIES.get(index)
        if not ex: return False
        dt = datetime.strptime(ex, "%d %b %Y")
        today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()
        return dt.date() == today
    except Exception:
        return False

def detect_gamma_squeeze(index, df):
    try:
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume']); 
        high = ensure_series(df['High']); low = ensure_series(df['Low'])
        if len(close) < 6: return None
        
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg>0 else 1)
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        
        try:
            url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            df_s = pd.DataFrame(requests.get(url,timeout=10).json())
            df_s['symbol'] = df_s['symbol'].str.upper()
            df_index = df_s[df_s['symbol'].str.contains(index)]
            df_index['oi'] = pd.to_numeric(df_index.get('oi',0), errors='coerce').fillna(0)
            ce_oi = df_index[df_index['symbol'].str.endswith("CE")]['oi'].sum()
            pe_oi = df_index[df_index['symbol'].str.endswith("PE")]['oi'].sum()
        except Exception:
            ce_oi = pe_oi = 0
        
        if vol_ratio > GAMMA_VOL_SPIKE_THRESHOLD and abs(speed) > 0.002:
            if speed > 0:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (ce_oi / (pe_oi+1e-6)) * 0.1)
                return {'side':'CE','confidence':conf}
            else:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (pe_oi / (ce_oi+1e-6)) * 0.1)
                return {'side':'PE','confidence':conf}
    except Exception:
        return None
    return None

# 🚨 LAYER 3: SMART-MONEY DIVERGENCE 🚨
def smart_money_divergence(df):
    try:
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume'])
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()
        if len(close) < 10: return None
        
        p_short = close.iloc[-5]; p_now = close.iloc[-1]
        rsi_short = rsi.iloc[-5]; rsi_now = rsi.iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        vol_now = volume.iloc[-1]
        
        if p_now < p_short and rsi_now > rsi_short and vol_now > vol_avg*1.1:
            return "CE"
        if p_now > p_short and rsi_now < rsi_short and vol_now > vol_avg*1.1:
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 4: STOP-HUNT DETECTOR 🚨
def detect_stop_hunt(df):
    try:
        high = ensure_series(df['High']); low = ensure_series(df['Low']); 
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume'])
        if len(close) < 6: return None
        
        recent_high = high.iloc[-6:-1].max(); recent_low = low.iloc[-6:-1].min()
        last_high = high.iloc[-1]; last_low = low.iloc[-1]; last_close = close.iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        
        if last_high > recent_high * 1.002 and last_close < recent_high and volume.iloc[-1] > vol_avg*1.2:
            return "PE"
        if last_low < recent_low * 0.998 and last_close > recent_low and volume.iloc[-1] > vol_avg*1.2:
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 5: INSTITUTIONAL CONTINUATION 🚨
def detect_institutional_continuation(df):
    try:
        close = ensure_series(df['Close']); high = ensure_series(df['High']); 
        low = ensure_series(df['Low']); volume = ensure_series(df['Volume'])
        if len(close) < 10: return None
        
        atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.2 and speed > 0.004:
            return "CE"
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.2 and speed < -0.004:
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 6: PULLBACK REVERSAL 🚨
def detect_pullback_reversal(df):
    try:
        close = ensure_series(df['Close'])
        ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        if len(close) < 6:
            return None

        if close.iloc[-6] > ema21.iloc[-6] and close.iloc[-3] <= ema21.iloc[-3] and close.iloc[-1] > ema9.iloc[-1] and rsi.iloc[-1] > 50:
            return "CE"

        if close.iloc[-6] < ema21.iloc[-6] and close.iloc[-3] >= ema21.iloc[-3] and close.iloc[-1] < ema9.iloc[-1] and rsi.iloc[-1] < 50:
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 7: ORDERFLOW MIMIC LOGIC 🚨
def mimic_orderflow_logic(df):
    try:
        close = ensure_series(df['Close']); high = ensure_series(df['High']); 
        low = ensure_series(df['Low']); volume = ensure_series(df['Volume'])
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        if len(close) < 4:
            return None

        body = (high - low).abs(); wick_top = (high - close).abs(); wick_bottom = (close - low).abs()
        body_last = body.iloc[-1] if body.iloc[-1] != 0 else 1.0
        wick_top_ratio = wick_top.iloc[-1] / body_last
        wick_bottom_ratio = wick_bottom.iloc[-1] / body_last
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg and vol_avg > 0 else 1)

        if close.iloc[-1] > close.iloc[-3] and rsi.iloc[-1] < rsi.iloc[-3] and wick_top_ratio > 0.6 and vol_ratio > 1.2:
            return "PE"

        if close.iloc[-1] < close.iloc[-3] and rsi.iloc[-1] > rsi.iloc[-3] and wick_bottom_ratio > 0.6 and vol_ratio > 1.2:
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 8: VCP (Volatility Contraction Pattern) 🚨
def detect_vcp_pattern(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        atr = ta.volatility.AverageTrueRange(high, low, close, 5).average_true_range()
        
        recent_atr = atr.iloc[-1]
        prev_atr = atr.iloc[-5]
        
        recent_vol = volume.iloc[-5:].mean()
        prev_vol = volume.iloc[-10:-5].mean()
        
        if (recent_atr < prev_atr * VCP_CONTRACTION_RATIO and 
            recent_vol < prev_vol and
            close.iloc[-1] > close.iloc[-5]):
            return "CE"
        elif (recent_atr < prev_atr * VCP_CONTRACTION_RATIO and 
              recent_vol < prev_vol and
              close.iloc[-1] < close.iloc[-5]):
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 9: FAULTY BASES 🚨
def detect_faulty_bases(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 8:
            return None
            
        recent_high = high.iloc[-8:-3].max()
        recent_low = low.iloc[-8:-3].min()
        current_close = close.iloc[-1]
        
        if (high.iloc[-4] > recent_high * (1 + FAULTY_BASE_BREAK_THRESHOLD/100) and
            current_close < recent_high and
            volume.iloc[-4] > volume.iloc[-5:].mean()):
            return "PE"
            
        if (low.iloc[-4] < recent_low * (1 - FAULTY_BASE_BREAK_THRESHOLD/100) and
            current_close > recent_low and
            volume.iloc[-4] > volume.iloc[-5:].mean()):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 10: WYCKOFF SCHEMATICS 🚨
def detect_wyckoff_schematic(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 15:
            return None
            
        spring_low = low.iloc[-5]
        support_level = low.iloc[-10:-5].min()
        spring_volume = volume.iloc[-5]
        avg_volume = volume.iloc[-10:].mean()
        
        if (spring_low < support_level * 0.995 and
            close.iloc[-1] > support_level and
            spring_volume > avg_volume * WYCKOFF_VOLUME_SPRING):
            return "CE"
            
        upthrust_high = high.iloc[-5]
        resistance_level = high.iloc[-10:-5].max()
        upthrust_volume = volume.iloc[-5]
        
        if (upthrust_high > resistance_level * 1.005 and
            close.iloc[-1] < resistance_level and
            upthrust_volume > avg_volume * WYCKOFF_VOLUME_SPRING):
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 11: LIQUIDITY SWEEPS 🚨
def detect_liquidity_sweeps(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        recent_highs = high.iloc[-10:-2]
        recent_lows = low.iloc[-10:-2]
        
        liquidity_high = recent_highs.max()
        liquidity_low = recent_lows.min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        if (current_high > liquidity_high * (1 + LIQUIDITY_SWEEP_DISTANCE) and
            current_close < liquidity_high and
            volume.iloc[-1] > volume.iloc[-10:-1].mean()):
            return "PE"
            
        if (current_low < liquidity_low * (1 - LIQUIDITY_SWEEP_DISTANCE) and
            current_close > liquidity_low and
            volume.iloc[-1] > volume.iloc[-10:-1].mean()):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 12: PEAK REJECTION 🚨
def detect_peak_rejection(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 5:
            return None
            
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        body_size = abs(current_close - close.iloc[-2])
        upper_wick = current_high - max(close.iloc[-1], close.iloc[-2])
        lower_wick = min(close.iloc[-1], close.iloc[-2]) - current_low
        
        if (upper_wick > body_size * PEAK_REJECTION_WICK_RATIO and
            current_close < (current_high + current_low) / 2):
            return "PE"
            
        if (lower_wick > body_size * PEAK_REJECTION_WICK_RATIO and
            current_close > (current_high + current_low) / 2):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 13: FAIR VALUE GAP (FVG) 🚨
def detect_fair_value_gap(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        
        if len(close) < 3:
            return None
            
        if (low.iloc[-1] > high.iloc[-2] * (1 + FVG_GAP_THRESHOLD) and
            close.iloc[-1] > close.iloc[-2]):
            return "CE"
            
        if (high.iloc[-1] < low.iloc[-2] * (1 - FVG_GAP_THRESHOLD) and
            close.iloc[-1] < close.iloc[-2]):
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 14: VOLUME GAP IMBALANCE 🚨
def detect_volume_gap_imbalance(df):
    try:
        volume = ensure_series(df['Volume'])
        close = ensure_series(df['Close'])
        
        if len(volume) < 20:
            return None
            
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-20:].mean()
        price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        
        if (current_volume > avg_volume * VOLUME_GAP_IMBALANCE and
            abs(price_change) > 0.002):
            if price_change > 0:
                return "CE"
            else:
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 15: OTE (Optimal Trade Entry) 🚨
def detect_ote_retracement(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        
        if len(close) < 15:
            return None
            
        swing_high = high.iloc[-15:-5].max()
        swing_low = low.iloc[-15:-5].min()
        swing_range = swing_high - swing_low
        
        current_price = close.iloc[-1]
        
        for level in OTE_RETRACEMENT_LEVELS:
            ote_level = swing_high - (swing_range * level)
            
            if (abs(current_price - ote_level) / ote_level < 0.002 and
                close.iloc[-1] > close.iloc[-2]):
                return "CE"
                
            ote_level = swing_low + (swing_range * level)
            if (abs(current_price - ote_level) / ote_level < 0.002 and
                close.iloc[-1] < close.iloc[-2]):
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 16: DEMAND AND SUPPLY ZONES 🚨
def detect_demand_supply_zones(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < DEMAND_SUPPLY_ZONE_LOOKBACK + 5:
            return None
            
        lookback = DEMAND_SUPPLY_ZONE_LOOKBACK
        
        demand_lows = low.rolling(3, center=True).min().dropna()
        significant_demand = demand_lows[demand_lows == demand_lows.rolling(5).min()]
        
        supply_highs = high.rolling(3, center=True).max().dropna()
        significant_supply = supply_highs[supply_highs == supply_highs.rolling(5).max()]
        
        current_price = close.iloc[-1]
        
        for zone in significant_demand.iloc[-5:]:
            if (abs(current_price - zone) / zone < 0.003 and
                close.iloc[-1] > close.iloc[-2] and
                volume.iloc[-1] > volume.iloc[-5:].mean()):
                return "CE"
                
        for zone in significant_supply.iloc[-5:]:
            if (abs(current_price - zone) / zone < 0.003 and
                close.iloc[-1] < close.iloc[-2] and
                volume.iloc[-1] > volume.iloc[-5:].mean()):
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 17: BOTTOM-FISHING 🚨
def detect_bottom_fishing(index, df):
    try:
        close = ensure_series(df['Close'])
        low = ensure_series(df['Low'])
        high = ensure_series(df['High'])
        volume = ensure_series(df['Volume'])
        if len(close) < 6: 
            return None

        bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
        last_close = float(close.iloc[-1])

        wick = last_close - low.iloc[-1]
        body = abs(close.iloc[-1] - close.iloc[-2])
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)

        if wick > body * 1.5 and vol_ratio > 1.2:
            for zone in bull_liq:
                if abs(last_close - zone) <= 5:
                    return "CE"

        bear_wick = high.iloc[-1] - last_close
        if bear_wick > body * 1.5 and vol_ratio > 1.2:
            for zone in bear_liq:
                if abs(last_close - zone) <= 5:
                    return "PE"
    except:
        return None
    return None

# 🚨 LAYER 18: INSTITUTIONAL TRAP 🚨
def detect_institutional_trap(df, lookback=10):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        if len(close) < lookback + 2:
            return None
        recent_high = high.rolling(lookback).max().iloc[-2]
        recent_low = low.rolling(lookback).min().iloc[-2]
        avg_vol = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()

        if high.iloc[-1] > recent_high and close.iloc[-1] < recent_high and (avg_vol is None or volume.iloc[-1] > (avg_vol * 1.2)):
            return "PE"

        if low.iloc[-1] < recent_low and close.iloc[-1] > recent_low and (avg_vol is None or volume.iloc[-1] > (avg_vol * 1.2)):
            return "CE"
    except Exception:
        return None
    return None

# --------- UPDATED STRATEGY CHECK WITH ALL LAYERS ---------
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    last_close = float(close5.iloc[-1])
    prev_close = float(close5.iloc[-2])

    # 🚨 LAYER 0: OPENING-PLAY PRIORITY 🚨
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        t = ist_now.time()
        opening_range_bias = OPENING_PLAY_ENABLED and (OPENING_START <= t <= OPENING_END)
        if opening_range_bias:
            op_sig = institutional_opening_play(index, df5)
            if op_sig:
                fakeout = False
                high_zone, low_zone = detect_liquidity_zone(df5, lookback=10)
                try:
                    if op_sig == "CE" and last_close >= high_zone: fakeout = True
                    if op_sig == "PE" and last_close <= low_zone: fakeout = True
                except:
                    fakeout = False
                return op_sig, df5, fakeout
    except Exception:
        pass

    # 🚨 LAYER 1: EXPIRY / GAMMA DETECTION 🚨
    try:
        gamma = detect_gamma_squeeze(index, df5)
        if gamma:
            gamma_msg = f"⚡ GAMMA-LIKE EVENT DETECTED: {index} {gamma['side']} (conf {gamma['confidence']:.2f})"
            send_telegram(gamma_msg)
            if is_expiry_day_for_index(index) and EXPIRY_ACTIONABLE and not EXPIRY_INFO_ONLY:
                cand = gamma['side']
                oi_flow = oi_delta_flow_signal(index)
                if institutional_flow_confirm(index, cand, df5):
                    return cand, df5, False
                if gamma['confidence'] > 0.45 and oi_flow == cand:
                    return cand, df5, False
    except Exception:
        pass

    # 🚨 LAYER 2: LIQUIDITY SWEEPS (Highest Priority) 🚨
    sweep_sig = detect_liquidity_sweeps(df5)
    if sweep_sig:
        return sweep_sig, df5, True

    # 🚨 LAYER 3: WYCKOFF SCHEMATICS 🚨
    wyckoff_sig = detect_wyckoff_schematic(df5)
    if wyckoff_sig:
        return wyckoff_sig, df5, False

    # 🚨 LAYER 4: VCP PATTERN 🚨
    vcp_sig = detect_vcp_pattern(df5)
    if vcp_sig:
        return vcp_sig, df5, False

    # 🚨 LAYER 5: FAULTY BASES 🚨
    faulty_sig = detect_faulty_bases(df5)
    if faulty_sig:
        return faulty_sig, df5, True

    # 🚨 LAYER 6: PEAK REJECTION 🚨
    peak_sig = detect_peak_rejection(df5)
    if peak_sig:
        return peak_sig, df5, True

    # 🚨 LAYER 7: SMART-MONEY DIVERGENCE 🚨
    sm_sig = smart_money_divergence(df5)
    if sm_sig:
        return sm_sig, df5, False

    # 🚨 LAYER 8: STOP-HUNT DETECTOR 🚨
    stop_sig = detect_stop_hunt(df5)
    if stop_sig:
        return stop_sig, df5, True

    # 🚨 LAYER 9: INSTITUTIONAL CONTINUATION 🚨
    cont_sig = detect_institutional_continuation(df5)
    if cont_sig:
        if institutional_flow_confirm(index, cont_sig, df5):
            return cont_sig, df5, False

    # 🚨 LAYER 10: FAIR VALUE GAP 🚨
    fvg_sig = detect_fair_value_gap(df5)
    if fvg_sig:
        return fvg_sig, df5, False

    # 🚨 LAYER 11: VOLUME GAP IMBALANCE 🚨
    volume_sig = detect_volume_gap_imbalance(df5)
    if volume_sig:
        return volume_sig, df5, False

    # 🚨 LAYER 12: OTE RETRACEMENT 🚨
    ote_sig = detect_ote_retracement(df5)
    if ote_sig:
        return ote_sig, df5, False

    # 🚨 LAYER 13: DEMAND & SUPPLY ZONES 🚨
    ds_sig = detect_demand_supply_zones(df5)
    if ds_sig:
        return ds_sig, df5, False

    # 🚨 LAYER 14: PULLBACK REVERSAL 🚨
    pull_sig = detect_pullback_reversal(df5)
    if pull_sig:
        return pull_sig, df5, False

    # 🚨 LAYER 15: ORDERFLOW MIMIC 🚨
    flow_sig = mimic_orderflow_logic(df5)
    if flow_sig:
        return flow_sig, df5, False

    # 🚨 LAYER 16: BOTTOM-FISHING 🚨
    bottom_sig = detect_bottom_fishing(index, df5)
    if bottom_sig:
        return bottom_sig, df5, False

    # 🚨 LAYER 17: INSTITUTIONAL TRAP 🚨
    trap_sig = detect_institutional_trap(df5)
    if trap_sig:
        return trap_sig, df5, True

    # Final fallback: Liquidity-based entry
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
    liquidity_side = liquidity_zone_entry_check(last_close, bull_liq, bear_liq)
    if liquidity_side:
        return liquidity_side, df5, False

    return None

# --------- SYMBOL FORMAT FOR ALL INDICES ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    dt=datetime.strptime(expiry_str,"%d %b %Y")
    
    if index == "SENSEX":
        year_short = dt.strftime("%y")
        month_code = dt.strftime("%b").upper()
        day = dt.strftime("%d")
        return f"SENSEX{year_short}{month_code}{strike}{opttype}"
    elif index == "FINNIFTY":
        return f"FINNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
    elif index == "MIDCPNIFTY":
        return f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
    else:
        return f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"

# --------- INSTITUTIONAL FLOW CHECKS ---------
def institutional_flow_signal(index, df5):
    try:
        last_close = float(ensure_series(df5["Close"]).iloc[-1])
        prev_close = float(ensure_series(df5["Close"]).iloc[-2])
    except:
        return None

    vol5 = ensure_series(df5["Volume"])
    vol_latest = float(vol5.iloc[-1])
    vol_avg = float(vol5.rolling(20).mean().iloc[-1]) if len(vol5) >= 20 else float(vol5.mean())

    if vol_latest > vol_avg*1.5 and abs(last_close-prev_close)/prev_close>0.003:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg:
        return "PE"
    
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)
    try:
        if last_close>=high_zone: return "PE"
        elif last_close<=low_zone: return "CE"
    except:
        return None
    return None

# --------- OI + DELTA FLOW DETECTION ---------
def oi_delta_flow_signal(index):
    try:
        url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        df_index=df[df['symbol'].str.contains(index)]
        if 'oi' not in df_index.columns:
            return None
        df_index['oi'] = pd.to_numeric(df_index['oi'], errors='coerce').fillna(0)
        df_index['oi_change'] = df_index['oi'].diff().fillna(0)
        ce_sum = df_index[df_index['symbol'].str.endswith("CE")]['oi_change'].sum()
        pe_sum = df_index[df_index['symbol'].str.endswith("PE")]['oi_change'].sum()
        if ce_sum>pe_sum*DELTA_OI_RATIO: return "CE"
        if pe_sum>ce_sum*DELTA_OI_RATIO: return "PE"
        if ce_sum>0 and pe_sum>0: return "BOTH"
    except:
        return None

# --------- SIMPLIFIED CONFIRMATION ---------
def institutional_confirmation_layer(index, df5, base_signal):
    try:
        close = ensure_series(df5['Close'])
        last_close = float(close.iloc[-1])
        
        high_zone, low_zone = detect_liquidity_zone(df5, lookback=20)
        if base_signal == 'CE' and last_close >= high_zone:
            return False
        if base_signal == 'PE' and last_close <= low_zone:
            return False

        return True
    except Exception:
        return False

def institutional_flow_confirm(index, base_signal, df5):
    flow = institutional_flow_signal(index, df5)
    oi_flow = oi_delta_flow_signal(index)

    if flow and flow != 'BOTH' and flow != base_signal:
        return False
    if oi_flow and oi_flow != 'BOTH' and oi_flow != base_signal:
        return False

    if not institutional_confirmation_layer(index, df5, base_signal):
        return False

    return True

# --------- MONITOR WITH THREAD UPDATES ---------
def monitor_price_live(symbol,entry,targets,sl,fakeout,thread_id):
    last_high = entry
    weakness_sent = False
    in_trade=False
    for idx, val in active_trades.items():
        if val and val.get("symbol") == symbol:
            active_trades[idx]["status"] = "OPEN"
            break

    while True:
        if should_stop_trading():
            global STOP_SENT
            if not STOP_SENT:
                send_telegram(f"🛑 Market closed - Stopping monitoring for {symbol}", reply_to=thread_id)
                STOP_SENT = True
            for idx, val in active_trades.items():
                if val and val.get("symbol") == symbol:
                    active_trades[idx]["status"] = "CLOSED"
            break
            
        price=fetch_option_price(symbol)
        if not price: time.sleep(10); continue
        price=round(price)
        if not in_trade:
            if price >= entry:
                send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                in_trade=True
                last_high=price
        else:
            if price > last_high:
                send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                last_high=price
            elif not weakness_sent and price < sl*1.05:
                send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                weakness_sent=True
            if price>=targets[0]:
                send_telegram(f"🌟 {symbol}: First Target {targets[0]} hit", reply_to=thread_id)
                for idx, val in active_trades.items():
                    if val and val.get("symbol") == symbol:
                        active_trades[idx]["status"] = "CLOSED"
                break
            if price<=sl:
                send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade.", reply_to=thread_id)
                for idx, val in active_trades.items():
                    if val and val.get("symbol") == symbol:
                        active_trades[idx]["status"] = "CLOSED"
                break
        time.sleep(10)

# --------- EXPIRY CONFIG FOR ALL INDICES ---------
EXPIRIES = {
    "NIFTY": "20 OCT 2025",
    "BANKNIFTY": "28 OCT 2025", 
    "SENSEX": "23 OCT 2025",
    "FINNIFTY": "28 OCT 2025",
    "MIDCPNIFTY": "28 OCT 2025",
    "EICHERMOT": "28 OCT 2025",
    "TRENT": "28 OCT 2025", 
    "RELIANCE": "28 OCT 2025"
}

# ACTIVE TRACKING FOR ALL INDICES
active_trades = {
    "NIFTY": None, "BANKNIFTY": None, "SENSEX": None,
    "FINNIFTY": None, "MIDCPNIFTY": None, "EICHERMOT": None,
    "TRENT": None, "RELIANCE": None
}

# --------- THREAD FUNCTION ---------
def trade_thread(index):
    global active_trades
    if active_trades[index] and isinstance(active_trades[index], dict) and active_trades[index].get("status") == "OPEN":
        return

    sig=analyze_index_signal(index)
    side=None; fakeout=False; df=None
    if sig: 
        if isinstance(sig, tuple) and len(sig) == 3:
            side, df, fakeout = sig
        elif isinstance(sig, tuple) and len(sig) == 2:
            side, df = sig
            fakeout = False
        else:
            side = sig

    df5=fetch_index_data(index,"5m","2d")
    inst_signal = institutional_flow_signal(index, df5) if df5 is not None else None
    oi_signal = oi_delta_flow_signal(index)
    final_signal = oi_signal or inst_signal or side

    if final_signal=="BOTH":
        for s in ["CE","PE"]:
            if institutional_flow_confirm(index, s, df5):
                send_signal(index,s,df,fakeout)
        return
    elif final_signal:
        if df is None: df=df5
        if institutional_flow_confirm(index, final_signal, df5):
            send_signal(index,final_signal,df,fakeout)
    else:
        return

# --------- SEND SIGNAL ---------
def send_signal(index,side,df,fakeout):
    ltp=float(ensure_series(df["Close"]).iloc[-1])
    
    strike=round_strike(index,ltp)
    
    if strike is None:
        send_telegram(f"⚠️ {index}: could not determine strike (ltp missing). Signal skipped.")
        return
        
    symbol=get_option_symbol(index,EXPIRIES[index],strike,side)
    
    price=fetch_option_price(symbol)
    if not price: return
    high=ensure_series(df["High"])
    low=ensure_series(df["Low"])
    close=ensure_series(df["Close"])
    atr=float(ta.volatility.AverageTrueRange(high,low,close,14).average_true_range().iloc[-1])
    entry=round(price+5)
    sl=round(price-atr)
    targets=[round(price+atr*1.5),round(price+atr*2)]
    
    msg=(f" GIT🔊 {index} {side} VSSIGNAL CONFIRMED\n"
         f"🔹 Strike: {strike}\n"
         f"🟩 Buy Above ₹{entry}\n"
         f"🔵 SL: ₹{sl}\n"
         f"🌟 Targets: {targets[0]} / {targets[1]}\n"
         f"⚡ Fakeout: {'YES' if fakeout else 'NO'}")
         
    thread_id=send_telegram(msg)
    active_trades[index]={"symbol":symbol,"entry":entry,"sl":sl,"targets":targets,"thread":thread_id,"status":"OPEN"}
    monitor_price_live(symbol,entry,targets,sl,fakeout,thread_id)
    active_trades[index]=None

# --------- MAIN LOOP (ALL INDICES PARALLEL) ---------
def run_algo_parallel():
    if not is_market_open(): 
        print("❌ Market closed - skipping iteration")
        return
        
    if should_stop_trading():
        global STOP_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
        return
        
    threads=[]
    all_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "EICHERMOT", "TRENT", "RELIANCE"]
    
    for index in all_indices:
        t=threading.Thread(target=trade_thread,args=(index,))
        t.start()
        threads.append(t)
    for t in threads: t.join()

# --------- START ---------
while True:
    try:
        if not STARTED_SENT and is_market_open():
            send_telegram("🚀 GIT ULTIMATE MASTER ALGO STARTED - All 8 Indices Running with 18 Institutional Layers!")
            STARTED_SENT = True
            STOP_SENT = False
            
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached - Algorithm stopped automatically")
                STOP_SENT = True
                STARTED_SENT = False
            break
            
        if is_market_open():
            run_algo_parallel()
            
        time.sleep(30)
    except Exception as e:
        send_telegram(f"⚠️ Error in main loop: {e}")
        time.sleep(60)
