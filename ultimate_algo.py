# ultimate_algo.py - INSTITUTIONAL ANTICIPATION ENGINE - CROSS-INDEX FIXED
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
GAMMA_VOL_SPIKE_THRESHOLD = 2.0
DELTA_OI_RATIO = 2.0
MOMENTUM_VOL_AMPLIFIER = 1.5

# STRONGER CONFIRMATION THRESHOLDS
VCP_CONTRACTION_RATIO = 0.6
FAULTY_BASE_BREAK_THRESHOLD = 0.25
WYCKOFF_VOLUME_SPRING = 2.2
LIQUIDITY_SWEEP_DISTANCE = 0.005
PEAK_REJECTION_WICK_RATIO = 0.8
FVG_GAP_THRESHOLD = 0.0025
VOLUME_GAP_IMBALANCE = 2.5
OTE_RETRACEMENT_LEVELS = [0.618, 0.786]
DEMAND_SUPPLY_ZONE_LOOKBACK = 20

# INSTITUTIONAL THRESHOLDS
SWEEP_ORDER_MIN_LOTS = 50000
UNUSUAL_PREMIUM_THRESHOLD = 5000000
GAMMA_FLIP_THRESHOLD = 0.7
LIQUIDITY_GRAB_DISTANCE = 0.003

# --------- EXPIRIES FOR ALL INDICES ---------
EXPIRIES = {
    "NIFTY": "11 NOV 2025",
    "BANKNIFTY": "25 NOV 2025",
    "SENSEX": "13 NOV 2025",
    "FINNIFTY": "25 NOV 2025",
    "MIDCPNIFTY": "25 NOV 2025",
    "EICHERMOT": "25 NOV 2025",
    "TRENT": "25 NOV 2025",
    "RELIANCE": "25 NOV 2025"
}

# --------- STRATEGY TRACKING ---------
STRATEGY_NAMES = {
    # INSTITUTIONAL STRATEGIES
    "sweep_order_detection": "SWEEP ORDER DETECTION",
    "unusual_options_flow": "UNUSUAL OPTIONS FLOW", 
    "volume_profile_breakout": "VOLUME PROFILE BREAKOUT",
    "futures_roll_pressure": "FUTURES ROLL PRESSURE",
    "liquidity_grab_setup": "LIQUIDITY GRAB SETUP",
    "gamma_exposure_flip": "GAMMA EXPOSURE FLIP",
    "smart_money_divergence": "SMART MONEY DIVERGENCE",
    "institutional_accumulation": "INSTITUTIONAL ACCUMULATION",
    
    # EXISTING STRATEGIES
    "institutional_price_action": "INSTITUTIONAL PRICE ACTION",
    "opening_play": "OPENING PLAY", 
    "gamma_squeeze": "GAMMA SQUEEZE",
    "liquidity_sweeps": "LIQUIDITY SWEEP",
    "wyckoff_schematic": "WYCKOFF SCHEMATIC",
    "vcp_pattern": "VCP PATTERN",
    "faulty_bases": "FAULTY BASES",
    "peak_rejection": "PEAK REJECTION",
    "stop_hunt": "STOP HUNT",
    "institutional_continuation": "INSTITUTIONAL CONTINUATION",
    "fair_value_gap": "FAIR VALUE GAP",
    "volume_gap_imbalance": "VOLUME GAP IMBALANCE",
    "ote_retracement": "OTE RETRACEMENT",
    "demand_supply_zones": "DEMAND SUPPLY ZONES",
    "pullback_reversal": "PULLBACK REVERSAL",
    "orderflow_mimic": "ORDERFLOW MIMIC",
    "bottom_fishing": "BOTTOM FISHING",
    "liquidity_zone": "LIQUIDITY ZONE"
}

# Track all signals for end-of-day report
daily_signals = []
signal_counter = 0
all_generated_signals = []

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now() if TOTP_SECRET else None

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP) if API_KEY and CLIENT_CODE and PASSWORD and TOTP else None
feedToken = client.getfeedToken() if session else None

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STARTED_SENT = False
STOP_SENT = False
EOD_REPORT_SENT = False
MARKET_CLOSED_SENT = False

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

# 🚨 INSTITUTIONAL ANTICIPATION ENGINE - FIXED CROSS-CONTAMINATION 🚨
def anticipate_institutional_move(index, df):
    """
    PREDICT where institutions will move NEXT
    Think like institution: Hunt stops → Get fills → Reverse
    """
    try:
        current_price = float(ensure_series(df['Close']).iloc[-1])
        
        # 1. GAMMA TRAP DETECTION (Institutions are forced to hedge)
        gamma_trap = detect_gamma_trap(index, current_price)
        if gamma_trap:
            return gamma_trap, "GAMMA_TRAP"
        
        # 2. MAX PAIN PRESSURE (Expiry day manipulation)
        if is_expiry_day_for_index(index):
            max_pain_signal = detect_max_pain_pressure(index, current_price)
            if max_pain_signal:
                return max_pain_signal, "MAX_PAIN"
        
        # 3. LIQUIDITY HUNT SETUP (Institutions hunting retail stops)
        liquidity_hunt = detect_liquidity_hunt_setup(index, df, current_price)
        if liquidity_hunt:
            return liquidity_hunt, "LIQUIDITY_HUNT"
        
        return None, None
    except Exception as e:
        return None, None

def detect_gamma_trap(index, current_price):
    """Detect where market makers are trapped and MUST move price"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df_chain = pd.DataFrame(requests.get(url, timeout=10).json())
        # 🚨 CRITICAL FIX: Exact index matching to prevent cross-contamination
        df_chain = df_chain[df_chain['symbol'].str.startswith(index)]
        
        df_chain['oi'] = pd.to_numeric(df_chain['oi'], errors='coerce')
        df_chain['strike'] = df_chain['symbol'].str.extract(r'(\d+)')[0].astype(float)
        
        # Find high OI strikes near current price (gamma walls)
        near_strikes = df_chain[
            (df_chain['strike'] >= current_price * 0.99) & 
            (df_chain['strike'] <= current_price * 1.01)
        ]
        
        if len(near_strikes) > 0:
            # Check if market maker is short gamma (needs to hedge)
            total_oi = near_strikes['oi'].sum()
            avg_oi = df_chain['oi'].mean()
            
            if total_oi > avg_oi * 2.0:  # High gamma concentration
                highest_oi_strike = near_strikes.loc[near_strikes['oi'].idxmax(), 'strike']
                
                if current_price > highest_oi_strike:
                    # Market maker short calls - will push DOWN to hedge
                    return "PE"
                else:
                    # Market maker short puts - will push UP to hedge  
                    return "CE"
    except:
        return None
    return None

def detect_max_pain_pressure(index, current_price):
    """Detect expiry day max pain manipulation"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df_chain = pd.DataFrame(requests.get(url, timeout=10).json())
        # 🚨 CRITICAL FIX: Exact index matching
        df_chain = df_chain[df_chain['symbol'].str.startswith(index)]
        
        df_chain['oi'] = pd.to_numeric(df_chain['oi'], errors='coerce')
        df_chain['strike'] = df_chain['symbol'].str.extract(r'(\d+)')[0].astype(float)
        
        # Simple max pain calculation
        strikes = sorted(df_chain['strike'].unique())
        pain_values = []
        
        for strike in strikes:
            if strike <= current_price:
                # ITM puts pain
                put_oi = df_chain[(df_chain['strike'] == strike) & 
                                 (df_chain['symbol'].str.endswith("PE"))]['oi'].sum()
                pain = (current_price - strike) * put_oi
            else:
                # ITM calls pain  
                call_oi = df_chain[(df_chain['strike'] == strike) & 
                                  (df_chain['symbol'].str.endswith("CE"))]['oi'].sum()
                pain = (strike - current_price) * call_oi
            pain_values.append(pain)
        
        if pain_values:
            max_pain_strike = strikes[pain_values.index(max(pain_values))]
            
            # If close to max pain, predict direction
            if abs(current_price - max_pain_strike) / current_price < 0.005:
                if current_price < max_pain_strike:
                    return "CE"  # Push UP to max pain
                else:
                    return "PE"  # Push DOWN to max pain
    except:
        return None
    return None

def detect_liquidity_hunt_setup(index, df, current_price):
    """Detect when institutions are setting up stop hunts"""
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        # Find recent liquidity zones (where retail stops cluster)
        recent_high = high.rolling(10).max().iloc[-2]
        recent_low = low.rolling(10).min().iloc[-2]
        
        # Check if price is approaching liquidity zones with volume
        vol_avg = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        if (abs(current_price - recent_high) / recent_high < 0.003 and 
            current_vol > vol_avg * 1.8):
            # Approaching resistance with volume - institution hunting upside stops
            return "PE"  # They'll hunt stops then reverse
            
        elif (abs(current_price - recent_low) / recent_low < 0.003 and 
              current_vol > vol_avg * 1.8):
            # Approaching support with volume - institution hunting downside stops  
            return "CE"  # They'll hunt stops then reverse
              
    except:
        return None
    return None

# 🚨 UPDATED INSTITUTIONAL FLOW DETECTORS WITH ANTICIPATION - FIXED CROSS-CONTAMINATION 🚨

def detect_sweep_orders(index):
    """Detect when institutions sweep multiple strikes simultaneously"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df_chain = pd.DataFrame(requests.get(url, timeout=10).json())
        # 🚨 CRITICAL FIX: Exact index matching
        df_chain = df_chain[df_chain['symbol'].str.startswith(index)]
        
        df_chain['oi'] = pd.to_numeric(df_chain['oi'], errors='coerce')
        df_chain['oi_change'] = df_chain.groupby('symbol')['oi'].diff().fillna(0)
        
        ce_sweeps = df_chain[
            (df_chain['symbol'].str.endswith("CE")) & 
            (df_chain['oi_change'] > SWEEP_ORDER_MIN_LOTS)
        ]
        
        pe_sweeps = df_chain[
            (df_chain['symbol'].str.endswith("PE")) &
            (df_chain['oi_change'] > SWEEP_ORDER_MIN_LOTS) 
        ]
        
        if len(ce_sweeps) > len(pe_sweeps) * 1.5:
            return "CE"
        elif len(pe_sweeps) > len(ce_sweeps) * 1.5:
            return "PE"
            
    except Exception as e:
        print(f"Sweep order error: {e}")
    return None

def detect_unusual_options_flow(index):
    """Detect large premium trades and block deals"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df_chain = pd.DataFrame(requests.get(url, timeout=10).json())
        # 🚨 CRITICAL FIX: Exact index matching
        df_chain = df_chain[df_chain['symbol'].str.startswith(index)]
        
        df_chain['ltp'] = pd.to_numeric(df_chain['ltp'], errors='coerce')
        df_chain['oi'] = pd.to_numeric(df_chain['oi'], errors='coerce')
        df_chain['premium_flow'] = df_chain['ltp'] * df_chain['oi'] * 25
        
        unusual_ce = df_chain[
            (df_chain['symbol'].str.endswith("CE")) &
            (df_chain['premium_flow'] > UNUSUAL_PREMIUM_THRESHOLD)
        ]
        
        unusual_pe = df_chain[
            (df_chain['symbol'].str.endswith("PE")) & 
            (df_chain['premium_flow'] > UNUSUAL_PREMIUM_THRESHOLD)
        ]
        
        if len(unusual_ce) > len(unusual_pe):
            return "CE"
        elif len(unusual_pe) > len(unusual_ce):
            return "PE"
            
    except Exception as e:
        print(f"Unusual flow error: {e}")
    return None

def detect_volume_profile_breakout(df):
    """Volume Profile Point of Control breakouts"""
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close']) 
        volume = ensure_series(df['Volume'])
        
        price_levels = np.linspace(float(low.min()), float(high.max()), 20)
        volume_profile = {}
        
        for i in range(len(close)):
            price = float(close.iloc[i])
            vol = float(volume.iloc[i])
            level = min(price_levels, key=lambda x: abs(x - price))
            volume_profile[level] = volume_profile.get(level, 0) + vol
        
        if volume_profile:
            poc = max(volume_profile, key=volume_profile.get)
            current_price = float(close.iloc[-1])
            
            if (current_price > poc and 
                float(volume.iloc[-1]) > float(volume.rolling(20).mean().iloc[-1]) * 1.8):
                return "CE"
                
            elif (current_price < poc and
                  float(volume.iloc[-1]) > float(volume.rolling(20).mean().iloc[-1]) * 1.8):
                return "PE"
                  
    except Exception as e:
        print(f"Volume profile error: {e}")
    return None

def detect_futures_roll_pressure(index):
    """Detect institutional futures roll activity"""
    try:
        expiry_date = datetime.strptime(EXPIRIES[index], "%d %b %Y")
        days_to_expiry = (expiry_date - datetime.now()).days
        
        if days_to_expiry <= 2:
            df_fut = fetch_index_data(index, "5m", "1d")
            if df_fut is not None:
                volume = ensure_series(df_fut['Volume'])
                current_vol = float(volume.iloc[-1])
                avg_vol = float(volume.rolling(20).mean().iloc[-1])
                
                if current_vol > avg_vol * 2.0:
                    close = ensure_series(df_fut['Close'])
                    if float(close.iloc[-1]) > float(close.iloc[-2]):
                        return "CE"
                    else:
                        return "PE"
                        
    except Exception as e:
        print(f"Futures roll error: {e}")
    return None

def detect_liquidity_grab_setup(df):
    """Detect institutional stop hunts and liquidity grabs"""
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        recent_high = float(high.rolling(10).max().iloc[-2])
        recent_low = float(low.rolling(10).min().iloc[-2])
        current_price = float(close.iloc[-1])
        
        if (current_price > recent_high * (1 + LIQUIDITY_GRAB_DISTANCE) and
            float(volume.iloc[-1]) > float(volume.rolling(20).mean().iloc[-1]) * 1.5):
            return "PE"
            
        elif (current_price < recent_low * (1 - LIQUIDITY_GRAB_DISTANCE) and
              float(volume.iloc[-1]) > float(volume.rolling(20).mean().iloc[-1]) * 1.5):
            return "CE"
              
    except Exception as e:
        print(f"Liquidity grab error: {e}")
    return None

def detect_gamma_exposure_flip(index):
    """Detect gamma squeeze conditions"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df_chain = pd.DataFrame(requests.get(url, timeout=10).json())
        # 🚨 CRITICAL FIX: Exact index matching
        df_chain = df_chain[df_chain['symbol'].str.startswith(index)]
        
        df_chain['oi'] = pd.to_numeric(df_chain['oi'], errors='coerce')
        df_chain['strike'] = df_chain['symbol'].str.extract(r'(\d+)')[0].astype(float)
        
        df_index = fetch_index_data(index, "5m", "1d")
        if df_index is None:
            return None
            
        current_price = float(ensure_series(df_index['Close']).iloc[-1])
        
        near_strikes = df_chain[
            (df_chain['strike'] >= current_price * 0.98) & 
            (df_chain['strike'] <= current_price * 1.02)
        ]
        
        if len(near_strikes) > 0:
            total_oi = near_strikes['oi'].sum()
            avg_oi = df_chain['oi'].mean()
            
            if total_oi > avg_oi * GAMMA_FLIP_THRESHOLD:
                if current_price > near_strikes['strike'].max():
                    return "CE"
                elif current_price < near_strikes['strike'].min():
                    return "PE"
                    
    except Exception as e:
        print(f"Gamma exposure error: {e}")
    return None

def detect_institutional_accumulation(df):
    """Detect where big money is accumulating positions"""
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        current_volume = float(volume.iloc[-1])
        avg_volume = float(volume.rolling(20).mean().iloc[-1])
        price_range = float(high.iloc[-1]) - float(low.iloc[-1])
        
        if (current_volume > avg_volume * 1.5 and 
            price_range < (float(high.rolling(10).mean().iloc[-1]) - float(low.rolling(10).mean().iloc[-1])) * 0.6):
            
            if float(close.iloc[-1]) > float(close.iloc[-2]):
                return "CE"
            elif float(close.iloc[-1]) < float(close.iloc[-2]):
                return "PE"
                
    except Exception as e:
        print(f"Accumulation error: {e}")
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

# 🚨 NEW: INSTITUTIONAL PRICE ACTION LAYER 🚨
def institutional_price_action_signal(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        recent_high = high.iloc[-10:-1].max()
        recent_low = low.iloc[-10:-1].min()
        current_close = close.iloc[-1]
        
        vol_avg = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        if (current_close > recent_high and 
            current_vol > vol_avg * 1.8 and
            current_close > close.iloc[-2] and
            close.iloc[-2] > close.iloc[-3]):
            return "CE"
            
        if (current_close < recent_low and
            current_vol > vol_avg * 1.8 and
            current_close < close.iloc[-2] and
            close.iloc[-2] < close.iloc[-3]):
            return "PE"
            
        current_body = abs(close.iloc[-1] - close.iloc[-2])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], close.iloc[-2])
        lower_wick = min(close.iloc[-1], close.iloc[-2]) - low.iloc[-1]
        
        if (upper_wick > current_body * 1.5 and
            current_vol > vol_avg * 1.5 and
            close.iloc[-1] < close.iloc[-2]):
            return "PE"
            
        if (lower_wick > current_body * 1.5 and
            current_vol > vol_avg * 1.5 and
            close.iloc[-1] > close.iloc[-2]):
            return "CE"
            
    except Exception:
        return None
    return None

# 🚨 NEW: INSTITUTIONAL MOMENTUM CONFIRMATION 🚨
def institutional_momentum_confirmation(index, df, proposed_signal):
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 5:
            return False
            
        if proposed_signal == "CE":
            if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
                return False
            if (high.iloc[-1] - low.iloc[-1]) < (high.iloc[-2] - low.iloc[-2]) * 0.7:
                return False
                
        elif proposed_signal == "PE":
            if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
                return False
            if (high.iloc[-1] - low.iloc[-1]) < (high.iloc[-2] - low.iloc[-2]) * 0.7:
                return False
                
        return True
        
    except Exception:
        return False

# 🚨 LAYER 1: OPENING-RANGE INSTITUTIONAL PLAY 🚨
def institutional_opening_play(index, df):
    try:
        prev_high = float(ensure_series(df['High']).iloc[-2])
        prev_low = float(ensure_series(df['Low']).iloc[-2])
        prev_close = float(ensure_series(df['Close']).iloc[-2])
        current_price = float(ensure_series(df['Close']).iloc[-1])
    except Exception:
        return None
        
    volume = ensure_series(df['Volume'])
    vol_avg = volume.rolling(10).mean().iloc[-1] if len(volume) >= 10 else volume.mean()
    vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)
    
    if current_price > prev_high + 15 and vol_ratio > 1.3: return "CE"
    if current_price < prev_low - 15 and vol_ratio > 1.3: return "PE"
    if current_price > prev_close + 25 and vol_ratio > 1.2: return "CE"
    if current_price < prev_close - 25 and vol_ratio > 1.2: return "PE"
    return None

# 🚨 LAYER 2: GAMMA SQUEEZE / EXPIRY LAYER 🚨
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
            # 🚨 CRITICAL FIX: Exact index matching
            df_index = df_s[df_s['symbol'].str.startswith(index)]
            df_index['oi'] = pd.to_numeric(df_index.get('oi',0), errors='coerce').fillna(0)
            ce_oi = df_index[df_index['symbol'].str.endswith("CE")]['oi'].sum()
            pe_oi = df_index[df_index['symbol'].str.endswith("PE")]['oi'].sum()
        except Exception:
            ce_oi = pe_oi = 0
        
        if vol_ratio > GAMMA_VOL_SPIKE_THRESHOLD and abs(speed) > 0.003:
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
        
        if p_now < p_short and rsi_now > rsi_short + 5 and vol_now > vol_avg*1.3:
            return "CE"
        if p_now > p_short and rsi_now < rsi_short - 5 and vol_now > vol_avg*1.3:
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
        
        if last_high > recent_high * 1.003 and last_close < recent_high and volume.iloc[-1] > vol_avg*1.5:
            return "PE"
        if last_low < recent_low * 0.997 and last_close > recent_low and volume.iloc[-1] > vol_avg*1.5:
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
        
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.5 and speed > 0.006:
            return "CE"
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.5 and speed < -0.006:
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

        if (close.iloc[-6] > ema21.iloc[-6] and close.iloc[-3] <= ema21.iloc[-3] and 
            close.iloc[-1] > ema9.iloc[-1] and rsi.iloc[-1] > 55 and 
            close.iloc[-1] > close.iloc[-2]):
            return "CE"

        if (close.iloc[-6] < ema21.iloc[-6] and close.iloc[-3] >= ema21.iloc[-3] and 
            close.iloc[-1] < ema9.iloc[-1] and rsi.iloc[-1] < 45 and 
            close.iloc[-1] < close.iloc[-2]):
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

        if (close.iloc[-1] > close.iloc[-3] and rsi.iloc[-1] < rsi.iloc[-3] - 3 and 
            wick_top_ratio > 0.7 and vol_ratio > 1.5):
            return "PE"

        if (close.iloc[-1] < close.iloc[-3] and rsi.iloc[-1] > rsi.iloc[-3] + 3 and 
            wick_bottom_ratio > 0.7 and vol_ratio > 1.5):
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
            recent_vol < prev_vol * 0.8 and
            close.iloc[-1] > close.iloc[-5] and
            volume.iloc[-1] > recent_vol * 1.3):
            return "CE"
        elif (recent_atr < prev_atr * VCP_CONTRACTION_RATIO and 
              recent_vol < prev_vol * 0.8 and
              close.iloc[-1] < close.iloc[-5] and
              volume.iloc[-1] > recent_vol * 1.3):
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
            current_close < recent_high * 0.998 and
            volume.iloc[-4] > volume.iloc[-5:].mean() * 1.4):
            return "PE"
            
        if (low.iloc[-4] < recent_low * (1 - FAULTY_BASE_BREAK_THRESHOLD/100) and
            current_close > recent_low * 1.002 and
            volume.iloc[-4] > volume.iloc[-5:].mean() * 1.4):
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
        
        if (spring_low < support_level * 0.992 and
            close.iloc[-1] > support_level * 1.005 and
            spring_volume > avg_volume * WYCKOFF_VOLUME_SPRING and
            volume.iloc[-1] > avg_volume * 1.2):
            return "CE"
            
        upthrust_high = high.iloc[-5]
        resistance_level = high.iloc[-10:-5].max()
        upthrust_volume = volume.iloc[-5]
        
        if (upthrust_high > resistance_level * 1.008 and
            close.iloc[-1] < resistance_level * 0.995 and
            upthrust_volume > avg_volume * WYCKOFF_VOLUME_SPRING and
            volume.iloc[-1] > avg_volume * 1.2):
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
            current_close < liquidity_high * 0.998 and
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.6):
            return "PE"
            
        if (current_low < liquidity_low * (1 - LIQUIDITY_SWEEP_DISTANCE) and
            current_close > liquidity_low * 1.002 and
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.6):
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
            current_close < (current_high + current_low) / 2 * 0.995 and
            volume.iloc[-1] > volume.iloc[-5:].mean() * 1.3):
            return "PE"
            
        if (lower_wick > body_size * PEAK_REJECTION_WICK_RATIO and
            current_close > (current_high + current_low) / 2 * 1.005 and
            volume.iloc[-1] > volume.iloc[-5:].mean() * 1.3):
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
            close.iloc[-1] > close.iloc[-2] and
            close.iloc[-1] > (high.iloc[-2] + low.iloc[-2]) / 2):
            return "CE"
            
        if (high.iloc[-1] < low.iloc[-2] * (1 - FVG_GAP_THRESHOLD) and
            close.iloc[-1] < close.iloc[-2] and
            close.iloc[-1] < (high.iloc[-2] + low.iloc[-2]) / 2):
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
            abs(price_change) > 0.004):
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
            
            if (abs(current_price - ote_level) / ote_level < 0.0015 and
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3]):
                return "CE"
                
            ote_level = swing_low + (swing_range * level)
            if (abs(current_price - ote_level) / ote_level < 0.0015 and
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3]):
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
            if (abs(current_price - zone) / zone < 0.002 and
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3] and
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.4):
                return "CE"
                
        for zone in significant_supply.iloc[-5:]:
            if (abs(current_price - zone) / zone < 0.002 and
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3] and
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.4):
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

        if wick > body * 2.0 and vol_ratio > 1.5:
            for zone in bull_liq:
                if zone and abs(last_close - zone) <= 3:
                    return "CE"

        bear_wick = high.iloc[-1] - last_close
        if bear_wick > body * 2.0 and vol_ratio > 1.5:
            for zone in bear_liq:
                if zone and abs(last_close - zone) <= 3:
                    return "PE"
    except:
        return None
    return None

# 🚨 UPDATED: INSTITUTIONAL ANTICIPATION INTEGRATION - FIXED CROSS-CONTAMINATION 🚨
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    last_close = float(close5.iloc[-1])
    prev_close = float(close5.iloc[-2])

    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time = ist_now.time()
        if current_time >= dtime(14, 45):
            return None
    except:
        pass

    # 🚨 INSTITUTIONAL ANTICIPATION ENGINE - HIGHEST PRIORITY 🚨
    anticipated_signal, anticipation_type = anticipate_institutional_move(index, df5)
    if anticipated_signal:
        send_telegram(f"🎯 INSTITUTIONAL ANTICIPATION: {index} {anticipated_signal} - {anticipation_type}")
        return anticipated_signal, df5, False, f"ANTICIPATION_{anticipation_type}"

    # 🚨 EXPIRY DAY GAMMA PRIORITY 🚨
    if is_expiry_day_for_index(index):
        gamma_signal = detect_gamma_exposure_flip(index)
        if gamma_signal:
            send_telegram(f"⚡ EXPIRY GAMMA PRIORITY: {index} {gamma_signal}")
            return gamma_signal, df5, False, "EXPIRY_GAMMA"

    # 🚨 INSTITUTIONAL FLOW DETECTION 🚨
    sweep_signal = detect_sweep_orders(index)
    if sweep_signal:
        return sweep_signal, df5, False, "sweep_order_detection"

    unusual_signal = detect_unusual_options_flow(index) 
    if unusual_signal:
        return unusual_signal, df5, False, "unusual_options_flow"

    gamma_exp_signal = detect_gamma_exposure_flip(index)
    if gamma_exp_signal:
        return gamma_exp_signal, df5, False, "gamma_exposure_flip"

    volume_profile_signal = detect_volume_profile_breakout(df5)
    if volume_profile_signal:
        return volume_profile_signal, df5, False, "volume_profile_breakout"

    futures_signal = detect_futures_roll_pressure(index)
    if futures_signal:
        return futures_signal, df5, False, "futures_roll_pressure"

    liquidity_grab_signal = detect_liquidity_grab_setup(df5)
    if liquidity_grab_signal:
        return liquidity_grab_signal, df5, True, "liquidity_grab_setup"

    accumulation_signal = detect_institutional_accumulation(df5)
    if accumulation_signal:
        return accumulation_signal, df5, False, "institutional_accumulation"

    # Continue with your existing strategies
    institutional_pa_signal = institutional_price_action_signal(df5)
    if institutional_pa_signal:
        if institutional_momentum_confirmation(index, df5, institutional_pa_signal):
            return institutional_pa_signal, df5, False, "institutional_price_action"

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
                return op_sig, df5, fakeout, "opening_play"
    except Exception:
        pass

    try:
        gamma = detect_gamma_squeeze(index, df5)
        if gamma:
            gamma_msg = f"⚡ GAMMA-LIKE EVENT DETECTED: {index} {gamma['side']} (conf {gamma['confidence']:.2f})"
            send_telegram(gamma_msg)
            if is_expiry_day_for_index(index) and EXPIRY_ACTIONABLE and not EXPIRY_INFO_ONLY:
                cand = gamma['side']
                oi_flow = oi_delta_flow_signal(index)
                if institutional_flow_confirm(index, cand, df5):
                    return cand, df5, False, "gamma_squeeze"
                if gamma['confidence'] > 0.6 and oi_flow == cand:
                    return cand, df5, False, "gamma_squeeze"
    except Exception:
        pass

    sweep_sig = detect_liquidity_sweeps(df5)
    if sweep_sig:
        if institutional_momentum_confirmation(index, df5, sweep_sig):
            return sweep_sig, df5, True, "liquidity_sweeps"

    wyckoff_sig = detect_wyckoff_schematic(df5)
    if wyckoff_sig:
        if institutional_momentum_confirmation(index, df5, wyckoff_sig):
            return wyckoff_sig, df5, False, "wyckoff_schematic"

    vcp_sig = detect_vcp_pattern(df5)
    if vcp_sig:
        if institutional_momentum_confirmation(index, df5, vcp_sig):
            return vcp_sig, df5, False, "vcp_pattern"

    faulty_sig = detect_faulty_bases(df5)
    if faulty_sig:
        if institutional_momentum_confirmation(index, df5, faulty_sig):
            return faulty_sig, df5, True, "faulty_bases"

    peak_sig = detect_peak_rejection(df5)
    if peak_sig:
        if institutional_momentum_confirmation(index, df5, peak_sig):
            return peak_sig, df5, True, "peak_rejection"

    sm_sig = smart_money_divergence(df5)
    if sm_sig:
        if institutional_momentum_confirmation(index, df5, sm_sig):
            return sm_sig, df5, False, "smart_money_divergence"

    stop_sig = detect_stop_hunt(df5)
    if stop_sig:
        if institutional_momentum_confirmation(index, df5, stop_sig):
            return stop_sig, df5, True, "stop_hunt"

    cont_sig = detect_institutional_continuation(df5)
    if cont_sig:
        if institutional_flow_confirm(index, cont_sig, df5):
            return cont_sig, df5, False, "institutional_continuation"

    fvg_sig = detect_fair_value_gap(df5)
    if fvg_sig:
        if institutional_momentum_confirmation(index, df5, fvg_sig):
            return fvg_sig, df5, False, "fair_value_gap"

    volume_sig = detect_volume_gap_imbalance(df5)
    if volume_sig:
        if institutional_momentum_confirmation(index, df5, volume_sig):
            return volume_sig, df5, False, "volume_gap_imbalance"

    ote_sig = detect_ote_retracement(df5)
    if ote_sig:
        if institutional_momentum_confirmation(index, df5, ote_sig):
            return ote_sig, df5, False, "ote_retracement"

    ds_sig = detect_demand_supply_zones(df5)
    if ds_sig:
        if institutional_momentum_confirmation(index, df5, ds_sig):
            return ds_sig, df5, False, "demand_supply_zones"

    pull_sig = detect_pullback_reversal(df5)
    if pull_sig:
        if institutional_momentum_confirmation(index, df5, pull_sig):
            return pull_sig, df5, False, "pullback_reversal"

    flow_sig = mimic_orderflow_logic(df5)
    if flow_sig:
        if institutional_momentum_confirmation(index, df5, flow_sig):
            return flow_sig, df5, False, "orderflow_mimic"

    bottom_sig = detect_bottom_fishing(index, df5)
    if bottom_sig:
        if institutional_momentum_confirmation(index, df5, bottom_sig):
            return bottom_sig, df5, False, "bottom_fishing"

    bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
    liquidity_side = liquidity_zone_entry_check(last_close, bull_liq, bear_liq)
    if liquidity_side:
        return liquidity_side, df5, False, "liquidity_zone"

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

# 🚨 FIXED: INSTITUTIONAL FLOW CHECKS - NO CROSS-CONTAMINATION 🚨
def institutional_flow_signal(index, df5):
    try:
        last_close = float(ensure_series(df5["Close"]).iloc[-1])
        prev_close = float(ensure_series(df5["Close"]).iloc[-2])
    except:
        return None

    vol5 = ensure_series(df5["Volume"])
    vol_latest = float(vol5.iloc[-1])
    vol_avg = float(vol5.rolling(20).mean().iloc[-1]) if len(vol5) >= 20 else float(vol5.mean())

    # 🚨 FIX: Only use data from the current index, no cross-checking
    if vol_latest > vol_avg*2.0 and abs(last_close-prev_close)/prev_close>0.005:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg*1.5:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg*1.5:
        return "PE"
    
    # 🚨 FIX: Use only current index's liquidity zones
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)
    try:
        if last_close>=high_zone: return "PE"
        elif last_close<=low_zone: return "CE"
    except:
        return None
    return None

# 🚨 FIXED: OI + DELTA FLOW DETECTION - NO CROSS-CONTAMINATION 🚨
def oi_delta_flow_signal(index):
    try:
        url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        
        # 🚨 CRITICAL FIX: Exact index matching using startswith instead of contains
        df_index=df[df['symbol'].str.startswith(index)]
        
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

# --------- TRADE MONITORING AND TRACKING ---------
active_trades = {}

def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        weakness_sent = False
        in_trade = False
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        
        while True:
            if should_stop_trading():
                try:
                    final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                except Exception:
                    final_pnl = "0"
                signal_data.update({
                    "entry_status": "NOT_ENTERED" if not entry_price_achieved else "ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price_reached,
                    "zero_targets": sum(targets_hit) == 0,
                    "no_new_highs": max_price_reached <= entry,
                    "final_pnl": final_pnl
                })
                daily_signals.append(signal_data)
                break
                
            price = fetch_option_price(symbol)
            if not price: 
                time.sleep(10)
                continue
                
            price = round(price)
            
            if price > max_price_reached:
                max_price_reached = price
            
            if not in_trade:
                if price >= entry:
                    send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                    in_trade = True
                    entry_price_achieved = True
                    last_high = price
                    signal_data["entry_status"] = "ENTERED"
            else:
                if price > last_high:
                    send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                    last_high = price
                elif not weakness_sent and price < sl * 1.05:
                    send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                    weakness_sent = True
                
                for i, target in enumerate(targets):
                    if price >= target and not targets_hit[i]:
                        send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                        targets_hit[i] = True
                
                if price <= sl:
                    send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade.", reply_to=thread_id)
                    try:
                        final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                    except Exception:
                        final_pnl = "0"
                    signal_data.update({
                        "targets_hit": sum(targets_hit),
                        "max_price_reached": max_price_reached,
                        "zero_targets": sum(targets_hit) == 0,
                        "no_new_highs": max_price_reached <= entry,
                        "final_pnl": final_pnl
                    })
                    daily_signals.append(signal_data)
                    break
                    
                if all(targets_hit):
                    send_telegram(f"🏆 {symbol}: ALL TARGETS HIT! Trade completed successfully!", reply_to=thread_id)
                    try:
                        final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                    except Exception:
                        final_pnl = "0"
                    signal_data.update({
                        "targets_hit": len(targets),
                        "max_price_reached": max_price_reached,
                        "zero_targets": False,
                        "no_new_highs": False,
                        "final_pnl": final_pnl
                    })
                    daily_signals.append(signal_data)
                    break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

def calculate_pnl(entry, max_price, targets, targets_hit, sl):
    try:
        if targets is None or len(targets) == 0:
            diff = max_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        
        if not isinstance(targets_hit, (list, tuple)):
            targets_hit = list(targets_hit) if targets_hit is not None else [False]*len(targets)
        if len(targets_hit) < len(targets):
            targets_hit = list(targets_hit) + [False] * (len(targets) - len(targets_hit))
        
        achieved_prices = [target for i, target in enumerate(targets) if targets_hit[i]]
        if achieved_prices:
            exit_price = achieved_prices[-1]
            diff = exit_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        else:
            if max_price <= sl:
                diff = sl - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
            else:
                diff = max_price - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
    except Exception:
        return "0"

# --------- NEW INDIVIDUAL SIGNAL REPORTING ---------
def send_individual_signal_reports():
    """Send each signal in separate detailed messages after market hours"""
    global daily_signals, all_generated_signals
    
    # 🚨 CRITICAL FIX: Combine both signal sources
    all_signals = daily_signals + all_generated_signals
    
    # Remove duplicates based on signal_id
    seen_ids = set()
    unique_signals = []
    for signal in all_signals:
        sid = signal.get('signal_id')
        if not sid:
            continue
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_signals.append(signal)
    
    if not unique_signals:
        send_telegram("📊 END OF DAY REPORT\nNo signals generated today.")
        return
    
    # Send header message
    send_telegram(f"🕒 END OF DAY SIGNAL REPORT - { (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y') }\n"
                  f"📈 Total Signals: {len(unique_signals)}\n"
                  f"─────────────────────────────")
    
    # Send each signal in separate message
    for i, signal in enumerate(unique_signals, 1):
        targets_hit_list = []
        if signal.get('targets_hit', 0) > 0:
            for j in range(signal.get('targets_hit', 0)):
                if j < len(signal.get('targets', [])):
                    targets_hit_list.append(str(signal['targets'][j]))
        
        targets_for_disp = signal.get('targets', [])
        while len(targets_for_disp) < 4:
            targets_for_disp.append('-')
        
        msg = (f"📊 SIGNAL #{i} - {signal.get('index','?')} {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"─────────────────────────────\n"
               f"📅 Date: {(datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y')}\n"
               f"🕒 Time: {signal.get('timestamp','?')}\n"
               f"📈 Index: {signal.get('index','?')}\n"
               f"🎯 Strike: {signal.get('strike','?')}\n"
               f"🔰 Type: {signal.get('option_type','?')}\n"
               f"🏷️ Strategy: {signal.get('strategy','?')}\n\n"
               
               f"💰 ENTRY: ₹{signal.get('entry_price','?')}\n"
               f"🎯 TARGETS: {targets_for_disp[0]} // {targets_for_disp[1]} // {targets_for_disp[2]} // {targets_for_disp[3]}\n"
               f"🛑 STOP LOSS: ₹{signal.get('sl','?')}\n\n"
               
               f"📊 PERFORMANCE:\n"
               f"• Entry Status: {signal.get('entry_status', 'PENDING')}\n"
               f"• Targets Hit: {signal.get('targets_hit', 0)}/4\n")
        
        if targets_hit_list:
            msg += f"• Targets Achieved: {', '.join(targets_hit_list)}\n"
        
        msg += (f"• Max Price Reached: ₹{signal.get('max_price_reached', signal.get('entry_price','?'))}\n"
                f"• Final P&L: {signal.get('final_pnl', '0')} points\n\n"
                
                f"⚡ Fakeout: {'YES' if signal.get('fakeout') else 'NO'}\n"
                f"📈 Index Price at Signal: {signal.get('index_price','?')}\n"
                f"🆔 Signal ID: {signal.get('signal_id','?')}\n"
                f"─────────────────────────────")
        
        send_telegram(msg)
        time.sleep(1)
    
    # Send summary
    total_pnl = 0.0
    successful_trades = 0
    for signal in unique_signals:
        pnl_str = signal.get("final_pnl", "0")
        try:
            if isinstance(pnl_str, str) and pnl_str.startswith("+"):
                total_pnl += float(pnl_str[1:])
                successful_trades += 1
            elif isinstance(pnl_str, str) and pnl_str.startswith("-"):
                total_pnl -= float(pnl_str[1:])
        except:
            pass
    
    summary_msg = (f"📈 DAY SUMMARY\n"
                   f"─────────────────────────────\n"
                   f"• Total Signals: {len(unique_signals)}\n"
                   f"• Successful Trades: {successful_trades}\n"
                   f"• Success Rate: {(successful_trades/len(unique_signals))*100:.1f}%\n"
                   f"• Total P&L: ₹{total_pnl:+.2f}\n"
                   f"─────────────────────────────")
    
    send_telegram(summary_msg)
    
    # 🚨 COMPULSORY CONFIRMATION
    send_telegram("✅ END OF DAY REPORTS COMPLETED! See you tomorrow at 9:15 AM! 🚀")

# 🚨 FIXED: UPDATED SIGNAL SENDING - NO CROSS-CONTAMINATION 🚨
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    # 🚨 FIX: Each index only processes its own data
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
    strike = round_strike(index, signal_detection_price)
    
    # 🚨 ADDED: VALIDATION CHECK FOR STRIKE PRICE
    if strike is None or strike <= 0:
        send_telegram(f"⚠️ {index}: Invalid strike price detected. Signal skipped.")
        return
        
    # 🚨 ADDED: CHECK IF STRIKE IS REASONABLE (within 20% of current price)
    if abs(strike - signal_detection_price) / signal_detection_price > 0.20:
        send_telegram(f"⚠️ {index}: Strike {strike} too far from current price {signal_detection_price}. Signal skipped.")
        return
        
    # 🚨 FIX: Each index only sends its own symbol
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    option_price = fetch_option_price(symbol)
    if not option_price or option_price <= 1: 
        send_telegram(f"⚠️ {index}: Option price too low or unavailable. Signal skipped.")
        return
    
    entry = round(option_price)
    
    # 🚨 ADDED: VALIDATION FOR ENTRY PRICE
    if entry <= 1:
        send_telegram(f"⚠️ {index}: Entry price too low (₹{entry}). Signal skipped.")
        return
    
    high = ensure_series(df["High"])
    low = ensure_series(df["Low"])
    close = ensure_series(df["Close"])
    atr = float(ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1])
    
    atr_multiplier = 0.3
    base_target = option_price
    
    targets = [
        round(base_target + (atr * atr_multiplier * 1.0)),
        round(base_target + (atr * atr_multiplier * 1.5)),
        round(base_target + (atr * atr_multiplier * 2.2)),
        round(base_target + (atr * atr_multiplier * 3.0))
    ]
    
    sl = round(option_price - (atr * 0.8))
    
    # 🚨 ADDED: VALIDATION FOR STOP LOSS
    if sl <= 0:
        sl = round(option_price * 0.8)  # Fallback SL
    
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key.upper())
    
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    signal_data = {
        "signal_id": signal_id,
        "timestamp": (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": side,
        "strategy": strategy_name,
        "entry_price": entry,
        "targets": targets,
        "sl": sl,
        "fakeout": fakeout,
        "index_price": signal_detection_price,
        "entry_status": "PENDING",
        "targets_hit": 0,
        "max_price_reached": entry,
        "zero_targets": True,
        "no_new_highs": True,
        "final_pnl": "0"
    }
    
    # 🚨 CRITICAL FIX: Track signal immediately
    all_generated_signals.append(signal_data.copy())
    
    msg = (f"🟢 GIT🔊 {index} {strike} {side} - {strategy_name}\n"
           f"🔹 Strike: {strike}\n"
           f"🟩 Buy Above ₹{entry}\n"
           f"🔴 SL: ₹{sl}\n"
           f"🎯 TGT: {targets_str}\n"
           f"⚡ Fakeout: {'YES' if fakeout else 'NO'}\n"
           f"📊 Layer: {strategy_name} | Index: {signal_detection_price}\n"
           f"🆔 Signal ID: {signal_id}")
         
    thread_id = send_telegram(msg)
    
    trade_id = f"{symbol}_{int(time.time())}"
    active_trades[trade_id] = {
        "symbol": symbol, 
        "entry": entry, 
        "sl": sl, 
        "targets": targets, 
        "thread": thread_id, 
        "status": "OPEN",
        "index": index,
        "signal_data": signal_data
    }
    
    monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# 🚨 FIXED: UPDATED TRADE THREAD - NO CROSS-CONTAMINATION 🚨
def trade_thread(index):
    """🚨 FIX: Each index thread operates independently"""
    result = analyze_index_signal(index)
    
    if not result:
        return
        
    if len(result) == 4:
        side, df, fakeout, strategy_key = result
    else:
        side, df, fakeout = result
        strategy_key = "unknown"
    
    # 🚨 FIX: Each index only uses its own data for confirmation
    df5 = fetch_index_data(index, "5m", "2d")
    inst_signal = institutional_flow_signal(index, df5) if df5 is not None else None
    oi_signal = oi_delta_flow_signal(index)
    final_signal = oi_signal or inst_signal or side

    if final_signal == "BOTH":
        for s in ["CE", "PE"]:
            if institutional_flow_confirm(index, s, df5):
                send_signal(index, s, df, fakeout, strategy_key)
        return
    elif final_signal:
        if df is None: 
            df = df5
        if institutional_flow_confirm(index, final_signal, df5):
            send_signal(index, final_signal, df, fakeout, strategy_key)
    else:
        return

# --------- MAIN LOOP (ALL INDICES PARALLEL) ---------
def run_algo_parallel():
    if not is_market_open(): 
        print("❌ Market closed - skipping iteration")
        return
        
    threads = []
    all_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "EICHERMOT", "TRENT", "RELIANCE"]
    
    for index in all_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# --------- FIXED MAIN LOOP WITH GUARANTEED EOD REPORTS ---------
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False
STARTED_SENT = False
STOP_SENT = False

while True:
    try:
        # Get current IST time
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        current_datetime_ist = ist_now
        
        # Check if market is open
        market_open = is_market_open()
        
        # 🚨 MARKET CLOSED BEHAVIOR
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market is currently closed. Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            # 🚨 COMPULSORY EOD REPORT TRIGGER BETWEEN 3:30 PM - 4:00 PM
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
                time.sleep(10)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ EOD Report completed! Algorithm will resume tomorrow.")
            
            time.sleep(30)
            continue
        
        # 🚨 MARKET OPEN BEHAVIOR
        if not STARTED_SENT:
            send_telegram("🚀 GIT ULTIMATE MASTER ALGO STARTED - All 8 Indices Running\n"
                         "✅ INSTITUTIONAL ANTICIPATION ENGINE ACTIVATED 🎯\n"
                         "✅ CROSS-INDEX CONTAMINATION FIXED 🎯\n"
                         "✅ Gamma Priority on Expiry Days\n"
                         "✅ Real-time Signal Tracking\n"
                         "✅ Comprehensive P&L Analysis")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        # 🚨 MARKET CLOSE DETECTION WITH GUARANTEED EOD REPORT
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached! Preparing EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            # 🚨 GUARANTEED EOD REPORT - NO EXCEPTIONS
            if not EOD_REPORT_SENT:
                send_telegram("📊 FINALIZING TRADES...")
                time.sleep(20)  # Extra time for all threads to complete
                try:
                    send_individual_signal_reports()
                except Exception as e:
                    send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                    time.sleep(10)
                    send_individual_signal_reports()  # Retry once
                EOD_REPORT_SENT = True
                send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
            time.sleep(60)
            continue
            
        # 🚨 RUN MAIN ALGORITHM DURING MARKET HOURS
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)
