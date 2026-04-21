import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import pytz
import requests
import gspread
from google.oauth2.service_account import Credentials
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'
TG_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TG_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

try:
    model_bull = joblib.load('model_bull.joblib')
    model_bear = joblib.load('model_bear.joblib')
except Exception as e:
    print(f"❌ Model load failed: {e}")
    exit()

WATCHLIST = [
    'NVDA', 'META', 'TSLA', 'AMD', 'MU', 'PLTR', 
    'LRCX', 'AMAT', 'CRM', 'UBER', 
    'GOOGL', 'AMZN', 'MSFT', 'AVGO', 'AAPL', 'ORCL', 
    'TXN', 'ADI', 'MRVL', 'NET', 'ON', 'MCHP', 'TSM', 
    'ASML', 'QCOM', 'ARM', 'PANW', 'SNOW', 'DDOG', 'NOW'
]

# ==========================================
# 2. TIME VALIDATION (เช็คเวลาทำงาน)
# ==========================================
def check_trading_window():
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    h, m = now_ny.hour, now_ny.minute
    print(f"🗽 เวลาปัจจุบันของ NY: {now_ny.strftime('%H:%M:%S')}")
    
    # ถ้ากดรันเองผ่าน GitHub Actions ให้ผ่านตลอด
    if os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch':
        print("🛠️ Manual Test Detected: ข้ามการตรวจเวลา")
        return True
        
    # รันช่วง 15:40 - 16:15 NY Time (ขยายเวลาเผื่อ Server ดีเลย์)
    if (h == 15 and m >= 40) or (h == 16 and m <= 15):
        return True
        
    return False

if not check_trading_window():
    print("⏰ นอกเวลาทำการ บอทจะหยุดทำงาน")
    exit()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def send_telegram(message):
    if TG_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': TG_CHAT_ID, 'text': message, 'parse_mode': 'HTML'})

def get_latest_data(symbol):
    # ดึงข้อมูลจาก yfinance เพื่อทำ Indicator (ใส่ Logic Indicator ของคุณตรงนี้)
    df = yf.download(symbol, period='1y', interval='1d', progress=False)
    if df.empty: return None
    
    # คำนวณ ATR, EMA และ Feature ต่างๆ ที่ AI ต้องใช้
    df['ATR'] = df['High'].sub(df['Low']).rolling(14).mean() # ตัวอย่างอย่างง่าย
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_13'] = df['Close'].ewm(span=13, adjust=False).mean()
    
    # เพิ่มการคำนวณ Features ของคุณที่นี่ (เช่น OBV_Slope, Divergence_Factor, ฯลฯ)
    # df['Price_vs_SMA20'] = ...
    
    return df

def calculate_shares(equity, prob, atr, price):
    risk_pct = 0.03 if prob >= 0.90 else 0.02 if prob >= 0.75 else 0.01
    risk_amount = equity * risk_pct
    stop_distance = 2.1 * atr
    if stop_distance <= 0: return 0
    shares = int(risk_amount / stop_distance)
    max_shares = int((equity * 0.20) / price)
    return min(shares, max_shares)

# ==========================================
# 4. MAIN TRADING LOGIC
# ==========================================
account = api.get_account()
account_equity = float(account.equity)
print(f"💰 Equity: ${account_equity:,.2f}")

buy_log = []
sell_log = []

# --- ระบบเช็คสภาวะตลาด (Regime) ---
spy = yf.download('SPY', period='3mo', progress=False)
spy['EMA_20'] = spy['Close'].ewm(span=20).mean()
spy['EMA_50'] = spy['Close'].ewm(span=50).mean()
is_bull_market = spy['EMA_20'].iloc[-1].item() > spy['EMA_50'].iloc[-1].item()
regime_label = "BULL" if is_bull_market else "BEAR"
model = model_bull if is_bull_market else model_bear

print(f"📊 Market Regime: {regime_label}")

# ---------------------------------------------------------
# STEP A: DUAL-STOP EXIT (ตรวจสอบหุ้นที่ถืออยู่และตั้งจุดขาย)
# ---------------------------------------------------------
positions = api.list_positions()
open_symbols = [p.symbol for p in positions]

for position in positions:
    symbol = position.symbol
    qty = position.qty
    current_price = float(position.current_price)
    avg_entry = float(position.avg_entry_price)
    profit_pct = (current_price - avg_entry) / avg_entry
    
    df = get_latest_data(symbol)
    if df is None: continue
    latest = df.iloc[-1]
    
    # 1. HARD STOP (-4.0%)
    if profit_pct <= -0.04:
        reason = f"Hard Stop (-4%)"
        print(f"🛑 {symbol}: {reason} ({profit_pct*100:.2f}%)")
        api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
        sell_log.append(f"🔴 SELL {symbol}: {reason}")
        continue

    # 2. DYNAMIC 2.1x ATR TRAILING STOP
    buy_orders = api.list_orders(status='filled', symbols=[symbol], limit=20)
    buy_orders = [o for o in buy_orders if o.side == 'buy']
    
    if buy_orders:
        last_buy_date = buy_orders[0].filled_at.strftime('%Y-%m-%d')
        try:
            # หาจุดสูงสุดนับตั้งแต่ซื้อมา
            highest_high = df.loc[last_buy_date:]['High'].max().item()
        except:
            highest_high = current_price
            
        trailing_stop_price = highest_high - (2.1 * latest['ATR'].item())
        
        if current_price < trailing_stop_price:
            reason = f"ATR Trailing Stop"
            print(f"📉 {symbol}: {reason} ที่ ${current_price:.2f}")
            api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
            sell_log.append(f"🔴 SELL {symbol}: {reason}")

# ---------------------------------------------------------
# ---------------------------------------------------------
# STEP B: AI ENTRY & COOLDOWN (ค้นหาหุ้นเข้าซื้อ)
# ---------------------------------------------------------
for symbol in WATCHLIST:
    if symbol in open_symbols:
        continue # ข้ามตัวที่มีอยู่แล้ว
        
    df = get_latest_data(symbol)
    if df is None: continue
    latest = df.iloc[-1]
    
    # --- 1. สมอง AI ของจริง (เอาบรรทัด 0.65 ออกแล้วใช้ของจริง) ---
    try:
        features = ['Price_vs_SMA20', 'OBV_Slope', 'ATR', 'Regime', 'Divergence_Factor', 'Upper_Wick_Ratio', 'Yield10Y', 'VIX']
        X_live = pd.DataFrame([latest[features]])
        ai_prob = model.predict_proba(X_live)[:, 1][0]
    except KeyError as e:
        print(f"⚠️ คำนวณข้าม {symbol} เนื่องจากขาดข้อมูล Indicator: {e}")
        continue
    # --------------------------------------------------------

    if ai_prob >= 0.60:
        # 2. COOLDOWN FILTER
        recent_orders = api.list_orders(status='filled', symbols=[symbol], limit=10)
        sell_orders = [o for o in recent_orders if o.side == 'sell']
        
        cooldown_active = False
        if sell_orders:
            last_sell_date = sell_orders[0].filled_at
            now_utc = datetime.now(pytz.utc)
            if (now_utc - last_sell_date).days <= 1:
                print(f"⏳ ข้าม {symbol} (ติด Cooldown 1 วัน)")
                cooldown_active = True
                
        if not cooldown_active:
            shares = calculate_shares(account_equity, ai_prob, latest['ATR'].item(), latest['Close'].item())
            if shares > 0:
                tier = "🔥🔥🔥" if ai_prob >= 0.90 else "🔥🔥" if ai_prob >= 0.75 else "🔥"
                
                # --- 3. ใส่ระบบกันกระแทก (Try...Except) ป้องกันเงินหมดพอร์ตแล้วแครช ---
                try:
                    api.submit_order(symbol=symbol, qty=shares, side='buy', type='market', time_in_force='day')
                    print(f"{tier} สั่งซื้อ {symbol} จำนวน {shares} หุ้น (AI {ai_prob*100:.1f}%)")
                    buy_log.append(f"🟢 BUY {symbol} x{shares} ({tier} AI: {ai_prob*100:.1f}%)")
                except Exception as e:
                    print(f"❌ ข้าม {symbol} สั่งซื้อไม่สำเร็จ (เงินอาจจะหมดพอร์ต): {e}")
                # ------------------------------------------------------------------

# ==========================================
# 5. SUMMARY NOTIFICATION
# ==========================================
msg = f"🤖 <b>AI Trading V4.0 (Single-Pass)</b>\n"
msg += f"📊 Regime: {regime_label}\n"
msg += f"💰 Equity: ${account_equity:,.2f}\n\n"

if buy_log:
    msg += "<b>🛒 รายการซื้อ:</b>\n" + "\n".join(buy_log) + "\n\n"
if sell_log:
    msg += "<b>🚪 รายการขาย:</b>\n" + "\n".join(sell_log) + "\n"

if not buy_log and not sell_log:
    msg += "<i>ไม่มีแอคชั่นในวันนี้</i>"

send_telegram(msg)
print("✅ ทำงานเสร็จสมบูรณ์")
