import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import pytz  # <--- เพิ่มตัวจัดการเขตเวลา
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ 1. ตั้งค่า API (ใช้ Secrets จาก GitHub)
# ==========================================
API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# โหลดสมอง AI
try:
    optimized_ai_v3 = joblib.load('super_model.joblib')
except Exception as e:
    print(f"❌ โหลดสมอง AI ไม่สำเร็จ: {e}")
    exit()

WATCHLIST = ['PLTR', 'AMD', 'TSM', 'MU', 'NVDA', 'META', 'NFLX', 'ASML', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'CRWD', 'AVGO']
THRESHOLD = 0.60
MAX_CAPITAL_PER_STOCK = 5000

# ==========================================
# 🕒 2. ฟังก์ชันเช็คเวลาตลาดหุ้น (New York Time)
# ==========================================
def is_market_closing_soon():
    # กำหนดเขตเวลา New York
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # ตลาดหุ้นสหรัฐฯ ปิดเวลา 16:00 น. ของที่นั่น
    market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # คำนวณว่าเหลืออีกกี่นาทีจะปิดตลาด
    diff_minutes = (market_close - now_ny).total_seconds() / 60
    
    print(f"🕒 เวลาที่ New York: {now_ny.strftime('%H:%M:%S')}")
    print(f"⏳ เหลืออีก {diff_minutes:.2f} นาทีก่อนตลาดปิด")

    # เงื่อนไข: ยอมให้เทรดเฉพาะช่วง 20 นาทีสุดท้ายก่อนตลาดปิด (15:40 - 16:00)
    if 0 <= diff_minutes <= 20:
        return True
    return False

def run_bot():
    # เช็คด่านแรก: ถ้าไม่ใช่เวลาใกล้ปิดตลาด ให้หยุดทำงานทันที
    if not is_market_closing_soon():
        print("🛑 ยังไม่ถึงช่วง 20 นาทีสุดท้ายก่อนตลาดปิด... บอทขอนั่งทับมือรอรอบถัดไป!")
        return

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 สตาร์ทเครื่องลุยตลาดช่วงโค้งสุดท้าย...")
    
    # --- (ส่วนดึงข้อมูล Macro เหมือนเดิม) ---
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']
    yield10y = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']
    
    for symbol in WATCHLIST:
        try:
            # เช็คสถานะพอร์ต
            try:
                position = api.get_position(symbol)
                qty = float(position.qty)
                is_holding = True
            except:
                is_holding = False

            # ดึงข้อมูลและคำนวณ Technical Features
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            
            # ... (ส่วนคำนวณ Indicator ทั้งหมดที่คุณมีอยู่เดิม Price_vs_SMA20, ATR, etc.) ...
            # หมายเหตุ: ตรงนี้ใส่สูตรคำนวณ Indicator เดิมของคุณกลับเข้าไปให้ครบนะครับ
            
            # สมมติว่าได้แถวล่าสุดมาแล้ว
            latest = df.iloc[-1]
            
            # --- ลอจิกการขาย ---
            if is_holding:
                if latest['EMA_5'] < latest['EMA_13']:
                    print(f"🚨 ขาย {symbol} ล้างพอร์ต")
                    api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
                continue
            
            # --- ลอจิกการซื้อ ---
            if latest['EMA_5'] > latest['EMA_13']: # เช็คสัญญาณเบื้องต้น
                features = ['Price_vs_SMA20', 'OBV_Slope', 'ATR', 'Regime', 'Divergence_Factor', 'Upper_Wick_Ratio', 'Yield10Y', 'VIX']
                X_live = pd.DataFrame([latest[features]])
                ai_prob = optimized_ai_v3.predict_proba(X_live)[:, 1][0]
                
                if ai_prob > THRESHOLD:
                    invest = MAX_CAPITAL_PER_STOCK * ((ai_prob - THRESHOLD) / (1.0 - THRESHOLD))
                    shares = int(invest // latest['Close'])
                    if shares > 0:
                        print(f"🔥 สั่งซื้อ {symbol} จำนวน {shares} หุ้น")
                        api.submit_order(symbol=symbol, qty=shares, side='buy', type='market', time_in_force='gtc')
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดกับ {symbol}: {e}")
            continue

if __name__ == "__main__":
    run_bot()
