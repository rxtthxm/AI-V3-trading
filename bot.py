import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import pytz
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ 1. ตั้งค่า API
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
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    diff_minutes = (market_close - now_ny).total_seconds() / 60
    
    print(f"🕒 เวลาที่ New York: {now_ny.strftime('%H:%M:%S')}")
    print(f"⏳ เหลืออีก {diff_minutes:.2f} นาทีก่อนตลาดปิด")

    if 0 <= diff_minutes <= 20:
        return True
    return False

# ==========================================
# 🚀 3. ระบบเทรดหลัก
# ==========================================
def run_bot():
    if not is_market_closing_soon():
        print("🛑 ยังไม่ถึงช่วง 20 นาทีสุดท้ายก่อนตลาดปิด... บอทขอนั่งทับมือรอรอบถัดไป!")
        return

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 สตาร์ทเครื่องลุยตลาดช่วงโค้งสุดท้าย...")
    
    # ดึงข้อมูล Macro (VIX, 10Y Yield) เผื่อไว้ใช้กับทุกหุ้น
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    macro_tickers = ['^VIX', '^TNX']
    macro_data = yf.download(macro_tickers, start=start_date, end=end_date, progress=False)['Close']
    macro_data.columns = ['VIX', 'Yield10Y']
    
    for symbol in WATCHLIST:
        try:
            # เช็คสถานะพอร์ต
            try:
                position = api.get_position(symbol)
                qty = float(position.qty)
                is_holding = True
            except:
                is_holding = False

            # ดึงข้อมูลราคาย้อนหลัง
            raw = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if raw.empty: continue
            if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
            
            # เตรียมข้อมูลพื้นฐาน
            df = pd.DataFrame(index=raw.index)
            c = raw['Close']
            h = raw['High']
            l = raw['Low']
            o = raw['Open']
            v = raw['Volume']
            df['Close'] = c
            
            # --------------------------------------------------
            # 🧠 คำนวณ Indicator สำหรับ AI v3
            # --------------------------------------------------
            df['Price_vs_SMA20'] = c - c.rolling(window=20).mean()
            
            tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=14).mean()
            
            obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
            df['OBV_Slope'] = obv.diff(5) / (obv.rolling(window=20).std() + 1e-9)
            
            df['Regime'] = (c > c.rolling(window=200).mean()).astype(int)
            
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            df['Divergence_Factor'] = c.diff(5) - rsi.diff(5)
            
            body_max = np.maximum(o, c)
            df['Upper_Wick_Ratio'] = (h - body_max) / (h - l + 1e-9)
            
            df = df.join(macro_data, how='left').ffill()
            
            # --------------------------------------------------
            # 🎯 คำนวณเส้น EMA 5 และ 13 ตามที่คุณต้องการ
            # --------------------------------------------------
            df['EMA_5'] = c.ewm(span=5, adjust=False).mean()
            df['EMA_13'] = c.ewm(span=13, adjust=False).mean()
            
            df.dropna(inplace=True)
            if df.empty: continue
            
            latest = df.iloc[-1]
            
            # --------------------------------------------------
            # ⚖️ ลอจิกการขาย (Sell)
            # --------------------------------------------------
            if is_holding:
                if latest['EMA_5'] < latest['EMA_13']:
                    print(f"🚨 ขาย {symbol} ล้างพอร์ต (เทรนด์ EMA 5/13 เปลี่ยน)")
                    api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
                continue
            
            # --------------------------------------------------
            # 🔥 ลอจิกการซื้อ (Buy)
            # --------------------------------------------------
            if latest['EMA_5'] > latest['EMA_13']:
                features = ['Price_vs_SMA20', 'ATR', 'OBV_Slope', 'Regime', 'Divergence_Factor', 'Upper_Wick_Ratio', 'VIX', 'Yield10Y']
                X_live = pd.DataFrame([latest[features]])
                
                ai_prob = optimized_ai_v3.predict_proba(X_live)[:, 1][0]
                
                if ai_prob > THRESHOLD:
                    invest = MAX_CAPITAL_PER_STOCK * ((ai_prob - THRESHOLD) / (1.0 - THRESHOLD))
                    shares = int(invest // latest['Close'])
                    if shares > 0:
                        print(f"🔥 สั่งซื้อ {symbol} จำนวน {shares} หุ้น (AI มั่นใจ {ai_prob*100:.2f}%)")
                        api.submit_order(symbol=symbol, qty=shares, side='buy', type='market', time_in_force='gtc')
                        
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดกับ {symbol}: {e}")
            continue

if __name__ == "__main__":
    run_bot()
