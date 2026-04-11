import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ 1. ตั้งค่า Alpaca API (นำ Key จากเว็บ Alpaca มาใส่ตรงนี้)
# ==========================================
import os
API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# โหลดสมอง AI ที่เทรนไว้
try:
    print("🧠 กำลังโหลดสมอง Super Model...")
    optimized_ai_v3 = joblib.load('super_model.joblib')
except Exception as e:
    print(f"❌ โหลดสมอง AI ไม่สำเร็จ: {e}")
    exit()

# ดรีมทีมหุ้น Quality Tech
WATCHLIST = ['PLTR', 'AMD', 'TSM', 'MU', 'NVDA', 'META', 'NFLX', 'ASML', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'CRWD', 'AVGO']
THRESHOLD = 0.60
MAX_CAPITAL_PER_STOCK = 5000
ATR_MULTIPLIER = 1.5

def run_bot():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 สตาร์ทเครื่อง Super Bot บนน่านฟ้า Cloud...")
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']
    yield10y = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']
    
    for symbol in WATCHLIST:
        print(f"\n" + "-"*30)
        print(f"🎯 วิเคราะห์: {symbol}")
        
        try:
            position = api.get_position(symbol)
            qty, current_price = float(position.qty), float(position.current_price)
            is_holding = True
        except:
            is_holding = False

        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            
            df['VIX'] = vix
            df['Yield10Y'] = yield10y
            df.ffill(inplace=True)
            
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_13'] = df['Close'].ewm(span=13, adjust=False).mean()
            df['Base_Signal'] = (df['EMA_5'] > df['EMA_13']).astype(int)
            df['Buy_Entry'] = (df['Base_Signal'].diff() == 1).astype(int)
            
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['Price_vs_SMA20'] = df['Close'] - df['SMA_20']
            
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['Regime'] = (df['Close'] > df['SMA_200']).astype(int)
            
            cr = df['High'] - df['Low']
            df['Upper_Wick_Ratio'] = np.where(cr == 0, 0, (df['High'] - df[['Open', 'Close']].max(axis=1)) / cr)
            df['OBV_Slope'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum().diff(5)
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            df['Divergence_Factor'] = df['Close'].pct_change(14) - df['RSI'].pct_change(14)
            
            df = df.dropna()
            latest = df.iloc[-1]
            
        except Exception as e:
            continue 
            
        if is_holding:
            if latest['Base_Signal'] == 0:
                print(f"🚨 ขายล้างพอร์ต {symbol}")
                api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
            else:
                print(f"✅ ถือ {symbol} รันเทรนด์")
            continue
            
        if latest['Buy_Entry'] == 1:
            features = ['Price_vs_SMA20', 'OBV_Slope', 'ATR', 'Regime', 'Divergence_Factor', 'Upper_Wick_Ratio', 'Yield10Y', 'VIX']
            X_live = pd.DataFrame([latest[features]])
            ai_prob = optimized_ai_v3.predict_proba(X_live)[:, 1][0]
            
            if ai_prob > THRESHOLD:
                invest = MAX_CAPITAL_PER_STOCK * ((ai_prob - THRESHOLD) / (1.0 - THRESHOLD))
                shares = int(invest // latest['Close'])
                if shares > 0:
                    print(f"🔥 ยิงออเดอร์ซื้อ {symbol} | {shares} หุ้น")
                    api.submit_order(symbol=symbol, qty=shares, side='buy', type='market', time_in_force='gtc')
            else:
                print(f"🛑 AI ปฏิเสธการเข้าซื้อ {symbol}")
        else:
            print(f"⏳ รอสัญญาณ {symbol}")

    print("\n🏁 สแกนตลาดเรียบร้อย!")

if __name__ == "__main__":
    run_bot()