import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import pytz
import warnings
warnings.filterwarnings(‘ignore’)

API_KEY = os.environ.get(‘ALPACA_API_KEY’)
API_SECRET = os.environ.get(‘ALPACA_SECRET_KEY’)
BASE_URL = ‘https://paper-api.alpaca.markets’
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version=‘v2’)

try:
model_bull = joblib.load(‘final_strategy_bull.joblib’)
model_bear = joblib.load(‘final_strategy_bear.joblib’)
except Exception as e:
print(f”❌ โหลดโมเดลไม่สำเร็จ: {e}”)
exit()

WATCHLIST = [
‘NVDA’, ‘META’, ‘TSLA’, ‘AMD’, ‘MU’, ‘PLTR’,
‘LRCX’, ‘AMAT’, ‘CRM’, ‘UBER’,
‘GOOGL’, ‘AMZN’, ‘MSFT’, ‘AVGO’, ‘AAPL’, ‘ORCL’,
‘TXN’, ‘ADI’, ‘MRVL’, ‘NET’, ‘ON’, ‘MCHP’, ‘TSM’
]

ATR_MULTIPLIER = 2.1
ACCOUNT_RISK_PER_TRADE = {
‘tier1’: 0.01,
‘tier2’: 0.02,
‘tier3’: 0.03,
}

def is_market_closing_soon():
tz_ny = pytz.timezone(‘America/New_York’)
now_ny = datetime.now(tz_ny)
market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
diff_minutes = (market_close - now_ny).total_seconds() / 60
print(f”🕒 เวลาที่ New York: {now_ny.strftime(’%H:%M:%S’)}”)
print(f”⏳ เหลืออีก {diff_minutes:.2f} นาทีก่อนตลาดปิด”)
return 0 <= diff_minutes <= 20

def get_risk_tier(ai_prob):
if ai_prob >= 0.90:
return ACCOUNT_RISK_PER_TRADE[‘tier3’]
elif ai_prob >= 0.75:
return ACCOUNT_RISK_PER_TRADE[‘tier2’]
elif ai_prob >= 0.60:
return ACCOUNT_RISK_PER_TRADE[‘tier1’]
return 0

def calculate_shares(account_equity, ai_prob, atr, price):
risk_pct = get_risk_tier(ai_prob)
if risk_pct == 0:
return 0
risk_amount = account_equity * risk_pct
stop_distance = ATR_MULTIPLIER * atr
shares = int(risk_amount // stop_distance)
max_shares = int((account_equity * 0.20) // price)
return min(shares, max_shares)

def run_bot():
if not is_market_closing_soon():
print(“🛑 ยังไม่ถึงช่วง 20 นาทีสุดท้ายก่อนตลาดปิด… บอทขอนั่งทับมือรอรอบถัดไป!”)
return

```
print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 สตาร์ทเครื่องลุยตลาดช่วงโค้งสุดท้าย...")

account = api.get_account()
account_equity = float(account.equity)
print(f"💰 Portfolio Equity: ${account_equity:,.2f}")

open_orders = [o.symbol for o in api.list_orders(status='open')]

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

macro_tickers = ['^VIX', '^TNX', 'SPY']
macro_raw = yf.download(macro_tickers, start=start_date, end=end_date, progress=False)['Close']
macro_raw.columns = ['SPY', 'VIX', 'Yield10Y']

macro_data = pd.DataFrame(index=macro_raw.index)
macro_data['VIX'] = macro_raw['VIX']
macro_data['Yield10Y'] = macro_raw['Yield10Y']
macro_data['VIX_ROC'] = macro_raw['VIX'].pct_change(5)
macro_data['SPY_MOM_5'] = macro_raw['SPY'].pct_change(5)
macro_data['SPY_MOM_20'] = macro_raw['SPY'].pct_change(20)
macro_data['SPY_Regime'] = (macro_raw['SPY'].pct_change(60) > 0).astype(int)

latest_macro = macro_data.iloc[-1]
is_bull = latest_macro['SPY_Regime'] == 1
model = model_bull if is_bull else model_bear
regime_label = "BULL 🐂" if is_bull else "BEAR 🐻"
print(f"📊 Regime ปัจจุบัน: {regime_label}")

for symbol in WATCHLIST:
    try:
        if symbol in open_orders:
            print(f"⏭️ {symbol} มี order ค้างอยู่ ข้ามไป")
            continue

        try:
            position = api.get_position(symbol)
            qty = float(position.qty)
            avg_entry = float(position.avg_entry_price)
            is_holding = True
        except:
            is_holding = False

        raw = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        df = pd.DataFrame(index=raw.index)
        c = raw['Close']
        h = raw['High']
        l = raw['Low']
        o = raw['Open']
        v = raw['Volume']
        df['Close'] = c

        df['Price_vs_SMA20'] = c - c.rolling(window=20).mean()

        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        df['OBV_Slope'] = obv.diff(5) / (obv.rolling(window=20).std() + 1e-9)

        df['Regime'] = (c - c.rolling(window=200).mean()) / (c.rolling(window=200).std() + 1e-9)

        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        df['Divergence_Factor'] = c.diff(5) - rsi.diff(5)

        body_max = np.maximum(o, c)
        df['Upper_Wick_Ratio'] = (h - body_max) / (h - l + 1e-9)

        df['EMA_5'] = c.ewm(span=5, adjust=False).mean()
        df['EMA_13'] = c.ewm(span=13, adjust=False).mean()

        df = df.join(macro_data, how='left').ffill()
        df.dropna(inplace=True)
        if df.empty:
            continue

        latest = df.iloc[-1]

        if is_holding:
            trailing_stop = avg_entry - (ATR_MULTIPLIER * latest['ATR'])
            current_price = latest['Close']
            ema_cross_down = latest['EMA_5'] < latest['EMA_13']
            stop_hit = current_price < trailing_stop

            if ema_cross_down or stop_hit:
                reason = "EMA 5/13 เปลี่ยนทิศ" if ema_cross_down else f"ATR Trailing Stop (${trailing_stop:.2f})"
                print(f"🚨 ขาย {symbol} ({reason})")
                api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
            continue

        if latest['EMA_5'] > latest['EMA_13']:
            features = ['Price_vs_SMA20', 'OBV_Slope', 'ATR', 'Regime', 'Divergence_Factor', 'Upper_Wick_Ratio', 'Yield10Y', 'VIX']
            X_live = pd.DataFrame([latest[features]])
            ai_prob = model.predict_proba(X_live)[:, 1][0]

            if ai_prob >= 0.60:
                shares = calculate_shares(account_equity, ai_prob, latest['ATR'], latest['Close'])
                if shares > 0:
                    tier = "🔥🔥🔥" if ai_prob >= 0.90 else "🔥🔥" if ai_prob >= 0.75 else "🔥"
                    print(f"{tier} [{regime_label}] สั่งซื้อ {symbol} จำนวน {shares} หุ้น (AI มั่นใจ {ai_prob*100:.2f}%)")
                    api.submit_order(symbol=symbol, qty=shares, side='buy', type='market', time_in_force='day')

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดกับ {symbol}: {e}")
        continue
```

if **name** == “**main**”:
run_bot()
