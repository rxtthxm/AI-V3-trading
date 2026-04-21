import alpaca_trade_api as tradeapi
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import pytz
import requests
import warnings
warnings.filterwarnings('ignore')

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

FEATURES = [
    'Price_vs_SMA20', 'OBV_Slope', 'ATR', 'Regime_SMA200',
    'Divergence_Factor', 'Upper_Wick_Ratio', 'Yield10Y', 'VIX',
    'Vol_Spike', 'Rel_Strength', 'SPY_3Mo_Ret', 'Market_Stress',
    'Fed_Slope', 'Mom_20_60', 'Vol_Trend_20d', 'VIX_ROC',
    'SPY_Mom_Cross', 'Price_Accel'
]

def check_trading_window():
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    h, m = now_ny.hour, now_ny.minute
    print(f"🗽 NY Time: {now_ny.strftime('%H:%M:%S')}")
    if os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch':
        print("🛠️ Manual run: skipping time check")
        return True
    return (h == 15 and m >= 40) or (h == 16 and m <= 15)

if not check_trading_window():
    print("⏰ Outside trading window")
    exit()

def send_telegram(message):
    if TG_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': TG_CHAT_ID, 'text': message, 'parse_mode': 'HTML'})

def get_latest_data(symbol, vix_data, spy_data, tnx_data):
    df = yf.download(symbol, period='2y', interval='1d', progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.join(vix_data, how='left').join(spy_data, how='left').join(tnx_data, how='left').ffill()

    df['VIX_ROC'] = df['VIX'].pct_change(5)
    df['SPY_Mom_Cross'] = df['SPY'].pct_change(5) - df['SPY'].pct_change(20)
    df['SPY_3Mo_Ret'] = df['SPY'].pct_change(63)
    df['Fed_Slope'] = df['Yield10Y'].diff(20)
    df['Market_Stress'] = df['VIX'].rolling(20).mean()
    df['Regime_SMA200'] = (df['Close'] > df['Close'].rolling(200).mean()).astype(int)
    df['Vol_Spike'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Vol_Trend_20d'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
    df['Mom_20_60'] = df['Close'].pct_change(20) - df['Close'].pct_change(63)
    df['Rel_Strength'] = df['Close'].pct_change(20) - df['SPY'].pct_change(20)
    df['Price_vs_SMA20'] = df['Close'] - df['Close'].rolling(20).mean()
    df['OBV_Slope'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum().diff(5)
    df['Price_Accel'] = df['Close'].pct_change(5) - df['Close'].pct_change(20)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df['Divergence_Factor'] = df['Close'].pct_change(14) - rsi.pct_change(14)
    df['Upper_Wick_Ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'] + 1e-9)
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_13'] = df['Close'].ewm(span=13, adjust=False).mean()
    df.dropna(inplace=True)
    return df if not df.empty else None

def calculate_shares(equity, prob, atr, price):
    risk_pct = 0.03 if prob >= 0.90 else 0.02 if prob >= 0.75 else 0.01
    risk_amount = equity * risk_pct
    stop_distance = 2.1 * atr
    if stop_distance <= 0: return 0
    shares = int(risk_amount / stop_distance)
    max_shares = int((equity * 0.20) / price)
    return min(shares, max_shares)

account = api.get_account()
account_equity = float(account.equity)
print(f"💰 Equity: ${account_equity:,.2f}")

vix_data = yf.download('^VIX', period='2y', progress=False)[['Close']].rename(columns={'Close': 'VIX'})
spy_data = yf.download('SPY', period='2y', progress=False)[['Close']].rename(columns={'Close': 'SPY'})
tnx_data = yf.download('^TNX', period='2y', progress=False)[['Close']].rename(columns={'Close': 'Yield10Y'})
for d in [vix_data, spy_data, tnx_data]:
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)

is_bull_market = spy_data['SPY'].pct_change(63).iloc[-1] > 0
regime_label = "BULL 🐂" if is_bull_market else "BEAR 🐻"
model = model_bull if is_bull_market else model_bear
print(f"📊 Regime: {regime_label}")

buy_log = []
sell_log = []

positions = api.list_positions()
open_symbols = [p.symbol for p in positions]
open_orders = [o.symbol for o in api.list_orders(status='open')]

for position in positions:
    symbol = position.symbol
    qty = position.qty
    current_price = float(position.current_price)
    avg_entry = float(position.avg_entry_price)
    profit_pct = (current_price - avg_entry) / avg_entry

    df = get_latest_data(symbol, vix_data, spy_data, tnx_data)
    if df is None: continue
    latest = df.iloc[-1]

    if profit_pct <= -0.04:
        print(f"🛑 {symbol}: Hard Stop ({profit_pct*100:.2f}%)")
        api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
        sell_log.append(f"🔴 SELL {symbol}: Hard Stop -4%")
        continue

    buy_orders = api.list_orders(status='filled', symbols=[symbol], limit=20)
    buy_orders = [o for o in buy_orders if o.side == 'buy']
    if buy_orders:
        last_buy_date = buy_orders[0].filled_at.strftime('%Y-%m-%d')
        try:
            highest_high = df.loc[last_buy_date:]['High'].max().item()
        except:
            highest_high = current_price
        trailing_stop_price = highest_high - (2.1 * latest['ATR'].item())
        if current_price < trailing_stop_price:
            print(f"📉 {symbol}: ATR Trailing Stop at ${current_price:.2f}")
            api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
            sell_log.append(f"🔴 SELL {symbol}: ATR Trailing Stop")

for symbol in WATCHLIST:
    if symbol in open_symbols or symbol in open_orders:
        continue

    df = get_latest_data(symbol, vix_data, spy_data, tnx_data)
    if df is None: continue
    latest = df.iloc[-1]

    if latest['EMA_5'] <= latest['EMA_13']:
        continue

    try:
        X_live = pd.DataFrame([latest[FEATURES]])
        ai_prob = model.predict_proba(X_live)[:, 1][0]
    except KeyError as e:
        print(f"⚠️ {symbol}: missing feature {e}")
        continue

    if ai_prob >= 0.60:
        recent_orders = api.list_orders(status='filled', symbols=[symbol], limit=10)
        sell_orders = [o for o in recent_orders if o.side == 'sell']
        if sell_orders:
            last_sell_date = sell_orders[0].filled_at
            now_utc = datetime.now(pytz.utc)
            if (now_utc - last_sell_date).days <= 1:
                print(f"⏳ {symbol}: Cooldown active")
                continue

        shares = calculate_shares(account_equity, ai_prob, latest['ATR'].item(), latest['Close'].item())
        if shares > 0:
            tier = "🔥🔥🔥" if ai_prob >= 0.90 else "🔥🔥" if ai_prob >= 0.75 else "🔥"
            try:
                api.submit_order(symbol=symbol, qty=shares, side='buy', type='market', time_in_force='day')
                print(f"{tier} BUY {symbol} x{shares} (AI {ai_prob*100:.1f}%)")
                buy_log.append(f"🟢 BUY {symbol} x{shares} ({tier} AI: {ai_prob*100:.1f}%)")
            except Exception as e:
                print(f"❌ {symbol}: order failed — {e}")

msg = f"🤖 <b>AI Trading V4.0</b>\n📊 Regime: {regime_label}\n💰 Equity: ${account_equity:,.2f}\n\n"
if buy_log:
    msg += "<b>🛒 รายการซื้อ:</b>\n" + "\n".join(buy_log) + "\n\n"
if sell_log:
    msg += "<b>🚪 รายการขาย:</b>\n" + "\n".join(sell_log) + "\n"
if not buy_log and not sell_log:
    msg += "<i>ไม่มีแอคชั่นในวันนี้</i>"

send_telegram(msg)
print("✅ Done")
