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
    print(f"Model load failed: {e}")
    exit()

WATCHLIST = [
    'NVDA', 'META', 'TSLA', 'AMD', 'MU', 'PLTR', 
    'LRCX', 'AMAT', 'CRM', 'UBER', 
    'GOOGL', 'AMZN', 'MSFT', 'AVGO', 'AAPL', 'ORCL', 
    'TXN', 'ADI', 'MRVL', 'NET', 'ON', 'MCHP', 'TSM', 
    'ASML', 'QCOM', 'ARM', 'PANW', 'SNOW', 'DDOG', 'NOW'
]

def send_telegram(message):
    if TG_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': TG_CHAT_ID, 'text': message, 'parse_mode': 'HTML'})

def get_latest_data(symbol):
    df = yf.download(symbol, period='1y', interval='1d', progress=False)
    if df.empty: return None
    
    df['ATR'] = df['High'].sub(df['Low']).rolling(14).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_13'] = df['Close'].ewm(span=13, adjust=False).mean()
    
    return df

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
print(f"Equity: ${account_equity:,.2f}")

buy_log = []
sell_log = []

spy = yf.download('SPY', period='3mo', progress=False)
spy['EMA_20'] = spy['Close'].ewm(span=20).mean()
spy['EMA_50'] = spy['Close'].ewm(span=50).mean()
is_bull_market = spy['EMA_20'].iloc[-1].item() > spy['EMA_50'].iloc[-1].item()
regime_label = "BULL" if is_bull_market else "BEAR"
model = model_bull if is_bull_market else model_bear

print(f"Market Regime: {regime_label}")

positions = api.list_positions()
open_symbols = [p.symbol for p in positions]
open_orders = api.list_orders(status='open')
open_sell_symbols = [o.symbol for o in open_orders if o.side == 'sell']

for position in positions:
    symbol = position.symbol
    qty = position.qty
    
    if symbol not in open_sell_symbols:
        df = get_latest_data(symbol)
        if df is None: continue
        latest = df.iloc[-1]
        
        trail_amount = round(2.1 * latest['ATR'].item(), 2)
        
        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='trailing_stop',
                time_in_force='gtc',
                trail_price=trail_amount
            )
            print(f"Trailing Stop {symbol}: ${trail_amount}")
            sell_log.append(f"🛡️ TRAILING STOP {symbol}: ${trail_amount}")
        except Exception as e:
            print(f"Trailing Stop error {symbol}: {e}")

for symbol in WATCHLIST:
    if symbol in open_symbols:
        continue
        
    df = get_latest_data(symbol)
    if df is None: continue
    latest = df.iloc[-1]
    
    try:
        features = ['Price_vs_SMA20', 'OBV_Slope', 'ATR', 'Regime', 'Divergence_Factor', 'Upper_Wick_Ratio', 'Yield10Y', 'VIX']
        X_live = pd.DataFrame([latest[features]])
        ai_prob = model.predict_proba(X_live)[:, 1][0]
    except KeyError as e:
        print(f"Skip {symbol}: {e}")
        continue

    if ai_prob >= 0.60:
        recent_orders = api.list_orders(status='filled', symbols=[symbol], limit=10)
        sell_orders = [o for o in recent_orders if o.side == 'sell']
        
        cooldown_active = False
        if sell_orders:
            last_sell_date = sell_orders[0].filled_at
            now_utc = datetime.now(pytz.utc)
            if (now_utc - last_sell_date).days <= 1:
                print(f"Cooldown {symbol}")
                cooldown_active = True
                
        if not cooldown_active:
            shares = calculate_shares(account_equity, ai_prob, latest['ATR'].item(), latest['Close'].item())
            if shares > 0:
                tier = "🔥🔥🔥" if ai_prob >= 0.90 else "🔥🔥" if ai_prob >= 0.75 else "🔥"
                target_price = round(latest['Close'].item() * 1.002, 2)
                
                try:
                    api.submit_order(
                        symbol=symbol, 
                        qty=shares, 
                        side='buy', 
                        type='limit', 
                        time_in_force='day',
                        limit_price=target_price
                    )
                    print(f"Queued Buy {symbol} x{shares} @ ${target_price}")
                    buy_log.append(f"🟢 LIMIT BUY {symbol} x{shares} @ ${target_price} ({tier} AI: {ai_prob*100:.1f}%)")
                except Exception as e:
                    print(f"Buy error {symbol}: {e}")

msg = f"🤖 <b>AI Trading V5.0 (Queued EOD)</b>\n"
msg += f"📊 Regime: {regime_label}\n"
msg += f"💰 Equity: ${account_equity:,.2f}\n\n"

if buy_log:
    msg += "<b>🛒 รายการตั้งซื้อ (Limit):</b>\n" + "\n".join(buy_log) + "\n\n"
if sell_log:
    msg += "<b>🛡️ รายการตั้งขาย (Trailing):</b>\n" + "\n".join(sell_log) + "\n"

if not buy_log and not sell_log:
    msg += "<i>ไม่มีแอคชั่นในวันนี้</i>"

send_telegram(msg)
print("Done")
