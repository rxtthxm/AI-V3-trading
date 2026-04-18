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

# ── CREDENTIALS ────────────────────────────────────────
API_KEY    = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')
BASE_URL   = 'https://paper-api.alpaca.markets'
TG_TOKEN   = os.environ.get('TELEGRAM_TOKEN')
TG_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
SHEET_ID   = '1IybSLjKmM3W1HwVGaHLlv30tSsOJkkicGfSIUq-9Enw'
SHEET_NAME = 'Trades'

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# ── MODELS ─────────────────────────────────────────────
try:
    model_bull = joblib.load('final_strategy_bull.joblib')
    model_bear = joblib.load('final_strategy_bear.joblib')
except Exception as e:
    print(f"❌ Model load failed: {e}")
    exit()

# ── CONFIG ─────────────────────────────────────────────
WATCHLIST = [
    'NVDA','META','TSLA','AMD','MU','PLTR',
    'LRCX','AMAT','CRM','UBER','GOOGL','AMZN',
    'MSFT','AVGO','AAPL','ORCL','TXN','ADI',
    'MRVL','NET','ON','MCHP','TSM'
]
ATR_MULTIPLIER = 2.1
HARD_STOP_PCT  = 0.04
ACCOUNT_RISK   = {'tier1': 0.01, 'tier2': 0.02, 'tier3': 0.03}
FEATURES       = ['Price_vs_SMA20','OBV_Slope','ATR','Regime',
                  'Divergence_Factor','Upper_Wick_Ratio','Yield10Y','VIX']

# ── TELEGRAM ───────────────────────────────────────────
def tg(msg):
    try:
        requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id': TG_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception as e:
        print(f"Telegram error: {e}")

# ── GOOGLE SHEETS ──────────────────────────────────────
def get_sheet():
    creds_dict = {
        "type": "service_account",
        "project_id": os.environ.get('GCP_PROJECT_ID'),
        "private_key_id": os.environ.get('GCP_PRIVATE_KEY_ID'),
        "private_key": os.environ.get('GCP_PRIVATE_KEY', '').replace('\\n', '\n'),
        "client_email": os.environ.get('GCP_CLIENT_EMAIL'),
        "client_id": os.environ.get('GCP_CLIENT_ID'),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.environ.get('GCP_CLIENT_CERT_URL'),
        "universe_domain": "googleapis.com"
    }
    scopes = ['https://spreadsheets.google.com/feeds',
              'https://www.googleapis.com/auth/drive']
    creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc     = gspread.authorize(creds)
    return gc.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

def log_trade(action, symbol, shares, price, reason, ai_prob, equity):
    try:
        sheet = get_sheet()
        now   = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([now, action, symbol, shares, round(price, 2),
                          reason, round(ai_prob, 4), round(equity, 2)])
    except Exception as e:
        print(f"Sheets error: {e}")

def init_sheet():
    try:
        sheet = get_sheet()
        if not sheet.get_all_values():
            sheet.append_row(['Timestamp','Action','Symbol','Shares',
                               'Price','Reason','AI_Prob','Equity'])
    except Exception as e:
        print(f"Sheet init error: {e}")

# ── BEAR DETECTION ─────────────────────────────────────
def is_bear_market(macro_raw):
    spy       = macro_raw['SPY']
    vix       = macro_raw['VIX']
    sma200    = spy.rolling(200).mean().iloc[-1]
    vix_now   = vix.iloc[-1]
    spy_20h   = spy.rolling(20).max().iloc[-1]
    spy_drop  = (spy.iloc[-1] - spy_20h) / spy_20h
    return (spy.iloc[-1] < sma200) or (vix_now > 25) or (spy_drop < -0.07)

# ── HELPERS ────────────────────────────────────────────
def is_market_closing_soon():
    tz_ny     = pytz.timezone('America/New_York')
    now_ny    = datetime.now(tz_ny)
    mkt_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    diff      = (mkt_close - now_ny).total_seconds() / 60
    print(f"🕒 NY: {now_ny.strftime('%H:%M:%S')} | {diff:.1f} min to close")
    return 0 <= diff <= 20

def get_risk_tier(prob):
    if prob >= 0.90: return ACCOUNT_RISK['tier3']
    if prob >= 0.75: return ACCOUNT_RISK['tier2']
    if prob >= 0.60: return ACCOUNT_RISK['tier1']
    return 0

def calculate_shares(equity, prob, atr, price):
    risk_pct  = get_risk_tier(prob)
    if risk_pct == 0: return 0
    stop_dist = max(ATR_MULTIPLIER * atr, price * HARD_STOP_PCT)
    shares    = int((equity * risk_pct) // stop_dist)
    max_sh    = int((equity * 0.20) // price)
    return min(shares, max_sh)

def verify_fill(symbol, order_id, side, timeout=60):
    import time
    for _ in range(timeout):
        try:
            order = api.get_order(order_id)
            if order.status == 'filled':
                fill_price = float(order.filled_avg_price)
                print(f"✅ {side} {symbol} filled @ ${fill_price:.2f}")
                return fill_price
            if order.status in ['cancelled','expired','rejected']:
                print(f"❌ Order {order.status}: {symbol}")
                return None
        except: pass
        time.sleep(1)
    print(f"⚠️ Fill timeout: {symbol}")
    return None

# ── MAIN ───────────────────────────────────────────────
def run_bot():
    run_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not is_market_closing_soon():
        tg(f"🛑 Bot skipped — not in window\n{run_time}")
        return

    try:
        init_sheet()
        account        = api.get_account()
        account_equity = float(account.equity)
        tg(f"🚀 <b>Bot Started</b>\nEquity: ${account_equity:,.2f}\n{run_time}")
    except Exception as e:
        tg(f"❌ Bot FAILED to start: {e}")
        raise

    open_orders = [o.symbol for o in api.list_orders(status='open')]
    end_date    = datetime.today().strftime('%Y-%m-%d')
    start_date  = (datetime.today() - timedelta(days=400)).strftime('%Y-%m-%d')

    macro_raw = yf.download(['^VIX','^TNX','SPY'], start=start_date,
                             end=end_date, progress=False)['Close']
    macro_raw.columns = ['SPY','VIX','Yield10Y']

    bear_mode = is_bear_market(macro_raw)
    if bear_mode:
        tg("🐻 <b>BEAR DETECTED</b> — no new longs")

    macro_data = pd.DataFrame(index=macro_raw.index)
    macro_data['VIX']        = macro_raw['VIX']
    macro_data['Yield10Y']   = macro_raw['Yield10Y']
    macro_data['SPY_Regime'] = (macro_raw['SPY'].pct_change(60) > 0).astype(int)

    latest_macro = macro_data.iloc[-1]
    is_bull      = latest_macro['SPY_Regime'] == 1
    model        = model_bull if is_bull else model_bear
    regime_label = "BULL 🐂" if is_bull else "BEAR 🐻"

    buy_log  = []
    sell_log = []

    for symbol in WATCHLIST:
        try:
            if symbol in open_orders:
                continue

            try:
                position   = api.get_position(symbol)
                qty        = float(position.qty)
                avg_entry  = float(position.avg_entry_price)
                is_holding = True
            except:
                is_holding = False

            raw = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if raw.empty: continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            c,h,l,o,v = raw['Close'],raw['High'],raw['Low'],raw['Open'],raw['Volume']
            df = pd.DataFrame(index=raw.index)
            df['Close']             = c
            df['Price_vs_SMA20']    = c - c.rolling(20).mean()
            tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
            df['ATR']               = tr.rolling(14).mean()
            obv = (np.sign(c.diff())*v).fillna(0).cumsum()
            df['OBV_Slope']         = obv.diff(5)/(obv.rolling(20).std()+1e-9)
            df['Regime']            = (c-c.rolling(200).mean())/(c.rolling(200).std()+1e-9)
            delta = c.diff()
            gain  = delta.where(delta>0,0).rolling(14).mean()
            loss  = (-delta.where(delta<0,0)).rolling(14).mean()
            rsi   = 100-(100/(1+(gain/(loss+1e-9))))
            df['Divergence_Factor'] = c.diff(5)-rsi.diff(5)
            body_max = np.maximum(o,c)
            df['Upper_Wick_Ratio']  = (h-body_max)/(h-l+1e-9)
            df['EMA_5']             = c.ewm(span=5,adjust=False).mean()
            df['EMA_13']            = c.ewm(span=13,adjust=False).mean()
            df = df.join(macro_data, how='left').ffill()
            df.dropna(inplace=True)
            if df.empty: continue

            latest = df.iloc[-1]

            # EXIT
            if is_holding:
                hard_stop  = avg_entry * (1 - HARD_STOP_PCT)
                trail_stop = avg_entry - (ATR_MULTIPLIER * latest['ATR'])
                eff_stop   = max(hard_stop, trail_stop)
                ema_down   = latest['EMA_5'] < latest['EMA_13']
                stop_hit   = latest['Close'] < eff_stop

                if ema_down or stop_hit:
                    reason = "EMA cross" if ema_down else f"Stop ${eff_stop:.2f}"
                    order  = api.submit_order(symbol=symbol, qty=qty, side='sell',
                                              type='market', time_in_force='day')
                    fill   = verify_fill(symbol, order.id, 'SELL')
                    fill_p = fill if fill else latest['Close']
                    sell_log.append(f"🚨 SELL {symbol} x{int(qty)} @ ${fill_p:.2f} | {reason}")
                    log_trade('SELL', symbol, qty, fill_p, reason, 0, account_equity)
                continue

            # ENTRY
            if bear_mode: continue
            if latest['EMA_5'] <= latest['EMA_13']: continue

            X_live  = pd.DataFrame([latest[FEATURES]])
            ai_prob = model.predict_proba(X_live)[:,1][0]
            if ai_prob < 0.60: continue

            shares = calculate_shares(account_equity, ai_prob, latest['ATR'], latest['Close'])
            if shares <= 0: continue

            limit_price = round(float(latest['Close']) * 1.001, 2)
            order = api.submit_order(symbol=symbol, qty=shares, side='buy',
                                     type='limit', time_in_force='day',
                                     limit_price=limit_price)
            fill = verify_fill(symbol, order.id, 'BUY', timeout=60)
            if fill:
                tier   = "🔥🔥🔥" if ai_prob>=0.90 else "🔥🔥" if ai_prob>=0.75 else "🔥"
                reason = f"AI {ai_prob*100:.1f}% | {regime_label}"
                buy_log.append(f"{tier} BUY {symbol} x{shares} @ ${fill:.2f} | {reason}")
                log_trade('BUY', symbol, shares, fill, reason, ai_prob, account_equity)
            else:
                try: api.cancel_order(order.id)
                except: pass

        except Exception as e:
            print(f"⚠️ {symbol}: {e}")
            continue

    # SUMMARY
    lines = [f"📊 <b>Run Complete</b> | {run_time}",
             f"💰 Equity: ${account_equity:,.2f} | {regime_label}"]
    if buy_log:  lines += ["\n<b>BUYS:</b>"]  + buy_log
    if sell_log: lines += ["\n<b>SELLS:</b>"] + sell_log
    if not buy_log and not sell_log:
        lines.append("💤 No trades today")
    tg("\n".join(lines))

if __name__ == "__main__":
    try:
        run_bot()
    except Exception as e:
        tg(f"❌ Bot CRASHED: {e}")
        raise
