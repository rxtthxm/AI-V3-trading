import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import json
import pytz
import requests
import gspread
from google.oauth2.service_account import Credentials
import warnings
warnings.filterwarnings('ignore')

API_KEY    = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')
BASE_URL   = 'https://paper-api.alpaca.markets'
TG_TOKEN   = os.environ.get('TELEGRAM_TOKEN')
TG_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
SHEET_ID   = '1IybSLjKmM3W1HwVGaHLlv30tSsOJkkicGfSIUq-9Enw'
SHEET_NAME = 'Trades'
SIGNAL_FILE = 'signals.json'

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

try:
    model_bull = joblib.load('final_strategy_bull.joblib')
    model_bear = joblib.load('final_strategy_bear.joblib')
except Exception as e:
    print(f"❌ Model load failed: {e}")
    exit()

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

def tg(msg):
    try:
        requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id': TG_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception as e:
        print(f"Telegram error: {e}")

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
        sheet.append_row([now, action, symbol, shares, round(price,2),
                          reason, round(ai_prob,4), round(equity,2)])
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

def is_bear_market(macro_raw):
    spy      = macro_raw['SPY']
    vix      = macro_raw['VIX']
    sma200   = spy.rolling(200).mean().iloc[-1]
    vix_now  = vix.iloc[-1]
    spy_20h  = spy.rolling(20).max().iloc[-1]
    spy_drop = (spy.iloc[-1] - spy_20h) / spy_20h
    return (spy.iloc[-1] < sma200) or (vix_now > 25) or (spy_drop < -0.07)

def get_market_phase():
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    h, m = now_ny.hour, now_ny.minute
    print(f"🗽 เวลาปัจจุบันของ NY: {now_ny.strftime('%H:%M:%S')}")
    
    # เช็คว่าเป็นการกดปุ่มรันด้วยมือ (Manual) บน GitHub Actions หรือไม่
    # ถ้ากดด้วยมือ เราจะบังคับให้รันเป็นโหมด Scanner เพื่อให้ทดสอบได้ทันที
    if os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch':
        print("🛠️ ตรวจพบการกดปุ่มรันด้วยมือ (Manual Test) -> บังคับเข้าโหมด Scanner")
        return 'scanner' # หรือเปลี่ยนเป็น 'executor' ถ้าอยากทดสอบฝั่งซื้อขาย
        
    # โหมด Scanner: ช่วงก่อนปิดตลาด (ขยายเวลาเผื่อ Server ดีเลย์เป็น 15:50 ถึง 16:15)
    if (h == 15 and m >= 50) or (h == 16 and m <= 15): 
        return 'scanner'
        
    # โหมด Executor: ช่วงเปิดตลาด (ขยายเวลาเผื่อ Server ดีเลย์เป็น 09:30 ถึง 09:55)
    if h == 9 and 30 <= m <= 55: 
        return 'executor'
        
    return 'none'


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

def build_features(symbol, start_date, end_date, macro_data):
    raw = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if raw.empty: return None
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
    return df

def verify_fill(symbol, order_id, side, timeout=60):
    import time
    for _ in range(timeout):
        try:
            order = api.get_order(order_id)
            if order.status == 'filled':
                fill_price = float(order.filled_avg_price)
                print(f"✅ {side} {symbol} @ ${fill_price:.2f}")
                return fill_price
            if order.status in ['cancelled','expired','rejected']:
                print(f"❌ {symbol} {order.status}")
                return None
        except: pass
        time.sleep(1)
    return None

def run_scanner():
    print("📡 SCANNER MODE")
    end_date   = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=400)).strftime('%Y-%m-%d')

    macro_raw = yf.download(['^VIX','^TNX','SPY'], start=start_date,
                             end=end_date, progress=False)['Close']
    macro_raw.columns = ['SPY','VIX','Yield10Y']

    bear_mode  = is_bear_market(macro_raw)
    macro_data = pd.DataFrame(index=macro_raw.index)
    macro_data['VIX']        = macro_raw['VIX']
    macro_data['Yield10Y']   = macro_raw['Yield10Y']
    macro_data['SPY_Regime'] = (macro_raw['SPY'].pct_change(60) > 0).astype(int)

    latest_macro = macro_data.iloc[-1]
    is_bull      = latest_macro['SPY_Regime'] == 1
    regime_label = "BULL 🐂" if is_bull else "BEAR 🐻"
    model        = model_bull if is_bull else model_bear

    account        = api.get_account()
    account_equity = float(account.equity)

    exit_signals = []
    try:
        for pos in api.list_positions():
            symbol    = pos.symbol
            qty       = float(pos.qty)
            avg_entry = float(pos.avg_entry_price)
            df = build_features(symbol, start_date, end_date, macro_data)
            if df is None or df.empty: continue
            latest     = df.iloc[-1]
            hard_stop  = avg_entry * (1 - HARD_STOP_PCT)
            trail_stop = avg_entry - (ATR_MULTIPLIER * latest['ATR'])
            eff_stop   = max(hard_stop, trail_stop)
            ema_down   = latest['EMA_5'] < latest['EMA_13']
            stop_hit   = latest['Close'] < eff_stop
            if ema_down or stop_hit or bear_mode:
                reason = "BEAR" if bear_mode else ("EMA cross" if ema_down else f"Stop ${eff_stop:.2f}")
                exit_signals.append({'symbol': symbol, 'qty': qty, 'reason': reason})
    except Exception as e:
        print(f"Position check error: {e}")

    buy_signals = []
    if not bear_mode:
        for symbol in WATCHLIST:
            try:
                df = build_features(symbol, start_date, end_date, macro_data)
                if df is None or df.empty: continue
                latest = df.iloc[-1]
                if latest['EMA_5'] <= latest['EMA_13']: continue
                X_live  = pd.DataFrame([latest[FEATURES]])
                ai_prob = model.predict_proba(X_live)[:,1][0]
                if ai_prob < 0.60: continue
                shares = calculate_shares(account_equity, ai_prob,
                                          latest['ATR'], latest['Close'])
                if shares <= 0: continue
                buy_signals.append({
                    'symbol': symbol, 'shares': shares,
                    'ai_prob': round(float(ai_prob), 4),
                    'close': round(float(latest['Close']), 2)
                })
            except Exception as e:
                print(f"{symbol}: {e}")

    signals = {
        'date': end_date, 'bear_mode': bear_mode,
        'regime': regime_label, 'equity': account_equity,
        'buy': buy_signals, 'exit': exit_signals
    }
    with open(SIGNAL_FILE, 'w') as f:
        json.dump(signals, f)

    lines = [f"📡 <b>Scanner Complete</b> | {end_date}",
             f"💰 Equity: ${account_equity:,.2f} | {regime_label}"]
    if bear_mode:
        lines.append("🐻 <b>BEAR — exits queued for open</b>")
    if buy_signals:
        lines.append(f"\n<b>Buy signals ({len(buy_signals)}):</b>")
        for s in buy_signals:
            tier = "🔥🔥🔥" if s['ai_prob']>=0.90 else "🔥🔥" if s['ai_prob']>=0.75 else "🔥"
            lines.append(f"{tier} {s['symbol']} x{s['shares']} | AI {s['ai_prob']*100:.1f}%")
    if exit_signals:
        lines.append(f"\n<b>Exit signals ({len(exit_signals)}):</b>")
        for e in exit_signals:
            lines.append(f"🚨 {e['symbol']} | {e['reason']}")
    if not buy_signals and not exit_signals:
        lines.append("💤 No signals today")
    lines.append("\n⏰ Executes at tomorrow 9:35am ET")
    tg("\n".join(lines))

def run_executor():
    print("⚡ EXECUTOR MODE")
    init_sheet()

    if not os.path.exists(SIGNAL_FILE):
        tg("⚠️ No signal file — scanner may have failed")
        return

    with open(SIGNAL_FILE, 'r') as f:
        signals = json.load(f)

    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    today     = datetime.today().strftime('%Y-%m-%d')
    if signals.get('date') not in [yesterday, today]:
        tg(f"⚠️ Stale signals ({signals.get('date')}) — skipping")
        return

    account        = api.get_account()
    account_equity = float(account.equity)
    open_orders    = [o.symbol for o in api.list_orders(status='open')]
    buy_log        = []
    sell_log       = []

    for sig in signals.get('exit', []):
        symbol = sig['symbol']
        if symbol in open_orders: continue
        try:
            order  = api.submit_order(symbol=symbol, qty=sig['qty'], side='sell',
                                      type='market', time_in_force='day')
            fill   = verify_fill(symbol, order.id, 'SELL')
            fill_p = fill if fill else 0
            sell_log.append(f"🚨 SELL {symbol} x{int(sig['qty'])} @ ${fill_p:.2f} | {sig['reason']}")
            log_trade('SELL', symbol, sig['qty'], fill_p, sig['reason'], 0, account_equity)
        except Exception as e:
            print(f"Exit error {symbol}: {e}")

    if not signals.get('bear_mode', False):
        for sig in signals.get('buy', []):
            symbol = sig['symbol']
            if symbol in open_orders: continue
            try:
                api.get_position(symbol)
                continue
            except: pass
            try:
                limit_price = round(sig['close'] * 1.002, 2)
                order = api.submit_order(symbol=symbol, qty=sig['shares'], side='buy',
                                         type='limit', time_in_force='day',
                                         limit_price=limit_price)
                fill = verify_fill(symbol, order.id, 'BUY', timeout=120)
                if fill:
                    tier   = "🔥🔥🔥" if sig['ai_prob']>=0.90 else "🔥🔥" if sig['ai_prob']>=0.75 else "🔥"
                    reason = f"AI {sig['ai_prob']*100:.1f}% | {signals['regime']}"
                    buy_log.append(f"{tier} BUY {symbol} x{sig['shares']} @ ${fill:.2f} | {reason}")
                    log_trade('BUY', symbol, sig['shares'], fill, reason,
                              sig['ai_prob'], account_equity)
                else:
                    try: api.cancel_order(order.id)
                    except: pass
            except Exception as e:
                print(f"Entry error {symbol}: {e}")

    try: os.remove(SIGNAL_FILE)
    except: pass

    lines = [f"⚡ <b>Executor Complete</b>",
             f"💰 Equity: ${account_equity:,.2f}"]
    if buy_log:  lines += ["\n<b>BUYS:</b>"]  + buy_log
    if sell_log: lines += ["\n<b>SELLS:</b>"] + sell_log
    if not buy_log and not sell_log:
        lines.append("💤 No orders executed")
    tg("\n".join(lines))

if __name__ == "__main__":
    try:
        phase = get_market_phase()
        if phase == 'scanner':
            run_scanner()
        elif phase == 'executor':
            run_executor()
        else:
            print("⏰ Outside trading windows")
            tg("⏰ Bot ran outside window — check cron")
    except Exception as e:
        tg(f"❌ Bot CRASHED: {e}")
        raise
