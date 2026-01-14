import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
DB_NAME = "momentum_backtest.db"
CSV_NAME = "backtest_results.csv"
START_DATE_DEFAULT = "2017-01-01" 
INITIAL_CAPITAL = 1000000
MAX_STOCKS = 15
TRAILING_SL_PCT = 0.15
BENCHMARK = "^NSEI"

# ==========================================
# LOAD UNIVERSE
# ==========================================
def load_universe():
    universe = []
    if not os.path.exists("universe.txt"):
        print("❌ Error: 'universe.txt' not found.")
        return []
    try:
        with open("universe.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    universe.append(line)
        print(f"✅ Loaded {len(universe)} stocks from universe.txt")
        return universe
    except Exception as e:
        print(f"❌ Error reading universe.txt: {e}")
        return []

TICKERS = load_universe()
if not TICKERS:
    print("⚠️ Warning: Universe is empty. Exiting.")
    exit()

# ==========================================
# 2. DATABASE MANAGEMENT
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
                 id INTEGER PRIMARY KEY, ticker TEXT, entry_date TEXT, entry_price REAL, qty INTEGER,
                 exit_date TEXT, exit_price REAL, pnl_abs REAL, pnl_pct REAL,
                 exit_reason TEXT, holding_days INTEGER)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS state_meta (
                 id INTEGER PRIMARY KEY, last_run_date TEXT, current_cash REAL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS state_holdings (
                 ticker TEXT PRIMARY KEY, qty INTEGER, entry_price REAL, 
                 entry_date TEXT, high_price REAL)''')
    conn.commit()
    return conn

def load_state(conn):
    c = conn.cursor()
    c.execute("SELECT last_run_date, current_cash FROM state_meta WHERE id=1")
    meta = c.fetchone()
    
    if meta:
        last_date = meta[0]
        cash = meta[1]
        c.execute("SELECT * FROM state_holdings")
        rows = c.fetchall()
        holdings = {}
        for r in rows:
            holdings[r[0]] = {
                'qty': r[1], 'entry_price': r[2], 
                'entry_date': datetime.strptime(r[3], '%Y-%m-%d'), 
                'high_price': r[4]
            }
        return last_date, cash, holdings
    else:
        return None, INITIAL_CAPITAL, {}

def save_state(conn, date_str, cash, holdings):
    c = conn.cursor()
    c.execute("DELETE FROM state_meta")
    c.execute("INSERT INTO state_meta (id, last_run_date, current_cash) VALUES (1, ?, ?)", (date_str, cash))
    c.execute("DELETE FROM state_holdings")
    for t, h in holdings.items():
        c.execute("INSERT INTO state_holdings VALUES (?, ?, ?, ?, ?)",
                  (t, h['qty'], h['entry_price'], h['entry_date'].strftime('%Y-%m-%d'), h['high_price']))
    conn.commit()

# ==========================================
# 3. DATE LOGIC (STRICT MONTHLY CUTOFF)
# ==========================================
conn = init_db()
last_run_date_str, current_cash, current_holdings = load_state(conn)

# Calculate Cutoff: Last Day of Previous Month
today = datetime.today()
first_of_current_month = today.replace(day=1)
cutoff_date = first_of_current_month - timedelta(days=1)
cutoff_str = cutoff_date.strftime('%Y-%m-%d')

print(f"🛑 STRICT CUTOFF DATE: {cutoff_str} (Last Completed Month End)")

if last_run_date_str:
    print(f"🔄 Resuming from Checkpoint: {last_run_date_str}")
    # If already up to date, stop early
    if datetime.strptime(last_run_date_str, '%Y-%m-%d').date() >= cutoff_date.date():
        print("✅ Database is already up to date with the last completed month.")
        conn.close()
        exit()
        
    fetch_start_date = (datetime.strptime(last_run_date_str, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
    sim_start_date = datetime.strptime(last_run_date_str, '%Y-%m-%d')
else:
    print("✨ No checkpoint found. Starting fresh from 2017.")
    fetch_start_date = (datetime.strptime(START_DATE_DEFAULT, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
    sim_start_date = datetime.strptime(START_DATE_DEFAULT, '%Y-%m-%d')

# ==========================================
# 4. FETCH DATA (LIMITED TO CUTOFF)
# ==========================================
print(f"📥 Fetching Market Data (Up to {cutoff_str})...")
try:
    # yfinance 'end' is exclusive, so we add 1 day to include the cutoff date
    download_end = cutoff_date + timedelta(days=1)
    data = yf.download(TICKERS + [BENCHMARK], start=fetch_start_date, end=download_end, progress=False, threads=True)['Close']
    data = data.dropna(axis=1, how='all')
except Exception as e:
    print(f"❌ Error fetching data: {e}")
    exit()

# Pre-Calculate Indicators
stock_data = data.drop(columns=[BENCHMARK], errors='ignore').ffill()

print("🧮 Calculating Indicators...")
mom_6m = stock_data.pct_change(126)
mom_3m = stock_data.pct_change(63)
mom_1m = stock_data.pct_change(21)
scores = (mom_6m * 0.4 + mom_3m * 0.4 + mom_1m * 0.2).fillna(-999)

sma_100 = stock_data.rolling(window=100).mean()
trend_filter = (stock_data > sma_100)

# ==========================================
# 5. TRADING SIMULATION LOOP
# ==========================================
portfolio = {'cash': current_cash, 'holdings': current_holdings}
trade_log = []
dates = stock_data.index

print(f"▶️ Processing...")

processed_any = False
final_date = None
warmup_needed = (last_run_date_str is None)

for i, date in enumerate(dates):
    # Skip already processed
    if date <= sim_start_date: continue
    
    # Redundant safety: Break if past cutoff
    if date.date() > cutoff_date.date(): break

    # Warmup check
    if warmup_needed and i < 150: continue

    processed_any = True
    final_date = date
    date_str = date.strftime('%Y-%m-%d')
    current_prices = stock_data.iloc[i]

    # --- A. DAILY EXITS (SL) ---
    stocks_to_sell = []
    for ticker, pos in portfolio['holdings'].items():
        price = current_prices.get(ticker)
        if pd.isna(price): continue

        if price > pos['high_price']:
            portfolio['holdings'][ticker]['high_price'] = price
        
        if price < pos['high_price'] * (1 - TRAILING_SL_PCT):
            stocks_to_sell.append((ticker, "STOP_LOSS"))

    for ticker, reason in stocks_to_sell:
        price = current_prices[ticker]
        pos = portfolio['holdings'][ticker]
        revenue = price * pos['qty']
        pnl = revenue - (pos['entry_price'] * pos['qty'])
        pnl_pct = (price - pos['entry_price']) / pos['entry_price']
        days_held = (date - pos['entry_date']).days
        
        trade_log.append((
            ticker, pos['entry_date'].strftime('%Y-%m-%d'), pos['entry_price'], pos['qty'],
            date_str, price, pnl, pnl_pct * 100, reason, days_held
        ))
        portfolio['cash'] += revenue
        del portfolio['holdings'][ticker]

    # --- B. MONTHLY REBALANCE (Strictly 1st Trading Day) ---
    # Logic: If month of 'date' != month of 'prev_date', it is the 1st trading day
    is_new_month = (date.month != dates[i-1].month)
    
    if is_new_month:
        current_score = scores.iloc[i]
        current_trend = trend_filter.iloc[i]
        current_mom_1m = mom_1m.iloc[i]
        
        valid_mask = (current_trend) & (current_mom_1m > -0.05)
        candidates = current_score[valid_mask].sort_values(ascending=False).index.tolist()
        
        # Sell Weak
        current_keys = list(portfolio['holdings'].keys())
        for ticker in current_keys:
            if ticker not in candidates[:25]:
                price = current_prices.get(ticker)
                if not pd.isna(price):
                    pos = portfolio['holdings'][ticker]
                    revenue = price * pos['qty']
                    pnl = revenue - (pos['entry_price'] * pos['qty'])
                    pnl_pct = (price - pos['entry_price']) / pos['entry_price']
                    days_held = (date - pos['entry_date']).days
                    
                    trade_log.append((
                        ticker, pos['entry_date'].strftime('%Y-%m-%d'), pos['entry_price'], pos['qty'],
                        date_str, price, pnl, pnl_pct * 100, "REBALANCE", days_held
                    ))
                    portfolio['cash'] += revenue
                    del portfolio['holdings'][ticker]

        # Buy New
        slots = MAX_STOCKS - len(portfolio['holdings'])
        if slots > 0 and portfolio['cash'] > 0:
            alloc = portfolio['cash'] / slots
            for ticker in candidates:
                if slots == 0: break
                if ticker in portfolio['holdings']: continue
                
                price = current_prices.get(ticker)
                if pd.isna(price) or price <= 0: continue
                
                qty = int(alloc / price)
                if qty > 0:
                    portfolio['cash'] -= qty * price
                    portfolio['holdings'][ticker] = {
                        'qty': qty, 'entry_price': price, 
                        'entry_date': date, 'high_price': price
                    }
                    slots -= 1

# ==========================================
# 6. SAVE & EXPORT
# ==========================================
if processed_any and final_date:
    print(f"\n💾 Appending {len(trade_log)} new trades to Database...")
    c = conn.cursor()
    c.executemany('''
        INSERT INTO trades (ticker, entry_date, entry_price, qty, exit_date, exit_price, pnl_abs, pnl_pct, exit_reason, holding_days)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', trade_log)
    save_state(conn, final_date.strftime('%Y-%m-%d'), portfolio['cash'], portfolio['holdings'])
    print(f"✅ Database updated up to {final_date.strftime('%Y-%m-%d')}.")
    
    current_prices_latest = stock_data.iloc[-1]
    holdings_val = sum(h['qty'] * current_prices_latest.get(t, 0) for t, h in portfolio['holdings'].items())
    curr_val = portfolio['cash'] + holdings_val
    print(f"💰 Portfolio Value (as of {final_date.date()}): ₹{curr_val:,.2f}")
else:
    print("\n⚠️ No new complete months found. Database unchanged.")

try:
    df_all = pd.read_sql("SELECT * FROM trades ORDER BY exit_date DESC", conn)
    df_all.to_csv(CSV_NAME, index=False)
except: pass

conn.close()
