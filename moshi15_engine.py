import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import os
import warnings
import time  # Added for retry logic

# --- 1. CLEANUP: Suppress Warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None 

# ==========================================
# 2. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "moshi15_backtest-live.db")
CSV_NAME = os.path.join(BASE_DIR, "moshi15_backtest-live.csv")

START_DATE_DEFAULT = "2017-01-01" 
INITIAL_CAPITAL = 1000000 
MAX_STOCKS = 15
TRAILING_SL_PCT = 0.15
BENCHMARK = "^NSEI"

# ==========================================
# 3. DATABASE & STATE
# ==========================================
def load_universe():
    universe = []
    txt_path = os.path.join(BASE_DIR, "universe.txt")
    if not os.path.exists(txt_path):
        print("❌ Error: 'universe.txt' not found.")
        return []
    try:
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    universe.append(line)
        return universe
    except: return []

TICKERS = load_universe()

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
                  slno INTEGER PRIMARY KEY AUTOINCREMENT,
                  trade_month TEXT, ticker TEXT, original_entry_date TEXT,
                  months_in_momentum INTEGER, latest_entry_date TEXT,
                  latest_entry_price REAL, qty INTEGER, exit_date TEXT,
                  exit_price REAL, pnl_abs REAL, pnl_pct REAL,
                  exit_reason TEXT, holding_days INTEGER)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS state_meta (
                  id INTEGER PRIMARY KEY, last_run_date TEXT, current_cash REAL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS state_holdings (
                  ticker TEXT PRIMARY KEY, qty INTEGER, entry_price REAL, 
                  entry_date TEXT, high_price REAL, orig_entry_date TEXT, months_active INTEGER)''')
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
        holdings = {r[0]: {'qty': r[1], 'entry_price': r[2], 'entry_date': datetime.strptime(r[3], '%Y-%m-%d'), 
                            'high_price': r[4], 'orig_entry_date': r[5], 'months_active': r[6]} for r in rows}
        return last_date, cash, holdings
    return None, INITIAL_CAPITAL, {}

def save_state(conn, date_str, cash, holdings):
    c = conn.cursor()
    c.execute("DELETE FROM state_meta")
    c.execute("INSERT INTO state_meta (id, last_run_date, current_cash) VALUES (1, ?, ?)", (date_str, cash))
    c.execute("DELETE FROM state_holdings")
    for t, h in holdings.items():
        c.execute("INSERT INTO state_holdings VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (t, h['qty'], h['entry_price'], h['entry_date'].strftime('%Y-%m-%d'), h['high_price'], h['orig_entry_date'], h['months_active']))
    conn.commit()

# ==========================================
# 4. METRICS & STATS
# ==========================================
def calculate_stats(conn):
    print("\n" + "="*65)
    print(f"📊 STRATEGY PERFORMANCE REPORT")
    print(f"   (Stocks Identified in Universe: {len(TICKERS)})")
    print("="*65)
    
    df_trades = pd.read_sql("SELECT * FROM trades", conn)
    if df_trades.empty:
        print("⚠️ No trades found yet.")
        return

    df_trades['exit_date_dt'] = pd.to_datetime(df_trades['exit_date'])
    yearly_strat_pnl = df_trades.groupby(df_trades['exit_date_dt'].dt.year)['pnl_abs'].sum()
    
    start_date = df_trades['latest_entry_date'].min()
    nifty = yf.download(BENCHMARK, start=start_date, progress=False)['Close']
    if isinstance(nifty, pd.Series): nifty = nifty.to_frame()
    elif isinstance(nifty, pd.DataFrame): nifty = nifty.iloc[:, 0].to_frame()
    
    nifty.columns = ['Close']
    nifty['Year'] = nifty.index.year
    nifty_annual = nifty.groupby('Year')['Close'].agg(['first', 'last'])
    nifty_annual['Return'] = ((nifty_annual['last'] / nifty_annual['first']) - 1) * 100
    
    current_capital = INITIAL_CAPITAL
    print(f"{'Year':<6} | {'Strategy Return %':<18} | {'Nifty Return %':<15}")
    print("-" * 45)
    
    for year in yearly_strat_pnl.index:
        pnl = yearly_strat_pnl.get(year, 0)
        year_ret_pct = (pnl / current_capital) * 100 if current_capital > 0 else 0.0
        n_ret = nifty_annual.loc[year, 'Return'] if year in nifty_annual.index else 0.0
        
        print(f"{year:<6} | {year_ret_pct:>16.2f}% | {n_ret:>13.2f}%")
        current_capital += pnl

    total_years = (df_trades['exit_date_dt'].max() - df_trades['exit_date_dt'].min()).days / 365.25
    if total_years < 1: total_years = 1
    
    strat_total_ret = ((current_capital / INITIAL_CAPITAL) - 1) * 100
    strat_cagr = ((current_capital / INITIAL_CAPITAL) ** (1/total_years) - 1) * 100
    
    n_start = nifty['Close'].iloc[0]
    n_end = nifty['Close'].iloc[-1]
    nifty_total_ret = ((n_end / n_start) - 1) * 100
    nifty_cagr = ((n_end / n_start) ** (1/total_years) - 1) * 100

    print("-" * 65)
    print(f"{'Metric':<20} | {'Strategy':<18} | {'Nifty 50':<15}")
    print("-" * 65)
    print(f"{'Starting Capital':<20} | ₹{INITIAL_CAPITAL:,.0f}")
    print(f"{'Ending Capital':<20} | ₹{current_capital:,.0f}")
    print(f"{'Total Trades':<20} | {len(df_trades):<18} | {'-':<15}")
    print(f"{'Overall Return':<20} | {strat_total_ret:>17.2f}% | {nifty_total_ret:>13.2f}%")
    print(f"{'CAGR':<20} | {strat_cagr:>17.2f}% | {nifty_cagr:>13.2f}%")
    
    print("-" * 65)
    print(f"✅ Full trade history saved to: {CSV_NAME}")
    print("="*65 + "\n")

# ==========================================
# 5. EXECUTION ENGINE
# ==========================================
conn = init_db()
last_run_date_str, current_cash, current_holdings = load_state(conn)

today = datetime.today()
cutoff_date = today 
cutoff_str = cutoff_date.strftime('%Y-%m-%d')

if last_run_date_str and datetime.strptime(last_run_date_str, '%Y-%m-%d').date() >= cutoff_date.date():
    calculate_stats(conn)
    conn.close()
    exit()

fetch_start_date = (datetime.strptime(last_run_date_str or START_DATE_DEFAULT, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')

print(f"⏳ Fetching data for {len(TICKERS)} stocks...")

# --- ADDED: Robust Download with Retries ---
def robust_download(tickers, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
            if not data.empty:
                return data
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
        time.sleep(5)
    return pd.DataFrame()

data = robust_download(TICKERS + [BENCHMARK], fetch_start_date, cutoff_date + timedelta(days=1))

# --- REPORT MISSING TICKERS ---
frame_list = []
missing_tickers = []

for ticker in TICKERS + [BENCHMARK]:
    try:
        if ticker in data.columns:
            s = data[ticker]['Close']
            if s.isnull().all():
                missing_tickers.append(ticker)
            else:
                s.name = ticker
                frame_list.append(s)
        elif ticker == BENCHMARK and '^NSEI' in data.columns:
             s = data['^NSEI']['Close']
             s.name = BENCHMARK
             frame_list.append(s)
        else:
            if ticker != BENCHMARK: missing_tickers.append(ticker)
    except: 
        if ticker != BENCHMARK: missing_tickers.append(ticker)

if missing_tickers:
    print(f"\n❌ SKIPPED {len(missing_tickers)} TICKERS (No Data / Invalid Symbol):")
    print(f"   {', '.join(missing_tickers)}")
    print("-" * 65)

if not frame_list:
    print("❌ Critical Error: No data downloaded. Signal failure to GitHub Actions.")
    exit(1)

stock_data = pd.concat(frame_list, axis=1).ffill()

# --- ADDED: Data Integrity Check ---
# If more than 50% of the required data points are missing, abort to prevent corrupting the database.
if stock_data.isnull().values.sum() > (len(stock_data) * len(TICKERS) * 0.5):
    print("🚨 Critical Alert: Over 50% of downloaded data is missing. Aborting to protect state.")
    exit(1)

stock_data = stock_data.drop(columns=[BENCHMARK], errors='ignore')

# Indicators
mom_6m = stock_data.pct_change(126)
mom_3m = stock_data.pct_change(63)
mom_1m = stock_data.pct_change(21)
scores = (mom_6m * 0.4 + mom_3m * 0.4 + mom_1m * 0.2).fillna(-999)
trend_filter = (stock_data > stock_data.rolling(100).mean())

portfolio = {'cash': current_cash, 'holdings': current_holdings}
dates = stock_data.index
sim_start = datetime.strptime(last_run_date_str or START_DATE_DEFAULT, '%Y-%m-%d')

processed_any = False

for i, date in enumerate(dates):
    if date <= sim_start or date.date() > cutoff_date.date(): continue
    
    processed_any = True
    date_str = date.strftime('%Y-%m-%d')
    current_prices = stock_data.iloc[i]
    
    is_last_day_of_month = (i < len(dates)-1 and date.month != dates[i+1].month)
    is_first_day_of_month = (i > 0 and date.month != dates[i-1].month)

    # A. Stop Loss
    stocks_to_sell = []
    for ticker, pos in portfolio['holdings'].items():
        price = current_prices.get(ticker)
        if pd.isna(price): continue
        if price > pos['high_price']: portfolio['holdings'][ticker]['high_price'] = price
        if price < pos['high_price'] * (1 - TRAILING_SL_PCT):
            stocks_to_sell.append((ticker, "STOP_LOSS"))

    # B. Month End Reset
    if is_last_day_of_month:
        for ticker in list(portfolio['holdings'].keys()):
            if ticker not in [s[0] for s in stocks_to_sell]:
                stocks_to_sell.append((ticker, "MONTH_END_RESET"))

    # Execute Sells
    for ticker, reason in stocks_to_sell:
        if ticker not in portfolio['holdings']: continue
        price = current_prices[ticker]
        pos = portfolio['holdings'][ticker]
        revenue = price * pos['qty']
        pnl_pct = (price - pos['entry_price']) / pos['entry_price']
        
        conn.cursor().execute('''INSERT INTO trades 
            (trade_month, ticker, original_entry_date, months_in_momentum, latest_entry_date, latest_entry_price, qty, exit_date, exit_price, pnl_abs, pnl_pct, exit_reason, holding_days)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (date.strftime('%b-%Y'), ticker, pos['orig_entry_date'], pos['months_active'], 
             pos['entry_date'].strftime('%Y-%m-%d'), pos['entry_price'], pos['qty'], 
             date_str, price, revenue - (pos['entry_price'] * pos['qty']), pnl_pct * 100, reason, (date - pos['entry_date']).days))
        
        portfolio['cash'] += revenue
        del portfolio['holdings'][ticker]

    # C. Fresh Entry
    if is_first_day_of_month:
        current_score = scores.iloc[i]
        valid_mask = (trend_filter.iloc[i]) & (mom_1m.iloc[i] > -0.05)
        candidates = current_score[valid_mask].sort_values(ascending=False).index.tolist()[:MAX_STOCKS]
        
        if candidates and portfolio['cash'] > 0:
            alloc = portfolio['cash'] / len(candidates)
            for ticker in candidates:
                price = current_prices.get(ticker)
                if pd.isna(price) or price <= 0: continue
                qty = int(alloc / price)
                if qty > 0:
                    portfolio['cash'] -= qty * price
                    
                    c = conn.cursor()
                    c.execute("SELECT months_in_momentum, original_entry_date, exit_date FROM trades WHERE ticker=? ORDER BY exit_date DESC LIMIT 1", (ticker,))
                    prev = c.fetchone()
                    
                    is_continuation = False
                    if prev:
                        prev_mom, prev_orig_date, prev_exit_str = prev
                        prev_exit_date = datetime.strptime(prev_exit_str, '%Y-%m-%d')
                        if (date - prev_exit_date).days <= 25:
                            is_continuation = True

                    if is_continuation:
                        months_active = prev_mom + 1
                        orig_date = prev_orig_date
                    else:
                        months_active = 1
                        orig_date = date_str 

                    portfolio['holdings'][ticker] = {
                        'qty': qty, 'entry_price': price, 'entry_date': date, 
                        'high_price': price, 'orig_entry_date': orig_date, 'months_active': months_active
                    }

    save_state(conn, date_str, portfolio['cash'], portfolio['holdings'])

calculate_stats(conn)
try:
    pd.read_sql("SELECT * FROM trades", conn).to_csv(CSV_NAME, index=False)
except: pass

conn.close()
