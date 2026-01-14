import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import os


# ==========================================
# 1. CONFIGURATION & BRANDING
# ==========================================

DB_NAME = "momentum_backtest.db"
ORIGINAL_BACKTEST_CAPITAL = 1000000  # Base capital (₹10L)
TRAILING_SL_PCT = 0.15 # 15% Strategy SL

# These lines MUST come before the 'if page ==' logic
st.set_page_config(page_title="Momentum Shield 15", layout="wide")
st.sidebar.title("🛡️ Momentum Shield 15")
st.sidebar.markdown(f"### *Shreesha S*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Current Portfolio", "Backtest Analytics"])
st.sidebar.markdown("---")

# --- Helper: Indian Currency Formatting ---
def format_indian(n):
    """Formats a number into Indian Lakh/Crore style (e.g., 10,00,000)."""
    try:
        n = float(n)
        is_negative = n < 0
        n = abs(n)
        s = "{:.0f}".format(n)
        if len(s) <= 3: res = s
        else:
            res = s[-3:]
            s = s[:-3]
            while len(s) > 2:
                res = s[-2:] + "," + res
                s = s[:-2]
            res = s + "," + res
        return "₹" + ("-" if is_negative else "") + res
    except: return "₹0"

# --- A. Load Universe from Master File ---
def load_universe():
    universe = []
    if not os.path.exists("universe.txt"):
        return []
    try:
        with open("universe.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    universe.append(line)
        return universe
    except:
        return []

UNIVERSE_TICKERS = load_universe()

# --- B. Sector Map ---
SECTOR_MAP = {
    "HDFCBANK.NS": "Financials", "ICICIBANK.NS": "Financials", "SBIN.NS": "Financials",
    "AXISBANK.NS": "Financials", "KOTAKBANK.NS": "Financials", "BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Financials", "SHRIRAMFIN.NS": "Financials", "CHOLAFIN.NS": "Financials",
    "PFC.NS": "Financials", "RECLTD.NS": "Financials", "SBILIFE.NS": "Financials",
    "LICI.NS": "Financials", "ABCAPITAL.NS": "Financials", "FEDERALBNK.NS": "Financials",
    "IDFCFIRSTB.NS": "Financials", "BANKBARODA.NS": "Financials", "CANBK.NS": "Financials",
    "PNB.NS": "Financials", "INDUSINDBK.NS": "Financials", "JIOFIN.NS": "Financials",
    "SBICARD.NS": "Financials", "POONAWALLA.NS": "Financials", "MCX.NS": "Financials",
    "BSE.NS": "Financials", "PAYTM.NS": "Financials", "PBFINTECH.NS": "Financials",
    "ICICIGI.NS": "Financials", "ICICIPRULI.NS": "Financials", "HDFCLIFE.NS": "Financials",
    "TCS.NS": "Technology", "INFY.NS": "Technology", "HCLTECH.NS": "Technology",
    "WIPRO.NS": "Technology", "TECHM.NS": "Technology", "LTIM.NS": "Technology",
    "PERSISTENT.NS": "Technology", "COFORGE.NS": "Technology", "KPITTECH.NS": "Technology",
    "MPHASIS.NS": "Technology", "OFSS.NS": "Technology", "TATACOMM.NS": "Technology",
    "NAUKRI.NS": "Technology", "ZOMATO.NS": "Technology", "POLICYBZR.NS": "Technology",
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "NTPC.NS": "Power",
    "POWERGRID.NS": "Power", "COALINDIA.NS": "Energy", "BPCL.NS": "Energy",
    "IOC.NS": "Energy", "GAIL.NS": "Energy", "ADANIGREEN.NS": "Power",
    "ATGL.NS": "Energy", "TATAPOWER.NS": "Power", "HINDPETRO.NS": "Energy",
    "IGL.NS": "Energy", "GUJGASLTD.NS": "Energy",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto",
    "EICHERMOT.NS": "Auto", "TVSMOTOR.NS": "Auto", "ASHOKLEY.NS": "Auto",
    "MOTHERSON.NS": "Auto", "MRF.NS": "Auto", "BOSCHLTD.NS": "Auto",
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "TITAN.NS": "Consumer", "ASIANPAINT.NS": "Consumer",
    "TRENT.NS": "Consumer", "VBL.NS": "FMCG", "GODREJCP.NS": "FMCG",
    "DABUR.NS": "FMCG", "MARICO.NS": "FMCG", "COLPAL.NS": "FMCG",
    "BERGEPAINT.NS": "Consumer", "PAGEIND.NS": "Consumer", "HAVELLS.NS": "Consumer",
    "TATACONSUM.NS": "FMCG", "POLYCAB.NS": "Consumer",
    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "VEDL.NS": "Metals", "JINDALSTEL.NS": "Metals", "ADANIENT.NS": "Metals",
    "LT.NS": "Infra", "ULTRACEMCO.NS": "Materials", "GRASIM.NS": "Materials",
    "AMBUJACEM.NS": "Materials", "DALBHARAT.NS": "Materials", "ADANIPORTS.NS": "Infra",
    "DLF.NS": "Realty", "LODHA.NS": "Realty", "GODREJPROP.NS": "Realty",
    "OBEROIRLTY.NS": "Realty", "PHOENIXLTD.NS": "Realty", "PRESTIGE.NS": "Realty",
    "HAL.NS": "Defense", "BEL.NS": "Defense", "MAZDOCK.NS": "Defense",
    "COCHINSHIP.NS": "Defense", "ABB.NS": "Cap Goods", "SIEMENS.NS": "Cap Goods",
    "CUMMINSIND.NS": "Cap Goods", "RVNL.NS": "Rail", "IRFC.NS": "Rail",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
    "DIVISLAB.NS": "Pharma", "APOLLOHOSP.NS": "Healthcare", "MAXHEALTH.NS": "Healthcare",
    "LUPIN.NS": "Pharma", "ALKEM.NS": "Pharma", "ZYDUSLIFE.NS": "Pharma",
    "ABBOTINDIA.NS": "Pharma"
}

def get_sector(ticker):
    return SECTOR_MAP.get(ticker, "Others")

@st.cache_data
def get_db_data():
    try:
        conn = sqlite3.connect(DB_NAME)
        trades = pd.read_sql("SELECT * FROM trades", conn)
        trades['entry_date'] = pd.to_datetime(trades['entry_date'])
        trades['exit_date'] = pd.to_datetime(trades['exit_date'])
        trades['sector'] = trades['ticker'].apply(get_sector)
        # Ensure qty is numeric for scaling
        trades['qty'] = pd.to_numeric(trades['qty'], errors='coerce').fillna(0)
        meta = pd.read_sql("SELECT * FROM state_meta", conn)
        conn.close()
        return trades, meta
    except: return None, None

@st.cache_data(ttl=3600)
def fetch_nifty_data_final():
    """
    Bulletproof Benchmark Fetcher.
    1. Fetches MAX history.
    2. Strips Timezones immediately.
    3. Falls back to Nifty Bees if Index fails.
    """
    try:
        # Try Primary
        df = yf.download("^NSEI", period="max", progress=False)['Close']
        if isinstance(df, pd.Series): df = df.to_frame()
        
        # Check if we got data or if it's too short (e.g. starts in 2024)
        if df.empty or (not df.empty and df.index[0].year > 2020):
            # Fallback
            df = yf.download("NIFTYBEES.NS", period="max", progress=False)['Close']
            if isinstance(df, pd.Series): df = df.to_frame()
            
        df.columns = ["Nifty 50"]
        # CRITICAL FIX: Remove timezone
        if not df.empty:
            df.index = df.index.tz_localize(None)
            
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def run_scanner():
    if not UNIVERSE_TICKERS: return None, None, {}, {}, {}
    
    # Download 2 years of data
    data = yf.download(UNIVERSE_TICKERS, period="2y", progress=False)['Close']
    
    # Calc Indicators
    mom_6m = data.pct_change(126).iloc[-1]
    mom_3m = data.pct_change(63).iloc[-1]
    mom_1m_series = data.pct_change(21)
    mom_1m = mom_1m_series.iloc[-1]
    
    scores = (mom_6m * 0.4 + mom_3m * 0.4 + mom_1m * 0.2).fillna(-999)
    sma_100_series = data.rolling(window=100).mean()
    latest_prices = data.iloc[-1]
    
    # Filters
    history_mask = (data > sma_100_series) & (mom_1m_series > -0.05)
    trend_filter = latest_prices > sma_100_series.iloc[-1]
    mom_filter = mom_1m > -0.05
    
    valid = scores[trend_filter & mom_filter].sort_values(ascending=False).head(15)
    
    # Calculate Streak Details
    streaks = {}
    streak_starts = {}
    streak_entry_prices = {}
    
    for ticker in valid.index:
        if ticker in history_mask.columns:
            bool_series = history_mask[ticker]
            days_active = 0
            # Count backwards
            for was_met in bool_series.iloc[::-1]:
                if was_met: days_active += 1
                else: break
            
            streaks[ticker] = days_active
            if days_active > 0:
                start_date = bool_series.index[-days_active]
                streak_starts[ticker] = start_date.strftime('%Y-%m-%d')
                try:
                    streak_entry_prices[ticker] = data.loc[start_date, ticker]
                except:
                    streak_entry_prices[ticker] = 0.0
            else:
                streak_starts[ticker] = "-"
                streak_entry_prices[ticker] = 0.0
        else:
            streaks[ticker] = 0
            streak_starts[ticker] = "-"
            streak_entry_prices[ticker] = 0.0
            
    return valid, latest_prices, streaks, streak_starts, streak_entry_prices

def is_first_trading_day():
    return datetime.today().day <= 5

# ==========================================
# PAGE 1: CURRENT PORTFOLIO
# ==========================================
if page == "Current Portfolio":
    st.header("📊 Current Portfolio")
    
    # Database Loading
    try:
        conn = sqlite3.connect(DB_NAME)
        df_holdings = pd.read_sql("SELECT * FROM state_holdings", conn)
        df_meta = pd.read_sql("SELECT * FROM state_meta", conn)
        conn.close()
        
        # Robust Date Logic for Sub-heading
        if not df_meta.empty:
            date_col = [c for c in df_meta.columns if 'date' in c.lower() or 'rebalance' in c.lower()][0]
            last_reb_raw = pd.to_datetime(df_meta[date_col].iloc[0]).replace(tzinfo=None)
            reb_display_date = last_reb_raw.replace(day=1)
            sub_head_date = reb_display_date.strftime('%d-%b-%Y')
        else:
            reb_display_date = pd.Timestamp.now().replace(day=1, hour=0, minute=0, second=0)
            sub_head_date = reb_display_date.strftime('%d-%b-%Y')
            
    except Exception as e:
        st.error(f"Error: {e}")
        sub_head_date = "N/A"
        reb_display_date = pd.Timestamp.now().replace(day=1)

    st.subheader(f"Rebalanced on {sub_head_date}") 
    st.warning("⚠️ **Strategy Note:** Only enter/rebalance on the **1st trading day**.")

    if not df_holdings.empty:
        portfolio_list = []
        tickers = df_holdings['ticker'].tolist()
        live_data = yf.download(tickers, period="5d", progress=False)['Close']

        for _, row in df_holdings.iterrows():
            ticker = row['ticker']
            price_at_reb = row['entry_price'] 
            high_price = row['high_price']
            
            try:
                ltp = live_data[ticker].iloc[-1]
            except:
                ltp = high_price
            
            # Fetch Trend Start Date (Avoids 1970/ancient dates)
            conn = sqlite3.connect(DB_NAME)
            query = f"""
                SELECT entry_date, entry_price 
                FROM trades 
                WHERE ticker='{ticker}' 
                AND entry_date <= '{reb_display_date.strftime('%Y-%m-%d')}' 
                AND (exit_date IS NULL OR exit_date >= '{reb_display_date.strftime('%Y-%m-%d')}')
                ORDER BY entry_date ASC LIMIT 1
            """
            original_data = pd.read_sql(query, conn)
            conn.close()
            
            orig_date = pd.to_datetime(original_data['entry_date'].iloc[0]) if not original_data.empty else pd.to_datetime(row['entry_date'])
            orig_price = original_data['entry_price'].iloc[0] if not original_data.empty else price_at_reb

            days_active = (pd.Timestamp.now() - orig_date).days
            trailing_sl = high_price * (1 - TRAILING_SL_PCT)
            fall_from_high = ((ltp - high_price) / high_price) * 100
            dist_from_sl = ((ltp - trailing_sl) / ltp) * 100
            status = "🟢 LIVE" if ltp >= trailing_sl else "🔴 STOPPED OUT"
            
            portfolio_list.append({
                "Stock Name": ticker,
                "Original Entry Date": orig_date.strftime('%Y-%m-%d'),
                "Days Active": days_active,
                "Original Entry Price": orig_price,
                "Price at Rebalance": price_at_reb,
                "LTP": ltp,
                "High Price": high_price,
                "Fall from High (%)": fall_from_high,
                "Trailing SL": trailing_sl,
                "Dist. from SL (%)": dist_from_sl,
                "Status": status
            })
        
        df_p = pd.DataFrame(portfolio_list)
        
        # --- THE FIX: use_container_width=True makes it fit the page width ---
        st.dataframe(
            df_p.style.format({
                "Original Entry Price": "₹{:.2f}",
                "Price at Rebalance": "₹{:.2f}",
                "LTP": "₹{:.2f}",
                "High Price": "₹{:.2f}",
                "Fall from High (%)": "{:.2f}%",
                "Trailing SL": "₹{:.2f}",
                "Dist. from SL (%)": "{:.2f}%"
            })
            .background_gradient(subset=['Dist. from SL (%)'], cmap="RdYlGn", vmin=0, vmax=15)
            .background_gradient(subset=['Fall from High (%)'], cmap="RdYlGn", vmin=-15, vmax=0),
            use_container_width=True, # THIS REMOVES THE HORIZONTAL SCROLL
            hide_index=True
        )

        # 3. Order Generator Summary (Calculates Remaining Cash)
        st.divider()
        st.subheader("🛒 Order Generator")
        col_cap, _ = st.columns([1, 2])
        with col_cap:
            port_capital = st.number_input("Capital to Deploy (₹)", value=100000, step=10000, key="port_cap_final")
        
        if port_capital > 0:
            alloc = port_capital / 15
            order_data = []
            total_deployed = 0
            for i, row in df_p.iterrows():
                qty = int(alloc / row['LTP']) if row['LTP'] > 0 else 0
                val = qty * row['LTP']
                total_deployed += val
                order_data.append({"Sl No": i+1, "Stock": row['Stock Name'], "Quantity": qty, "Est. Value": val})
            
            # Metrics Display
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Capital", format_indian(port_capital))
            m2.metric("Deployed", format_indian(total_deployed))
            m3.metric("Remaining Cash", format_indian(port_capital - total_deployed))
            
            st.table(pd.DataFrame(order_data).style.format({"Est. Value": "₹{:,.0f}"}))

# ==========================================
# PAGE 2: BACKTEST ANALYTICS (ADVANCED)
# ==========================================
elif page == "Backtest Analytics":
    st.title("📊 Strategy Performance Center")
    trades_df, meta_df = get_db_data()
    
    if trades_df is None or trades_df.empty:
        st.error(f"⚠️ Database '{DB_NAME}' missing/empty. Run 'incremental_backtest.py'.")
    else:
        # --- INPUTS ---
        st.sidebar.markdown("### ⚙️ Backtest Settings")
        initial_capital = st.sidebar.number_input("Starting Capital (₹)", value=100000, step=10000)
        
        # Calculate Scaling Factor
        scale_factor = initial_capital / ORIGINAL_BACKTEST_CAPITAL
        
        # --- PRESETS ---
        st.sidebar.markdown("### 🗓️ Period Presets")
        preset = st.sidebar.selectbox("Quick Select", ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "Current FY", "All Time"])
        
        today = datetime.today().date()
        min_db_date = trades_df['entry_date'].min().date()
        
        if preset == "1 Month": start_default = today - timedelta(days=30)
        elif preset == "3 Months": start_default = today - timedelta(days=90)
        elif preset == "6 Months": start_default = today - timedelta(days=180)
        elif preset == "1 Year": start_default = today - timedelta(days=365)
        elif preset == "2 Years": start_default = today - timedelta(days=365*2)
        elif preset == "Current FY":
            start_default = datetime(today.year, 4, 1).date() if today.month >= 4 else datetime(today.year - 1, 4, 1).date()
        elif preset == "All Time":
            start_default = min_db_date
        else: start_default = min_db_date
            
        if start_default < min_db_date: start_default = min_db_date
            
        # FIX: Define columns BEFORE using them
        col_d1, col_d2 = st.sidebar.columns(2)
        with col_d1: start_date_input = st.date_input("Start", start_default, min_value=min_db_date, max_value=today)
        with col_d2: end_date_input = st.date_input("End", today, min_value=min_db_date, max_value=today)
        
        # Align to 1st of month logic
        if start_date_input.day > 1:
            if start_date_input.month == 12: next_month = datetime(start_date_input.year + 1, 1, 1).date()
            else: next_month = datetime(start_date_input.year, start_date_input.month + 1, 1).date()
            if next_month <= today:
                st.info(f"ℹ️ Aligned start date from {start_date_input} to **{next_month}** (1st of month) for accuracy.")
                start_date = next_month
            else: start_date = start_date_input
        else: start_date = start_date_input
        end_date = end_date_input
        
        s_date, e_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
        mask = (trades_df['exit_date'] >= s_date) & (trades_df['exit_date'] <= e_date)
        period_df = trades_df.loc[mask].copy()
        
        # Scale the PnL & Qty
        period_df['Scaled_PnL'] = period_df['pnl_abs'] * scale_factor
        period_df['Scaled_Qty'] = (period_df['qty'].astype(float) * scale_factor).astype(int)
        
        # --- CALCULATIONS ---
        if period_df.empty:
            st.info("No trades found in this period.")
        else:
            # 1. Capital Stats
            total_pnl = period_df['Scaled_PnL'].sum()
            ending_capital = initial_capital + total_pnl
            return_pct = (total_pnl / initial_capital) * 100
            
            # --- BENCHMARK DATA: Fetch FULL history ---
            bench_full = fetch_nifty_data_final()
            nifty_ret = 0.0
            
            if not bench_full.empty:
                try:
                    # Filter only for the metrics calculation
                    metrics_bench = bench_full[(bench_full.index >= s_date) & (bench_full.index <= e_date)]
                    if not metrics_bench.empty:
                        nifty_val_start = metrics_bench.iloc[0]['Nifty 50']
                        nifty_val_end = metrics_bench.iloc[-1]['Nifty 50']
                        nifty_ret = ((nifty_val_end / nifty_val_start) - 1) * 100
                except: nifty_ret = 0.0

            # 2. Trade Stats
            total_trades = len(period_df)
            winning_trades = len(period_df[period_df['Scaled_PnL'] > 0])
            accuracy = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # 3. Avg Stats
            wins = period_df[period_df['Scaled_PnL'] > 0]['Scaled_PnL']
            losses = period_df[period_df['Scaled_PnL'] <= 0]['Scaled_PnL']
            
            avg_gain = wins.mean() if not wins.empty else 0
            avg_loss = losses.mean() if not losses.empty else 0
            gl_ratio = avg_gain / abs(avg_loss) if avg_loss != 0 else 0
            
            # 4. Extreme Stats
            max_gain = period_df['Scaled_PnL'].max()
            max_loss = period_df['Scaled_PnL'].min()
            profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 0
            
            # 5. Risk Stats
            period_df = period_df.sort_values('exit_date')
            period_df['Cumulative_PnL'] = period_df['Scaled_PnL'].cumsum()
            period_df['Equity'] = initial_capital + period_df['Cumulative_PnL']
            period_df['Peak'] = period_df['Equity'].cummax()
            period_df['Drawdown_Abs'] = period_df['Equity'] - period_df['Peak']
            
            max_dd = period_df['Drawdown_Abs'].min()
            
            # Days to Recover
            recovery_days = 0
            try:
                mdd_idx = period_df['Drawdown_Abs'].idxmin()
                mdd_date = period_df.loc[mdd_idx, 'exit_date']
                peak_at_mdd = period_df.loc[mdd_idx, 'Peak']
                recovery_df = period_df[period_df['exit_date'] > mdd_date]
                recovered = recovery_df[recovery_df['Equity'] >= peak_at_mdd]
                if not recovered.empty:
                    rec_date = recovered.iloc[0]['exit_date']
                    recovery_days = (rec_date - mdd_date).days
                else: recovery_days = -1
            except: recovery_days = 0
            rec_str = f"{recovery_days} days" if recovery_days >= 0 else "Not Recovered"

            # Sortino & VaR
            downside_returns = period_df[period_df['pnl_pct'] < 0]['pnl_pct']
            downside_dev = downside_returns.std()
            avg_ret = period_df['pnl_pct'].mean()
            sortino = avg_ret/downside_dev if downside_dev > 0 else 0
            var_95 = np.percentile(period_df['pnl_pct'], 5)

            # --- DISPLAY GRID ---
            st.markdown("### 📈 Performance Summary")
            l1_c1, l1_c2, l1_c3 = st.columns(3)
            l1_c1.metric("Ending Capital", format_indian(ending_capital))
            l1_c2.metric("Return %", f"{return_pct:.2f}%")
            l1_c3.metric("Nifty Return", f"{nifty_ret:.2f}%")
            
            l2_c1, l2_c2, l2_c3 = st.columns(3)
            l2_c1.metric("Trades Taken", total_trades)
            l2_c2.metric("Winning Trades", winning_trades)
            l2_c3.metric("Accuracy", f"{accuracy:.1f}%")
            
            l3_c1, l3_c2, l3_c3 = st.columns(3)
            l3_c1.metric("Avg Gain", format_indian(avg_gain))
            l3_c2.metric("Avg Loss", format_indian(avg_loss))
            l3_c3.metric("Gain/Loss Ratio", f"{gl_ratio:.2f}")
            
            # Line 4 (Gain/Loss)
            l4_c1, l4_c2, l4_c3 = st.columns(3)
            l4_c1.metric("Max Gain", format_indian(max_gain))
            l4_c2.metric("Max Loss", format_indian(max_loss))
            l4_c3.metric("Profit Factor", f"{profit_factor:.2f}")

            # Line 5 (Risk)
            l5_c1, l5_c2, l5_c3 = st.columns(3)
            l5_c1.metric("Max Drawdown", format_indian(max_dd))
            l5_c2.metric("Days to Recover", rec_str)
            l5_c3.metric("Sortino Ratio", f"{sortino:.2f}")
            
            st.markdown("---")
            st.subheader("📈 Equity Curve vs Nifty 50")
            
            # --- CHARTING (PERCENTAGE BASED) ---
            fig = go.Figure()
            
            # 1. Strategy Line (% Growth)
            chart_equity = period_df.set_index('exit_date')['Equity']
            
            if not chart_equity.empty:
                # Growth % relative to initial_capital
                strategy_pct = ((chart_equity - initial_capital) / initial_capital) * 100
                
                # Add Day 0 point (0%)
                start_row = pd.Series([0.0], index=[s_date])
                strategy_pct = pd.concat([start_row, strategy_pct]).sort_index()
                
                fig.add_trace(go.Scatter(x=strategy_pct.index, y=strategy_pct, mode='lines', name='Strategy (%)', line=dict(color='#FFD700', width=3)))
            
            # 2. Benchmark Line (% Growth)
            if not bench_full.empty:
                # Slicing master data to the chart window
                bench_plot = bench_full[bench_full.index >= s_date].copy()
                if not bench_plot.empty:
                    start_val = bench_plot.iloc[0]['Nifty 50']
                    nifty_pct = ((bench_plot['Nifty 50'] - start_val) / start_val) * 100
                    fig.add_trace(go.Scatter(x=nifty_pct.index, y=nifty_pct, mode='lines', name='Nifty 50 (%)', line=dict(color='#0078FF', width=2)))
            
            fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Growth (%)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- FIX: MONTHLY HEATMAP ---
            st.markdown("---")
            st.subheader("📅 Monthly Returns Heatmap")
            
            try:
                # 1. Reconstruct Daily Equity Curve strictly for Heatmap
                daily_idx = pd.date_range(start=s_date, end=e_date, freq='D')
                daily_pnl = period_df.groupby('exit_date')['Scaled_PnL'].sum()
                daily_pnl = daily_pnl.reindex(daily_idx).fillna(0)
                heatmap_equity = daily_pnl.cumsum() + initial_capital
                
                # 2. Resample to Monthly Returns
                try:
                    monthly_ret = heatmap_equity.resample('ME').last().pct_change() * 100
                except:
                    # Fallback for older pandas
                    monthly_ret = heatmap_equity.resample('M').last().pct_change() * 100
                
                # FIX: Explicitly convert to DataFrame and set name 'Equity'
                monthly_df = monthly_ret.to_frame(name='Equity')
                
                monthly_df['Year'] = monthly_df.index.year
                monthly_df['Month'] = monthly_df.index.strftime('%b')
                
                heatmap = monthly_df.pivot(index='Year', columns='Month', values='Equity')
                
                # 4. Sort Columns
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                heatmap = heatmap.reindex(columns=months)
                
                st.dataframe(
                    heatmap.style
                    .background_gradient(cmap='RdYlGn', vmin=-10, vmax=10)
                    .format("{:.2f}%", na_rep=""),
                    use_container_width=True
                )
            except Exception as e:
                st.info(f"Not enough data to generate heatmap yet. (Need > 1 month). Error: {e}")

            st.subheader(f"📜 Trade Log ({start_date} to {end_date})")
            max_pnl_scale = max(abs(period_df['Scaled_PnL'].min()), abs(period_df['Scaled_PnL'].max())) if not period_df.empty else 1
            
            st.dataframe(
                period_df.sort_values('exit_date', ascending=False)
                [['ticker', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'Scaled_Qty', 'Scaled_PnL', 'pnl_pct', 'sector', 'exit_reason']]
                .rename(columns={'Scaled_Qty': 'Qty', 'Scaled_PnL': 'PnL'})
                .style
                .format({'PnL': format_indian, 'entry_price': '₹{:.2f}', 'exit_price': '₹{:.2f}', 'pnl_pct': '{:.2f}%'})
                .background_gradient(subset=['PnL'], cmap='RdYlGn', vmin=-max_pnl_scale, vmax=max_pnl_scale),
                use_container_width=True
            )
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Sector Allocation")
                st.plotly_chart(px.pie(period_df['sector'].value_counts().reset_index(), values='count', names='sector'), use_container_width=True)
            with c2:
                st.subheader("Top Winners")
                winners = period_df.groupby('ticker')['Scaled_PnL'].sum().reset_index().sort_values('Scaled_PnL', ascending=False).head(10)
                st.dataframe(winners.style.format({'Scaled_PnL': format_indian}).background_gradient(cmap='Greens'), use_container_width=True, hide_index=True)

        # --- FULL HISTORY ---
        st.markdown("---")
        with st.expander("📂 View Full History (All Trades)"):
            st.info(f"**NOTE:** Quantities below are based on the original database capital of ₹{ORIGINAL_BACKTEST_CAPITAL/100000:.0f} Lakhs. PnL is scaled to your input.")
            full_df = trades_df.copy().sort_values('exit_date', ascending=False)
            full_df['Scaled_PnL'] = full_df['pnl_abs'] * scale_factor
            
            max_full_scale = max(abs(full_df['Scaled_PnL'].min()), abs(full_df['Scaled_PnL'].max())) if not full_df.empty else 1
            
            st.dataframe(
                full_df[['ticker', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'qty', 'Scaled_PnL', 'pnl_pct', 'sector', 'exit_reason']]
                .rename(columns={'qty': 'Qty (Base 10L)', 'Scaled_PnL': 'PnL'})
                .style
                .format({'PnL': format_indian, 'entry_price': '₹{:.2f}', 'exit_price': '₹{:.2f}', 'pnl_pct': '{:.2f}%'})
                .background_gradient(subset=['PnL'], cmap='RdYlGn', vmin=-max_full_scale, vmax=max_full_scale),
                use_container_width=True
            )
