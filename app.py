import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import calendar 
from datetime import datetime, timedelta
import os

# ==========================================
# 1. CONFIGURATION & BRANDING
# ==========================================

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Force connection to the file in that specific directory
DB_NAME = os.path.join(BASE_DIR, "moshi15_backtest-live.db")
ORIGINAL_BACKTEST_CAPITAL = 1000000  # Base capital (₹10L)
TRAILING_SL_PCT = 0.15 # 15% Strategy SL

# These lines MUST come before any UI elements
st.set_page_config(page_title="Momentum Shield 15", layout="wide")

# --- CUSTOM CSS FOR REGAL, CENTERED, & STICKY UI ---
st.markdown("""
    <style>
    /* 1. Regal Centered Title */
    h1 {
        text-align: center;
        font-family: 'Playfair Display', serif; /* Serif font for regal look */
        font-size: 4rem !important;
        font-weight: 700 !important;
        color: #DAA520; /* GoldenRod color */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5); /* Shadow for depth */
        padding-bottom: 0rem;
        margin-bottom: 0rem;
    }
    
    /* Center the caption/author */
    div[data-testid="stCaptionContainer"] {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.2rem;
        letter-spacing: 2px;
        color: #aaa;
        padding-bottom: 2rem;
    }

    /* 2. Tabs: Bigger and Centered */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        width: 100%;
        gap: 20px; /* Space between tabs */
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 25px;
        padding-right: 25px;
        font-size: 1.3rem; /* Larger font */
        font-weight: 600;
    }

    /* 3. Sticky Tabs (Freeze Pane Effect) */
    /* Target the container holding the tab list */
    div[data-testid="stTabs"] > div:first-child {
        position: sticky;
        top: 2.8rem; /* Offset for Streamlit's top header */
        z-index: 1000; /* Ensure it sits on top of content */
        background-color: var(--primary-background-color); /* Match theme bg so content hides behind it */
        padding-top: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.1); /* Subtle separator line */
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER (Replaces Sidebar) ---
st.title("🛡️ Momentum Shield 15 🛡️")
st.caption("A Framework For Systematic Momentum Investing")

# --- TABS FOR NAVIGATION ---
tab_portfolio, tab_analytics = st.tabs(["Current Portfolio", "Backtest Analytics"])

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

# --- Helper: TradingView Watchlist Generator ---
def copy_tv(data):
    if not data: return
    unique_tickers = []
    for x in data:
        # Catch both "Stock Name" (Live Tab) and "ticker" (Backtest Tab)
        ticker = x.get('Stock Name', x.get('ticker', ''))
        if ticker:
            # Strip out .NS or .ns before appending
            clean_ticker = str(ticker).replace('.NS', '').replace('.ns', '')
            unique_tickers.append(f"NSE:{clean_ticker}")
            
    unique_tickers = list(dict.fromkeys(unique_tickers))
    batches = [", ".join(unique_tickers[i:i+30]) for i in range(0, len(unique_tickers), 30)]
    st.markdown("### 📋 TradingView Watchlist")
    for b in batches: st.code(b, language="text")

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

# ==========================================
# DATABASE CONNECTION & VALIDATION
# ==========================================
def get_db_connection():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "moshi15_backtest-live.db")
    return sqlite3.connect(db_path, check_same_thread=False)

try:
    conn = get_db_connection()
    check_df = pd.read_sql("SELECT count(*) as cnt FROM trades", conn)
    conn.close()
    
    if check_df['cnt'].iloc[0] == 0:
        st.error(f"⚠️ Database found at {DB_NAME}, but the 'trades' table is empty.")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Critical Error: Could not read database at: {DB_NAME}")
    st.code(f"Error details: {e}")
    st.info("💡 Tip: Ensure 'moshi15_engine.py' has been run successfully.")
    st.stop()

def get_db_data():
    try:
        conn = get_db_connection()
        trades = pd.read_sql("SELECT * FROM trades", conn)
        meta = pd.read_sql("SELECT * FROM state_meta", conn)
        conn.close()

        rename_map = {
            'latest_entry_date': 'entry_date',
            'latest_entry_price': 'entry_price' 
        }
        trades.rename(columns=rename_map, inplace=True)

        if 'entry_date' in trades.columns:
            trades['entry_date'] = pd.to_datetime(trades['entry_date'])
        
        if 'exit_date' in trades.columns:
            trades['exit_date'] = pd.to_datetime(trades['exit_date'])
            
        trades['qty'] = pd.to_numeric(trades['qty'], errors='coerce').fillna(0)
        
        return trades, meta
    except Exception as e: 
        st.error(f"Error reading detailed data: {e}")
        return None, None

@st.cache_data(ttl=3600)
def fetch_nifty_data_final():
    try:
        df = yf.download("^NSEI", period="max", progress=False)['Close']
        if isinstance(df, pd.Series): df = df.to_frame()
        
        if df.empty or (not df.empty and df.index[0].year > 2020):
            df = yf.download("NIFTYBEES.NS", period="max", progress=False)['Close']
            if isinstance(df, pd.Series): df = df.to_frame()
            
        df.columns = ["Nifty 50"]
        if not df.empty:
            df.index = df.index.tz_localize(None)
            
        return df
    except: return pd.DataFrame()

# ==========================================
# TAB 1: CURRENT PORTFOLIO
# ==========================================
with tab_portfolio:
    st.header("📊 Current Portfolio")
    
    try:
        conn = get_db_connection()
        df_holdings = pd.read_sql("SELECT * FROM state_holdings", conn)
        df_meta = pd.read_sql("SELECT * FROM state_meta", conn)
        conn.close()
        
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
            price_at_rebalance = row['entry_price'] 
            high_price = row['high_price']
            
            try:
                ltp = live_data[ticker].iloc[-1]
            except:
                ltp = price_at_rebalance
            
            if 'orig_entry_date' in row and row['orig_entry_date']:
                orig_date = pd.to_datetime(row['orig_entry_date'])
            else:
                orig_date = pd.to_datetime(row['entry_date'])

            orig_price_val = price_at_rebalance 
            if orig_date != pd.to_datetime(row['entry_date']):
                try:
                    conn_lookup = get_db_connection()
                    q = f"SELECT latest_entry_price FROM trades WHERE ticker='{ticker}' AND latest_entry_date='{orig_date.strftime('%Y-%m-%d')}' LIMIT 1"
                    res = pd.read_sql(q, conn_lookup)
                    conn_lookup.close()
                    if not res.empty:
                        orig_price_val = res.iloc[0]['latest_entry_price']
                except:
                    pass

            months_active_count = row['months_active'] if 'months_active' in row else 1
            
            trailing_sl = high_price * (1 - TRAILING_SL_PCT)
            fall_from_high = ((ltp - high_price) / high_price) * 100
            dist_from_sl = ((ltp - trailing_sl) / ltp) * 100
            running_return_pct = ((ltp - price_at_rebalance) / price_at_rebalance) * 100
            status = "🟢 LIVE" if ltp >= trailing_sl else "🔴 STOPPED OUT"
            
            portfolio_list.append({
                "Stock Name": ticker,
                "Orig. Date": orig_date.strftime('%Y-%m-%d'),
                "Orig. Price": orig_price_val,
                "Months Active": months_active_count,
                "Rebal. Price": price_at_rebalance,
                "Running Return (%)": running_return_pct, 
                "LTP": ltp,
                "High Price": high_price,
                "Fall from High (%)": fall_from_high,
                "Trailing SL": trailing_sl,
                "Status": status
            })
        
        df_p = pd.DataFrame(portfolio_list)
        
        st.dataframe(
            df_p.style.format({
                "Orig. Price": "₹{:.2f}",
                "Rebal. Price": "₹{:.2f}",
                "Running Return (%)": "{:.2f}%", 
                "LTP": "₹{:.2f}",
                "High Price": "₹{:.2f}",
                "Fall from High (%)": "{:.2f}%",
                "Trailing SL": "₹{:.2f}",
            })
            .background_gradient(subset=['Running Return (%)'], cmap="RdYlGn", vmin=-10, vmax=10) 
            .background_gradient(subset=['Fall from High (%)'], cmap="RdYlGn", vmin=-15, vmax=0),
            use_container_width=True,
            hide_index=True
        )

        stopped_trades = df_p[df_p['Status'] == "🔴 STOPPED OUT"].copy()
        
        st.markdown("") 
        if not stopped_trades.empty:
            st.error(f"🛑 STOPPED TRADES DETECTED ({len(stopped_trades)})")
            st.markdown("The following trades have hit their trailing stop loss. **Exit immediately.**")
            
            st.dataframe(
                stopped_trades[['Stock Name', 'Rebal. Price', 'LTP', 'Trailing SL', 'Fall from High (%)']]
                .style.format({
                    "Rebal. Price": "₹{:.2f}",
                    "LTP": "₹{:.2f}",
                    "Trailing SL": "₹{:.2f}",
                    "Fall from High (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("✅ All Trades are LIVE (No Stop Losses Hit)")
            
        # --- GENERATE TRADINGVIEW WATCHLIST FOR LIVE TRADES ---
        st.markdown("<br>", unsafe_allow_html=True)
        copy_tv(df_p.to_dict('records'))

    else:
        st.info("No active holdings found. Ensure the automated backtest has run successfully.")

# ==========================================
# TAB 2: BACKTEST ANALYTICS
# ==========================================
with tab_analytics:
    st.header("📊 Backtest Analytics")
    trades_df, meta_df = get_db_data()
    
    if trades_df is None or trades_df.empty:
        st.error(f"⚠️ Database '{DB_NAME}' missing/empty. Run 'moshi15_engine.py'.")
    else:
        # --- NEW LOCATION: SETTINGS IN BODY ---
        st.markdown("### ⚙️ Settings")
        c_set1, c_set2 = st.columns(2)
        
        with c_set1:
            capital = st.number_input("Starting Capital (₹)", value=100000, step=100000)
            scale_factor = capital / ORIGINAL_BACKTEST_CAPITAL
        
        with c_set2:
            preset = st.selectbox("Quick Select Period", ["Custom", "Specific Month", "Previous 1 Month", "Previous 3 Months", "Previous 6 Months", "Previous 1 Year", "Previous 2 Years", "Current FY", "All Time"])
        
        today = datetime.today().date()
        min_db_date = trades_df['entry_date'].min().date()
        
        start_default = min_db_date
        end_default = today

        # --- Date Logic ---
        if preset == "Specific Month":
            min_year = trades_df['entry_date'].min().year
            max_year = today.year
            years = list(range(min_year, max_year + 1))
            c_y, c_m = st.columns(2) 
            with c_y:
                sel_year = st.selectbox("Year", years, index=len(years)-1)
            with c_m:
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                sel_month_str = st.selectbox("Month", months, index=today.month-1)
            sel_month_int = months.index(sel_month_str) + 1
            start_default = datetime(sel_year, sel_month_int, 1).date()
            last_day = calendar.monthrange(sel_year, sel_month_int)[1]
            end_default = datetime(sel_year, sel_month_int, last_day).date()
            if end_default > today: end_default = today

        elif preset == "Previous 1 Month": 
            first_day_this_month = today.replace(day=1)
            end_default = first_day_this_month - timedelta(days=1)
            start_default = end_default.replace(day=1)

        elif preset == "Previous 3 Months": start_default = today - timedelta(days=90)
        elif preset == "Previous 6 Months": start_default = today - timedelta(days=180)
        elif preset == "Previous 1 Year": start_default = today - timedelta(days=365)
        elif preset == "Previous 2 Years": start_default = today - timedelta(days=365*2)
        elif preset == "Current FY":
            start_default = datetime(today.year, 4, 1).date() if today.month >= 4 else datetime(today.year - 1, 4, 1).date()
        elif preset == "All Time":
            start_default = min_db_date
            
        if start_default < min_db_date: start_default = min_db_date
            
        # Display Date Pickers in body
        col_d1, col_d2 = st.columns(2)
        with col_d1: start_date_input = st.date_input("Start Date", start_default, min_value=min_db_date, max_value=today)
        with col_d2: end_date_input = st.date_input("End Date", end_default, min_value=min_db_date, max_value=today)
        
        # Date alignment logic
        if preset != "Specific Month" and preset != "Previous 1 Month" and start_date_input.day > 1:
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
        
        period_df['Scaled_PnL'] = period_df['pnl_abs'] * scale_factor
        period_df['Scaled_Qty'] = (period_df['qty'].astype(float) * scale_factor).astype(int)
        
        if period_df.empty:
            st.info("No trades found in this period.")
        else:
            # --- CALCULATIONS INSIDE BLOCK ---
            total_pnl = period_df['Scaled_PnL'].sum()
            ending_capital = capital + total_pnl
            return_pct = (total_pnl / capital) * 100
            
            total_days = (e_date - s_date).days
            if total_days > 0:
                cagr = ((ending_capital / capital) ** (365.25 / total_days)) - 1
            else:
                cagr = 0.0

            bench_full = fetch_nifty_data_final()
            nifty_ret = 0.0
            
            if not bench_full.empty:
                try:
                    metrics_bench = bench_full[(bench_full.index >= s_date) & (bench_full.index <= e_date)]
                    if not metrics_bench.empty:
                        nifty_val_start = metrics_bench.iloc[0]['Nifty 50']
                        nifty_val_end = metrics_bench.iloc[-1]['Nifty 50']
                        nifty_ret = ((nifty_val_end / nifty_val_start) - 1) * 100
                except: nifty_ret = 0.0

            total_trades = len(period_df)
            winning_trades = len(period_df[period_df['Scaled_PnL'] > 0])
            accuracy = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            wins = period_df[period_df['Scaled_PnL'] > 0]['Scaled_PnL']
            losses = period_df[period_df['Scaled_PnL'] <= 0]['Scaled_PnL']
            
            avg_gain = wins.mean() if not wins.empty else 0
            avg_loss = losses.mean() if not losses.empty else 0
            gl_ratio = avg_gain / abs(avg_loss) if avg_loss != 0 else 0
            
            max_gain = period_df['Scaled_PnL'].max()
            max_loss = period_df['Scaled_PnL'].min()
            profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 0
            
            period_df = period_df.sort_values('exit_date')
            period_df['Cumulative_PnL'] = period_df['Scaled_PnL'].cumsum()
            period_df['Equity'] = capital + period_df['Cumulative_PnL']
            period_df['Peak'] = period_df['Equity'].cummax()
            period_df['Drawdown_Abs'] = period_df['Equity'] - period_df['Peak']
            period_df['Drawdown_Pct'] = (period_df['Drawdown_Abs'] / period_df['Peak']) * 100
            
            max_dd = period_df['Drawdown_Abs'].min()
            max_dd_pct = period_df['Drawdown_Pct'].min()
            
            recovery_days = 0
            rec_str = "Not Recovered"
            try:
                mdd_idx = period_df['Drawdown_Abs'].idxmin()
                mdd_date = period_df.loc[mdd_idx, 'exit_date']
                peak_at_mdd = period_df.loc[mdd_idx, 'Peak']
                recovery_df = period_df[period_df['exit_date'] > mdd_date]
                recovered_mask = recovery_df['Equity'] >= peak_at_mdd
                if recovered_mask.any():
                    rec_date = recovery_df.loc[recovered_mask.idxmax(), 'exit_date']
                    recovery_days = (rec_date - mdd_date).days
                    rec_str = f"{recovery_days} Days ({rec_date.strftime('%d-%b-%Y')})"
                else:
                    recovery_days = -1
                    rec_str = "Not Recovered"
            except: 
                recovery_days = 0
                rec_str = "N/A"

            downside_returns = period_df[period_df['pnl_pct'] < 0]['pnl_pct']
            downside_dev = downside_returns.std()
            avg_ret = period_df['pnl_pct'].mean()
            sortino = avg_ret/downside_dev if downside_dev > 0 else 0

            # --- RENDER METRICS ---
            st.markdown("### 📈 Performance Summary")
            l1_c1, l1_c2, l1_c3, l1_c4 = st.columns(4) 
            l1_c1.metric("Ending Capital", format_indian(ending_capital), delta=format_indian(ending_capital - capital))
            l1_c2.metric("Total Return", f"{return_pct:.2f}%", delta=f"{return_pct:.2f}%")
            l1_c3.metric("CAGR (XIRR)", f"{cagr*100:.2f}%", delta=f"{cagr*100:.2f}%") 
            l1_c4.metric("Nifty Return", f"{nifty_ret:.2f}%", delta=f"{nifty_ret:.2f}%")
            
            # --- UPDATED: Added 'Period Backtested' to Row 2 ---
            l2_c1, l2_c2, l2_c3, l2_c4 = st.columns(4)
            l2_c1.metric("Backtest Period", f"{total_days} Days")
            l2_c2.metric("Trades Taken", total_trades)
            l2_c3.metric("Winning Trades", winning_trades)
            l2_c4.metric("Accuracy", f"{accuracy:.1f}%", delta=f"{accuracy-50:.1f}%")
            
            l3_c1, l3_c2, l3_c3 = st.columns(3)
            l3_c1.metric("Avg Gain", format_indian(avg_gain), delta=format_indian(avg_gain))
            l3_c2.metric("Avg Loss", format_indian(avg_loss), delta=format_indian(avg_loss), delta_color="inverse")
            l3_c3.metric("Gain/Loss Ratio", f"{gl_ratio:.2f}", delta=f"{gl_ratio-1:.2f}")
            
            l4_c1, l4_c2, l4_c3 = st.columns(3)
            l4_c1.metric("Max Gain", format_indian(max_gain), delta=format_indian(max_gain))
            l4_c2.metric("Max Loss", format_indian(max_loss), delta=format_indian(max_loss), delta_color="inverse")
            l4_c3.metric("Profit Factor", f"{profit_factor:.2f}", delta=f"{profit_factor-1.0:.2f}")

            l5_c1, l5_c2, l5_c3 = st.columns(3)
            l5_c1.metric("Max Drawdown", f"{format_indian(max_dd)} ({max_dd_pct:.2f}%)", delta=format_indian(max_dd), delta_color="inverse")
            rec_delta = None if recovery_days < 0 else f"{recovery_days} days"
            l5_c2.metric("Days to Breakeven", rec_str, delta=rec_delta, delta_color="inverse")
            l5_c3.metric("Sortino Ratio", f"{sortino:.2f}", delta=f"{sortino:.2f}")
            
            st.markdown("---")
            st.subheader("📈 Equity Curve vs Nifty 50")
            
            current_date = datetime.now().date()
            first_of_current_month = current_date.replace(day=1)
            last_completed_month_end = first_of_current_month - timedelta(days=1)
            
            chart_df = period_df[period_df['exit_date'].dt.date <= last_completed_month_end].copy()
            fig = go.Figure()
            chart_equity = chart_df.set_index('exit_date')['Equity']
            
            if not chart_equity.empty:
                strategy_pct = ((chart_equity - capital) / capital) * 100
                start_row = pd.Series([0.0], index=[s_date])
                strategy_pct = pd.concat([start_row, strategy_pct]).sort_index()
                fig.add_trace(go.Scatter(x=strategy_pct.index, y=strategy_pct, mode='lines', name='Strategy (%)', line=dict(color='#FFD700', width=3)))
            
            if not bench_full.empty:
                if not chart_equity.empty:
                    chart_end_date = chart_equity.index.max()
                else:
                    chart_end_date = e_date 
                
                bench_plot = bench_full[(bench_full.index >= s_date) & (bench_full.index <= chart_end_date)].copy()
                if not bench_plot.empty:
                    start_val = bench_plot.iloc[0]['Nifty 50']
                    nifty_pct = ((bench_plot['Nifty 50'] - start_val) / start_val) * 100
                    fig.add_trace(go.Scatter(x=nifty_pct.index, y=nifty_pct, mode='lines', name='Nifty 50 (%)', line=dict(color='#0078FF', width=2)))
            
            fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Growth (%)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🌊 Drawdown (Underwater Chart)")
            if not chart_equity.empty:
                dd_series = (chart_equity / chart_equity.cummax()) - 1
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dd_series.index, 
                    y=dd_series * 100, 
                    mode='lines', 
                    name='Drawdown',
                    line=dict(color='#FF4B4B', width=1),
                    fill='tozeroy'
                ))
                fig_dd.update_layout(height=300, xaxis_title="Date", yaxis_title="Drawdown (%)", hovermode="x unified")
                st.plotly_chart(fig_dd, use_container_width=True)

            st.markdown("---")
            st.subheader("📅 Monthly Returns Heatmap")
            
            try:
                daily_idx = pd.date_range(start=s_date, end=e_date, freq='D')
                daily_pnl = period_df.groupby('exit_date')['Scaled_PnL'].sum()
                daily_pnl = daily_pnl.reindex(daily_idx).fillna(0)
                anchor_date = s_date - timedelta(days=1)
                daily_pnl.loc[anchor_date] = 0 
                daily_pnl = daily_pnl.sort_index()
                
                heatmap_equity = daily_pnl.cumsum() + capital
                try:
                    monthly_equity = heatmap_equity.resample('ME').last()
                except:
                    monthly_equity = heatmap_equity.resample('M').last()
                
                monthly_ret = monthly_equity.pct_change() * 100
                monthly_ret = monthly_ret.dropna()
                
                monthly_df = monthly_ret.to_frame(name='Equity')
                monthly_df['Year'] = monthly_df.index.year
                monthly_df['Month'] = monthly_df.index.strftime('%b')
                
                heatmap = monthly_df.pivot(index='Year', columns='Month', values='Equity')
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
            
            # --- FIXED: Removed 'sector' column from display ---
            st.dataframe(
                period_df.sort_values('exit_date', ascending=False)
                [['ticker', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'qty', 'Scaled_PnL', 'pnl_pct', 'exit_reason']]
                .rename(columns={'qty': 'Qty (Base 10L)', 'Scaled_PnL': 'PnL'})
                .style
                .format({'PnL': format_indian, 'entry_price': '₹{:.2f}', 'exit_price': '₹{:.2f}', 'pnl_pct': '{:.2f}%'})
                .background_gradient(subset=['PnL'], cmap='RdYlGn', vmin=-max_pnl_scale, vmax=max_pnl_scale),
                use_container_width=True
            )
            
            # --- GENERATE TRADINGVIEW WATCHLIST FOR PERIOD LOG ---
            st.markdown("<br>", unsafe_allow_html=True)
            copy_tv(period_df.to_dict('records'))

        st.markdown("---")
        with st.expander("📂 View Full History (All Trades)"):
            st.info(f"**NOTE:** Quantities below are based on the original database capital of ₹{ORIGINAL_BACKTEST_CAPITAL/100000:.0f} Lakhs. PnL is scaled to your input.")
            full_df = trades_df.copy().sort_values('exit_date', ascending=False)
            full_df['Scaled_PnL'] = full_df['pnl_abs'] * scale_factor
            max_full_scale = max(abs(full_df['Scaled_PnL'].min()), abs(full_df['Scaled_PnL'].max())) if not full_df.empty else 1
            
            # --- FIXED: Removed 'sector' column from display ---
            st.dataframe(
                full_df[['ticker', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'qty', 'Scaled_PnL', 'pnl_pct', 'exit_reason']]
                .rename(columns={'qty': 'Qty (Base 10L)', 'Scaled_PnL': 'PnL'})
                .style
                .format({'PnL': format_indian, 'entry_price': '₹{:.2f}', 'exit_price': '₹{:.2f}', 'pnl_pct': '{:.2f}%'})
                .background_gradient(subset=['PnL'], cmap='RdYlGn', vmin=-max_full_scale, vmax=max_full_scale),
                use_container_width=True
            )
            
            # --- GENERATE TRADINGVIEW WATCHLIST FOR FULL LOG ---
            st.markdown("<br>", unsafe_allow_html=True)
            copy_tv(full_df.to_dict('records'))
