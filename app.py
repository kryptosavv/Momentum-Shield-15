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
st.caption("BY SHREESHA S")

# --- TABS FOR NAVIGATION ---
# UPDATED: Renamed 3rd tab to "Stress Test Simulator"
tab_portfolio, tab_analytics, tab_sim = st.tabs(["Current Portfolio", "Backtest Analytics", "Stress Test Simulator"])

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
                order_data.append({
                    "Stock": row['Stock Name'], 
                    "Rebal. Price": row['Rebal. Price'], 
                    "Quantity": qty, 
                    "Buy Value": val
                })
            
            df_orders = pd.DataFrame(order_data)
            # Sort DESCENDING (Largest first)
            df_orders = df_orders.sort_values(by="Quantity", ascending=False).reset_index(drop=True)
            
            # --- FINAL COLUMN SELECTION (NO SL NO) ---
            df_orders = df_orders[["Stock", "Rebal. Price", "Quantity", "Buy Value"]]
            
            st.table(df_orders.style.format({"Buy Value": "₹{:,.0f}", "Rebal. Price": "₹{:.2f}"}))
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
            c_y, c_m = st.columns(2) # Changed to st.columns for body layout
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

# ==========================================
# TAB 3: STRESS TEST SIMULATOR (FIXED & FULL)
# ==========================================
with tab_sim:
    # --- UPDATED: Changed from st.title to st.header to match style ---
    st.header("📉 Stress Test Simulator")
    trades_df, _ = get_db_data()
    
    if trades_df is None or trades_df.empty: st.error("Data missing.")
    else:
        # 1. Build Daily Equity Series (Base)
        start_dt = trades_df['entry_date'].min()
        end_dt = datetime.today()
        dates = pd.date_range(start_dt, end_dt)
        
        daily_pnl = trades_df.groupby('exit_date')['pnl_abs'].sum().reindex(dates).fillna(0)
        base_cap = ORIGINAL_BACKTEST_CAPITAL
        equity_curve = base_cap + daily_pnl.cumsum()
        
        # 2. Get Nifty Benchmark & ALIGN
        nifty_raw = fetch_nifty_data_final()
        
        # --- FIX 1: STRICT ALIGNMENT ---
        common_idx = equity_curve.index.intersection(nifty_raw.index)
        
        if not common_idx.empty:
            strat_aligned = equity_curve.loc[common_idx]
            nifty_aligned = nifty_raw.loc[common_idx]['Nifty 50']
            # Normalize Nifty to Strategy Capital
            nifty_aligned = (nifty_aligned / nifty_aligned.iloc[0]) * base_cap
        else:
            st.error("Error aligning Strategy and Nifty dates.")
            st.stop()

        # --- PART 1: ROLLING RETURNS ---
        st.subheader("a. Rolling Returns Analysis")
        
        c1, c2 = st.columns([1, 3])
        with c1:
            # Added "1 Month" to options
            roll_window = st.selectbox("Select Rolling Window", 
                ["1 Month", "2 Months", "3 Months", "6 Months", "9 Months", "1 Year", "2 Years", "3 Years"], index=5, key="roll_win_sel")
            
            w_map = {
                "1 Month": 21, "2 Months": 42, "3 Months": 63, "6 Months": 126, 
                "9 Months": 189, "1 Year": 252, "2 Years": 504, "3 Years": 756
            }
            days = w_map[roll_window]
            st.markdown("---")
            sim_date = st.date_input("📅 Simulate Entry Date", value=datetime(2023, 1, 1), min_value=start_dt, max_value=end_dt, key="sim_dt_in")

        # Calculate Rolling Returns on ALIGNED data
        strat_roll = strat_aligned.pct_change(days) * 100
        nifty_roll = nifty_aligned.pct_change(days) * 100
        
        with c2:
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(x=strat_roll.index, y=strat_roll, mode='lines', name=f'Strategy {roll_window}', line=dict(color='#FFD700')))
            fig_roll.add_trace(go.Scatter(x=nifty_roll.index, y=nifty_roll, mode='lines', name=f'Nifty {roll_window}', line=dict(color='#0078FF', width=1)))
            
            # Add marker for selected date
            try:
                idx_loc = strat_roll.index.get_indexer([pd.Timestamp(sim_date)], method='nearest')[0]
                if idx_loc >= 0:
                    val_at_date = strat_roll.iloc[idx_loc]
                    date_at_loc = strat_roll.index[idx_loc]
                    fig_roll.add_trace(go.Scatter(
                        x=[date_at_loc], y=[val_at_date],
                        mode='markers', marker=dict(color='red', size=10),
                        name=f"Entry: {sim_date.strftime('%d-%b-%y')}"
                    ))
            except: pass

            fig_roll.update_layout(title=f"Rolling {roll_window} Returns", xaxis_title="Date", yaxis_title="Return (%)", height=400)
            st.plotly_chart(fig_roll, use_container_width=True, key=f"roll_chart_{roll_window}")

        try:
            target_date = pd.Timestamp(sim_date) + timedelta(days=days * 1.45)
            # Use searchsorted/get_indexer logic for robustness
            res_idx = strat_aligned.index.get_indexer([target_date], method='nearest')[0]
            start_idx = strat_aligned.index.get_indexer([pd.Timestamp(sim_date)], method='nearest')[0]
            
            start_val = strat_aligned.iloc[start_idx]
            end_val = strat_aligned.iloc[res_idx]
            actual_ret = ((end_val - start_val) / start_val) * 100
            
            # Nifty Comparison Logic
            n_start = nifty_aligned.iloc[start_idx]
            n_end = nifty_aligned.iloc[res_idx]
            nifty_sim_ret = ((n_end - n_start) / n_start) * 100
            
            msg = f"If you invested on **{sim_date.strftime('%d-%b-%Y')}**, your {roll_window} return would be **{actual_ret:.2f}%** vs Nifty50 return of **{nifty_sim_ret:.2f}%**."
            
            if actual_ret >= nifty_sim_ret:
                st.success(f"✅ {msg}")
            else:
                st.error(f"📉 {msg}")
                
        except: st.warning("Not enough future data.")

        st.markdown("---")

        # --- PART 2: BEST AND WORST ROLLING RETURNS (UPDATED) ---
        st.subheader("b. Best and Worst Rolling Returns")
        
        # New Filters (Keys added to prevent dup ID)
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            alpha_type = st.radio("Select Performance Type", ["Positive Alpha (Outperformance)", "Negative Alpha (Underperformance)"], horizontal=True, key="alpha_radio")
        with c_f2:
            # --- UPDATED: Added 6 Months ---
            period_sel = st.selectbox("Select Rolling Period", ["6 Months", "1 Year", "2 Years", "3 Years"], key="period_radio")

        # --- UPDATED: Map includes 6M ---
        p_map = {"6 Months": 6, "1 Year": 12, "2 Years": 24, "3 Years": 36}
        months = p_map[period_sel]

        # 1. Resample to Month Start (One entry per month)
        s_m = strat_aligned.resample('MS').first()
        n_m = nifty_aligned.resample('MS').first()

        # 2. Calculate Rolling Returns
        df_res = pd.DataFrame({
            'Strategy': s_m.pct_change(months) * 100,
            'Nifty': n_m.pct_change(months) * 100
        }).dropna()

        df_res['Alpha'] = df_res['Strategy'] - df_res['Nifty']
        # --- UPDATED: Explicit Start & End Dates ---
        df_res['End Date'] = df_res.index
        df_res['Start Date'] = df_res.index - pd.DateOffset(months=months)

        # 3. Filter
        if "Positive" in alpha_type:
            df_filt = df_res[df_res['Alpha'] > 0].sort_values(by='Alpha', ascending=False)
            color_map = 'Greens'
        else:
            df_filt = df_res[df_res['Alpha'] < 0].sort_values(by='Alpha', ascending=True) # Most negative first
            # --- UPDATED: Correct Gradient (Red for Worst) ---
            color_map = 'YlOrRd_r'

        # 4. Display (No Limit)
        if not df_filt.empty:
            df_display = df_filt.reset_index(drop=True)
            df_display.insert(0, "Sl No", df_display.index + 1)
            # Reorder columns
            df_display = df_display[['Sl No', 'Start Date', 'End Date', 'Strategy', 'Nifty', 'Alpha']]
            
            st.dataframe(
                df_display.style.format({
                    'Strategy': '{:.2f}%', 'Nifty': '{:.2f}%', 'Alpha': '{:.2f}%', 
                    'Start Date': '{:%d-%b-%Y}', 'End Date': '{:%d-%b-%Y}'
                })
                .background_gradient(subset=['Alpha'], cmap=color_map),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No data found for selected criteria.")

        st.markdown("---")

        # --- PART 3: SIGNIFICANT DRAWDOWN SIMULATOR (FIXED) ---
        st.subheader("c. Significant DD Simulator (DD > 5%)")
        
        peak = strat_aligned.cummax()
        dd_series = (strat_aligned - peak) / peak
        
        is_dd = dd_series < 0
        dd_id = (is_dd != is_dd.shift()).cumsum()
        dd_groups = dd_series[is_dd].groupby(dd_id)
        
        dd_stats = []
        for _, group in dd_groups:
            if group.empty: continue
            start_d = group.index[0]
            end_d = group.index[-1]
            depth = group.min()
            bottom_date = group.idxmin()
            duration = (end_d - start_d).days
            
            # Recovery relative to start date peak
            if strat_aligned.loc[end_d] >= peak.loc[start_d]:
                rec_status = "Recovered"
            else:
                rec_status = "Not Recovered"
            
            if depth < -0.05: 
                dd_stats.append({
                    "Start Date": start_d, 
                    "Bottom Date": bottom_date, 
                    "Recovery Date": end_d,
                    "Recovery Status": rec_status,
                    "Depth (%)": depth * 100, 
                    "Days to Recover": duration
                })
        
        if dd_stats:
            df_dd = pd.DataFrame(dd_stats).sort_values("Depth (%)", ascending=True)
            top_dds = df_dd.reset_index(drop=True)
            top_dds.insert(0, "Sl No", top_dds.index + 1)
            
            # Dropdown Presets
            presets = top_dds.head(10)
            presets['Label'] = presets.apply(lambda x: f"{x['Start Date'].strftime('%b-%Y')} (Depth: {x['Depth (%)']:.2f}%)", axis=1)
            
            c_sel, c_chart = st.columns([1, 3])
            with c_sel:
                st.markdown("##### Select Stress Period")
                if not presets.empty:
                    sel_dd_idx = st.selectbox("Choose a historical crash:", presets.index, format_func=lambda x: presets.loc[x, 'Label'], key="dd_select")
                    sel_row = presets.loc[sel_dd_idx]
                    
                    st.error(f"📉 Max Depth: **{sel_row['Depth (%)']:.2f}%**")
                    st.warning(f"⏳ Days Under: **{sel_row['Days to Recover']} days**")
                    st.success(f"✅ Recovered By: **{sel_row['Recovery Date'].strftime('%d-%b-%Y')}**")
                else:
                    st.info("No major crashes to simulate.")

            with c_chart:
                if not presets.empty:
                    zoom_start = sel_row['Start Date'] - timedelta(days=30)
                    zoom_end = sel_row['Recovery Date'] + timedelta(days=60)
                    
                    # --- FIX 2: SLICING BY DATE (Prevents IndexError) ---
                    # Using the ALIGNED series, so indices match perfectly.
                    z_strat = strat_aligned.loc[zoom_start:zoom_end]
                    z_nifty = nifty_aligned.loc[zoom_start:zoom_end]
                    
                    if not z_strat.empty:
                        # Rebase to 100
                        base_z = z_strat.iloc[0]
                        z_strat_plot = (z_strat / base_z) * 100
                        
                        fig_z = go.Figure()
                        fig_z.add_trace(go.Scatter(x=z_strat_plot.index, y=z_strat_plot, mode='lines', name='Strategy', line=dict(color='#FF4B4B', width=3)))
                        
                        if not z_nifty.empty:
                            base_n = z_nifty.iloc[0]
                            z_nifty_plot = (z_nifty / base_n) * 100
                            fig_z.add_trace(go.Scatter(x=z_nifty_plot.index, y=z_nifty_plot, mode='lines', name='Nifty', line=dict(color='#0078FF', dash='dot')))
                        
                        fig_z.update_layout(title=f"Stress Test: {sel_row['Label']}", xaxis_title="Date", yaxis_title="Rebased Value (100)", hovermode="x unified")
                        st.plotly_chart(fig_z, use_container_width=True, key="dd_chart")
            
            st.markdown("##### 📋 Log of All Drawdowns > 5%")
            st.table(top_dds[['Sl No', 'Start Date', 'Bottom Date', 'Recovery Date', 'Depth (%)', 'Days to Recover']].style.format({
                'Depth (%)': '{:.2f}%', 'Start Date': '{:%d-%b-%Y}', 'Bottom Date': '{:%d-%b-%Y}', 'Recovery Date': '{:%d-%b-%Y}'
            }))
        else:
            st.success("No significant drawdowns (>5%) detected in history.")