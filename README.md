Momentum-Shield-15
Momentum Shield 15 is a rules-based equity momentum strategy that ranks stocks using 6M, 3M, and 1M returns, applies trend and momentum filters, and builds an equal-weight portfolio of up to 15 stocks. Risk is managed through monthly rebalancing and a daily 15% trailing stop-loss.

**🚀 Strategy Mechanics**

**A. Selection & Ranking**
1. Universe: Liquid stocks defined in universe.txt.
2. Ranking Logic: Composite momentum score calculated as:
2.1 40% Weight: 6-Month Returns (126 trading days)
2.2 40% Weight: 3-Month Returns (63 trading days)
2.3 20% Weight: 1-Month Returns (21 trading days)

**B. Entry Filters (The Gatekeepers)**
A stock is eligible for entry only if it meets all three criteria:
1. Trend Filter: Current Price must be above the 100-day Simple Moving Average (SMA).
2. Momentum Filter: 1-Month Return must be greater than -5% (to avoid "falling knives").
3. Timing: Fresh entries occur strictly on the 1st trading day of each month.

**C. Exit Rules**
1. Trailing Stop Loss: 15% below the highest price achieved since entry. This is checked daily, and exits happen immediately mid-month if hit.
2. Monthly Rebalance: During the monthly review (1st trading day), any stock that has fallen out of the Top 25 ranks is sold to make room for stronger candidates.

**📁 Project Structure**
1. app.py: The Streamlit dashboard. Features a Live Scanner, Order Generator (with Sl No numbering), and Backtest Analytics.
2. moshi15_engine.py: The engine that runs simulations. It strictly appends data only for the last day of a completed month to ensure data integrity.
3. universe.txt: A simple text file containing the list of stock tickers (Yahoo Finance format).
4. moshi15_backtest-live.db: SQLite database storing all trade history and portfolio state.
