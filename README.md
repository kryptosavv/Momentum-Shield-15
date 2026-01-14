Momentum-Shield-15
Momentum Shield 15 is a rules-based equity momentum strategy that ranks stocks using 6M, 3M, and 1M returns, applies trend and momentum filters, and builds an equal-weight portfolio of up to 15 stocks. Risk is managed through monthly rebalancing and a daily 15% trailing stop-loss.

🚀 Strategy Mechanics
1. Selection & Ranking
Universe: Defined in universe.txt.

Ranking Logic: Composite momentum score based on:

40% Weight: 6-Month Returns (126 trading days)

40% Weight: 3-Month Returns (63 trading days)

20% Weight: 1-Month Returns (21 trading days)

2. Entry Filters
Trend Filter: Price must be above the 100-day Simple Moving Average (SMA).

Momentum Filter: 1-Month Return must be greater than -5%.

Timing: Fresh entries occur strictly on the 1st trading day of the month.

3. Exit Rules
Trailing Stop Loss: 15% below the highest price since entry (checked daily).

Monthly Rebalance: Sell any stock that falls out of the Top 25 ranks.

📁 Project Structure
app.py: The Streamlit dashboard for live scanning and backtest visualization.

incremental_backtest.py: The engine that runs simulations and updates the database.

universe.txt: List of stocks to be scanned.

momentum_backtest.db: SQLite database storing trade history.

📊 Key Metrics Tracked
The dashboard provides a comprehensive performance summary including:

Capital Stats: Ending Capital, Return %, and Nifty Benchmark Return.

Trade Stats: Total Trades, Winning Trades, and Accuracy.

Efficiency: Avg Gain/Loss, Gain/Loss Ratio, and Profit Factor.

Risk Metrics: Max Drawdown (₹), Days to Recover, and Sortino Ratio.
