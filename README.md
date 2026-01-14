Momentum-Shield-15
Momentum Shield 15 is a rules-based equity momentum strategy that ranks stocks using 6M, 3M, and 1M returns, applies trend and momentum filters, and builds an equal-weight portfolio of up to 15 stocks. Risk is managed through monthly rebalancing and a daily 15% trailing stop-loss.

🚀 Strategy Mechanics
1. Selection & Ranking
Universe: Liquid stocks defined in universe.txt.

Ranking Logic: Composite momentum score calculated as:

40% Weight: 6-Month Returns (126 trading days)

40% Weight: 3-Month Returns (63 trading days)

20% Weight: 1-Month Returns (21 trading days)

2. Entry Filters (The Gatekeepers)
A stock is eligible for entry only if it meets all three criteria:

Trend Filter: Current Price must be above the 100-day Simple Moving Average (SMA).

Momentum Filter: 1-Month Return must be greater than -5% (to avoid "falling knives").

Timing: Fresh entries occur strictly on the 1st trading day of each month.

3. Exit Rules
Trailing Stop Loss: 15% below the highest price achieved since entry. This is checked daily, and exits happen immediately mid-month if hit.

Monthly Rebalance: During the monthly review (1st trading day), any stock that has fallen out of the Top 25 ranks is sold to make room for stronger candidates.

📊 Performance Metrics
The dashboard provides 15 key metrics for deep strategy analysis:

Capital Stats: Ending Capital, Total Return %, and Nifty Benchmark Return.

Trade Stats: Total Trades Taken, Winning Trades, and Accuracy (Win %).

Efficiency: Average Gain, Average Loss, Gain/Loss Ratio, and Profit Factor.

Risk: Max Gain/Loss (₹), Max Drawdown (₹), Days to Recover, and Sortino Ratio.

📁 Project Structure
app.py: The Streamlit dashboard. Features a Live Scanner, Order Generator (with Sl No numbering), and Backtest Analytics.

incremental_backtest.py: The engine that runs simulations. It strictly appends data only for the last day of a completed month to ensure data integrity.

universe.txt: A simple text file containing the list of stock tickers (Yahoo Finance format).

momentum_backtest.db: SQLite database storing all trade history and portfolio state.

🛠️ Installation & Usage
Install Dependencies:

Bash

pip install streamlit pandas yfinance plotly
Update Database: Run the backtest engine to process the latest completed months:

Bash

python incremental_backtest.py
Launch Dashboard:

Bash

streamlit run app.py
