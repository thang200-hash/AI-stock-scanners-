# AI STOCK SCANNER MVP2 FINAL

This package is a Streamlit-based U.S. stock screening prototype designed around a 50/50 philosophy:
- Growth stocks
- Stable quality stocks

## Files
- `app.py` — main Streamlit application
- `requirements.txt` — Python dependencies
- `scanner_config.json` — version/config metadata
- `README.md` — setup instructions

## MVP2 FINAL features
- Live/public stock data using `yfinance`
- Data completeness / quality check
- Quality score
- Growth score
- Stability score
- Future Growth score
- Valuation score
- Growth vs Stable classification
- Preliminary fair-value estimate
- Buy zones
- Red-flag checks
- Position sizing based on portfolio value and risk per trade
- Suggested stop reference
- CSV export

## Important
This is NOT a guaranteed-profit system.
No app can guarantee 10–20% yearly returns.
The fair-value model remains an MVP approximation and should be validated using paper trading and backtesting before any live-money use.

## Deploy on Streamlit Community Cloud
1. Create or open your GitHub repository, for example `ai-stock-scanner`.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `scanner_config.json`
3. Open Streamlit Community Cloud.
4. Create a new app from the GitHub repository.
5. Use `app.py` as the main file.
6. Click Deploy.
7. Open the generated Streamlit link.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Recommended next stage
After MVP2 runs reliably:
1. Backtest scoring logic.
2. Add full DCF and peer valuation.
3. Add earnings-date guard.
4. Add portfolio-level risk limits.
5. Connect to IBKR paper trading with manual confirmation.
6. Only consider live trading after sufficient testing.
