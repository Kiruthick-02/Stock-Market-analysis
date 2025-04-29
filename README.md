# Stock Market Analysis using Time Series Relational Model

## Project Structure

```
stock_market_analysis/
│
├── data/                      # Sample or user datasets
│   └── example_stock_data.csv (not included - will be user provided)
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading from user path
│   ├── preprocessing.py       # Clean, transform, and prepare data
│   ├── model_arima.py         # Build and train ARIMA model
│   ├── model_garch.py         # Build and train GARCH model
│   ├── evaluation.py          # Compare models and metrics
│   └── visualization.py       # Plot results and model outputs
│
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview and instructions
```

## Setup Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```
   python main.py
   ```
3. When prompted, provide the path to your stock data CSV file (or use the provided `Cleaned_StockMarket_Data.csv` for testing)

## Data Format Requirements

The input CSV file should contain at least:

- A 'Date' column (YYYY-MM-DD format or DD-MM-YYYY with time optional)
- A 'Close' column with closing prices
- Optional: 'Open', 'High', 'Low', 'Volume' columns

Example CSV format:

```
Date,Open,High,Low,Close,Volume
2020-01-02,294.80,303.95,293.80,300.35,47660500
2020-01-03,297.15,300.58,296.50,297.43,41570900
```

## Features

- ARIMA model for price trend prediction
- GARCH model for volatility forecasting
- Data preprocessing and stationarity testing
- Model evaluation and comparison
- Visualization of forecasts and volatility

## How to Access Output

- **Console Output**: The script prints progress and summary stats to the terminal.
- **Forecast Output File**: A CSV file `forecast_results.csv` is generated with ARIMA forecast values.
- **Graphs**: Multiple plots are generated and displayed, including:
  - Raw and transformed time series
  - ARIMA forecasts with confidence intervals
  - GARCH volatility forecasts

## Function Descriptions

### `model_arima.py`

- `train_arima_model(data, forecast_periods, test_ratio)`:
  - Automatically selects ARIMA (p,d,q)
  - Trains the model on historical price data
  - Forecasts future prices
  - Returns forecasts and confidence intervals

### `model_garch.py`

- `train_garch_model(data)`:
  - Converts price to returns
  - Trains a GARCH(1,1) model to capture volatility clustering
  - Returns forecasted volatility

### `visualization.py`

- `plot_forecast(train, test, forecast, conf_int)`:
  - Shows ARIMA forecast with training/test data and confidence bounds
- `plot_volatility(actual_returns, forecast_volatility)`:
  - Plots actual vs forecasted volatility from GARCH model

## Summary

This project provides a full pipeline for analyzing stock market data using time series models. ARIMA predicts price trends, while GARCH models volatility. The modular structure makes it easy to adapt to other datasets or integrate additional models.

To get started quickly, use the cleaned example dataset and follow the prompts when running `main.py`. Forecasts will be printed, plotted, and saved automatically.

