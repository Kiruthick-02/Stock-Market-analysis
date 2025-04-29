"""
Stock Market Analysis using Time Series Relational Model
ARIMA and GARCH Time Series Forecasting

This script performs time series analysis on stock market data
using ARIMA for price forecasting and GARCH for volatility modeling.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Handle path for importing local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Updated: Import the new hardcoded loader
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model_arima import run_arima
from src.model_garch import run_garch
from src.visualization import (
    visualize_data, 
    visualize_arima_results, 
    visualize_garch_forecast,
    visualize_comparison
)
from src.evaluation import evaluate_models

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    """Main execution function for the stock market analysis project."""
    
    print("\n" + "="*70)
    print("STOCK MARKET ANALYSIS USING TIME SERIES RELATIONAL MODEL")
    print("="*70)
    print("\nThis program performs time series analysis on stock market data using:")
    print("  - ARIMA (AutoRegressive Integrated Moving Average) for price forecasting")
    print("  - GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) for volatility modeling\n")
    
    try:
        # Step 1: Load the data
        print("\nSTEP 1: LOADING DATA")
        df = load_data()  # Updated: No more user input
        
        if df is None:
            print("Error loading data. Exiting program.")
            return 1
        
        # Step 2: Preprocess the data
        print("\nSTEP 2: PREPROCESSING DATA")
        df_processed = preprocess_data(df)
        
        # Step 3: Visualize the data
        print("\nSTEP 3: VISUALIZING DATA")
        visualize_data(df_processed)
        
        # Step 4: Run ARIMA model
        print("\nSTEP 4: RUNNING ARIMA MODEL")
        forecast_periods = int(input("\nEnter the number of periods to forecast (default: 30): ") or 30)
        test_size = float(input("Enter the proportion of data to use for testing (0.0-1.0, default: 0.2): ") or 0.2)
        
        arima_results = run_arima(df_processed, forecast_periods=forecast_periods, test_size=test_size)
        visualize_arima_results(arima_results)
        
        # Step 5: Run GARCH model
        print("\nSTEP 5: RUNNING GARCH MODEL")
        garch_results = run_garch(df_processed, forecast_periods=forecast_periods, test_size=test_size)

        visualize_garch_forecast(
            train_data=garch_results.train_data,
            cond_vol=garch_results.conditional_volatility,
            forecast_dates=garch_results.forecast_dates,
            forecast_vol=garch_results.forecast_volatility,
            garch_results=garch_results
        )
        
        # Step 6: Compare models
        print("\nSTEP 6: COMPARING MODELS")
        visualize_comparison(arima_results, garch_results, df_processed)
        
        # Step 7: Evaluate models
        print("\nSTEP 7: EVALUATING MODELS")
        evaluate_models(arima_results, garch_results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nForecasts have been saved to 'forecast_results.csv'")
        
    except Exception as e:
        print("\nERROR: An error occurred during analysis.")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
