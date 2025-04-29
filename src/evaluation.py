import pandas as pd
import numpy as np
import os

def evaluate_models(arima_results, garch_results):
    """
    Evaluate ARIMA and GARCH models and save forecasts to CSV.
    
    Args:
        arima_results: ARIMAResults object
        garch_results: GARCHResults object
    """
    print("\nEvaluation of model forecasts:")
    print("-" * 40)
    
    # Create a results dictionary
    results = {}
    
    # Add ARIMA metrics
    if hasattr(arima_results, 'rmse') and arima_results.rmse is not None:
        results['ARIMA RMSE'] = arima_results.rmse
    if hasattr(arima_results, 'mae') and arima_results.mae is not None:
        results['ARIMA MAE'] = arima_results.mae
    if hasattr(arima_results, 'aic') and arima_results.aic is not None:
        results['ARIMA AIC'] = arima_results.aic
    if hasattr(arima_results, 'bic') and arima_results.bic is not None:
        results['ARIMA BIC'] = arima_results.bic
    
    # Add GARCH metrics
    if hasattr(garch_results, 'rmse') and garch_results.rmse is not None:
        results['GARCH RMSE'] = garch_results.rmse
    if hasattr(garch_results, 'mae') and garch_results.mae is not None:
        results['GARCH MAE'] = garch_results.mae
    if hasattr(garch_results, 'aic') and garch_results.aic is not None:
        results['GARCH AIC'] = garch_results.aic
    if hasattr(garch_results, 'bic') and garch_results.bic is not None:
        results['GARCH BIC'] = garch_results.bic
    
    # Print evaluation results
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Create a DataFrame with forecast dates
    try:
        forecast_df = pd.DataFrame()
        
        # ARIMA price forecasts
        if hasattr(arima_results, 'forecast') and arima_results.forecast is not None:
            forecast_df['Date'] = arima_results.forecast_dates
            forecast_df['Price_Forecast'] = arima_results.forecast
        
        # GARCH volatility forecasts
        if hasattr(garch_results, 'forecast_volatility') and garch_results.forecast_volatility is not None:
            # Make sure we have the date column
            if 'Date' not in forecast_df.columns and hasattr(garch_results, 'forecast_dates'):
                forecast_df['Date'] = garch_results.forecast_dates
            forecast_df['Volatility_Forecast'] = garch_results.forecast_volatility
        
        # Set the date as index
        if 'Date' in forecast_df.columns:
            forecast_df.set_index('Date', inplace=True)
        
        # Print forecast summary
        if not forecast_df.empty:
            print("\nForecast summary:")
            print(forecast_df.head())
            
            # Save forecasts to CSV
            try:
                forecast_df.to_csv('forecast_results.csv')
                print("\nForecasts saved to 'forecast_results.csv'")
            except Exception as e:
                print(f"\nWarning: Could not save forecasts to CSV: {e}")
                # Try alternative location
                try:
                    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
                    csv_path = os.path.join(desktop_path, 'forecast_results.csv')
                    forecast_df.to_csv(csv_path)
                    print(f"Forecasts saved to '{csv_path}'")
                except:
                    print("Could not save forecasts to any location.")
    
    except Exception as e:
        print(f"Error creating forecast summary: {e}")
        
    return results