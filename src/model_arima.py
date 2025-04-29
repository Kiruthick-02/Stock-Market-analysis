import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import itertools
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

class ARIMAResults:
    """Class to store ARIMA model results."""
    def __init__(self):
        self.model = None
        self.order = None
        self.forecast = None
        self.forecast_dates = None
        self.train_data = None
        self.test_data = None
        self.residuals = None
        self.rmse = None
        self.mae = None
        self.aic = None
        self.bic = None

def determine_arima_order(series, max_p=3, max_d=2, max_q=3):
    """
    Determine the best ARIMA order based on AIC.
    
    Args:
        series (pd.Series): Time series data
        max_p (int): Maximum AR order
        max_d (int): Maximum differencing order
        max_q (int): Maximum MA order
        
    Returns:
        tuple: Best (p,d,q) order
    """
    print("\nFinding optimal ARIMA parameters...")
    
    # Check if we have enough data
    if len(series) < 10:
        print("Not enough data for automatic order determination. Using default order (1,1,0).")
        return (1, 1, 0)
    
    # If series is price, use differencing
    if series.mean() > 1:  # Heuristic to detect price vs returns
        d_range = range(1, max_d + 1)
    else:  # For returns, we might already have stationarity
        d_range = range(0, max_d + 1)
    
    best_aic = float('inf')
    best_order = None
    
    # We'll only try a subset of combinations to speed up the process
    p_range = range(0, max_p + 1)
    q_range = range(0, max_q + 1)
    
    # Only try some combinations to save time
    pdq_combinations = list(itertools.product(p_range, d_range, q_range))
    
    # If there are too many combinations, sample a subset
    if len(pdq_combinations) > 20:
        import random
        random.seed(42)
        pdq_combinations = random.sample(pdq_combinations, 20)
    
    for order in pdq_combinations:
        try:
            model = ARIMA(series, order=order)
            results = model.fit()
            
            # If the model is better (lower AIC), save it
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                
            print(f"ARIMA{order} - AIC: {results.aic:.2f}")
        except:
            continue
    
    if best_order is None:
        # If no models were successfully fit, use a default order
        best_order = (1, 1, 0)
        print(f"\nFailed to fit ARIMA models, using default order: {best_order}")
    else:
        print(f"\nBest ARIMA order: {best_order} with AIC: {best_aic:.2f}")
    
    return best_order

def run_arima(df, forecast_periods=30, test_size=0.2):
    """
    Run ARIMA model on stock data.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with stock data
        forecast_periods (int): Number of periods to forecast
        test_size (float): Proportion of data to use for testing
        
    Returns:
        ARIMAResults: Object containing model and forecasts
    """
    print("\n" + "="*50)
    print("ARIMA MODELING")
    print("="*50)
    
    # Create result object
    results = ARIMAResults()
    
    # Get the series to model (we'll use Close price)
    series = df['Close'].copy()
    
    # Ensure we have enough data
    if len(series) < 10:
        error_msg = "Not enough data points for ARIMA modeling (minimum 10 required)"
        print(f"ERROR: {error_msg}")
        
        # Create mock results for very small datasets to allow visualization
        results.train_data = series
        results.test_data = pd.Series()
        results.order = (1, 1, 0)
        results.forecast = pd.Series(data=[series.iloc[-1]] * forecast_periods)
        results.forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_periods)
        results.residuals = pd.Series(data=[0] * len(series), index=series.index)
        results.rmse = np.nan
        results.mae = np.nan
        results.aic = np.nan
        results.bic = np.nan
        
        print("WARNING: Using simplified model due to insufficient data")
        return results
    
    # Split into train and test sets
    train_size = int(len(series) * (1 - test_size))
    # Ensure at least 8 points in train set
    train_size = max(train_size, min(8, len(series)-2))
    train_data = series[:train_size]
    test_data = series[train_size:] if train_size < len(series) else pd.Series(dtype=float)
    
    print(f"Training data: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} records)")
    if len(test_data) > 0:
        print(f"Testing data: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} records)")
    else:
        print("No testing data available (using all data for training)")
    
    # Store in results
    results.train_data = train_data
    results.test_data = test_data
    
    # Check if 'Returns' has valid data before plotting ACF/PACF
    if 'Returns' in df.columns and len(df['Returns'].dropna()) > 10:
        try:
            # Plot ACF and PACF to help identify potential orders
            plt.figure(figsize=(12, 6))
            
            plt.subplot(121)
            returns_data = df['Returns'].dropna()
            plot_acf(returns_data, lags=min(20, len(returns_data) - 1), ax=plt.gca(), title='ACF of Returns')
            
            plt.subplot(122)
            plot_pacf(returns_data, lags=min(20, len(returns_data) - 1), ax=plt.gca(), title='PACF of Returns')
            
            plt.tight_layout()
            plt.savefig('acf_pacf_plot.png')  # Save to file instead of showing
            plt.close()
        except Exception as e:
            print(f"Error plotting ACF/PACF: {e}")
            print("Continuing with model fitting...")
    else:
        print("Not enough returns data for ACF/PACF plots. Skipping.")
    
    # Determine best order
    best_order = determine_arima_order(train_data)
    results.order = best_order
    
    # Fit the ARIMA model
    print("\nFitting ARIMA model...")
    try:
        model = ARIMA(train_data, order=best_order)
        fitted_model = model.fit()
        
        print("\nARIMA Model Summary:")
        print(fitted_model.summary().tables[0].as_text())
        print(fitted_model.summary().tables[1].as_text())
        
        # Store model and diagnostics
        results.model = fitted_model
        results.residuals = fitted_model.resid
        results.aic = fitted_model.aic
        results.bic = fitted_model.bic
        
        # In-sample prediction
        in_sample_pred = fitted_model.predict(dynamic=False)
        
        # Forecast on test data
        if len(test_data) > 0:
            forecasts = fitted_model.forecast(steps=len(test_data))
            
            # Calculate error metrics
            results.rmse = np.sqrt(mean_squared_error(test_data, forecasts))
            results.mae = mean_absolute_error(test_data, forecasts)
            
            print(f"\nTest set metrics:")
            print(f"RMSE: {results.rmse:.2f}")
            print(f"MAE: {results.mae:.2f}")
        
        # Generate future forecasts
        print(f"\nGenerating {forecast_periods} period forecast...")
        
        # Refit the model on the entire dataset for future forecasting
        full_model = ARIMA(series, order=best_order)
        full_fitted = full_model.fit()
        
        # Generate forecasts
        future_forecast = full_fitted.forecast(steps=forecast_periods)
        
        # Generate forecast dates
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods)
        
        # Store forecasts
        results.forecast = future_forecast
        results.forecast_dates = forecast_dates
        
    except Exception as e:
        print(f"Error in ARIMA modeling: {e}")
        # Create simple forecast as fallback
        results.forecast = pd.Series(data=[series.iloc[-1]] * forecast_periods)
        results.forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_periods)
        if not hasattr(results, 'residuals') or results.residuals is None:
            results.residuals = pd.Series(data=[0] * len(series), index=series.index)
    
    print("ARIMA modeling completed.")
    return results