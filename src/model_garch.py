import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

class GARCHResults:
    """Class to store GARCH model results."""
    def __init__(self):
        self.model = None
        self.order = None
        self.train_data = None
        self.test_data = None
        self.forecast_dates = None
        self.forecast_volatility = None
        self.conditional_volatility = None
        self.residuals = None
        self.rmse = None
        self.mae = None
        self.aic = None
        self.bic = None

def determine_garch_order(returns, max_p=2, max_q=2):
    """
    Determine the best GARCH order based on AIC.
    
    Args:
        returns (pd.Series): Return series
        max_p (int): Maximum ARCH order
        max_q (int): Maximum GARCH order
        
    Returns:
        tuple: Best (p, q) order
    """
    print("\nFinding optimal GARCH parameters...")
    
    best_aic = float('inf')
    best_order = None
    
    # Try different GARCH orders
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                # Create and fit GARCH model
                model = arch_model(returns, p=p, q=q, mean='Zero', vol='GARCH', dist='normal')
                results = model.fit(disp='off', show_warning=False)
                
                # If the model is better (lower AIC), save it
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, q)
                    
                print(f"GARCH({p},{q}) - AIC: {results.aic:.2f}")
            except Exception as e:
                # If fitting fails, just skip this order
                print(f"GARCH({p},{q}) - Failed to fit: {str(e)}")
                continue
    
    if best_order is None:
        # If no models were successfully fit, use a default order
        best_order = (1, 1)
        print(f"\nFailed to fit GARCH models, using default order: {best_order}")
    else:
        print(f"\nBest GARCH order: {best_order} with AIC: {best_aic:.2f}")
    
    return best_order

def run_garch(df, forecast_periods=30, test_size=0.2):
    """
    Run GARCH model on stock market returns.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with stock data
        forecast_periods (int): Number of periods to forecast
        test_size (float): Proportion of data to use for testing
        
    Returns:
        GARCHResults: Object containing model and forecasts
    """
    print("\n" + "="*50)
    print("GARCH MODELING FOR VOLATILITY")
    print("="*50)
    
    # Create result object
    results = GARCHResults()
    
    # Create return series - we'll try multiple options to ensure we have valid data
    if 'Log_Returns' in df.columns and not df['Log_Returns'].isna().all():
        returns = df['Log_Returns'].dropna()
    elif 'Returns' in df.columns and not df['Returns'].isna().all():
        returns = df['Returns'].dropna()
    else:
        # Calculate returns if not already present or all NaN
        if 'Close' in df.columns:
            returns = 100 * df['Close'].pct_change().dropna()
        else:
            raise ValueError("Required 'Close' column not found in dataset")
    
    # Ensure we have enough data
    if len(returns) < 10:
        # Try to generate some minimal sample data
        print("Not enough return data points for GARCH modeling (minimum 10 required)")
        print("Creating synthetic return data for demonstration...")
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 1, 100), 
                           index=pd.date_range(end=pd.Timestamp.now(), periods=100, freq='B'))
    
    # Split into train and test sets
    train_size = int(len(returns) * (1 - test_size))
    train_data = returns[:train_size]
    test_data = returns[train_size:]
    
    print(f"Training data: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} records)")
    if len(test_data) > 0:
        print(f"Testing data: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} records)")
    else:
        print("No testing data available")
    
    # Store in results
    results.train_data = train_data
    results.test_data = test_data
    
    # Determine best GARCH order
    best_order = determine_garch_order(train_data)
    results.order = best_order
    
    # Fit the GARCH model
    print("\nFitting GARCH model...")
    try:
        model = arch_model(train_data, p=best_order[0], q=best_order[1], mean='Zero', vol='GARCH', dist='normal')
        fitted_model = model.fit(disp='off', show_warning=False)
        
        print("\nGARCH Model Summary:")
        print(fitted_model.summary().tables[0].as_text())
        
        # Store model and diagnostics
        results.model = fitted_model
        results.conditional_volatility = fitted_model.conditional_volatility
        results.residuals = fitted_model.resid
        results.aic = fitted_model.aic
        results.bic = fitted_model.bic
        
        # Forecast on test data
        if len(test_data) > 0:
            forecasts = fitted_model.forecast(horizon=len(test_data))
            forecast_vol = np.sqrt(forecasts.variance.values[-1, :])
            
            # Calculate error metrics
            results.rmse = np.sqrt(mean_squared_error(test_data.values**2, forecast_vol**2))
            results.mae = mean_absolute_error(test_data.values**2, forecast_vol**2)
            
            print(f"\nTest set metrics for squared returns:")
            print(f"RMSE: {results.rmse:.6f}")
            print(f"MAE: {results.mae:.6f}")
        
        # Generate future forecasts
        print(f"\nGenerating {forecast_periods} period forecast...")
        
        # Refit the model on the entire dataset for future forecasting
        full_model = arch_model(returns, p=best_order[0], q=best_order[1], mean='Zero', vol='GARCH', dist='normal')
        full_fitted = full_model.fit(disp='off', show_warning=False)
        
        # Generate forecasts
        future_forecast = full_fitted.forecast(horizon=forecast_periods)
        forecast_volatility = np.sqrt(future_forecast.variance.values[-1, :])
        
        # Generate forecast dates
        last_date = returns.index[-1]
        # Use business day frequency for forecast dates
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods, freq='B')
        
        # Store forecasts
        results.forecast_volatility = forecast_volatility
        results.forecast_dates = forecast_dates
        
        print("GARCH modeling completed.")
        
    except Exception as e:
        print(f"Error in GARCH modeling: {e}")
        # Create dummy forecasts for demonstration
        print("Creating dummy forecasts for demonstration...")
        
        # Generate dummy forecast dates
        last_date = returns.index[-1] if len(returns) > 0 else pd.Timestamp.now()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods, freq='B')
        
        # Generate dummy volatility forecast (just a constant value)
        vol_mean = returns.std() if len(returns) > 0 else 1.0
        forecast_volatility = np.ones(forecast_periods) * vol_mean
        
        # Store dummy forecasts
        results.forecast_volatility = forecast_volatility
        results.forecast_dates = forecast_dates
        results.conditional_volatility = np.ones(len(train_data)) * vol_mean
        results.order = (1, 1)
    
    return results