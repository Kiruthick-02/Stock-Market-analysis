import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats

def preprocess_data(df):
    """
    Preprocess stock market data for time series analysis.
    
    Args:
        df (pd.DataFrame): Input DataFrame with stock data
        
    Returns:
        pd.DataFrame: Processed DataFrame ready for modeling
    """
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Check if we have enough data for analysis
    if len(processed_df) < 20:
        print(f"Warning: Limited data available ({len(processed_df)} records). Results may not be reliable.")
    
    # Ensure we have financial data
    required_cols = ['Close']
    optional_cols = ['Open', 'High', 'Low', 'Volume']
    
    for col in required_cols:
        if col not in processed_df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")
    
    # Resample to daily data if index has multiple entries per day
    if isinstance(processed_df.index, pd.DatetimeIndex):
        # Check if we have intraday data (multiple entries per day)
        dates = processed_df.index.date
        if len(dates) != len(set(dates)):
            print("Multiple entries per day detected. Resampling to daily data...")
            # Group by date and use OHLC aggregation
            processed_df = processed_df.groupby(processed_df.index.date).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            # Convert index back to datetime
            processed_df.index = pd.to_datetime(processed_df.index)
    
    # Convert any string columns to numeric
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                print(f"Converted column '{col}' to numeric")
            except:
                print(f"Could not convert column '{col}' to numeric")
    
    # Handle missing values
    for col in processed_df.columns:
        missing_count = processed_df[col].isnull().sum()
        if missing_count > 0:
            print(f"Filling {missing_count} missing values in '{col}' column")
            # Forward fill then backward fill to handle edges
            processed_df[col] = processed_df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Make sure data is sorted by date
    if isinstance(processed_df.index, pd.DatetimeIndex):
        processed_df = processed_df.sort_index()
    
    # Calculate daily returns with handling for zero or negative values
    processed_df['Returns'] = processed_df['Close'].pct_change() * 100
    
    # Calculate log returns with safety checks
    processed_df['Log_Returns'] = np.log(processed_df['Close'] / processed_df['Close'].shift(1).replace(0, np.nan)) * 100
    
    # Replace inf values that might occur
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Add some technical indicators with appropriate window sizes based on data length
    # Use smaller windows if we have limited data
    min_window = min(7, max(3, len(processed_df) // 10))
    med_window = min(30, max(5, len(processed_df) // 5))
    
    processed_df[f'MA_{min_window}'] = processed_df['Close'].rolling(window=min_window).mean()
    processed_df[f'MA_{med_window}'] = processed_df['Close'].rolling(window=med_window).mean()
    processed_df[f'Volatility_{min_window}'] = processed_df['Returns'].rolling(window=min_window).std()
    
    # Add renamed columns with standard names for compatibility
    processed_df['MA_7'] = processed_df[f'MA_{min_window}']
    processed_df['MA_30'] = processed_df[f'MA_{med_window}']
    processed_df['Volatility_30'] = processed_df['Returns'].rolling(window=med_window).std()
    
    # Check for outliers in returns
    if len(processed_df['Returns'].dropna()) > 5:  # Need at least a few points
        z_scores = stats.zscore(processed_df['Returns'].dropna())
        outliers = (abs(z_scores) > 3).sum()
        if outliers > 0:
            print(f"Detected {outliers} potential outliers in returns (|z-score| > 3)")
            # Could handle outliers here if desired
    
    # Perform stationarity test with error handling
    if len(processed_df) >= 10:
        try:
            check_stationarity(processed_df['Close'].dropna(), 'Close Price')
        except Exception as e:
            print(f"Stationarity test for Close Price failed: {e}")
        
        try:
            if len(processed_df['Returns'].dropna()) >= 10:
                check_stationarity(processed_df['Returns'].dropna(), 'Returns')
        except Exception as e:
            print(f"Stationarity test for Returns failed: {e}")
    else:
        print("Not enough data points for stationarity test (minimum 10 required)")
    
    # Keep original data for the first few rows
    processed_df_final = processed_df.copy()
    
    # Remove NaN values for modeling but keep at least 10 rows
    processed_df_no_nan = processed_df.dropna()
    if len(processed_df_no_nan) >= 10:
        processed_df_final = processed_df_no_nan
    else:
        # If dropping NaN leaves too few rows, fill them instead
        print(f"Warning: Dropping NaN values would leave only {len(processed_df_no_nan)} rows.")
        print("Using data with filled NaN values instead.")
        # We already filled NaNs earlier
    
    print("\nPreprocessed data summary:")
    print(processed_df_final.describe())
    
    # Print the number of records that will be used for modeling
    print(f"\nFinal dataset has {len(processed_df_final)} records after preprocessing")
    
    return processed_df_final

def check_stationarity(series, title):
    """
    Check stationarity of a time series using Augmented Dickey-Fuller test.
    Enhanced with better error handling and data validation.
    
    Args:
        series (pd.Series): Time series to test
        title (str): Title for the series being tested
    """
    print(f"\nStationarity Check for {title}:")
    
    # Ensure we have enough data
    if len(series) < 10:
        print(f"Not enough data points for {title} stationarity test (minimum 10 required)")
        return
    
    # Clean the series, drop any NA values
    clean_series = series.dropna()
    
    # Check if series has enough variation
    if clean_series.min() == clean_series.max():
        print(f"No variation in {title} series, stationarity test skipped")
        return
    
    try:
        # Perform ADF test
        result = adfuller(clean_series, maxlag=int(np.sqrt(len(clean_series))))
        
        # Extract and display results
        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        print(f"ADF Statistic: {adf_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print("Critical Values:")
        for key, value in critical_values.items():
            print(f"  {key}: {value:.4f}")
        
        # Interpretation
        if p_value < 0.05:
            print(f"Result: The {title} series is stationary (reject H0)")
        else:
            print(f"Result: The {title} series is non-stationary (fail to reject H0)")
            if title == 'Close Price':
                print("Note: Non-stationarity in price series is expected. Differencing or returns transformation recommended.")
    
    except Exception as e:
        print(f"Error in stationarity test for {title}: {e}")
        print("Stationarity test skipped. This does not affect other analyses.")