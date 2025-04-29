import pandas as pd
import os
import numpy as np

def load_data():
    """
    Load stock data from a specific hardcoded path.
    Now with better error handling and format detection.

    Returns:
        pd.DataFrame: DataFrame containing the loaded stock data
    """
    try:
        # Hardcoded dataset path - using a relative path is better for portability
        dataset_path = "./data/Sample StockMarket Data.csv"
        
        # For testing - fallback to absolute path if needed
        if not os.path.exists(dataset_path):
            dataset_path = "C:\\Users\\SUPERSTAR\\OneDrive\\Desktop\\FE Final project\\data\\Cleaned_StockMarket_Data (1).csv"
            
        if not os.path.exists(dataset_path):
            print(f"File not found: {dataset_path}")
            print("Creating sample data for testing...")
            return create_sample_data()
            
        # Load the dataset
        print(f"Loading data from: {dataset_path}")
        df = pd.read_csv("C:\\Users\\SUPERSTAR\\OneDrive\\Desktop\\FE Final project\\data\\Cleaned_StockMarket_Data (1).csv")
        
        # Check if required columns exist
        if 'Close' not in df.columns:
            print("Error: CSV file must contain a 'Close' column")
            return create_sample_data()  # Create sample data as fallback
        
        # Convert Date to datetime with flexible format detection
        if 'Date' in df.columns:
            try:
                # Try different date formats (DD-MM-YYYY or YYYY-MM-DD)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
                
                # If many NaT values, try alternate format
                if df['Date'].isna().sum() > len(df) / 2:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)
                    
                print(f"Date format detected and parsed. Sample: {df['Date'].iloc[0]}")
            except Exception as e:
                print(f"Date parsing error: {e}")
                return create_sample_data()
        else:
            print("Error: CSV file must contain a 'Date' column")
            return create_sample_data()
        
        # Handle combined date and time if needed
        if 'Time' in df.columns:
            print("Time column detected. Combining with Date...")
            try:
                # Convert Time to string to handle various formats
                df['Time'] = df['Time'].astype(str)
                
                # Try to combine Date and Time
                df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'], 
                                                errors='coerce')
                
                # Set DateTime as index
                df.set_index('DateTime', inplace=True)
            except Exception as e:
                print(f"Error combining date and time: {e}")
                # Fallback to just using Date
                df.set_index('Date', inplace=True)
        else:
            # Set Date as index
            df.set_index('Date', inplace=True)
        
        # Sort the data by index
        df = df.sort_index()
        
        # Handle ticker information if present
        if 'Ticker' in df.columns:
            tickers = df['Ticker'].unique()
            if len(tickers) > 1:
                print(f"Multiple tickers detected: {tickers}")
                # Ask user to select or use the first ticker
                selected_ticker = tickers[0]
                print(f"Using ticker: {selected_ticker}")
                df = df[df['Ticker'] == selected_ticker].copy()
                df.drop('Ticker', axis=1, inplace=True)
            else:
                print(f"Single ticker detected: {tickers[0]}")
                df.drop('Ticker', axis=1, inplace=True)
        
        # Ensure numeric data types for price columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for too many missing values
        missing_ratio = df['Close'].isna().mean()
        if missing_ratio > 0.3:  # If more than 30% missing
            print(f"Warning: High ratio of missing values in Close column: {missing_ratio:.2%}")
            print("Creating sample data as fallback...")
            return create_sample_data()
        
        # Fill missing values
        for col in df.columns:
            if df[col].isna().any():
                print(f"Filling {df[col].isna().sum()} missing values in {col}")
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Print information about the dataset
        print(f"\nData loaded successfully! Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print("\nPreview of the data:")
        print(df.head())
        
        # Ensure we have enough data
        if len(df) < 20:  # Need minimum data for modeling
            print(f"Warning: Dataset has only {len(df)} records, which is not enough for reliable analysis.")
            if len(df) < 10:
                print("Creating sample data as fallback...")
                return create_sample_data()
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data as fallback...")
        return create_sample_data()

def create_sample_data():
    """
    Create synthetic stock data for testing when real data isn't available.
    """
    print("Generating synthetic data for analysis...")
    
    # Create date range for the past year with business days
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=2)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Initialize with a starting price
    start_price = 100.0
    
    # Generate random price movements with some trend and volatility
    np.random.seed(42)  # For reproducibility
    price_changes = np.random.normal(0.0005, 0.012, len(dates))  # Mean positive drift
    
    # Add some autocorrelation and volatility clustering
    for i in range(3, len(price_changes)):
        price_changes[i] += 0.2 * price_changes[i-1]  # Autocorrelation
        if abs(price_changes[i-1]) > 0.02:  # Volatility clustering
            price_changes[i] *= 1.2
    
    # Calculate price series
    prices = start_price * (1 + price_changes).cumprod()
    
    # Create high, low, open prices based on close
    daily_volatility = np.random.uniform(0.005, 0.02, len(dates))
    high_prices = prices * (1 + daily_volatility)
    low_prices = prices * (1 - daily_volatility)
    open_prices = low_prices + np.random.random(len(dates)) * (high_prices - low_prices)
    
    # Generate volume - higher on volatile days
    volume = np.random.normal(1000000, 300000, len(dates))
    volume = volume * (1 + 5 * np.abs(price_changes))  # More volume on bigger moves
    volume = np.round(volume).astype(int)
    
    # Create dataframe
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': prices,
        'Volume': volume
    }, index=dates)
    
    print(f"Sample data created with {len(df)} trading days")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print("\nPreview of sample data:")
    print(df.head())
    
    return df