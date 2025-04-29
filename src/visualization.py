# visualization.py (updated)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.dates as mdates

def visualize_data(df, show_plots=True):
    """
    Visualize the stock market data with key indicators.
    
    Args:
        df (pd.DataFrame): Processed DataFrame with stock data
        show_plots (bool): If True, display plots instead of saving to file
    """
    try:
        # Check if we have enough data for visualization
        if len(df) < 2:
            print("Not enough data points for visualization (minimum 2 required)")
            return
            
        plt.figure(figsize=(14, 12))
        
        # Plot 1: Price and Moving Averages
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        
        # Only add MA lines if we have enough data
        if 'MA_7' in df.columns and df['MA_7'].notna().sum() > 0:
            plt.plot(df.index, df['MA_7'], label='7-day MA')
        if 'MA_30' in df.columns and df['MA_30'].notna().sum() > 0:
            plt.plot(df.index, df['MA_30'], label='30-day MA')
            
        plt.title('Stock Price and Moving Averages')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Plot 2: Daily Returns (if available)
        plt.subplot(3, 1, 2)
        if 'Returns' in df.columns and df['Returns'].notna().sum() > 0:
            plt.plot(df.index, df['Returns'], color='blue', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Daily Returns (%)')
        else:
            plt.text(0.5, 0.5, 'Insufficient data for Returns plot', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
            plt.title('Daily Returns (%) - Not Available')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Plot 3: Volatility (if available)
        plt.subplot(3, 1, 3)
        if 'Volatility_30' in df.columns and df['Volatility_30'].notna().sum() > 0:
            plt.plot(df.index, df['Volatility_30'], color='green')
            plt.title('30-Day Rolling Volatility')
        else:
            plt.text(0.5, 0.5, 'Insufficient data for Volatility plot', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
            plt.title('30-Day Rolling Volatility - Not Available')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.savefig('stock_analysis_overview.png')
        plt.close()
        
        # Additional volume plot (if available)
        if 'Volume' in df.columns and df['Volume'].notna().sum() > 0:
            plt.figure(figsize=(14, 6))
            plt.bar(df.index, df['Volume'], color='purple', alpha=0.6)
            plt.title('Trading Volume')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis for dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            
            if show_plots:
                plt.show()
            else:
                plt.savefig('volume_chart.png')
            plt.close()
        else:
            print("Volume data not available or insufficient, skipping volume plot")
        
        # Correlation heatmap
        try:
            plt.figure(figsize=(10, 8))
            correlation_columns = ['Close']
            
            # Only include columns with sufficient data
            for col in ['Returns', 'Log_Returns', 'Volatility_30', 'Volume']:
                if col in df.columns and df[col].notna().sum() > 0:
                    correlation_columns.append(col)
            
            if len(correlation_columns) > 1:  # Need at least 2 columns for correlation
                correlation = df[correlation_columns].corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                if show_plots:
                    plt.show()
                else:
                    plt.savefig('correlation_matrix.png')
            else:
                print("Not enough data for correlation matrix")
            
            plt.close()
        except Exception as e:
            print(f"Error creating correlation matrix: {e}")
        
        if show_plots:
            print("Data visualizations displayed")
        else:
            print("Data visualizations saved to current directory")
            print("Files: stock_analysis_overview.png, volume_chart.png, correlation_matrix.png")
        
    except Exception as e:
        print(f"Error in data visualization: {e}")
        import traceback
        traceback.print_exc()


def visualize_arima_results(arima_results, show_plots=True):
    """
    Visualize ARIMA model results and forecasts.
    
    Args:
        arima_results: ARIMAResults object
        show_plots (bool): If True, display plots instead of saving to file
    """
    try:
        # Check if we have the necessary attributes
        if not hasattr(arima_results, 'train_data') or arima_results.train_data is None:
            print("No training data available for ARIMA visualization")
            return
            
        if not hasattr(arima_results, 'forecast') or arima_results.forecast is None:
            print("No forecast data available for ARIMA visualization")
            return
            
        # Plot 1: Historical data and forecast
        plt.figure(figsize=(12, 8))
        
        # Historical data
        plt.plot(arima_results.train_data.index, arima_results.train_data, 
                color='blue', label='Training Data')
        
        # Test data if available
        if hasattr(arima_results, 'test_data') and arima_results.test_data is not None and len(arima_results.test_data) > 0:
            plt.plot(arima_results.test_data.index, arima_results.test_data, 
                    color='green', label='Test Data')
            
            # Forecast on test period if model is available
            if hasattr(arima_results, 'model') and arima_results.model is not None:
                try:
                    test_forecast = arima_results.model.forecast(steps=len(arima_results.test_data))
                    plt.plot(arima_results.test_data.index, test_forecast, 
                            color='red', linestyle='--', label='Test Forecast')
                except:
                    pass
        
        # Future forecast
        plt.plot(arima_results.forecast_dates, arima_results.forecast, 
                color='red', label=f'ARIMA{arima_results.order} Forecast')
        
        # Confidence intervals if we have residuals
        if hasattr(arima_results, 'model') and arima_results.model is not None and hasattr(arima_results, 'residuals'):
            try:
                std_err = np.std(arima_results.residuals) * np.sqrt(np.arange(1, len(arima_results.forecast) + 1))
                plt.fill_between(arima_results.forecast_dates,
                                arima_results.forecast - 1.96 * std_err,
                                arima_results.forecast + 1.96 * std_err,
                                color='pink', alpha=0.3, label='95% Confidence Interval')
            except Exception as e:
                print(f"Error calculating confidence intervals: {e}")
                # If error calculation fails, use a simple approximation
                std_err = arima_results.forecast.mean() * 0.1 * np.sqrt(np.arange(1, len(arima_results.forecast) + 1))
                plt.fill_between(arima_results.forecast_dates,
                                arima_results.forecast - 1.96 * std_err,
                                arima_results.forecast + 1.96 * std_err,
                                color='pink', alpha=0.3, label='Approximated Confidence Interval')
        
        plt.title('ARIMA Model Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.savefig('arima_forecast.png')
        plt.close()
        
        # Only show ARIMA diagnostics if we have enough data
        if (hasattr(arima_results, 'residuals') and 
            arima_results.residuals is not None and 
            len(arima_results.residuals) >= 8):
            
            plt.figure(figsize=(12, 10))
            
            # Residuals
            plt.subplot(3, 2, 1)
            plt.plot(arima_results.residuals.index, arima_results.residuals)
            plt.title('Residuals')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis for dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            
            # Residual histogram
            plt.subplot(3, 2, 2)
            sns.histplot(arima_results.residuals, kde=True, bins=min(30, len(arima_results.residuals)//2 + 1))
            plt.title('Residual Distribution')
            
            # ACF of residuals
            plt.subplot(3, 2, 3)
            try:
                plot_acf(arima_results.residuals, lags=min(20, len(arima_results.residuals) - 1), ax=plt.gca())
                plt.title('ACF of Residuals')
            except Exception as e:
                print(f"Error plotting ACF of residuals: {e}")
                plt.title('ACF of Residuals (Error)')
            
            # Q-Q plot
            plt.subplot(3, 2, 4)
            try:
                from scipy import stats
                stats.probplot(arima_results.residuals, dist="norm", plot=plt)
                plt.title('Q-Q plot')
            except Exception as e:
                print(f"Error creating Q-Q plot: {e}")
                plt.title('Q-Q Plot (Error)')
            
            # Residuals scatter if we have fitted values
            plt.subplot(3, 2, 5)
            if hasattr(arima_results, 'model') and arima_results.model is not None:
                try:
                    plt.scatter(arima_results.model.fittedvalues, arima_results.residuals)
                    plt.axhline(y=0, color='r', linestyle='-')
                    plt.title('Residuals vs Fitted')
                    plt.xlabel('Fitted values')
                    plt.ylabel('Residuals')
                except Exception as e:
                    print(f"Error plotting residuals vs fitted: {e}")
                    plt.title('Residuals vs Fitted (Error)')
            else:
                plt.text(0.5, 0.5, 'Model fit data not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
                plt.title('Residuals vs Fitted - Not Available')
            
            plt.tight_layout()
            if show_plots:
                plt.show()
            else:
                plt.savefig('arima_diagnostics.png')
            plt.close()
        else:
            print("Not enough data for ARIMA diagnostic plots")
        
        if show_plots:
            print("ARIMA visualizations displayed")
        else:
            print("ARIMA visualization saved to current directory")
            print("Files: arima_forecast.png, arima_diagnostics.png")
        
    except Exception as e:
        print(f"Error in ARIMA visualization: {e}")
        import traceback
        traceback.print_exc()


def visualize_garch_forecast(train_data, cond_vol, forecast_dates, forecast_vol, garch_results, show_plots=True):
    """
    Visualizes GARCH volatility forecast and diagnostic plots.
    
    Args:
        train_data (pd.Series): Training data series
        cond_vol (np.array): Conditional volatility array
        forecast_dates (pd.DatetimeIndex): Dates for forecast period
        forecast_vol (np.array): Volatility forecasts
        garch_results: GARCHResults object containing model results
        show_plots (bool): If True, display plots instead of saving to file
    """
    try:
        plt.figure(figsize=(12, 8))

        # Check if we have enough historical data
        last_days = min(30, len(train_data))
        if last_days > 0:
            historical_dates = train_data.index[-last_days:]
            historical_vol = cond_vol[-last_days:] if len(cond_vol) >= last_days else cond_vol
            plt.plot(historical_dates, historical_vol, color='blue', label='Historical Volatility')
        
        # Plot forecast if available
        if forecast_dates is not None and forecast_vol is not None and len(forecast_vol) > 0:
            plt.plot(forecast_dates, forecast_vol, color='red', label='Volatility Forecast')
            
            # Add confidence interval if we have enough data
            if len(cond_vol) > 0:
                std_err = np.std(cond_vol) * np.sqrt(np.arange(1, len(forecast_vol) + 1)) * 0.5
                plt.fill_between(forecast_dates,
                                forecast_vol - 1.96 * std_err,
                                forecast_vol + 1.96 * std_err,
                                color='pink', alpha=0.3)

        plt.title('GARCH Volatility Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.savefig('garch_forecast.png')
        plt.close()

        # GARCH Diagnostics if we have residuals
        if hasattr(garch_results, 'residuals') and garch_results.residuals is not None and len(garch_results.residuals) >= 8:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.plot(garch_results.residuals)
            plt.title('Standardized Residuals')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 2)
            sns.histplot(garch_results.residuals, kde=True, bins=min(30, len(garch_results.residuals)//2 + 1))
            plt.title('Residuals Distribution')

            plt.subplot(2, 2, 3)
            try:
                squared_resid = garch_results.residuals**2
                plot_acf(squared_resid, lags=min(20, len(squared_resid) - 1), ax=plt.gca())
                plt.title('ACF of Squared Residuals')
            except Exception as e:
                print(f"Error plotting ACF of squared residuals: {e}")
                plt.title('ACF of Squared Residuals (Error)')

            plt.subplot(2, 2, 4)
            plt.scatter(range(len(garch_results.residuals)), garch_results.residuals**2)
            plt.title('Squared Residuals')

            plt.tight_layout()
            if show_plots:
                plt.show()
            else:
                plt.savefig('garch_diagnostics.png')
            plt.close()
        else:
            print("Not enough data for GARCH diagnostic plots")

        if show_plots:
            print("GARCH visualizations displayed")
        else:
            print("GARCH visualization saved to current directory")
            print("Files: garch_forecast.png, garch_diagnostics.png")
        
    except Exception as e:
        print(f"Error in GARCH visualization: {e}")
        import traceback
        traceback.print_exc()


def visualize_comparison(arima_results, garch_results, df, show_plots=True):
    """
    Compare ARIMA and GARCH model forecasts visually.
    
    Args:
        arima_results: ARIMAResults object
        garch_results: GARCHResults object
        df (pd.DataFrame): Processed DataFrame with stock data
        show_plots (bool): If True, display plots instead of saving to file
    """
    try:
        plt.figure(figsize=(12, 10))

        # Price Forecast
        plt.subplot(2, 1, 1)
        
        # Check if we have the necessary data
        if (hasattr(arima_results, 'train_data') and arima_results.train_data is not None):
            
            # Plot historical price
            plt.plot(arima_results.train_data.index, arima_results.train_data, label='Historical Price')
            
            # Add test data if available
            if hasattr(arima_results, 'test_data') and arima_results.test_data is not None:
                all_data = pd.concat([arima_results.train_data, arima_results.test_data])
                plt.plot(all_data.index, all_data, label='All Historical Price')
            
            # Plot forecast if available
            if hasattr(arima_results, 'forecast') and arima_results.forecast is not None:
                plt.plot(arima_results.forecast_dates, arima_results.forecast,
                        color='red', label=f'ARIMA{arima_results.order} Forecast')
                
                # Add confidence interval if we have residuals
                if hasattr(arima_results, 'model') and arima_results.model is not None and hasattr(arima_results, 'residuals'):
                    try:
                        std_err = np.std(arima_results.residuals) * np.sqrt(np.arange(1, len(arima_results.forecast) + 1))
                        plt.fill_between(arima_results.forecast_dates,
                                        arima_results.forecast - 1.96 * std_err,
                                        arima_results.forecast + 1.96 * std_err,
                                        color='pink', alpha=0.3)
                    except Exception as e:
                        print(f"Error calculating ARIMA confidence intervals: {e}")
        else:
            plt.text(0.5, 0.5, 'Insufficient data for Price Forecast comparison', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)

        plt.title('Stock Price Forecast (ARIMA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()

        # Volatility Forecast
        plt.subplot(2, 1, 2)
        
        # Plot returns if available
        if 'Log_Returns' in df.columns and df['Log_Returns'].notna().sum() > 0:
            returns = df['Log_Returns']
            plt.plot(returns.index, returns, color='gray', alpha=0.5, label='Historical Returns')

        # Plot historical volatility if available
        if 'Volatility_30' in df.columns and df['Volatility_30'].notna().sum() > 0:
            hist_vol = df['Volatility_30'].dropna()
            plt.plot(hist_vol.index, hist_vol, color='blue', label='30-day Rolling Volatility')

        # Plot GARCH forecast if available
        if (hasattr(garch_results, 'forecast_dates') and garch_results.forecast_dates is not None and
            hasattr(garch_results, 'forecast_volatility') and garch_results.forecast_volatility is not None):
            
            plt.plot(garch_results.forecast_dates, garch_results.forecast_volatility,
                    color='red', label='GARCH Volatility Forecast')
            
            # Add confidence interval
            if hasattr(garch_results, 'conditional_volatility') and garch_results.conditional_volatility is not None:
                cond_vol = garch_results.conditional_volatility
                if len(cond_vol) > 0:
                    std_err = np.std(cond_vol) * np.sqrt(np.arange(1, len(garch_results.forecast_volatility) + 1)) * 0.5
                    plt.fill_between(garch_results.forecast_dates,
                                    garch_results.forecast_volatility - 1.96 * std_err,
                                    garch_results.forecast_volatility + 1.96 * std_err,
                                    color='pink', alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Insufficient data for Volatility Forecast comparison', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)

        plt.title('Volatility Forecast (GARCH)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.savefig('forecast_comparison.png')
        plt.close()
        
        if show_plots:
            print("Forecast comparison visualization displayed")
        else:
            print("Forecast comparison visualization saved to current directory")
            print("File: forecast_comparison.png")
        
    except Exception as e:
        print(f"Error in forecast comparison visualization: {e}")
        import traceback
        traceback.print_exc()