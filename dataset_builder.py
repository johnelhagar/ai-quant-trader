import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
from tqdm import tqdm

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

warnings.filterwarnings('ignore')

def process_history(ticker_dir):
    hist_path = os.path.join(ticker_dir, 'history.csv')
    if not os.path.exists(hist_path):
        return None
    try:
        df = pd.read_csv(hist_path)
    except Exception:
        return None
        
    if df.empty or 'Date' not in df.columns:
        return None
        
    # Standardize timezone to localized None
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
    df.set_index('Date', inplace=True)
    
    # Feature engineering for ML
    if 'Close' in df.columns:
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Volatility_21d'] = df['Return_1d'].rolling(21).std() * np.sqrt(252)
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['MA_200'] = df['Close'].rolling(200).mean()
        
    return df

def align_financials(ticker_dir, target_index):
    fin_dfs = []
    for file in ['financials.csv', 'balance_sheet.csv', 'cashflow.csv']:
        path = os.path.join(ticker_dir, file)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0)
                # Ensure the dataframe isn't completely empty before transposing
                if not df.empty and len(df.columns) > 0:
                    df = df.T
                    df.index = pd.to_datetime(df.index, utc=True).dt.tz_localize(None).dt.normalize()
                    df = df.sort_index()
                    df = df.dropna(axis=1, how='all')
                    if not df.empty:
                        fin_dfs.append(df)
            except Exception:
                pass
    
    if not fin_dfs:
        return pd.DataFrame(index=target_index)
        
    fin_combined = fin_dfs[0]
    for df in fin_dfs[1:]:
        # Use outer join and avoid duplicate columns if metrics overlap
        fin_combined = fin_combined.join(df, how='outer', rsuffix='_dup')
    
    fin_combined = fin_combined.loc[:, ~fin_combined.columns.str.endswith('_dup')]
    
    # Adjust for reporting lag: fundamentals are reported ~45 days after quarter-end date to prevent look-ahead bias
    fin_combined.index = fin_combined.index + pd.Timedelta(days=45)
    
    # Forward fill financial data up to ~400 trading days (over a year to allow some leeway on missing quarters)
    fin_aligned = fin_combined.reindex(target_index, method='ffill', limit=400)
    
    return fin_aligned

def process_earnings(ticker_dir, target_index):
    path = os.path.join(ticker_dir, 'earnings_dates.csv')
    if not os.path.exists(path):
        return pd.DataFrame(index=target_index)
    
    try:
        df = pd.read_csv(path)
        if 'Earnings Date' not in df.columns:
            return pd.DataFrame(index=target_index)
            
        df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], utc=True).dt.tz_localize(None).dt.normalize()
        df.set_index('Earnings Date', inplace=True)
        df = df.sort_index()
        
        cols_to_keep = ['EPS Estimate', 'Reported EPS', 'Surprise(%)']
        cols = [c for c in cols_to_keep if c in df.columns]
        df = df[cols]
        
        # Take the most recent earnings on days with duplicate indices
        df = df[~df.index.duplicated(keep='last')]
        
        # Forward fill up to 100 days (~1 quarter)
        df_aligned = df.reindex(target_index, method='ffill', limit=100)
        return df_aligned
    except Exception:
        return pd.DataFrame(index=target_index)

def process_news(ticker_dir, target_index):
    path = os.path.join(ticker_dir, 'news.json')
    if not os.path.exists(path):
         return pd.DataFrame(index=target_index, columns=['news_sentiment', 'news_count']).fillna(0)
         
    try:
        with open(path, 'r') as f:
            news_data = json.load(f)
            
        if not news_data:
             return pd.DataFrame(index=target_index, columns=['news_sentiment', 'news_count']).fillna(0)
             
        sia = SentimentIntensityAnalyzer()
        
        records = []
        for item in news_data:
            if 'providerPublishTime' in item:
                dt = pd.to_datetime(item['providerPublishTime'], unit='s', utc=True).tz_localize(None).normalize()
                title = item.get('title', '')
                try:
                    sentiment = sia.polarity_scores(title)['compound'] if title else 0
                except:
                    sentiment = 0
                records.append({'Date': dt, 'sentiment': sentiment})
                
        if not records:
             return pd.DataFrame(index=target_index, columns=['news_sentiment', 'news_count']).fillna(0)
             
        news_df = pd.DataFrame(records)
        daily_news = news_df.groupby('Date').agg(
            news_sentiment=('sentiment', 'mean'), 
            news_count=('sentiment', 'count')
        )
        
        # Forward filling news sentiment for up to 30 days
        news_aligned = daily_news.reindex(target_index, method='ffill', limit=30).fillna({'news_sentiment': 0, 'news_count': 0})
        return news_aligned
    except Exception:
        return pd.DataFrame(index=target_index, columns=['news_sentiment', 'news_count']).fillna(0)

def process_macro(macro_path, target_index):
    if not os.path.exists(macro_path):
        return pd.DataFrame(index=target_index)
        
    try:
        df = pd.read_csv(macro_path)
        df['DATE'] = pd.to_datetime(df['DATE'], utc=True).dt.tz_localize(None).dt.normalize()
        df.set_index('DATE', inplace=True)
        df = df.sort_index()
        
        # Forward fill macro data
        df_aligned = df.reindex(target_index, method='ffill')
        return df_aligned
    except Exception:
        return pd.DataFrame(index=target_index)

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "sp500_data")
    if not os.path.exists(base_dir):
        print("Data directory not found. Please run download_sp500_data.py first.")
        return
        
    macro_path = os.path.join(base_dir, 'macro_data.csv')
    
    all_stock_data = []
    
    try:
        tickers = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        # Exclude hidden folders if any exist
        tickers = [t for t in tickers if not t.startswith('.')]
        print(f"Found {len(tickers)} ticker directories.")
    except Exception as e:
        print(f"Error accessing directories: {e}")
        return
        
    for ticker in tqdm(tickers, desc="Consolidating Datasets"):
        ticker_dir = os.path.join(base_dir, ticker)
        
        hist_df = process_history(ticker_dir)
        if hist_df is None or hist_df.empty:
            continue
            
        target_index = hist_df.index
        
        # Merge modules
        fin_df = align_financials(ticker_dir, target_index)
        earn_df = process_earnings(ticker_dir, target_index)
        news_df = process_news(ticker_dir, target_index)
        macro_df = process_macro(macro_path, target_index)
        
        combined = pd.concat([hist_df, fin_df, earn_df, news_df, macro_df], axis=1)
        
        combined['Ticker'] = ticker
        combined.reset_index(inplace=True)
        
        all_stock_data.append(combined)
        
    if not all_stock_data:
        print("No valid data processed. Nothing to save.")
        return
        
    print("\nConcatenating all datasets...")
    final_dataset = pd.concat(all_stock_data, ignore_index=True)
    
    print("\nSorting MultiIndex (Date, Ticker)...")
    final_dataset.sort_values(['Date', 'Ticker'], inplace=True)
    final_dataset.set_index(['Date', 'Ticker'], inplace=True)
    
    # Clean column names (remove invalid characters for parquet)
    final_dataset.columns = final_dataset.columns.astype(str)
    
    print(f"Final dataset shape: {final_dataset.shape}")
    
    # Output to parent directory
    output_dir = os.path.dirname(__file__)
    output_csv = os.path.join(output_dir, "sp500_ml_dataset.csv")
    output_parquet = os.path.join(output_dir, "sp500_ml_dataset.parquet")
    
    print("Saving to CSV... (This might take a minute)")
    final_dataset.to_csv(output_csv)
    print(f"Successfully saved CSV to {output_csv}")
    
    try:
         print("Saving to Parquet...")
         final_dataset.to_parquet(output_parquet)
         print(f"Successfully saved Parquet to {output_parquet}")
    except Exception as e:
         print(f"Skipping parquet save: {e}")
         
    print("\nDataset consolidation complete! Ready for Machine Learning.")

if __name__ == '__main__':
    main()
