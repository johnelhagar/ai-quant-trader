import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import os
import time
from datetime import datetime, timedelta
import json
from tqdm import tqdm

def get_sp500_tickers():
    import requests
    import io
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    table = pd.read_html(io.StringIO(response.text))
    df = table[0]
    tickers = df['Symbol'].tolist()
    # Replace '.' with '-' for yfinance (e.g., BRK.B to BRK-B)
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

def download_macro_data(start_date, end_date, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Common FRED series
    series = {
        'GDP': 'GDP',
        'CPI': 'CPIAUCSL',
        'FedFundsRate': 'FEDFUNDS',
        '10YrTreasury': 'GS10',
        'UnemploymentRate': 'UNRATE',
        'M2': 'M2SL'
    }
    macro_data = pd.DataFrame()
    for name, code in series.items():
        try:
            df = web.DataReader(code, 'fred', start_date, end_date)
            df.columns = [name]
            if macro_data.empty:
                macro_data = df
            else:
                macro_data = macro_data.join(df, how='outer')
        except Exception as e:
            print(f"Failed to fetch {name}: {e}")
            
    if not macro_data.empty:
        macro_data.to_csv(os.path.join(output_dir, 'macro_data.csv'))
        print("Macro data saved.")
    else:
        print("No macro data could be fetched.")

def download_stock_data(ticker, start_date, end_date, output_dir):
    ticker_dir = os.path.join(output_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    t = yf.Ticker(ticker)
    
    # 1. History / OHLCV / Technicals Base Data
    try:
        hist = t.history(start=start_date, end=end_date)
        if not hist.empty:
            hist.to_csv(os.path.join(ticker_dir, 'history.csv'))
    except Exception as e:
        pass
        
    # 2. Fundamentals
    try:
        financials = t.financials
        if not financials.empty:
            financials.to_csv(os.path.join(ticker_dir, 'financials.csv'))
            
        balance_sheet = t.balance_sheet
        if not balance_sheet.empty:
            balance_sheet.to_csv(os.path.join(ticker_dir, 'balance_sheet.csv'))
            
        cashflow = t.cashflow
        if not cashflow.empty:
            cashflow.to_csv(os.path.join(ticker_dir, 'cashflow.csv'))
    except Exception as e:
        pass
        
    # 3. News
    try:
        news = t.news
        if news:
            with open(os.path.join(ticker_dir, 'news.json'), 'w') as f:
                json.dump(news, f, indent=4)
    except Exception as e:
        pass
        
    # 4. Earnings
    try:
        earnings = t.get_earnings_dates(limit=40) # Approximately 10 years of quarterly dates
        if earnings is not None and not earnings.empty:
            earnings.to_csv(os.path.join(ticker_dir, 'earnings_dates.csv'))
    except Exception as e:
        pass

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "sp500_data")
    os.makedirs(base_dir, exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    print(f"Data will be saved to: {base_dir}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    print("\nDownloading Macroeconomic data...")
    download_macro_data(start_date, end_date, base_dir)
    
    print("\nFetching S&P500 tickers...")
    try:
        tickers = get_sp500_tickers()
        print(f"Found {len(tickers)} tickers. Beginning downloads...")
    except Exception as e:
        print(f"Failed to fetch S&P500 list: {e}")
        return
    
    # Optional: for testing quickly, uncomment the line below:
    # tickers = tickers[:5] 
    
    for ticker in tqdm(tickers):
        download_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), base_dir)
        # Sleep to avoid excessive rate limiting from Yahoo Finance
        time.sleep(1.5)

if __name__ == '__main__':
    main()
