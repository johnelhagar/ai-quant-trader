import os
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from sklearn.preprocessing import StandardScaler

# Constants
SEQ_LEN = 10 # Predict 10 days out
TRAIN_START = '2016-03-15'
VAL_START = '2024-03-01'
VAL_END = '2026-03-13'
TOP_K_LONG = 20

def prepare_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'sp500_ml_dataset.parquet')
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading raw dataset...")
    df = pd.read_parquet(data_path)
    
    # 1. Target Engineering (Predicting 10-day forward return)
    # df is already multi-indexed by (Date, Ticker) and sorted
    print("Calculating 10-day forward returns...")
    
    # Group by ticker, shift Close price backwards by 10 days for future price
    df['Future_Close_10d'] = df.groupby(level='Ticker')['Close'].shift(-10)
    df['Target_Return_10d'] = (df['Future_Close_10d'] - df['Close']) / df['Close']
    
    # Drop rows where we don't have a future target
    df = df.dropna(subset=['Target_Return_10d'])
    
    # Define feature columns (dropping non-predictive/target metadata)
    ignore_cols = ['Open', 'High', 'Low', 'Close', 'Future_Close_10d', 'Target_Return_10d']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    # Drop rows with excessive missing features
    df = df.dropna(subset=feature_cols)
    
    # Reset index for easier splitting Operations
    df = df.reset_index()
    
    # 2. Time Series Split to prevent look-ahead bias
    print(f"Splitting Data: Train ({TRAIN_START} to {VAL_START}) | Val ({VAL_START} to {VAL_END})")
    df['Date'] = pd.to_datetime(df['Date'])
    
    train_mask = (df['Date'] >= pd.to_datetime(TRAIN_START)) & (df['Date'] < pd.to_datetime(VAL_START))
    val_mask = (df['Date'] >= pd.to_datetime(VAL_START)) & (df['Date'] <= pd.to_datetime(VAL_END))
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    
    # 3. Standardization
    print("Standardizing features...")
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    
    # 4. Save Tensors for quick loading in train.py
    print(f"Saving tensors to {out_dir}...")
    
    # Train set
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df['Target_Return_10d'].values, dtype=torch.float32)
    torch.save(X_train, os.path.join(out_dir, 'X_train.pt'))
    torch.save(y_train, os.path.join(out_dir, 'y_train.pt'))
    
    # Validation set needs extra metadata to reconstruct the portfolio simulation
    X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
    y_val = torch.tensor(val_df['Target_Return_10d'].values, dtype=torch.float32)
    torch.save(X_val, os.path.join(out_dir, 'X_val.pt'))
    torch.save(y_val, os.path.join(out_dir, 'y_val.pt'))
    
    # Save validation metadata (Dates and Tickers) to reconstruct daily cross-section
    val_meta = val_df[['Date', 'Ticker', 'Target_Return_10d']].copy()
    val_meta.to_parquet(os.path.join(out_dir, 'val_meta.parquet'))
    
    # 5. Fetch S&P500 Baseline for KPI calculation
    print("Fetching SPY benchmark for validation period...")
    spy = yf.download('SPY', start=VAL_START, end=VAL_END, progress=False)
    if not spy.empty:
        spy['SPY_Return_10d'] = spy['Close'].pct_change(10).shift(-10) # Align benchmark target format
        spy = spy.dropna()
        # Keep only date and return
        spy_df = pd.DataFrame({'Date': spy.index, 'Benchmark_Return': spy['SPY_Return_10d']})
        spy_df.to_parquet(os.path.join(out_dir, 'spy_benchmark.parquet'))
        print("SPY Benchmark saved.")
        
    print(f"Data Preparation Complete. Features: {len(feature_cols)}")
    print(f"Train Shape: {X_train.shape}")
    print(f"Val Shape: {X_val.shape}")

if __name__ == '__main__':
    prepare_data()
