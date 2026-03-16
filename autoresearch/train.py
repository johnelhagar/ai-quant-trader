import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Backtester Config
# -----------------------------------------------------------------------------
EPOCHS = 50
BATCH_SIZE = 4096
LR = 0.001
TOP_K_LONG = 20  # User specified: Long Only portfolio, top 20 stocks
TIMEOUT_SECONDS = 5 * 60  # 5 minutes wall clock timeout

# -----------------------------------------------------------------------------
# I/O setup
# -----------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# -----------------------------------------------------------------------------
# Baseline AI Trading Architecture
# -----------------------------------------------------------------------------
class AlphaNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1) # Predicts single floating point expected return
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

# -----------------------------------------------------------------------------
# Training & Evaluation Engine
# -----------------------------------------------------------------------------
def run_quant_experiment():
    start_time = time.time()
    
    # 1. Load Data
    X_train = torch.load(os.path.join(data_dir, 'X_train.pt'))
    y_train = torch.load(os.path.join(data_dir, 'y_train.pt'))
    X_val = torch.load(os.path.join(data_dir, 'X_val.pt'))
    
    val_meta = pd.read_parquet(os.path.join(data_dir, 'val_meta.parquet'))
    try:
        spy_bench = pd.read_parquet(os.path.join(data_dir, 'spy_benchmark.parquet'))
        spy_bench.set_index('Date', inplace=True)
    except:
        spy_bench = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = AlphaNet(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    print(f"Starting Training for Max {EPOCHS} epochs ({TIMEOUT_SECONDS}s limit)")

    # 2. Training Loop with Timeout
    model.train()
    best_val_excess_return = float('-inf')
    best_max_drawdown = float('inf')
    best_epoch_excess = float('-inf')
    best_epoch_drawdown = float('inf')
    
    for epoch in range(EPOCHS):
        if time.time() - start_time > TIMEOUT_SECONDS:
            print("Time budget exceeded. Halting training.")
            break
            
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        scheduler.step()

        # 3. Validation Portfolio Simulation — run every 5 epochs to save time budget
        if (epoch + 1) % 5 != 0 and epoch != EPOCHS - 1:
            print(f"Epoch {epoch+1:02d} | Train MSE: {total_loss/len(train_loader):.4f}")
            model.train()
            continue

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device)).cpu().numpy()
            
        val_meta['Prediction'] = val_preds
        
        # Portfolio Construction: Loop over every day to build a daily cross-section
        daily_returns = []
        dates = []
        
        for date, daily_df in val_meta.groupby('Date'):
            # Extract models positive predictions (Filter out negative expectations)
            positive_signals = daily_df[daily_df['Prediction'] > 0.0].copy()

            # Concentrate portfolio to TOP_K_LONG highest-conviction stocks
            if len(positive_signals) > TOP_K_LONG:
                positive_signals = positive_signals.nlargest(TOP_K_LONG, 'Prediction')

            if len(positive_signals) == 0:
                 # Market is entirely bearish, go 100% Cash (0% Return)
                 daily_returns.append(0.0)
                 dates.append(date)
                 continue
                 
            # Dynamic Weighting: proportional to prediction magnitude
            # If total conviction > 1.0: normalize to 100% invested. Otherwise keep as raw weights (partial cash).
            total_conviction = positive_signals['Prediction'].sum()
            if total_conviction > 1.0:
                positive_signals['Weight'] = positive_signals['Prediction'] / total_conviction
            else:
                positive_signals['Weight'] = positive_signals['Prediction']
                 
            # Calculate the daily weighted return of the allocated stocks
            portfolio_return = (positive_signals['Weight'] * positive_signals['Target_Return_10d']).sum()
            
            daily_returns.append(portfolio_return)
            dates.append(date)
            
        pf_df = pd.DataFrame({'Date': dates, 'Portfolio_Return': daily_returns}).set_index('Date')
        
        # Merge with SPY Benchmark
        if spy_bench is not None:
             # Using squeeze correctly for pandas > 2.0
            pf_df = pf_df.join(spy_bench, how='inner')
            pf_df['Benchmark_Return'] = pf_df['Benchmark_Return'].squeeze()
        else:
            pf_df['Benchmark_Return'] = 0.0
            
        # KPI Calculations — use compound returns for accuracy over the 2-year val period
        total_portfolio_return = (1 + pf_df['Portfolio_Return']).prod() - 1
        total_benchmark_return = (1 + pf_df['Benchmark_Return']).prod() - 1

        val_excess_return = total_portfolio_return - total_benchmark_return

        # Drawdown Calculation on compound cumulative curve
        pf_df['Cum_Ret'] = (1 + pf_df['Portfolio_Return']).cumprod() - 1
        pf_df['Peak'] = pf_df['Cum_Ret'].cummax()
        pf_df['Drawdown'] = pf_df['Cum_Ret'] - pf_df['Peak']
        val_max_drawdown = pf_df['Drawdown'].min() # Negative number
        
        print(f"Epoch {epoch+1:02d} | Train MSE: {total_loss/len(train_loader):.4f} | val_excess_return: {val_excess_return:.4f} | val_max_drawdown: {val_max_drawdown:.4f}")
        
        # Save absolute best model weights meeting criteria
        abs_drawdown = abs(val_max_drawdown)
        if val_excess_return > best_val_excess_return and abs_drawdown <= 0.08:
            best_val_excess_return = val_excess_return
            best_max_drawdown = abs_drawdown
            best_epoch_excess = val_excess_return
            best_epoch_drawdown = abs_drawdown
            torch.save(model.state_dict(), 'best_model.pt')
        
        model.train()
        
    # If no epoch met the drawdown constraint, report last-epoch values so the orchestrator
    # can still log the run (it will be rejected by the accept/reject logic, not by parse failure)
    if best_epoch_excess == float('-inf'):
        best_epoch_excess = val_excess_return
        best_epoch_drawdown = abs(val_max_drawdown)

    print("\n[FINAL RESULTS]")
    print(f"val_excess_return={best_epoch_excess:.4f}")
    print(f"val_max_drawdown={best_epoch_drawdown:.4f}")

if __name__ == '__main__':
    run_quant_experiment()
