# Algorithmic Trading Autoresearch Instructions

You are an expert Quantitative AI Researcher. I have provided you the skeleton of a robust PyTorch backtester testing over 10 years of S&P500 fundamental, technical, and macroeconomic data. 

Your job is to optimize the predictive logic in `train.py` to design the most profitable **Long-Only** trading portfolio over the validation period (2024-2026).

## The Portfolio Rules
Every day in the validation set, your model predicts the expected 10-day forward return of every ticker in the S&P 500.
The portfolio engine then ranks these predictions and assigns portfolio weights. 

**Capital Allocation and Cash:** Your portfolio weights DO NOT have to sum to 100%. If your model predicts severe market downturns, it is completely free to hold 50%, 90%, or even 100% of the portfolio in Cash (Cash effectively earns 0% return over the 10-day period). It achieves this by allocating portfolio weights mathematically that total less than 1.0.
**Dynamic Stock Weighting:** Instead of equal-weighting every stock, you are encouraged to allow the model to size its positions based on your conviction. If it's extremely confident in one stock, it can weight it at 15% and another at 2%. Shorting is restricted (weights must be >= 0).

## Your Priority KPIs
You must measure your success on these two target metrics logged at the end of the `train.py` run:

1. **`val_excess_return` (Maximize - Primary Objective)**: The gap between your portfolio's cumulative 10-day returns and the S&P 500 (`SPY`) benchmark. If this is negative, your model loses to a passive index fund and is useless. Beat the market.
2. **`val_max_drawdown` (Minimize - Constraint)**: Must be less than `0.15` (15%). If your strategy beats the market but routinely bombs with massive portfolio losses from the peak, it is too risky to deploy. 

## Permitted Research Avenues
Everything in `train.py` is yours to modify to achieve these KPIs. Here are research directions:
- **Architecture**: Change the vanilla Multi-Layer Perceptron into an RNN, Transformer, or swap PyTorch for XGBoost/LightGBM.
- **Hyperparameters**: Tune Learning Rates, Batch Sizes, Epochs, Dropout, Layer normalization.
- **Loss Functions**: MSE is naïve. Try writing a custom loss function that optimizes the structural ranking (Pairwise Loss) or directly correlates to Sharpe Ratio.
- **Regularization**: Add L1/L2 penalties. 

## Restrictions
1. **DO NOT** edit `prepare.py`. It is the source of truth for avoiding look-ahead bias.
2. **DO NOT** exceed the 5 minute execution timeout for `train.py`.
3. **DO NOT** short stocks (weights must be >= 0) or use leverage (sum of weights cannot exceed 1.0).

Optimize aggressively. If a change improves the `val_excess_return` without breaching the drawdown constraint, keep it. Otherwise, revert the code and try a new hypothesis.
