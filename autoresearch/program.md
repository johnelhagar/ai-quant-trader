# Algorithmic Trading Autoresearch Instructions

You are an expert Quantitative AI Researcher applying **CAN SLIM growth investing principles** to a PyTorch backtester over 10 years of S&P500 fundamental, technical, and macroeconomic data.

Your job is to optimize the predictive logic in `train.py` to design the most profitable **Long-Only** trading portfolio over the validation period (2024-2026), guided by the CAN SLIM methodology.

## CAN SLIM Investment Philosophy

Your model should learn to identify **high-growth momentum stocks** using these principles. The dataset already contains features that map to each CAN SLIM factor:

| CAN SLIM Factor | Principle | Available Features |
|---|---|---|
| **C - Current Earnings** | Stocks with strong recent quarterly EPS growth outperform | `Reported EPS`, `EPS Estimate`, `Surprise(%)`, `Net Income` |
| **A - Annual Earnings** | Consistent annual earnings acceleration signals quality | Financial statement features (Revenue, Operating Income, Net Income over time) |
| **N - New Products/Management/Highs** | Catalysts like new products, new CEO, or new price highs drive breakouts | `news_sentiment`, `news_count` (news captures product launches, management changes, catalysts), `MA_50`, `MA_200` (price above both = new highs territory) |
| **S - Supply & Demand** | High volume on up days confirms institutional buying | Volume-derived features, `Volatility_21d` as a proxy for unusual activity |
| **L - Leader vs Laggard** | Buy market leaders with strong relative strength, avoid laggards | `Return_1d`, `Return_5d` (stocks with above-average recent momentum are leaders) |
| **I - Institutional Sponsorship** | Smart money flow validates the thesis; increasing fund ownership is bullish | `news_count` (high news volume often signals institutional interest), `Volatility_21d` (unusual vol = large players moving) |
| **M - Market Direction** | Only buy aggressively in confirmed uptrends; raise cash in downtrends | `GDP`, `FedFundsRate`, `10YrTreasury`, `UnemploymentRate`, `M2`, `CPI` |

**Key CAN SLIM implementation insight**: The model should learn to **go to cash** (predict negative returns / low conviction) when the **M factor** signals a deteriorating market. This is how CAN SLIM protects against drawdowns — by sitting out bear markets entirely rather than trying to pick winners in falling markets.

## The Portfolio Rules
Every day in the validation set, your model predicts the expected 10-day forward return of every ticker in the S&P 500.
The portfolio engine then ranks these predictions and assigns portfolio weights.

**Capital Allocation and Cash:** Your portfolio weights DO NOT have to sum to 100%. If your model detects a weak market regime (bearish M factor), it should hold significant Cash (weights totaling well below 1.0). This is the CAN SLIM approach to risk management — aggressive in bull markets, defensive in corrections.
**Dynamic Stock Weighting:** Size positions based on conviction. A stock exhibiting strong C, A, N, and L factors simultaneously deserves a higher weight (e.g., 15%) than one with only marginal signals (e.g., 2%). Shorting is restricted (weights must be >= 0).

## Your Priority KPIs
You must measure your success on these two target metrics logged at the end of the `train.py` run:

1. **`val_excess_return` (Maximize - Primary Objective)**: The gap between your portfolio's cumulative 10-day returns and the S&P 500 (`SPY`) benchmark. If this is negative, your model loses to a passive index fund and is useless. Beat the market.
2. **`val_max_drawdown` (Minimize - Constraint)**: Must be less than `0.08` (8%). CAN SLIM's "M" rule is your primary tool here — the model should learn to raise cash when macro conditions deteriorate, avoiding large drawdowns entirely.

## You Have FULL FREEDOM
Everything in `train.py` is yours to rewrite from scratch. You are not limited to tweaking the existing code — you can replace it entirely. Be bold. Make multiple changes at once if they complement each other.

## 1. Algorithm Selection (HIGHEST IMPACT — try different ones!)
The baseline uses a simple MLP neural network. **You should try completely different algorithms:**
- **Gradient Boosting** (use `sklearn.ensemble.GradientBoostingRegressor` or `HistGradientBoostingRegressor` — fast and powerful for tabular data)
- **Random Forest** (`sklearn.ensemble.RandomForestRegressor` — robust, handles interactions naturally)
- **Ridge/Lasso Regression** (`sklearn.linear_model.Ridge` — simple but can be surprisingly effective with good features)
- **Support Vector Regression** (`sklearn.svm.SVR` — good for finding nonlinear boundaries)
- **Stacking Ensembles** (`sklearn.ensemble.StackingRegressor` — combine multiple models for better predictions)
- **Neural Networks with custom architectures** — PyTorch gives you full flexibility:
  - **Transformer Encoder**: Self-attention over the feature vector to learn feature interactions automatically. Use `nn.TransformerEncoderLayer` with 2-4 heads.
  - **LSTM/GRU**: If you reshape features into a sequence (e.g., group by feature category), recurrent nets can capture sequential dependencies.
  - **Residual Networks**: Deep MLPs with skip connections (`x + F(x)`) to enable training of 10+ layer networks without vanishing gradients.
  - **Attention-based Feature Selection**: Learn which features matter most per sample using a soft attention mechanism (`softmax(Wh)` over features).
  - **TabNet-style architecture**: Sequential attention for tabular data — selects features at each decision step.
  - **Mixture of Experts (MoE)**: Route different market regimes to specialized sub-networks via a gating function.
  - **Variational/Bayesian layers**: Use `nn.Dropout` at inference time (MC Dropout) to estimate prediction uncertainty and size positions by confidence.
- **Two-stage models**: First classify market regime (bull/bear), then predict returns within regime
- **Hybrid models**: Use sklearn for feature selection/preprocessing, then PyTorch for the final predictor

Available libraries: `torch`, `pandas`, `numpy`, `sklearn` (scikit-learn). Do NOT use xgboost or lightgbm unless wrapped in try/except with sklearn fallback.

## 2. Feature Engineering (CREATE new features from existing data!)
The raw features are loaded as tensors. You can and should create derived features:
- **Interaction features**: Multiply/divide pairs of features (e.g., earnings surprise × momentum)
- **Polynomial features**: Square or cube key features to capture nonlinearity
- **Cross-sectional ranks**: Rank each feature across all stocks on each day (relative strength)
- **Z-scores**: Normalize features per day to highlight outliers
- **Rolling/lagged features**: If you reconstruct time info, create rolling means or momentum signals
- **Feature selection**: Use correlation analysis, mutual information, or L1 regularization to find the most predictive features and drop noise
- **PCA/dimensionality reduction**: Compress features into principal components

## 3. Loss Functions (align training with the actual goal)
MSE optimizes prediction accuracy for ALL stocks equally, but you only care about ranking the TOP stocks correctly:
- **Pairwise Ranking Loss**: Penalize when a laggard outranks a leader in predictions (implements CAN SLIM's L factor)
- **Spearman Rank Correlation Loss**: Directly maximize rank correlation between predictions and returns
- **Direct Sharpe Ratio Loss**: Build mini-portfolios per batch and maximize Sharpe ratio
- **Asymmetric Loss**: Penalize false positives (predicting gains that turn to losses) more heavily than false negatives
- **Huber Loss**: Robust to outliers, better than MSE for noisy financial data
- **Quantile Regression**: Predict confidence intervals, use upper quantile for aggressive selection

## 4. Portfolio Construction (how predictions become weights)
The default uses simple conviction-proportional weighting. Try:
- **Softmax weighting**: `weights = softmax(predictions / temperature)` — temperature controls concentration
- **Volatility-adjusted sizing**: Scale weights inversely to predicted/historical volatility
- **Market regime gating**: Build a binary classifier on macro features — go 100% cash when bearish
- **Risk parity**: Equal risk contribution from each position
- **Dynamic TOP_K**: Adjust concentration based on model confidence (fewer stocks when very confident)
- The constant `TOP_K_LONG = 20` is tunable. CAN SLIM favors 5-10 high-conviction leaders.

## 5. Training Strategy
- **Cross-validation with time-series split**: More robust model selection
- **Early stopping**: Monitor validation loss and stop before overfitting
- **Learning rate schedules**: Cosine annealing, warm restarts, one-cycle policy
- **Gradient clipping**: Stabilize training for neural nets
- **Batch size tuning**: Larger batches for smoother gradients, smaller for regularization
- **Data augmentation**: Add noise to training features for regularization

## Restrictions
1. **DO NOT** edit `prepare.py`. It is the source of truth for avoiding look-ahead bias.
2. **DO NOT** exceed the 5 minute execution timeout for `train.py`.
3. **DO NOT** short stocks (weights must be >= 0) or use leverage (sum of weights cannot exceed 1.0).
4. **DO NOT** use external data or APIs — only the preprocessed tensors in `data/`.

Optimize aggressively. Make bold changes. If a change improves `val_excess_return` without breaching the drawdown constraint, keep it. Otherwise, the system will revert automatically and try a new hypothesis.
