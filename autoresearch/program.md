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

## Permitted Research Avenues
Everything in `train.py` is yours to modify to achieve these KPIs. Research directions **ranked by expected impact**:

1. **Loss Functions (HIGHEST IMPACT)**: MSE is naïve — it optimizes prediction accuracy for every stock equally, but CAN SLIM only cares about correctly identifying the top growth leaders. Replace MSE with:
   - A **Pairwise Ranking Loss** (penalize when a laggard stock ranks above a leader in your predictions — directly implements the L factor)
   - A **Spearman Rank Correlation Loss** (directly maximize rank correlation between predictions and actual returns per day)
   - A **Direct Sharpe Ratio Loss** (construct mini-portfolios from each batch and maximize their risk-adjusted return — naturally penalizes drawdown)
   - A **CAN SLIM-weighted Loss** (weight training samples by how many CAN SLIM factors they satisfy — stocks with strong C+A+N+L should have higher loss weight)

2. **Feature Engineering in the Model**: Create derived CAN SLIM signals inside `train.py` during training:
   - **Earnings Momentum**: Interaction features between `Surprise(%)` and `Return_5d` (earnings beat + price momentum = classic CAN SLIM buy)
   - **Trend Confirmation**: Binary signal for price > MA_50 > MA_200 (golden cross regime)
   - **Market Regime Filter**: Use macro features (`FedFundsRate`, `10YrTreasury`, `CPI`) to create a regime indicator that scales position sizes down in risk-off environments

3. **Portfolio Concentration (`TOP_K_LONG`)**: The constant `TOP_K_LONG = 20` controls how many stocks are held per day. CAN SLIM favors concentrated portfolios of 5-10 high-conviction leaders. Experiment with values from 5 (very concentrated, high alpha) to 20 (moderate diversification). Going above 20 dilutes the CAN SLIM edge.

4. **Architecture**: Change the vanilla Multi-Layer Perceptron to better capture CAN SLIM patterns:
   - **Attention mechanisms** can learn which CAN SLIM factors matter most per stock
   - **XGBoost/LightGBM** handles the interaction features (earnings x momentum) naturally
   - **Ensemble**: Combine a neural net for complex patterns with gradient boosting for feature interactions

5. **Hyperparameters**: Tune Learning Rates, Batch Sizes, Epochs, Dropout, Layer normalization.

6. **Regularization**: Add L1/L2 penalties to encourage the model to focus on the most predictive CAN SLIM factors.

## Restrictions
1. **DO NOT** edit `prepare.py`. It is the source of truth for avoiding look-ahead bias.
2. **DO NOT** exceed the 5 minute execution timeout for `train.py`.
3. **DO NOT** short stocks (weights must be >= 0) or use leverage (sum of weights cannot exceed 1.0).

Optimize aggressively. If a change improves the `val_excess_return` without breaching the drawdown constraint, keep it. Otherwise, revert the code and try a new hypothesis.
