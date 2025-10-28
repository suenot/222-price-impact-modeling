# Chapter 264: Price Impact Modeling with Machine Learning

## Introduction

Price impact -- the adverse price movement caused by a trade -- is one of the most critical yet underappreciated factors in quantitative trading. A strategy that looks profitable on paper can easily become unprofitable once realistic execution costs are accounted for. Understanding, measuring, and predicting price impact is essential for:

- **Optimal execution**: Minimizing the cost of liquidating or building a position
- **Transaction cost analysis (TCA)**: Evaluating broker and algorithm performance
- **Portfolio construction**: Incorporating realistic costs into optimization
- **Market making**: Managing inventory risk with accurate impact forecasts
- **Alpha research**: Separating genuine signal from market impact artifacts

This chapter explores the mathematical foundations of price impact, from classical microstructure theory to modern machine learning approaches, with full Rust implementation and Bybit integration for cryptocurrency markets.

## Theoretical Foundations

### What Is Price Impact?

When a trader submits an order to buy a large quantity of an asset, the price typically moves against them. This occurs through two mechanisms:

1. **Information leakage**: Market participants infer that a large buyer likely has positive information about the asset, causing them to revise their quotes upward
2. **Supply/demand imbalance**: Large orders consume available liquidity at successive price levels in the order book

Price impact is typically decomposed into:

- **Temporary impact**: The transient price displacement that reverses after the trade completes
- **Permanent impact**: The lasting price change reflecting new information incorporated into the price

### The Square-Root Law

One of the most robust empirical findings in market microstructure is that price impact scales approximately as the square root of order size. The square-root impact model is:

$$\Delta P / P = \sigma \cdot \text{sign}(Q) \cdot \eta \cdot \left(\frac{|Q|}{V}\right)^\delta$$

where:
- $\sigma$ is the daily volatility
- $Q$ is the signed order size (positive for buys)
- $V$ is the average daily volume (ADV)
- $\eta$ is the impact coefficient (typically 0.1 to 1.0)
- $\delta$ is the exponent (empirically close to 0.5)

The square-root relationship $\delta \approx 0.5$ has been observed across:
- Equities (NYSE, NASDAQ, LSE)
- Futures markets
- Foreign exchange
- Cryptocurrency markets

This universality suggests a deep connection to market microstructure theory. The theoretical justification comes from Kyle's lambda model and its extensions, where informed traders optimally split orders to minimize information leakage.

### Calibration via Log-Linearization

To calibrate the model from observed data, we take logarithms:

$$\ln\left(\frac{|\Delta P / P|}{\sigma}\right) = \ln(\eta) + \delta \cdot \ln\left(\frac{|Q|}{V}\right)$$

This is a simple linear regression $y = a + bx$ where:
- $y = \ln(|\Delta P / P| / \sigma)$
- $x = \ln(|Q| / V)$
- $a = \ln(\eta)$, so $\eta = e^a$
- $b = \delta$

### The Almgren-Chriss Framework

The Almgren-Chriss model (2000) provides the foundation for optimal trade execution. It models the problem of liquidating $X$ shares over $T$ periods while minimizing the trade-off between expected cost and cost variance.

#### Impact Decomposition

The model separates impact into:

- **Permanent impact**: $g(v) = \gamma \cdot v$ (proportional to trading rate)
- **Temporary impact**: $h(v) = \epsilon \cdot \text{sign}(v) + \eta \cdot v$ (fixed spread + linear)

where $v = n_j / \tau$ is the trading rate (shares per time period).

#### Optimal Trajectory

The optimal execution trajectory minimizes:

$$\min_{n_1, \ldots, n_N} \left[ E[\text{cost}] + \lambda \cdot \text{Var}[\text{cost}] \right]$$

subject to $\sum n_j = X$ (all shares must be executed).

The solution is:

$$X_j = X \cdot \frac{\sinh(\kappa(T - t_j)\tau)}{\sinh(\kappa T \tau)}$$

where:
- $\kappa = \sqrt{\lambda \sigma^2 / \tilde{\eta}}$
- $\tilde{\eta} = \eta - \gamma\tau/2$
- $\lambda$ is the risk aversion parameter

**Key insight**: When $\lambda$ is large (high risk aversion), the trader executes quickly to avoid uncertainty. When $\lambda$ is small, the trader spreads execution evenly to minimize impact.

#### Expected Cost and Variance

The expected cost of following the optimal trajectory is:

$$E[\text{cost}] = \frac{1}{2}\gamma X^2 + \sum_{j=1}^{N} \left[\epsilon |n_j| + \tilde{\eta} \frac{n_j^2}{\tau}\right]$$

The cost variance is:

$$\text{Var}[\text{cost}] = \sigma^2 \tau \sum_{j=1}^{N} X_{j-1}^2$$

### Machine Learning for Impact Prediction

While classical models capture the dominant effects, they miss important nonlinear interactions. ML models can incorporate:

1. **Order book features**: Bid-ask spread, depth imbalance, queue position
2. **Market regime**: Volatility clustering, momentum, mean-reversion state
3. **Temporal patterns**: Time of day, day of week, around-event effects
4. **Cross-asset signals**: Correlated assets' order flow

#### Linear Regression Baseline

The simplest ML approach is multivariate linear regression:

$$\hat{y} = w_0 + \sum_{i=1}^{n} w_i x_i$$

Features typically include:
- Participation rate: $|Q| / \text{ADV}$
- Relative spread: $\text{spread} / P_{\text{mid}}$
- Realized volatility
- Order sign (buy/sell indicator)
- Volume profile features

This is fitted using ordinary least squares (OLS) or gradient descent.

#### Neural Network Predictor

For capturing nonlinear relationships, we use a feedforward neural network:

$$h_1 = \text{ReLU}(W_1 x + b_1)$$
$$h_2 = \text{ReLU}(W_2 h_1 + b_2)$$
$$\hat{y} = W_3 h_2 + b_3$$

The network is trained with MSE loss:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

using backpropagation and stochastic gradient descent.

## Transaction Cost Analysis (TCA)

TCA is the post-trade analysis of execution quality. Key metrics include:

### Implementation Shortfall

The difference between the decision price (arrival price) and the actual execution price:

$$\text{IS} = \frac{(P_{\text{exec}} - P_{\text{arrival}}) \cdot \text{sign}(Q)}{P_{\text{arrival}}}$$

Positive IS means the execution was worse than the arrival price (i.e., the trader paid more for buys or received less for sells).

### VWAP Slippage

Comparison of execution price to the volume-weighted average price:

$$\text{VWAP Slippage} = \frac{(P_{\text{exec}} - \text{VWAP}) \cdot \text{sign}(Q)}{\text{VWAP}}$$

### Spread Cost

The cost of crossing the bid-ask spread:

$$\text{Spread Cost} = \frac{\text{spread}}{2 \cdot P_{\text{mid}}}$$

## Rust Implementation Walkthrough

Our Rust implementation provides four complementary models organized in a single library crate.

### Square-Root Model

The `SquareRootModel` struct encapsulates the classical impact model with calibration support:

```rust
pub struct SquareRootModel {
    pub eta: f64,              // Impact coefficient
    pub delta: f64,            // Exponent (~0.5)
    pub volatility: f64,       // Daily volatility
    pub avg_daily_volume: f64, // ADV
}
```

Key methods:
- `estimate_impact(order_size)` -- Computes fractional price change
- `estimate_cost(order_size, mid_price)` -- Dollar cost of impact
- `calibrate(sizes, impacts)` -- Fits eta and delta from observed data using log-linearization

### Almgren-Chriss Model

The `AlmgrenChrissModel` computes optimal execution trajectories:

```rust
pub struct AlmgrenChrissModel {
    pub gamma: f64,         // Permanent impact coefficient
    pub eta: f64,           // Temporary impact coefficient
    pub epsilon: f64,       // Half-spread
    pub volatility: f64,    // Daily volatility
    pub risk_aversion: f64, // Lambda
}
```

Key methods:
- `optimal_trajectory(shares, steps, tau)` -- Returns the optimal remaining-shares schedule
- `expected_cost(trajectory, tau)` -- Expected execution cost
- `cost_variance(trajectory, tau)` -- Variance of execution cost

### ML Impact Predictor

The `MLImpactPredictor` is a two-hidden-layer neural network for regression:

```rust
pub struct MLImpactPredictor {
    pub weights1: Array2<f64>,  // Input -> Hidden1
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,  // Hidden1 -> Hidden2
    pub bias2: Array1<f64>,
    pub weights3: Array2<f64>,  // Hidden2 -> Output (1 neuron)
    pub bias3: Array1<f64>,
    pub learning_rate: f64,
}
```

Key methods:
- `forward(input)` -- Predict impact value
- `train_step(input, target)` -- Single SGD step with MSE loss and full backpropagation
- `train_batch(inputs, targets, epochs)` -- Batch training

### Transaction Cost Analyzer

The `TransactionCostAnalyzer` provides end-to-end TCA:

```rust
pub struct TransactionCostAnalyzer {
    pub sqrt_model: SquareRootModel,
    pub ml_model: Option<MLImpactPredictor>,
}
```

It analyzes `TradeRecord` objects and produces `TCAResult` with implementation shortfall, VWAP slippage, spread cost, and estimated impact.

## Bybit Integration

The trading example fetches real BTCUSDT candle data from Bybit's public API:

```rust
async fn fetch_bybit_candles(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    // ... HTTP request and JSON parsing
}
```

The example then:
1. Computes realized volatility and average volume from candle data
2. Simulates trade records with realistic slippage
3. Calibrates the square-root model
4. Computes optimal Almgren-Chriss trajectories for different risk aversion levels
5. Trains both linear and neural network impact predictors
6. Runs full TCA analysis
7. Benchmarks inference speed across all models

If the API is unavailable, synthetic data is generated as a fallback.

## Applications in Cryptocurrency Trading

### Execution Algorithms

Price impact models drive execution algorithm design:

- **TWAP (Time-Weighted Average Price)**: Splits orders evenly. Corresponds to Almgren-Chriss with $\lambda = 0$ (no risk aversion).
- **VWAP**: Follows the expected volume profile. Impact model determines how much to deviate from the profile.
- **Implementation Shortfall (IS)**: Directly optimizes Almgren-Chriss objective. The impact model parameters determine the optimal schedule.
- **Adaptive algorithms**: Use real-time impact estimates to adjust execution speed.

### Market Making

Market makers use impact models to:
- Set bid-ask spreads wide enough to cover expected adverse selection
- Manage inventory by understanding the impact of their own hedge trades
- Detect toxic flow (orders from informed traders with high impact)

### Portfolio Optimization with Impact

Traditional portfolio optimization ignores trading costs. With impact awareness:

$$\max_{w} \left[\mu^T w - \frac{\lambda}{2} w^T \Sigma w - \text{ImpactCost}(w - w_{\text{current}})\right]$$

This penalizes large portfolio changes, leading to smoother rebalancing.

## Practical Considerations

### Cryptocurrency-Specific Factors

Crypto markets have unique impact characteristics:
- **Fragmented liquidity**: Liquidity spread across many exchanges
- **24/7 trading**: No closing auction to concentrate liquidity
- **Higher volatility**: Larger natural price movements can mask impact
- **Varying tick sizes**: Different precision across exchanges
- **Funding rates**: Perpetual futures have periodic funding that affects impact

### Model Limitations

- The square-root law is an empirical regularity, not a physical law. It can break down for very small or very large orders.
- Almgren-Chriss assumes constant parameters; in practice, volatility and liquidity change during execution.
- ML models can overfit to specific market regimes and may not generalize across regime changes.
- All models assume price impact is a function of trade characteristics only; in reality, impact depends on the full state of the order book and market context.

## Key Takeaways

1. **Price impact scales as the square root of order size** -- the most universal finding in market microstructure. This means doubling your order size increases impact by only ~41%, not 100%.

2. **The Almgren-Chriss framework** provides a rigorous way to trade off execution speed against cost uncertainty. Higher risk aversion leads to faster, front-loaded execution.

3. **Machine learning extends classical models** by capturing nonlinear relationships between trade features and realized impact. Neural networks can learn complex interactions that linear models miss.

4. **Transaction cost analysis** is essential for evaluating execution quality. Key metrics are implementation shortfall, VWAP slippage, and spread cost.

5. **Calibration matters more than model complexity**. A well-calibrated square-root model often outperforms a poorly trained neural network. Start simple, add complexity only when justified by out-of-sample performance.

6. **Impact models enable better trading strategies** -- from optimal execution algorithms to impact-aware portfolio optimization. Ignoring impact leads to overstated backtested returns and real-world losses.

## Running the Code

```bash
cd 264_price_impact_modeling/rust

# Run all tests (11 tests)
cargo test

# Run the trading example with Bybit data
cargo run --example trading_example
```

The example will attempt to fetch live BTCUSDT data from Bybit and fall back to synthetic data if the API is unavailable.

## References

1. Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." Econometrica, 53(6), 1315-1335.
2. Almgren, R., & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions." Journal of Risk, 3(2), 5-39.
3. Bershova, N., & Rakhlin, D. (2013). "The Non-Linear Market Impact of Large Trades." Annals of Finance, 9(4), 521-544.
4. Bacry, E., et al. (2015). "Market Impacts and the Life Cycle of Investor Orders." Market Microstructure and Liquidity, 1(2).
5. Toth, B., et al. (2011). "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets." Physical Review X, 1(2).
6. Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order Book Events." Journal of Financial Econometrics, 12(1), 47-88.
7. Bucci, F., et al. (2020). "Co-Impact: Crowding Effects in Institutional Trading Activity." Quantitative Finance, 20(2), 193-205.
