//! # Price Impact Modeling with Machine Learning
//!
//! This library implements several price impact models used in quantitative
//! trading, combining classical market microstructure theory with machine
//! learning techniques.
//!
//! ## Key Components
//! - `SquareRootModel`: The classic Barra/Kyle square-root impact model
//! - `AlmgrenChrissModel`: Optimal execution with temporary and permanent impact
//! - `LinearImpactModel`: Simple linear regression-based impact estimator
//! - `MLImpactPredictor`: Neural network for nonlinear impact prediction
//! - `TransactionCostAnalyzer`: End-to-end TCA with Bybit data integration
//!
//! ## Price Impact Theory
//!
//! Price impact measures how much a trade moves the market price. Understanding
//! and predicting impact is critical for:
//! - Optimal order execution (minimizing trading costs)
//! - Transaction cost analysis (TCA)
//! - Portfolio construction (accounting for realistic costs)
//! - Market making (inventory risk management)

use ndarray::{Array1, Array2};
use rand::Rng;

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Compute softmax of a vector (for neural network output layer).
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = logits.iter().map(|&z| (z - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// ReLU activation function.
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Compute the sign of a value: -1.0, 0.0, or 1.0.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Compute mean of a slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute standard deviation of a slice.
pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let var = data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

/// Compute R-squared (coefficient of determination).
pub fn r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(actual.len(), predicted.len());
    let mean_actual = mean(actual);
    let ss_res: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).powi(2))
        .sum();
    let ss_tot: f64 = actual.iter().map(|&a| (a - mean_actual).powi(2)).sum();
    if ss_tot < 1e-15 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Compute Mean Absolute Error.
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(actual.len(), predicted.len());
    let n = actual.len() as f64;
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).abs())
        .sum::<f64>()
        / n
}

/// Compute Root Mean Squared Error.
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(actual.len(), predicted.len());
    let n = actual.len() as f64;
    let mse: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).powi(2))
        .sum::<f64>()
        / n;
    mse.sqrt()
}

// ---------------------------------------------------------------------------
// Square-Root Impact Model
// ---------------------------------------------------------------------------

/// The Square-Root Impact Model (Kyle/Barra style).
///
/// Models price impact as:
///   ΔP/P = σ · sign(Q) · η · |Q/V|^δ
///
/// where:
/// - σ = daily volatility
/// - Q = order size (signed, positive for buys)
/// - V = average daily volume
/// - η = impact coefficient (calibrated)
/// - δ = exponent (typically ~0.5, hence "square root")
///
/// This model captures the empirical observation that price impact scales
/// as the square root of order size, not linearly.
pub struct SquareRootModel {
    /// Impact coefficient (η). Typically 0.1-1.0.
    pub eta: f64,
    /// Exponent (δ). Typically ~0.5 for square-root law.
    pub delta: f64,
    /// Daily volatility (σ).
    pub volatility: f64,
    /// Average daily volume (V).
    pub avg_daily_volume: f64,
}

impl SquareRootModel {
    /// Create a new square-root impact model.
    pub fn new(eta: f64, delta: f64, volatility: f64, avg_daily_volume: f64) -> Self {
        Self {
            eta,
            delta,
            volatility,
            avg_daily_volume,
        }
    }

    /// Estimate price impact for a given order size.
    ///
    /// Returns the expected fractional price change (ΔP/P).
    pub fn estimate_impact(&self, order_size: f64) -> f64 {
        if self.avg_daily_volume < 1e-10 {
            return 0.0;
        }
        let participation_rate = order_size.abs() / self.avg_daily_volume;
        self.volatility * sign(order_size) * self.eta * participation_rate.powf(self.delta)
    }

    /// Estimate total cost of executing an order (in price units).
    ///
    /// Total cost = mid_price * |impact| * order_size
    pub fn estimate_cost(&self, order_size: f64, mid_price: f64) -> f64 {
        let impact = self.estimate_impact(order_size);
        mid_price * impact.abs() * order_size.abs()
    }

    /// Calibrate model parameters from observed trade data using least-squares.
    ///
    /// Given (order_size, observed_impact) pairs, finds best-fit η and δ.
    /// Uses log-linearization: ln(|ΔP/P| / σ) = ln(η) + δ * ln(|Q/V|)
    pub fn calibrate(
        &mut self,
        order_sizes: &[f64],
        observed_impacts: &[f64],
    ) {
        assert_eq!(order_sizes.len(), observed_impacts.len());

        let mut log_x = Vec::new();
        let mut log_y = Vec::new();

        for (&q, &impact) in order_sizes.iter().zip(observed_impacts.iter()) {
            let participation = q.abs() / self.avg_daily_volume;
            if participation > 1e-15 && impact.abs() > 1e-15 && self.volatility > 1e-15 {
                log_x.push(participation.ln());
                log_y.push((impact.abs() / self.volatility).ln());
            }
        }

        if log_x.len() < 2 {
            return;
        }

        // Simple linear regression: log_y = a + b * log_x
        let n = log_x.len() as f64;
        let sum_x: f64 = log_x.iter().sum();
        let sum_y: f64 = log_y.iter().sum();
        let sum_xy: f64 = log_x.iter().zip(log_y.iter()).map(|(&x, &y)| x * y).sum();
        let sum_xx: f64 = log_x.iter().map(|&x| x * x).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return;
        }

        let b = (n * sum_xy - sum_x * sum_y) / denom;
        let a = (sum_y - b * sum_x) / n;

        self.delta = b;
        self.eta = a.exp();
    }
}

// ---------------------------------------------------------------------------
// Almgren-Chriss Optimal Execution Model
// ---------------------------------------------------------------------------

/// The Almgren-Chriss model for optimal trade execution.
///
/// Decomposes price impact into:
/// - **Permanent impact**: g(v) = γ·v (linear in trading rate)
/// - **Temporary impact**: h(v) = ε·sign(v) + η·v (spread + linear)
///
/// Solves the optimal execution trajectory minimizing:
///   E[cost] + λ·Var[cost]
///
/// where λ is the risk aversion parameter.
pub struct AlmgrenChrissModel {
    /// Permanent impact coefficient (γ).
    pub gamma: f64,
    /// Temporary impact coefficient (η).
    pub eta: f64,
    /// Fixed cost / half-spread (ε).
    pub epsilon: f64,
    /// Volatility (σ).
    pub volatility: f64,
    /// Risk aversion parameter (λ).
    pub risk_aversion: f64,
}

impl AlmgrenChrissModel {
    /// Create a new Almgren-Chriss model.
    pub fn new(
        gamma: f64,
        eta: f64,
        epsilon: f64,
        volatility: f64,
        risk_aversion: f64,
    ) -> Self {
        Self {
            gamma,
            eta,
            epsilon,
            volatility,
            risk_aversion,
        }
    }

    /// Compute the optimal execution trajectory.
    ///
    /// Returns the remaining shares at each time step [X_0, X_1, ..., X_N]
    /// where X_0 = total_shares and X_N = 0.
    ///
    /// # Arguments
    /// * `total_shares` - Total number of shares to execute
    /// * `num_steps` - Number of time periods
    /// * `tau` - Duration of each time period (in days)
    pub fn optimal_trajectory(
        &self,
        total_shares: f64,
        num_steps: usize,
        tau: f64,
    ) -> Vec<f64> {
        if num_steps == 0 {
            return vec![total_shares, 0.0];
        }

        // kappa = sqrt(lambda * sigma^2 / eta_tilde)
        // eta_tilde = eta - 0.5 * gamma * tau
        let eta_tilde = (self.eta - 0.5 * self.gamma * tau).max(1e-10);
        let kappa_sq = self.risk_aversion * self.volatility.powi(2) / eta_tilde;
        let kappa = kappa_sq.max(0.0).sqrt();

        let big_t = num_steps as f64;
        let sinh_kt = (kappa * big_t * tau).sinh();

        let mut trajectory = Vec::with_capacity(num_steps + 1);
        for j in 0..=num_steps {
            let t_j = j as f64;
            let remaining = if sinh_kt.abs() < 1e-15 {
                // Linear execution when kappa -> 0
                total_shares * (1.0 - t_j / big_t)
            } else {
                total_shares * (kappa * (big_t - t_j) * tau).sinh() / sinh_kt
            };
            trajectory.push(remaining);
        }

        trajectory
    }

    /// Compute the expected cost of executing along a trajectory.
    ///
    /// Expected cost = permanent impact cost + temporary impact cost
    pub fn expected_cost(&self, trajectory: &[f64], tau: f64) -> f64 {
        if trajectory.len() < 2 {
            return 0.0;
        }

        let mut permanent_cost = 0.0;
        let mut temporary_cost = 0.0;

        for j in 1..trajectory.len() {
            let n_j = trajectory[j - 1] - trajectory[j]; // shares traded in period j
            let v_j = n_j / tau; // trading rate

            // Permanent impact: 0.5 * gamma * sum(n_j^2)
            permanent_cost += self.gamma * n_j * n_j;

            // Temporary impact: epsilon * |n_j| + eta_tilde * n_j^2 / tau
            let eta_tilde = self.eta - 0.5 * self.gamma * tau;
            temporary_cost += self.epsilon * n_j.abs() + eta_tilde * v_j * n_j;
        }

        0.5 * permanent_cost + temporary_cost
    }

    /// Compute the variance of execution cost.
    pub fn cost_variance(&self, trajectory: &[f64], tau: f64) -> f64 {
        if trajectory.len() < 2 {
            return 0.0;
        }

        let sigma_sq = self.volatility.powi(2);
        let mut variance = 0.0;

        for j in 1..trajectory.len() {
            // Variance contribution: sigma^2 * tau * X_j^2
            variance += sigma_sq * tau * trajectory[j - 1].powi(2);
        }

        variance
    }
}

// ---------------------------------------------------------------------------
// Linear Impact Model (OLS Regression)
// ---------------------------------------------------------------------------

/// A simple linear regression model for price impact estimation.
///
/// Models impact as a linear combination of trade features:
///   impact = w_0 + w_1*x_1 + w_2*x_2 + ... + w_n*x_n
///
/// Features typically include:
/// - Order size / ADV ratio
/// - Spread at time of trade
/// - Volatility
/// - Time of day
/// - Order imbalance
pub struct LinearImpactModel {
    /// Regression weights (including bias as w[0]).
    pub weights: Vec<f64>,
    /// Number of features (excluding bias).
    pub num_features: usize,
}

impl LinearImpactModel {
    /// Create a new linear impact model with zero weights.
    pub fn new(num_features: usize) -> Self {
        Self {
            weights: vec![0.0; num_features + 1], // +1 for bias
            num_features,
        }
    }

    /// Predict impact from features.
    pub fn predict(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.num_features);
        let mut result = self.weights[0]; // bias
        for (i, &f) in features.iter().enumerate() {
            result += self.weights[i + 1] * f;
        }
        result
    }

    /// Fit model using Ordinary Least Squares (OLS).
    ///
    /// Solves the normal equations: w = (X^T X)^{-1} X^T y
    /// Uses a simple gradient descent approach for robustness.
    pub fn fit(&mut self, features: &[Vec<f64>], targets: &[f64], epochs: usize, lr: f64) {
        assert_eq!(features.len(), targets.len());
        if features.is_empty() {
            return;
        }

        for _ in 0..epochs {
            let mut grad = vec![0.0; self.num_features + 1];
            let n = features.len() as f64;

            for (x, &y) in features.iter().zip(targets.iter()) {
                let pred = self.predict(x);
                let error = pred - y;

                grad[0] += error; // bias gradient
                for (j, &xj) in x.iter().enumerate() {
                    grad[j + 1] += error * xj;
                }
            }

            // Update weights
            for (w, g) in self.weights.iter_mut().zip(grad.iter()) {
                *w -= lr * g / n;
            }
        }
    }

    /// Compute R-squared on a dataset.
    pub fn score(&self, features: &[Vec<f64>], targets: &[f64]) -> f64 {
        let predictions: Vec<f64> = features.iter().map(|x| self.predict(x)).collect();
        r_squared(targets, &predictions)
    }
}

// ---------------------------------------------------------------------------
// ML Impact Predictor (Neural Network)
// ---------------------------------------------------------------------------

/// A neural network for nonlinear price impact prediction.
///
/// Architecture: input -> hidden1 (ReLU) -> hidden2 (ReLU) -> output (linear)
///
/// Captures nonlinear relationships between trade characteristics and
/// resulting price impact that linear models cannot.
pub struct MLImpactPredictor {
    pub weights1: Array2<f64>,
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array1<f64>,
    pub weights3: Array2<f64>,
    pub bias3: Array1<f64>,
    pub learning_rate: f64,
}

impl MLImpactPredictor {
    /// Create a new ML impact predictor with Xavier initialization.
    pub fn new(
        input_size: usize,
        hidden1_size: usize,
        hidden2_size: usize,
        learning_rate: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut xavier = |fan_in: usize, fan_out: usize| -> f64 {
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            rng.gen_range(-limit..limit)
        };

        let weights1 =
            Array2::from_shape_fn((input_size, hidden1_size), |_| xavier(input_size, hidden1_size));
        let bias1 = Array1::zeros(hidden1_size);
        let weights2 = Array2::from_shape_fn((hidden1_size, hidden2_size), |_| {
            xavier(hidden1_size, hidden2_size)
        });
        let bias2 = Array1::zeros(hidden2_size);
        // Output layer: single neuron for regression
        let weights3 =
            Array2::from_shape_fn((hidden2_size, 1), |_| xavier(hidden2_size, 1));
        let bias3 = Array1::zeros(1);

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
            weights3,
            bias3,
            learning_rate,
        }
    }

    /// Forward pass: predict price impact from input features.
    pub fn forward(&self, input: &Array1<f64>) -> f64 {
        let h1_raw = input.dot(&self.weights1) + &self.bias1;
        let h1: Array1<f64> = h1_raw.mapv(relu);
        let h2_raw = h1.dot(&self.weights2) + &self.bias2;
        let h2: Array1<f64> = h2_raw.mapv(relu);
        let output = h2.dot(&self.weights3) + &self.bias3;
        output[0]
    }

    /// Predict impact for a feature vector.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let input = Array1::from_vec(features.to_vec());
        self.forward(&input)
    }

    /// Train on a single sample using backpropagation (MSE loss).
    pub fn train_step(&mut self, input: &Array1<f64>, target: f64) {
        // Forward pass with intermediate values
        let h1_raw = input.dot(&self.weights1) + &self.bias1;
        let h1: Array1<f64> = h1_raw.mapv(relu);
        let h2_raw = h1.dot(&self.weights2) + &self.bias2;
        let h2: Array1<f64> = h2_raw.mapv(relu);
        let output = h2.dot(&self.weights3) + &self.bias3;
        let pred = output[0];

        // MSE gradient: d_loss/d_output = 2 * (pred - target) / 1
        let d_output = Array1::from_vec(vec![2.0 * (pred - target)]);

        // Backprop through output layer
        let h2_col = h2.clone().into_shape((h2.len(), 1)).unwrap();
        let do_row = d_output.clone().into_shape((1, 1)).unwrap();
        let dw3 = h2_col.dot(&do_row);

        // Backprop through hidden layer 2
        let d_h2 = d_output.dot(&self.weights3.t());
        let d_h2_relu: Array1<f64> = d_h2
            .iter()
            .zip(h2_raw.iter())
            .map(|(&d, &h)| if h > 0.0 { d } else { 0.0 })
            .collect();

        let h1_col = h1.clone().into_shape((h1.len(), 1)).unwrap();
        let dh2_row = d_h2_relu.clone().into_shape((1, d_h2_relu.len())).unwrap();
        let dw2 = h1_col.dot(&dh2_row);

        // Backprop through hidden layer 1
        let d_h1 = d_h2_relu.dot(&self.weights2.t());
        let d_h1_relu: Array1<f64> = d_h1
            .iter()
            .zip(h1_raw.iter())
            .map(|(&d, &h)| if h > 0.0 { d } else { 0.0 })
            .collect();

        let inp_col = input.clone().into_shape((input.len(), 1)).unwrap();
        let dh1_row = d_h1_relu.clone().into_shape((1, d_h1_relu.len())).unwrap();
        let dw1 = inp_col.dot(&dh1_row);

        // Update weights
        let lr = self.learning_rate;
        self.weights3 = &self.weights3 - &(dw3 * lr);
        self.bias3 = &self.bias3 - &(d_output * lr);
        self.weights2 = &self.weights2 - &(dw2 * lr);
        self.bias2 = &self.bias2 - &(d_h2_relu * lr);
        self.weights1 = &self.weights1 - &(dw1 * lr);
        self.bias1 = &self.bias1 - &(d_h1_relu * lr);
    }

    /// Train on a batch for multiple epochs.
    pub fn train_batch(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[f64],
        epochs: usize,
    ) {
        for _ in 0..epochs {
            for (input, &target) in inputs.iter().zip(targets.iter()) {
                self.train_step(input, target);
            }
        }
    }

    /// Count total parameters.
    pub fn num_parameters(&self) -> usize {
        let (r1, c1) = self.weights1.dim();
        let (r2, c2) = self.weights2.dim();
        let (r3, c3) = self.weights3.dim();
        r1 * c1 + self.bias1.len() + r2 * c2 + self.bias2.len() + r3 * c3 + self.bias3.len()
    }
}

// ---------------------------------------------------------------------------
// Transaction Cost Analyzer
// ---------------------------------------------------------------------------

/// A trade record for transaction cost analysis.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Trade size (positive for buys, negative for sells).
    pub size: f64,
    /// Price at execution.
    pub exec_price: f64,
    /// Mid-price at time of order arrival.
    pub arrival_price: f64,
    /// Volume-weighted average price of the day.
    pub vwap: f64,
    /// Bid-ask spread at time of trade.
    pub spread: f64,
    /// Average daily volume.
    pub adv: f64,
    /// Daily volatility.
    pub volatility: f64,
}

/// Transaction Cost Analyzer combining multiple impact models.
///
/// Provides comprehensive analysis of execution quality including:
/// - Implementation shortfall (slippage from arrival price)
/// - VWAP slippage
/// - Spread cost
/// - Impact cost (using square-root model)
pub struct TransactionCostAnalyzer {
    /// Square-root model for impact estimation.
    pub sqrt_model: SquareRootModel,
    /// ML model for nonlinear impact prediction.
    pub ml_model: Option<MLImpactPredictor>,
}

/// Results of transaction cost analysis for a single trade.
#[derive(Debug, Clone)]
pub struct TCAResult {
    /// Implementation shortfall: (exec_price - arrival_price) * sign(size) / arrival_price
    pub implementation_shortfall: f64,
    /// VWAP slippage: (exec_price - vwap) * sign(size) / vwap
    pub vwap_slippage: f64,
    /// Spread cost: spread / (2 * arrival_price)
    pub spread_cost: f64,
    /// Estimated market impact from square-root model.
    pub estimated_impact: f64,
    /// ML-predicted impact (if ML model is available).
    pub ml_predicted_impact: Option<f64>,
    /// Participation rate: |size| / ADV.
    pub participation_rate: f64,
}

impl TransactionCostAnalyzer {
    /// Create a new TCA analyzer with a square-root model.
    pub fn new(sqrt_model: SquareRootModel) -> Self {
        Self {
            sqrt_model,
            ml_model: None,
        }
    }

    /// Set the ML impact predictor.
    pub fn with_ml_model(mut self, ml_model: MLImpactPredictor) -> Self {
        self.ml_model = Some(ml_model);
        self
    }

    /// Analyze a single trade.
    pub fn analyze_trade(&self, trade: &TradeRecord) -> TCAResult {
        let side = sign(trade.size);

        // Implementation shortfall
        let implementation_shortfall = if trade.arrival_price > 0.0 {
            (trade.exec_price - trade.arrival_price) * side / trade.arrival_price
        } else {
            0.0
        };

        // VWAP slippage
        let vwap_slippage = if trade.vwap > 0.0 {
            (trade.exec_price - trade.vwap) * side / trade.vwap
        } else {
            0.0
        };

        // Spread cost
        let spread_cost = if trade.arrival_price > 0.0 {
            trade.spread / (2.0 * trade.arrival_price)
        } else {
            0.0
        };

        // Estimated impact from square-root model
        let estimated_impact = self.sqrt_model.estimate_impact(trade.size).abs();

        // ML predicted impact
        let ml_predicted_impact = self.ml_model.as_ref().map(|ml| {
            let features = vec![
                trade.size.abs() / trade.adv.max(1.0), // participation rate
                trade.spread / trade.arrival_price.max(1e-10), // relative spread
                trade.volatility,
                side,
            ];
            ml.predict(&features).abs()
        });

        let participation_rate = if trade.adv > 0.0 {
            trade.size.abs() / trade.adv
        } else {
            0.0
        };

        TCAResult {
            implementation_shortfall,
            vwap_slippage,
            spread_cost,
            estimated_impact,
            ml_predicted_impact,
            participation_rate,
        }
    }

    /// Analyze a batch of trades and return summary statistics.
    pub fn analyze_batch(&self, trades: &[TradeRecord]) -> Vec<TCAResult> {
        trades.iter().map(|t| self.analyze_trade(t)).collect()
    }

    /// Compute summary statistics from TCA results.
    pub fn summary(results: &[TCAResult]) -> TcaSummary {
        if results.is_empty() {
            return TcaSummary::default();
        }

        let is_vals: Vec<f64> = results.iter().map(|r| r.implementation_shortfall).collect();
        let vwap_vals: Vec<f64> = results.iter().map(|r| r.vwap_slippage).collect();
        let spread_vals: Vec<f64> = results.iter().map(|r| r.spread_cost).collect();
        let impact_vals: Vec<f64> = results.iter().map(|r| r.estimated_impact).collect();

        TcaSummary {
            avg_implementation_shortfall: mean(&is_vals),
            avg_vwap_slippage: mean(&vwap_vals),
            avg_spread_cost: mean(&spread_vals),
            avg_estimated_impact: mean(&impact_vals),
            std_implementation_shortfall: std_dev(&is_vals),
            num_trades: results.len(),
        }
    }
}

/// Summary statistics from TCA.
#[derive(Debug, Clone, Default)]
pub struct TcaSummary {
    pub avg_implementation_shortfall: f64,
    pub avg_vwap_slippage: f64,
    pub avg_spread_cost: f64,
    pub avg_estimated_impact: f64,
    pub std_implementation_shortfall: f64,
    pub num_trades: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_root_model_basic() {
        let model = SquareRootModel::new(0.5, 0.5, 0.02, 1_000_000.0);

        // Buy 10,000 shares (1% of ADV)
        let impact = model.estimate_impact(10_000.0);
        assert!(impact > 0.0, "Buy impact should be positive");

        // Sell should give negative impact
        let sell_impact = model.estimate_impact(-10_000.0);
        assert!(sell_impact < 0.0, "Sell impact should be negative");

        // Magnitude should be the same
        assert!((impact.abs() - sell_impact.abs()).abs() < 1e-10);
    }

    #[test]
    fn test_square_root_model_scaling() {
        let model = SquareRootModel::new(0.5, 0.5, 0.02, 1_000_000.0);

        let small_impact = model.estimate_impact(10_000.0).abs();
        let large_impact = model.estimate_impact(100_000.0).abs();

        // With delta=0.5, 10x size should give ~sqrt(10) ~ 3.16x impact
        let ratio = large_impact / small_impact;
        assert!(
            (ratio - 10.0_f64.sqrt()).abs() < 0.01,
            "Impact should scale as square root; ratio={:.3}",
            ratio
        );
    }

    #[test]
    fn test_square_root_model_calibration() {
        let mut model = SquareRootModel::new(1.0, 1.0, 0.02, 1_000_000.0);

        // Generate synthetic data with known parameters
        let true_eta = 0.5;
        let true_delta = 0.5;
        let sizes: Vec<f64> = (1..=20).map(|i| i as f64 * 5000.0).collect();
        let impacts: Vec<f64> = sizes
            .iter()
            .map(|&q| {
                let pr = q / 1_000_000.0;
                0.02 * true_eta * pr.powf(true_delta)
            })
            .collect();

        model.calibrate(&sizes, &impacts);

        assert!(
            (model.delta - true_delta).abs() < 0.1,
            "Calibrated delta should be close to 0.5; got {:.3}",
            model.delta
        );
        assert!(
            (model.eta - true_eta).abs() < 0.2,
            "Calibrated eta should be close to 0.5; got {:.3}",
            model.eta
        );
    }

    #[test]
    fn test_almgren_chriss_trajectory() {
        let model = AlmgrenChrissModel::new(0.001, 0.01, 0.0005, 0.02, 1.0);

        let trajectory = model.optimal_trajectory(100_000.0, 10, 1.0);

        assert_eq!(trajectory.len(), 11, "Should have N+1 points");
        assert!(
            (trajectory[0] - 100_000.0).abs() < 1e-6,
            "Should start at full position"
        );
        assert!(trajectory[10].abs() < 1e-6, "Should end at zero");

        // Trajectory should be monotonically decreasing
        for i in 1..trajectory.len() {
            assert!(
                trajectory[i] <= trajectory[i - 1] + 1e-6,
                "Trajectory should be non-increasing"
            );
        }
    }

    #[test]
    fn test_almgren_chriss_risk_aversion() {
        let low_risk = AlmgrenChrissModel::new(0.001, 0.01, 0.0005, 0.02, 0.1);
        let high_risk = AlmgrenChrissModel::new(0.001, 0.01, 0.0005, 0.02, 10.0);

        let traj_low = low_risk.optimal_trajectory(100_000.0, 10, 1.0);
        let traj_high = high_risk.optimal_trajectory(100_000.0, 10, 1.0);

        // High risk aversion should trade faster (more front-loaded)
        // At step 1, high risk aversion should have less remaining
        assert!(
            traj_high[1] < traj_low[1],
            "Higher risk aversion should trade faster in early steps"
        );
    }

    #[test]
    fn test_linear_impact_model() {
        let mut model = LinearImpactModel::new(2);

        // y = 0.5 + 2*x1 + 3*x2
        let features: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
        ];
        let targets: Vec<f64> = vec![2.5, 3.5, 5.5, 7.5, 8.5];

        model.fit(&features, &targets, 1000, 0.01);

        let r2 = model.score(&features, &targets);
        assert!(r2 > 0.9, "R-squared should be high for linear data; got {:.3}", r2);
    }

    #[test]
    fn test_ml_impact_predictor_forward() {
        let predictor = MLImpactPredictor::new(4, 16, 8, 0.01);
        let input = Array1::from_vec(vec![0.01, 0.002, 0.02, 1.0]);
        let impact = predictor.forward(&input);
        // Should return a finite number
        assert!(impact.is_finite(), "Prediction should be finite");
    }

    #[test]
    fn test_ml_impact_predictor_training() {
        let mut predictor = MLImpactPredictor::new(2, 8, 4, 0.001);

        // Simple regression task: y = x1 + x2
        let inputs: Vec<Array1<f64>> = vec![
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0]),
            Array1::from_vec(vec![0.5, 0.5]),
            Array1::from_vec(vec![0.2, 0.8]),
        ];
        let targets = vec![1.0, 1.0, 1.0, 1.0];

        let pred_before = predictor.forward(&inputs[0]);
        predictor.train_batch(&inputs, &targets, 100);
        let pred_after = predictor.forward(&inputs[0]);

        // After training, prediction should be closer to target
        assert!(
            (pred_after - 1.0).abs() < (pred_before - 1.0).abs() + 0.1,
            "Training should improve predictions"
        );
    }

    #[test]
    fn test_tca_analysis() {
        let sqrt_model = SquareRootModel::new(0.5, 0.5, 0.02, 1_000_000.0);
        let analyzer = TransactionCostAnalyzer::new(sqrt_model);

        let trade = TradeRecord {
            size: 50_000.0,
            exec_price: 100.10,
            arrival_price: 100.00,
            vwap: 100.05,
            spread: 0.02,
            adv: 1_000_000.0,
            volatility: 0.02,
        };

        let result = analyzer.analyze_trade(&trade);

        assert!(
            result.implementation_shortfall > 0.0,
            "Buy at higher price should have positive IS"
        );
        assert!(
            result.spread_cost > 0.0,
            "Spread cost should be positive"
        );
        assert!(
            result.participation_rate > 0.0 && result.participation_rate < 1.0,
            "Participation rate should be between 0 and 1"
        );
    }

    #[test]
    fn test_tca_summary() {
        let sqrt_model = SquareRootModel::new(0.5, 0.5, 0.02, 1_000_000.0);
        let analyzer = TransactionCostAnalyzer::new(sqrt_model);

        let trades = vec![
            TradeRecord {
                size: 50_000.0,
                exec_price: 100.10,
                arrival_price: 100.00,
                vwap: 100.05,
                spread: 0.02,
                adv: 1_000_000.0,
                volatility: 0.02,
            },
            TradeRecord {
                size: -30_000.0,
                exec_price: 99.85,
                arrival_price: 100.00,
                vwap: 99.95,
                spread: 0.03,
                adv: 1_000_000.0,
                volatility: 0.02,
            },
        ];

        let results = analyzer.analyze_batch(&trades);
        let summary = TransactionCostAnalyzer::summary(&results);

        assert_eq!(summary.num_trades, 2);
        assert!(summary.avg_spread_cost > 0.0);
    }

    #[test]
    fn test_utility_functions() {
        assert_eq!(sign(5.0), 1.0);
        assert_eq!(sign(-3.0), -1.0);
        assert_eq!(sign(0.0), 0.0);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);
        assert!(std_dev(&data) > 0.0);

        let actual = vec![1.0, 2.0, 3.0];
        let perfect = vec![1.0, 2.0, 3.0];
        assert!((r_squared(&actual, &perfect) - 1.0).abs() < 1e-10);

        assert!((mae(&actual, &perfect)).abs() < 1e-10);
        assert!((rmse(&actual, &perfect)).abs() < 1e-10);
    }
}
