//! # Trading Example: Price Impact Modeling
//!
//! This example demonstrates end-to-end price impact analysis using
//! real market data from the Bybit API:
//!
//! 1. Fetch OHLCV candle data for BTCUSDT from Bybit
//! 2. Simulate trades and compute realized price impact
//! 3. Calibrate the square-root impact model
//! 4. Compute optimal execution trajectories (Almgren-Chriss)
//! 5. Train an ML impact predictor
//! 6. Run full transaction cost analysis
//!
//! Usage: cargo run --example trading_example

use anyhow::Result;
use ndarray::Array1;
use price_impact_modeling::*;
use rand::Rng;
use serde::Deserialize;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Bybit API Data Structures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone)]
struct Candle {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

// ---------------------------------------------------------------------------
// Data Fetching
// ---------------------------------------------------------------------------

async fn fetch_bybit_candles(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    println!("Fetching data from Bybit API...");
    println!("  URL: {}", url);

    let client = reqwest::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "ML-Trading-Bot/1.0")
        .send()
        .await?
        .json()
        .await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    open: row[1].as_str()?.parse().ok()?,
                    high: row[2].as_str()?.parse().ok()?,
                    low: row[3].as_str()?.parse().ok()?,
                    close: row[4].as_str()?.parse().ok()?,
                    volume: row[5].as_str()?.parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    candles.reverse(); // Bybit returns newest first
    println!("  Fetched {} candles for {}", candles.len(), symbol);
    Ok(candles)
}

// ---------------------------------------------------------------------------
// Feature Engineering & Simulation
// ---------------------------------------------------------------------------

/// Compute daily volatility from candle data.
fn compute_volatility(candles: &[Candle]) -> f64 {
    let returns: Vec<f64> = candles
        .windows(2)
        .filter_map(|w| {
            if w[0].close > 0.0 {
                Some((w[1].close / w[0].close).ln())
            } else {
                None
            }
        })
        .collect();
    std_dev(&returns)
}

/// Compute average daily volume from candle data.
fn compute_adv(candles: &[Candle]) -> f64 {
    if candles.is_empty() {
        return 0.0;
    }
    candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64
}

/// Simulate trade records from candle data.
///
/// For each candle, we simulate a market order arriving at the open
/// and filling at various points within the candle.
fn simulate_trades(candles: &[Candle], adv: f64, volatility: f64) -> Vec<TradeRecord> {
    let mut rng = rand::thread_rng();
    let mut trades = Vec::new();

    for i in 1..candles.len().saturating_sub(1) {
        let c = &candles[i];
        let _prev = &candles[i - 1];

        // Simulate order size as fraction of ADV
        let participation = rng.gen_range(0.001..0.05);
        let size = adv * participation * if rng.gen_bool(0.5) { 1.0 } else { -1.0 };

        // Simulated execution price (slipped from open)
        let slippage_bps = participation.sqrt() * volatility * 10000.0;
        let exec_price = c.open * (1.0 + sign(size) * slippage_bps * 0.0001);

        // VWAP approximation: (open + high + low + close) / 4
        let vwap = (c.open + c.high + c.low + c.close) / 4.0;

        // Spread approximation from high-low range
        let spread = (c.high - c.low) * 0.1;

        trades.push(TradeRecord {
            size,
            exec_price,
            arrival_price: c.open,
            vwap,
            spread,
            adv,
            volatility,
        });
    }

    trades
}

/// Generate synthetic data as fallback.
fn generate_synthetic_data() -> (Vec<Candle>, f64, f64) {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::new();
    let mut price = 50000.0;

    for _ in 0..200 {
        let ret: f64 = rng.gen_range(-0.02..0.02);
        let open: f64 = price;
        let close: f64 = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(500.0..2000.0);

        candles.push(Candle {
            open,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }

    let volatility = compute_volatility(&candles);
    let adv = compute_adv(&candles);

    (candles, adv, volatility)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    println!("=======================================================");
    println!("          Price Impact Modeling - Trading Example");
    println!("=======================================================\n");

    // --- Fetch or generate data ---
    let (candles, adv, volatility) = match fetch_bybit_candles("BTCUSDT", "15", 200).await {
        Ok(c) if c.len() > 20 => {
            println!("  Successfully fetched {} candles\n", c.len());
            let vol = compute_volatility(&c);
            let adv = compute_adv(&c);
            (c, adv, vol)
        }
        Ok(_) => {
            println!("  Insufficient candles, using synthetic data\n");
            generate_synthetic_data()
        }
        Err(e) => {
            println!("  API fetch failed ({}), using synthetic data\n", e);
            generate_synthetic_data()
        }
    };

    println!(
        "  Data: {} candles, ADV={:.1}, volatility={:.4}\n",
        candles.len(),
        adv,
        volatility
    );

    // -----------------------------------------------------------------------
    // Step 1: Square-Root Impact Model
    // -----------------------------------------------------------------------
    println!("--- Step 1: Square-Root Impact Model ---");
    let mut sqrt_model = SquareRootModel::new(0.5, 0.5, volatility, adv);

    // Simulate trades for calibration
    let trades = simulate_trades(&candles, adv, volatility);
    println!("  Simulated {} trades for calibration", trades.len());

    // Compute realized impacts for calibration
    let order_sizes: Vec<f64> = trades.iter().map(|t| t.size).collect();
    let realized_impacts: Vec<f64> = trades
        .iter()
        .map(|t| {
            if t.arrival_price > 0.0 {
                (t.exec_price - t.arrival_price) * sign(t.size) / t.arrival_price
            } else {
                0.0
            }
        })
        .collect();

    sqrt_model.calibrate(&order_sizes, &realized_impacts);
    println!(
        "  Calibrated: eta={:.4}, delta={:.4}",
        sqrt_model.eta, sqrt_model.delta
    );

    // Show impact for various order sizes
    println!("\n  Impact estimates (buy orders):");
    for pct in &[0.1, 0.5, 1.0, 2.0, 5.0] {
        let size = adv * pct / 100.0;
        let impact = sqrt_model.estimate_impact(size);
        println!(
            "    {:.1}% ADV ({:.0} units): {:.2} bps impact",
            pct,
            size,
            impact * 10000.0
        );
    }

    // -----------------------------------------------------------------------
    // Step 2: Almgren-Chriss Optimal Execution
    // -----------------------------------------------------------------------
    println!("\n--- Step 2: Almgren-Chriss Optimal Execution ---");
    let total_shares = adv * 0.05; // 5% of ADV
    println!("  Executing {:.0} units (5% of ADV)", total_shares);

    for &risk_aversion in &[0.1, 1.0, 10.0] {
        let ac_model = AlmgrenChrissModel::new(
            0.001,    // gamma: permanent impact
            0.01,     // eta: temporary impact
            0.0005,   // epsilon: half-spread
            volatility,
            risk_aversion,
        );

        let trajectory = ac_model.optimal_trajectory(total_shares, 10, 1.0);
        let cost = ac_model.expected_cost(&trajectory, 1.0);
        let variance = ac_model.cost_variance(&trajectory, 1.0);

        println!(
            "\n  Risk aversion lambda={:.1}:",
            risk_aversion
        );
        println!("    Expected cost: {:.2}", cost);
        println!("    Cost variance: {:.2}", variance);
        println!("    Trajectory (% remaining):");
        print!("    ");
        for (i, &x) in trajectory.iter().enumerate() {
            print!(
                "t{}={:.0}% ",
                i,
                x / total_shares * 100.0
            );
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Step 3: Linear Impact Model
    // -----------------------------------------------------------------------
    println!("\n--- Step 3: Linear Impact Model ---");
    let mut linear_model = LinearImpactModel::new(4);

    // Prepare features: participation_rate, relative_spread, volatility, side
    let features: Vec<Vec<f64>> = trades
        .iter()
        .map(|t| {
            vec![
                t.size.abs() / t.adv.max(1.0),
                t.spread / t.arrival_price.max(1e-10),
                t.volatility,
                sign(t.size),
            ]
        })
        .collect();

    let split = (features.len() * 4) / 5;
    let train_features = features[..split].to_vec();
    let train_targets = realized_impacts[..split].to_vec();
    let test_features = features[split..].to_vec();
    let test_targets = realized_impacts[split..].to_vec();

    let start = Instant::now();
    linear_model.fit(&train_features, &train_targets, 500, 0.01);
    let linear_time = start.elapsed();

    let r2_train = linear_model.score(&train_features, &train_targets);
    let r2_test = linear_model.score(&test_features, &test_targets);
    println!("  Train R-squared: {:.4}", r2_train);
    println!("  Test R-squared:  {:.4}", r2_test);
    println!("  Training time:   {:?}", linear_time);
    println!("  Weights: {:?}", linear_model.weights);

    // -----------------------------------------------------------------------
    // Step 4: ML Impact Predictor (Neural Network)
    // -----------------------------------------------------------------------
    println!("\n--- Step 4: ML Impact Predictor (Neural Network) ---");
    let mut ml_predictor = MLImpactPredictor::new(4, 16, 8, 0.0005);
    println!(
        "  Architecture: 4 -> 16 -> 8 -> 1 ({} params)",
        ml_predictor.num_parameters()
    );

    let train_inputs_nn: Vec<Array1<f64>> = train_features
        .iter()
        .map(|f| Array1::from_vec(f.clone()))
        .collect();
    let test_inputs_nn: Vec<Array1<f64>> = test_features
        .iter()
        .map(|f| Array1::from_vec(f.clone()))
        .collect();

    let start = Instant::now();
    ml_predictor.train_batch(&train_inputs_nn, &train_targets, 100);
    let nn_time = start.elapsed();

    let nn_train_preds: Vec<f64> = train_inputs_nn
        .iter()
        .map(|x| ml_predictor.forward(x))
        .collect();
    let nn_test_preds: Vec<f64> = test_inputs_nn
        .iter()
        .map(|x| ml_predictor.forward(x))
        .collect();

    let nn_r2_train = r_squared(&train_targets, &nn_train_preds);
    let nn_r2_test = r_squared(&test_targets, &nn_test_preds);
    let nn_rmse_test = rmse(&test_targets, &nn_test_preds);

    println!("  Train R-squared: {:.4}", nn_r2_train);
    println!("  Test R-squared:  {:.4}", nn_r2_test);
    println!("  Test RMSE:       {:.6}", nn_rmse_test);
    println!("  Training time:   {:?}", nn_time);

    // -----------------------------------------------------------------------
    // Step 5: Transaction Cost Analysis
    // -----------------------------------------------------------------------
    println!("\n--- Step 5: Transaction Cost Analysis ---");
    let tca_sqrt_model = SquareRootModel::new(sqrt_model.eta, sqrt_model.delta, volatility, adv);
    let analyzer = TransactionCostAnalyzer::new(tca_sqrt_model)
        .with_ml_model(MLImpactPredictor::new(4, 16, 8, 0.001));

    let results = analyzer.analyze_batch(&trades);
    let summary = TransactionCostAnalyzer::summary(&results);

    println!("  Number of trades:              {}", summary.num_trades);
    println!(
        "  Avg implementation shortfall:  {:.2} bps",
        summary.avg_implementation_shortfall * 10000.0
    );
    println!(
        "  Std implementation shortfall:  {:.2} bps",
        summary.std_implementation_shortfall * 10000.0
    );
    println!(
        "  Avg VWAP slippage:            {:.2} bps",
        summary.avg_vwap_slippage * 10000.0
    );
    println!(
        "  Avg spread cost:              {:.2} bps",
        summary.avg_spread_cost * 10000.0
    );
    println!(
        "  Avg estimated impact:         {:.2} bps",
        summary.avg_estimated_impact * 10000.0
    );

    // Show individual trade analysis for first 5 trades
    println!("\n  Sample trade analysis:");
    for (i, result) in results.iter().take(5).enumerate() {
        println!(
            "    Trade {}: IS={:.2}bps, VWAP={:.2}bps, spread={:.2}bps, impact={:.2}bps, PR={:.2}%",
            i + 1,
            result.implementation_shortfall * 10000.0,
            result.vwap_slippage * 10000.0,
            result.spread_cost * 10000.0,
            result.estimated_impact * 10000.0,
            result.participation_rate * 100.0,
        );
    }

    // -----------------------------------------------------------------------
    // Step 6: Inference Speed Comparison
    // -----------------------------------------------------------------------
    println!("\n--- Step 6: Inference Speed Comparison ---");
    let test_features_sample = vec![0.01, 0.0002, 0.02, 1.0];
    let test_input_nn = Array1::from_vec(test_features_sample.clone());
    let iterations = 10_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = sqrt_model.estimate_impact(adv * 0.01);
    }
    let sqrt_ns = start.elapsed().as_nanos() / iterations as u128;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = linear_model.predict(&test_features_sample);
    }
    let linear_ns = start.elapsed().as_nanos() / iterations as u128;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ml_predictor.forward(&test_input_nn);
    }
    let nn_ns = start.elapsed().as_nanos() / iterations as u128;

    println!("  Square-root model: {} ns/prediction", sqrt_ns);
    println!("  Linear model:      {} ns/prediction", linear_ns);
    println!("  Neural network:    {} ns/prediction", nn_ns);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=======================================================");
    println!("                      SUMMARY");
    println!("=======================================================");
    println!("  Square-root model: eta={:.4}, delta={:.4}", sqrt_model.eta, sqrt_model.delta);
    println!("  Linear model R2:   {:.4} (train), {:.4} (test)", r2_train, r2_test);
    println!("  NN model R2:       {:.4} (train), {:.4} (test)", nn_r2_train, nn_r2_test);
    println!(
        "  Avg IS:            {:.2} bps",
        summary.avg_implementation_shortfall * 10000.0
    );
    println!(
        "  Avg spread cost:   {:.2} bps",
        summary.avg_spread_cost * 10000.0
    );
    println!("=======================================================");

    Ok(())
}
