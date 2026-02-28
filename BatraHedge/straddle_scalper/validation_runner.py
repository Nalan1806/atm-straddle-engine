"""
Robustness & Edge Validation Runner
====================================
Orchestrates all tests and generates comprehensive report.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from .config import StrategyParams, OUTPUT_DIR
from .data_loader import load_data, build_intraday_panel
from .backtester import trades_to_dataframe, run_backtest
from .metrics import compute_metrics
from .robustness_validator import (
    parameter_sensitivity_grid,
    rolling_window_validation,
    regime_segmentation,
    cost_sensitivity_test,
    convexity_test,
    plot_sharpe_heatmap,
    plot_rolling_performance,
    plot_regime_performance,
    plot_cost_sensitivity,
    plot_convexity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_full_validation() -> None:
    """Execute the full robustness & edge validation suite."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ================================================================= LOAD
    logger.info("=" * 80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 80)
    ce_df, pe_df = load_data(strike_filter="ATM")
    panel = build_intraday_panel(ce_df, pe_df)
    logger.info("Panel shape: %d rows × %d columns", len(panel), len(panel.columns))
    
    # ================================================================= 1. PARAMETER GRID
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: PARAMETER SENSITIVITY GRID TEST")
    logger.info("=" * 80)
    try:
        results_grid, top_sharpe, top_pf = parameter_sensitivity_grid(panel)
    except Exception as e:
        logger.error("ERROR in parameter_sensitivity_grid: %s", str(e), exc_info=True)
        raise
    
    logger.info("\nTop 10 Configurations by Sharpe Ratio:")
    logger.info("\n%s", top_sharpe.to_string(index=False))
    
    logger.info("\nTop 10 Configurations by Profit Factor:")
    logger.info("\n%s", top_pf.to_string(index=False))
    
    # Export
    results_grid.to_csv(OUTPUT_DIR / "param_sensitivity_all.csv", index=False)
    top_sharpe.to_csv(OUTPUT_DIR / "param_sensitivity_top_sharpe.csv", index=False)
    top_pf.to_csv(OUTPUT_DIR / "param_sensitivity_top_pf.csv", index=False)
    logger.info("✓ Parameter sensitivity results saved.")
    
    # ================================================================= 2. ROLLING WINDOWS
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: ROLLING WINDOW VALIDATION")
    logger.info("=" * 80)
    
    best_config = top_sharpe.iloc[0] if len(top_sharpe) > 0 else None
    if best_config is not None:
        try:
            params_best = StrategyParams(
                profit_target=best_config["profit_target"],
                stop_loss=best_config["stop_loss"],
                exit_time=best_config["exit_time"],
                iv_momentum_threshold=best_config["iv_momentum_threshold"],
            )
            
            split_results, rolling_df = rolling_window_validation(panel, params_best, window_months=6)
            
            logger.info("\n50/50 Split Results:")
            logger.info("\n%s", split_results.to_string(index=False))
            
            logger.info("\nRolling Window (6-month) Results:")
            logger.info("\n%s", rolling_df.to_string(index=False) if not rolling_df.empty else "No rolling data")
            
            split_results.to_csv(OUTPUT_DIR / "rolling_split_validation.csv", index=False)
            rolling_df.to_csv(OUTPUT_DIR / "rolling_window_6month.csv", index=False)
            logger.info("✓ Rolling window results saved.")
        except Exception as e:
            logger.error("ERROR in rolling_window_validation: %s", str(e), exc_info=True)
            split_results = None
            rolling_df = pd.DataFrame()
    else:
        logger.warning("No best config found. Skipping rolling windows.")
        split_results = None
        rolling_df = pd.DataFrame()
    
    # ================================================================= 3. REGIME ANALYSIS
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: REGIME SEGMENTATION")
    logger.info("=" * 80)
    
    try:
        regime_res = regime_segmentation(panel, params_best)
        
        logger.info("\nRegime Analysis Results:")
        for regime_name, regime_metrics in regime_res.items():
            logger.info("\n  %s:", regime_name)
            for key, val in regime_metrics.items():
                logger.info("    %-20s : %s", key, val)
        
        regime_df = pd.DataFrame([{**{"regime": k}, **v} for k, v in regime_res.items()])
        regime_df.to_csv(OUTPUT_DIR / "regime_segmentation.csv", index=False)
        logger.info("✓ Regime segmentation saved.")
    except Exception as e:
        logger.error("ERROR in regime_segmentation: %s", str(e), exc_info=True)
        regime_res = {}
        regime_df = pd.DataFrame()
    
    # ================================================================= 4. COST SENSITIVITY
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: COST SENSITIVITY TEST")
    logger.info("=" * 80)
    
    try:
        top_configs_list = top_sharpe.head(5).to_dict("records")
        cost_df = cost_sensitivity_test(panel, top_configs_list, cost_multipliers=[1.0, 0.5, 0.0])
        
        logger.info("\nCost Sensitivity Results:")
        logger.info("\n%s", cost_df.to_string(index=False))
        cost_df.to_csv(OUTPUT_DIR / "cost_sensitivity.csv", index=False)
        logger.info("✓ Cost sensitivity results saved.")
    except Exception as e:
        logger.error("ERROR in cost_sensitivity_test: %s", str(e), exc_info=True)
        cost_df = pd.DataFrame()
    
    # ================================================================= 5. CONVEXITY
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: CONVEXITY ANALYSIS")
    logger.info("=" * 80)
    
    try:
        trades_best = run_backtest(panel, params_best)
        trade_df_best = trades_to_dataframe(trades_best)
        convexity_res = convexity_test(trade_df_best)
        
        logger.info("\nConvexity Analysis:")
        for key, val in convexity_res.items():
            logger.info("  %-35s : %s", key, val)
    except Exception as e:
        logger.error("ERROR in convexity_test: %s", str(e), exc_info=True)
        convexity_res = {}
    
    # ================================================================= 6. PLOTS
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    try:
        plot_sharpe_heatmap(results_grid, OUTPUT_DIR)
        logger.info("  ✓ Sharpe heatmaps generated")
    except Exception as e:
        logger.error("  ✗ Sharpe heatmap failed: %s", str(e))
    
    try:
        if not rolling_df.empty:
            plot_rolling_performance(split_results, rolling_df, OUTPUT_DIR)
            logger.info("  ✓ Rolling performance plots generated")
    except Exception as e:
        logger.error("  ✗ Rolling performance plot failed: %s", str(e))
    
    try:
        plot_regime_performance(regime_res, OUTPUT_DIR)
        logger.info("  ✓ Regime performance plots generated")
    except Exception as e:
        logger.error("  ✗ Regime performance plot failed: %s", str(e))
    
    try:
        if not cost_df.empty:
            plot_cost_sensitivity(cost_df, OUTPUT_DIR)
            logger.info("  ✓ Cost sensitivity plots generated")
    except Exception as e:
        logger.error("  ✗ Cost sensitivity plot failed: %s", str(e))
    
    try:
        plot_convexity(trade_df_best, OUTPUT_DIR)
        logger.info("  ✓ Pareto/convexity plot generated")
    except Exception as e:
        logger.error("  ✗ Pareto plot failed: %s", str(e))
    
    # ================================================================= FINAL REPORT
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY & EDGE ASSESSMENT")
    logger.info("=" * 80)
    
    # === Stability ===
    overall_sharpe_mean = results_grid["sharpe"].replace([float("inf"), float("-inf")], float("nan")).mean()
    overall_sharpe_std = results_grid["sharpe"].replace([float("inf"), float("-inf")], float("nan")).std()
    sharpe_stability = "STABLE" if overall_sharpe_std < abs(overall_sharpe_mean) else "UNSTABLE"
    
    logger.info("\n[1. PARAMETER STABILITY]")
    logger.info("  Mean Sharpe across all configs:        %.2f", overall_sharpe_mean)
    logger.info("  Std Dev of Sharpe:                     %.2f", overall_sharpe_std)
    logger.info("  Stability Assessment:                  %s", sharpe_stability)
    
    # === Rolling Performance ===
    if not rolling_df.empty:
        rolling_avg_return = rolling_df["total_return"].mean()
        rolling_return_positive_pct = (rolling_df["total_return"] > 0).sum() / len(rolling_df) * 100 if len(rolling_df) > 0 else 0
        
        logger.info("\n[2. ROLLING WINDOW CONSISTENCY]")
        logger.info("  Avg 6-month return:                   %.2f%%", rolling_avg_return * 100)
        logger.info("  %% periods with positive return:      %.1f%%", rolling_return_positive_pct)
        logger.info("  Consistency:                          %s", "CONSISTENT" if rolling_return_positive_pct > 50 else "INCONSISTENT")
    
    # === Regime Dependency ===
    if regime_res:
        iv_rising_return = regime_res.get("iv_rising", {}).get("return", 0)
        iv_falling_return = regime_res.get("iv_falling", {}).get("return", 0)
        regime_dependent = abs(iv_rising_return - iv_falling_return) > 0.05  # >5% difference
        
        logger.info("\n[3. REGIME DEPENDENCY]")
        logger.info("  Return in IV-Rising regime:          %.2f%%", iv_rising_return * 100)
        logger.info("  Return in IV-Falling regime:         %.2f%%", iv_falling_return * 100)
        logger.info("  Regime Dependency:                   %s", "DEPENDENT" if regime_dependent else "INDEPENDENT")
    
    # === Cost Impact ===
    if not cost_df.empty:
        has_gross_edge = (cost_df["gross_pnl"] > 0).any()
        has_net_edge = (cost_df["net_pnl"] > 0).any()
        
        logger.info("\n[4. COST SENSITIVITY]")
        logger.info("  Gross edge (before costs):           %s", "YES" if has_gross_edge else "NO")
        logger.info("  Net edge (after costs):              %s", "YES" if has_net_edge else "NO")
        cost_impact = "CRITICAL" if not has_net_edge else "MANAGEABLE"
        logger.info("  Cost Impact:                         %s", cost_impact)
    
    # === Convexity ===
    logger.info("\n[5. EDGE CONCENTRATION (CONVEXITY)]")
    logger.info("  Top 10 trades contribute:              %.1f%%", convexity_res.get("top_10_contribution_pct", 0))
    logger.info("  Is edge concentrated?                  %s", convexity_res.get("is_edge_concentrated", False))
    logger.info("  Trades needed for 80%% of PnL:         %d out of %d", 
               convexity_res.get("trades_for_80pct_of_pnl", 0),
               convexity_res.get("total_trades", 0))
    
    # === FINAL VERDICT ===
    logger.info("\n" + "=" * 80)
    logger.info("FINAL ASSESSMENT")
    logger.info("=" * 80)
    
    # Score card
    scores = {
        "structural_edge": "WEAK" if overall_sharpe_mean < -5 else "ABSENT" if overall_sharpe_mean < 0 else "PRESENT",
        "parameter_stability": sharpe_stability,
        "time_consistency": rolling_df["total_return"].mean() > 0 if not rolling_df.empty else False,
        "regime_independent": not regime_dependent if regime_res else None,
        "gross_edge_exists": has_gross_edge if not cost_df.empty else None,
        "net_edge_tradable": has_net_edge if not cost_df.empty else None,
        "edge_concentrated": convexity_res.get("is_edge_concentrated", False),
    }
    
    logger.info("\nStructural Edge:                         %s", scores["structural_edge"])
    logger.info("Parameter Stability:                    %s", scores["parameter_stability"])
    logger.info("Regime Independent:                     %s", scores["regime_independent"])
    logger.info("Gross Edge Exists:                      %s", scores["gross_edge_exists"])
    logger.info("Net Edge (After Costs):                 %s", scores["net_edge_tradable"])
    logger.info("Edge Concentrated (Fragile):            %s", scores["edge_concentrated"])
    
    # Final recommendation
    logger.info("\n" + "=" * 80)
    eligible_trade = (
        scores["structural_edge"] != "ABSENT" and 
        scores["gross_edge_exists"] and
        not scores["edge_concentrated"]
    )
    
    if eligible_trade:
        logger.info("RECOMMENDATION: POTENTIALLY TRADABLE")
        logger.info("  The strategy shows structural edge in specific regimes.")
        logger.info("  Requires careful parameter tuning and cost management.")
    else:
        logger.info("RECOMMENDATION: NOT RECOMMENDED FOR LIVE TRADING")
        logger.info("  The strategy does not exhibit robust structural edge.")
        logger.info("  Current negative Sharpe and cost burden outweigh benefits.")
    
    logger.info("=" * 80)
    logger.info("All validation results saved to: %s", OUTPUT_DIR)
    logger.info("=" * 80)


if __name__ == "__main__":
    run_full_validation()
