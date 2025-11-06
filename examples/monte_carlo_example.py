"""
Monte Carlo Simulation Example for Decline Curve Analysis

This example demonstrates:
1. Setting up probabilistic parameter distributions
2. Running Monte Carlo simulations
3. Analyzing P10/P50/P90 forecasts
4. Computing risk metrics
5. Visualizing uncertainty
6. Comparing deterministic vs probabilistic forecasts
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decline_analysis.models import ArpsParams, predict_arps
from decline_analysis.monte_carlo import (
    DistributionParams,
    MonteCarloParams,
    monte_carlo_forecast,
    plot_monte_carlo_results,
    risk_analysis,
    sensitivity_to_monte_carlo,
)


def example_1_basic_monte_carlo():
    """Example 1: Basic Monte Carlo simulation with uniform distributions."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Monte Carlo Simulation")
    print("=" * 70)
    print("\nUsing uniform distributions for all parameters...")

    # Define parameter distributions
    mc_params = MonteCarloParams(
        qi_dist=DistributionParams("uniform", min=1000, max=1500),
        di_dist=DistributionParams("uniform", min=0.10, max=0.30),
        b_dist=DistributionParams("uniform", min=0.3, max=0.8),
        price_dist=DistributionParams("uniform", min=50, max=90),
        opex_dist=DistributionParams("uniform", min=15, max=25),
        n_simulations=1000,
        seed=42,
    )

    # Run simulation
    results = monte_carlo_forecast(mc_params, verbose=True)

    # Display results
    print("\n" + "-" * 70)
    print("Probabilistic Forecast Results:")
    print("-" * 70)
    print(f"\nEUR (bbl):")
    print(f"  P90 (conservative): {results.p90_eur:>10,.0f}")
    print(f"  P50 (median):       {results.p50_eur:>10,.0f}")
    print(f"  P10 (optimistic):   {results.p10_eur:>10,.0f}")
    print(f"  Mean:               {results.mean_eur:>10,.0f}")

    print(f"\nNPV ($):")
    print(f"  P90 (conservative): {results.p90_npv:>10,.0f}")
    print(f"  P50 (median):       {results.p50_npv:>10,.0f}")
    print(f"  P10 (optimistic):   {results.p10_npv:>10,.0f}")
    print(f"  Mean:               {results.mean_npv:>10,.0f}")

    # Plot results
    plot_monte_carlo_results(results, title="Example 1: Basic Monte Carlo")

    return results


def example_2_lognormal_distributions():
    """Example 2: Monte Carlo with lognormal distributions (more realistic)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Monte Carlo with Lognormal Distributions")
    print("=" * 70)
    print("\nUsing lognormal distributions for reservoir parameters...")

    # Lognormal distributions are more realistic for reservoir properties
    mc_params = MonteCarloParams(
        qi_dist=DistributionParams("lognormal", mean=1200, std=0.3),
        di_dist=DistributionParams("lognormal", mean=0.15, std=0.2),
        b_dist=DistributionParams("triangular", min=0.3, mode=0.5, max=0.8),
        price_dist=DistributionParams("normal", mean=70, std=15),
        opex_dist=DistributionParams("normal", mean=20, std=3),
        n_simulations=1000,
        seed=42,
    )

    # Run simulation
    results = monte_carlo_forecast(mc_params, verbose=True)

    # Compute risk metrics
    risk_metrics = risk_analysis(results, threshold=0)

    print("\n" + "-" * 70)
    print("Risk Analysis:")
    print("-" * 70)
    print(f"EUR Coefficient of Variation: {risk_metrics['eur_cv']:.2%}")
    print(f"NPV Coefficient of Variation: {risk_metrics['npv_cv']:.2%}")
    print(f"Probability of Positive NPV:  {risk_metrics['prob_positive_npv']:.1%}")
    print(f"Value at Risk (5%):           ${risk_metrics['value_at_risk_npv']:,.0f}")

    # Plot results
    plot_monte_carlo_results(results, title="Example 2: Lognormal Distributions")

    return results


def example_3_correlated_parameters():
    """Example 3: Monte Carlo with correlated parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Monte Carlo with Correlated Parameters")
    print("=" * 70)
    print(
        "\nApplying correlation between qi and di (higher qi often means higher di)..."
    )

    # Define correlation matrix
    # Parameters: [qi, di, b, price]
    correlation_matrix = np.array(
        [
            [1.0, 0.6, 0.0, 0.0],  # qi: positive correlation with di
            [0.6, 1.0, 0.0, 0.0],  # di: positive correlation with qi
            [0.0, 0.0, 1.0, 0.0],  # b: independent
            [0.0, 0.0, 0.0, 1.0],  # price: independent
        ]
    )

    mc_params = MonteCarloParams(
        qi_dist=DistributionParams("lognormal", mean=1200, std=0.3),
        di_dist=DistributionParams("lognormal", mean=0.15, std=0.2),
        b_dist=DistributionParams("triangular", min=0.3, mode=0.5, max=0.8),
        price_dist=DistributionParams("normal", mean=70, std=15),
        opex_dist=DistributionParams("normal", mean=20, std=3),
        n_simulations=1000,
        correlation_matrix=correlation_matrix,
        seed=42,
    )

    # Run simulation
    results = monte_carlo_forecast(mc_params, verbose=True)

    # Compare with uncorrelated
    print("\n" + "-" * 70)
    print("Observed Correlation in Results:")
    print("-" * 70)
    corr_qi_di = results.parameters["qi"].corr(results.parameters["di"])
    print(f"qi-di correlation: {corr_qi_di:.3f} (target: 0.60)")

    plot_monte_carlo_results(results, title="Example 3: Correlated Parameters")

    return results


def example_4_deterministic_vs_probabilistic():
    """Example 4: Compare deterministic and probabilistic approaches."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Deterministic vs Probabilistic Comparison")
    print("=" * 70)

    # Deterministic case (single forecast)
    print("\n--- Deterministic Forecast ---")
    det_params = ArpsParams(qi=1200, di=0.15, b=0.5)
    t = np.arange(0, 240, 1.0)
    det_forecast = predict_arps(t, det_params)
    det_eur = np.trapz(det_forecast[det_forecast > 10], t[det_forecast > 10])
    print(f"Deterministic EUR: {det_eur:,.0f} bbl")

    # Probabilistic case (Monte Carlo)
    print("\n--- Probabilistic Forecast ---")
    mc_params = MonteCarloParams(
        qi_dist=DistributionParams("normal", mean=1200, std=120),
        di_dist=DistributionParams("normal", mean=0.15, std=0.03),
        b_dist=DistributionParams("normal", mean=0.5, std=0.1),
        price_dist=DistributionParams("normal", mean=70, std=10),
        opex_dist=DistributionParams("normal", mean=20, std=2),
        n_simulations=1000,
        seed=42,
    )

    results = monte_carlo_forecast(mc_params, verbose=False)

    print(f"Probabilistic EUR:")
    print(
        f"  P90: {results.p90_eur:,.0f} bbl ({results.p90_eur/det_eur:.1%} of deterministic)"
    )
    print(
        f"  P50: {results.p50_eur:,.0f} bbl ({results.p50_eur/det_eur:.1%} of deterministic)"
    )
    print(
        f"  P10: {results.p10_eur:,.0f} bbl ({results.p10_eur/det_eur:.1%} of deterministic)"
    )

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Forecast comparison
    axes[0].plot(t, det_forecast, "k-", linewidth=2, label="Deterministic")
    axes[0].fill_between(
        t,
        results.p90_forecast,
        results.p10_forecast,
        alpha=0.3,
        color="blue",
        label="P10-P90 range",
    )
    axes[0].plot(t, results.p50_forecast, "b-", linewidth=2, label="P50")
    axes[0].set_xlabel("Time (months)")
    axes[0].set_ylabel("Production Rate (bbl/month)")
    axes[0].set_title("Deterministic vs Probabilistic Forecast")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: EUR distribution with deterministic value
    axes[1].hist(
        results.eur_samples, bins=50, alpha=0.7, color="blue", edgecolor="black"
    )
    axes[1].axvline(
        det_eur,
        color="black",
        linestyle="-",
        linewidth=2,
        label=f"Deterministic: {det_eur:,.0f}",
    )
    axes[1].axvline(
        results.p50_eur,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"P50: {results.p50_eur:,.0f}",
    )
    axes[1].set_xlabel("EUR (bbl)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("EUR Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


def example_5_sensitivity_to_monte_carlo():
    """Example 5: Convert sensitivity ranges to Monte Carlo distributions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: From Sensitivity Analysis to Monte Carlo")
    print("=" * 70)
    print("\nConverting sensitivity ranges to probabilistic distributions...")

    # Define sensitivity ranges
    print("\nSensitivity Ranges:")
    print("  qi: 1000 - 1400 bbl/month")
    print("  di: 0.10 - 0.20 /year")
    print("  b:  0.3 - 0.7")
    print("  Price: $50 - $90 /bbl")

    # Convert to Monte Carlo parameters
    mc_params = sensitivity_to_monte_carlo(
        base_qi=1200,
        base_di=0.15,
        base_b=0.5,
        qi_range=(1000, 1400),
        di_range=(0.10, 0.20),
        b_range=(0.3, 0.7),
        price_range=(50, 90),
        n_simulations=1000,
    )

    # Run simulation
    results = monte_carlo_forecast(mc_params, verbose=True)

    # Compute probability of hitting targets
    target_eur = 150000  # Target EUR
    target_npv = 5000000  # Target NPV

    prob_eur = np.mean(results.eur_samples > target_eur)
    prob_npv = np.mean(results.npv_samples > target_npv)

    print("\n" + "-" * 70)
    print("Probability of Hitting Targets:")
    print("-" * 70)
    print(f"P(EUR > {target_eur:,.0f} bbl) = {prob_eur:.1%}")
    print(f"P(NPV > ${target_npv:,.0f}) = {prob_npv:.1%}")

    plot_monte_carlo_results(results, title="Example 5: Sensitivity to Monte Carlo")

    return results


def main():
    """Run all Monte Carlo examples."""
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION EXAMPLES")
    print("Decline Curve Analysis with Uncertainty Quantification")
    print("=" * 70)
    print("\nThese examples demonstrate probabilistic forecasting using")
    print("Monte Carlo simulation for risk assessment and decision-making.")
    print("\nNote: Simulations use Numba JIT and joblib parallelization")
    print("for fast execution (5-20x speedup).")

    # Run examples
    try:
        results1 = example_1_basic_monte_carlo()
        input("\nPress Enter to continue to Example 2...")

        results2 = example_2_lognormal_distributions()
        input("\nPress Enter to continue to Example 3...")

        results3 = example_3_correlated_parameters()
        input("\nPress Enter to continue to Example 4...")

        results4 = example_4_deterministic_vs_probabilistic()
        input("\nPress Enter to continue to Example 5...")

        results5 = example_5_sensitivity_to_monte_carlo()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
        return

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nMonte Carlo simulation provides:")
    print("  ✓ Probabilistic forecasts (P10/P50/P90)")
    print("  ✓ Uncertainty quantification")
    print("  ✓ Risk assessment metrics")
    print("  ✓ Better decision-making under uncertainty")
    print("  ✓ Fast execution with parallel processing")

    print("\nKey Functions:")
    print("  - monte_carlo_forecast(): Run simulation")
    print("  - plot_monte_carlo_results(): Visualize results")
    print("  - risk_analysis(): Compute risk metrics")
    print("  - sensitivity_to_monte_carlo(): Convert ranges to distributions")

    print("\nFor more information, see:")
    print("  - decline_analysis/monte_carlo.py")
    print("  - docs/MONTE_CARLO.md (documentation)")

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
