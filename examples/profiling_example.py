"""
Example: How to use profiling utilities to optimize your code.

This example demonstrates:
1. Using @profile decorator to find bottlenecks
2. Using profile_context for timing code blocks
3. Using @time_function for quick timing
4. Interpreting profiling results
"""

import logging

import numpy as np
import pandas as pd

from decline_curve import dca
from decline_curve.logging_config import configure_logging, get_logger
from decline_curve.profiling import (
    print_stats,
    profile,
    profile_context,
    save_stats,
    time_function,
)

logger = get_logger(__name__)


# Example 1: Profile a function with @profile decorator
@profile
def forecast_single_well(series):
    """
    Profile this function to see which lines are slow.

    After running, call print_stats() to see results.
    """
    # This will show up in profiling results
    forecast = dca.forecast(series, model="arps", kind="hyperbolic", horizon=12)
    metrics = dca.evaluate(series, forecast)
    return forecast, metrics


# Example 2: Use @time_function for simple timing
@time_function
def process_multiple_wells(df, n_wells=10):
    """
    Simple timing decorator - prints total execution time.
    """
    results = []
    for well_id in df["well_id"].unique()[:n_wells]:
        well_data = df[df["well_id"] == well_id]
        series = well_data.set_index("date")["oil_bbl"]
        forecast, metrics = forecast_single_well(series)
        results.append(metrics)
    return results


def generate_sample_data(n_wells=20, n_months=36):
    """Generate synthetic well data for testing."""
    data = []
    for well_id in range(n_wells):
        qi = np.random.uniform(800, 1500)
        di = np.random.uniform(0.10, 0.30)
        b = np.random.uniform(0.3, 0.8)

        t = np.arange(n_months)
        q = qi / ((1 + b * di * t) ** (1 / b))
        q += np.random.normal(0, q * 0.05)
        q = np.maximum(q, 0)

        dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")

        for date, production in zip(dates, q):
            data.append(
                {"well_id": f"WELL_{well_id:04d}", "date": date, "oil_bbl": production}
            )

    return pd.DataFrame(data)


def main():
    """Main profiling example."""
    configure_logging(level=logging.INFO)
    logger.info("PROFILING EXAMPLE")
    logger.info("This example shows how to profile your decline curve analysis code")
    logger.info("We'll generate synthetic data and profile various operations")

    # Generate sample data
    logger.info("Generating sample data")
    df = generate_sample_data(n_wells=20, n_months=36)
    logger.info(f"Created data for {len(df['well_id'].unique())} wells")

    # Example 3: Use profile_context for timing blocks
    logger.info("Example 1: Timing with profile_context")

    with profile_context("Data preparation") as timer:
        well_data = df[df["well_id"] == "WELL_0000"]
        series = well_data.set_index("date")["oil_bbl"]

    logger.info(f"Data prep took {timer['elapsed']:.4f} seconds")

    # Example 4: Profile individual function calls
    logger.info("Example 2: Profiling individual functions")
    logger.info("Running forecast_single_well (profiled function)")
    forecast, metrics = forecast_single_well(series)
    logger.info(f"Forecast completed. RMSE: {metrics['rmse']:.2f}")

    # Example 5: Time multiple operations
    logger.info("Example 3: Timing with @time_function decorator")
    logger.info("Processing 10 wells")
    results = process_multiple_wells(df, n_wells=10)
    logger.info(f"Processed {len(results)} wells")

    # Example 6: Compare sequential vs parallel
    logger.info("Example 4: Sequential vs Parallel Performance")
    logger.info("Sequential processing (n_jobs=1)")
    with profile_context("Sequential benchmark", print_time=True):
        results_seq = dca.benchmark(df, top_n=10, n_jobs=1, verbose=False)

    logger.info("Parallel processing (n_jobs=-1)")
    with profile_context("Parallel benchmark", print_time=True):
        results_par = dca.benchmark(df, top_n=10, n_jobs=-1, verbose=False)

    # Show profiling results
    logger.info("PROFILING RESULTS")
    logger.info("Line-by-line profiling for forecast_single_well()")
    print_stats()

    # Save results to file
    logger.info("Saving profiling results to file")
    save_stats("profiling_results.txt")
    logger.info("Results saved to profiling_results.txt")

    # Summary
    logger.info("SUMMARY")
    logger.info("Profiling Tools Used:")
    logger.info("  1. @profile decorator - Line-by-line profiling")
    logger.info("  2. @time_function decorator - Function timing")
    logger.info("  3. profile_context() - Block timing")
    logger.info("  4. print_stats() - Display profiling results")
    logger.info("  5. save_stats() - Save results to file")

    logger.info("How to Interpret Results:")
    logger.info("  - 'Time' column: seconds spent on each line")
    logger.info("  - 'Per Hit' column: average time per execution")
    logger.info("  - 'Hits' column: number of times line was executed")
    logger.info("  - Look for lines with high 'Time' values")

    logger.info("Next Steps:")
    logger.info("  1. Identify slow lines in profiling output")
    logger.info("  2. Consider parallelization (n_jobs=-1)")
    logger.info("  3. Check if Numba is accelerating numerical code")
    logger.info("  4. Profile your own code using these tools")

    logger.info("Example complete. Check profiling_results.txt for details")


if __name__ == "__main__":
    main()
