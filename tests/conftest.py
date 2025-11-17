"""
Test configuration and fixtures for decline curve analysis tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_production_data():
    """Generate sample production data for testing."""
    # Create 24 months of declining production data
    dates = pd.date_range(start="2020-01-01", periods=24, freq="MS")

    # Simulate hyperbolic decline: q(t) = qi / (1 + b * di * t)^(1/b)
    qi = 1000  # Initial production
    di = 0.1  # Decline rate
    b = 0.5  # Hyperbolic exponent

    t = np.arange(len(dates))
    production = qi / (1 + b * di * t) ** (1 / b)

    # Add some realistic noise
    np.random.seed(42)
    noise = np.random.normal(0, production * 0.05)
    production = np.maximum(production + noise, 0)

    return pd.Series(production, index=dates, name="oil_production")


@pytest.fixture
def sample_well_data():
    """Generate sample multi-well dataset for benchmarking tests."""
    np.random.seed(42)

    data = []
    well_ids = ["WELL_001", "WELL_002", "WELL_003"]

    for well_id in well_ids:
        dates = pd.date_range(start="2020-01-01", periods=36, freq="MS")

        # Different decline parameters for each well
        qi = np.random.uniform(500, 1500)
        di = np.random.uniform(0.05, 0.15)
        b = np.random.uniform(0.3, 0.7)

        t = np.arange(len(dates))
        production = qi / (1 + b * di * t) ** (1 / b)

        # Add noise
        noise = np.random.normal(0, production * 0.1)
        production = np.maximum(production + noise, 0)

        for date, prod in zip(dates, production):
            data.append({"well_id": well_id, "date": date, "oil_bbl": prod})

    return pd.DataFrame(data)


@pytest.fixture
def arps_parameters():
    """Sample Arps decline parameters for testing."""
    return {
        "exponential": {"qi": 1000, "di": 0.1, "b": 0.0, "kind": "exponential"},
        "harmonic": {"qi": 1000, "di": 0.1, "b": 1.0, "kind": "harmonic"},
        "hyperbolic": {"qi": 1000, "di": 0.1, "b": 0.5, "kind": "hyperbolic"},
    }


@pytest.fixture
def forecast_horizons():
    """Common forecast horizons for testing."""
    return [6, 12, 24]
