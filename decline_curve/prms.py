"""PRMS reserves classification using DCA-based probabilistic forecasting.

Implements SPE-PRMS (Society of Petroleum Engineers Petroleum Resources
Management System) 1P/2P/3P reserves classification via Monte Carlo EUR draws.

Reference:
    SPE-PRMS 2018 — Petroleum Resources Management System
    P90 → Proved (1P), P50 → Proved + Probable (2P), P10 → Proved + Probable +
    Possible (3P).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReservesClassification:
    """DCA-based PRMS 1P/2P/3P reserves classification.

    Attributes:
        p1: Proved reserves — P90 (conservative, high probability) in BOE.
        p2: Proved + Probable — P50 (best estimate) in BOE.
        p3: Proved + Probable + Possible — P10 (optimistic) in BOE.
        eur_distribution: Full array of EUR draws (sorted ascending).
        model: Decline model used (e.g. ``'modified_hyperbolic'``).
        kind: Arps kind or variant used.
        n_draws: Number of Monte Carlo draws.
    """

    p1: float
    p2: float
    p3: float
    eur_distribution: np.ndarray = field(repr=False)
    model: str = "arps"
    kind: str = "modified_hyperbolic"
    n_draws: int = 500

    @property
    def uncertainty_ratio(self) -> float:
        """P3/P1 ratio — dimensionless measure of resource uncertainty."""
        return self.p3 / self.p1 if self.p1 > 0 else float("nan")

    def to_dict(self) -> dict:
        """Return classification as a plain dict (excludes distribution array)."""
        return {
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
            "model": self.model,
            "kind": self.kind,
            "n_draws": self.n_draws,
            "uncertainty_ratio": self.uncertainty_ratio,
        }

    def to_series(self) -> pd.Series:
        """Return P1/P2/P3 as a labelled Series."""
        return pd.Series(
            {"1P (P90)": self.p1, "2P (P50)": self.p2, "3P (P10)": self.p3},
            name="reserves_boe",
        )


# ---------------------------------------------------------------------------
# EUR helpers
# ---------------------------------------------------------------------------


def _eur_from_forecast_array(
    q: np.ndarray,
    econ_limit: float = 0.0,
) -> float:
    """Numerical EUR from a production forecast array.

    Sums monthly production down to ``econ_limit`` (inclusive month).

    Args:
        q: Monthly production forecast array (BOE/month).
        econ_limit: Abandon rate (BOE/month). Months below this are excluded.

    Returns:
        Estimated Ultimate Recovery in BOE.
    """
    if econ_limit > 0:
        above = np.where(q >= econ_limit)[0]
        if len(above) == 0:
            return 0.0
        q = q[: int(above[-1]) + 1]
    return float(q.sum())


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


def classify_reserves(
    series: pd.Series,
    model: str = "arps",
    kind: str = "modified_hyperbolic",
    horizon: int = 360,
    n_draws: int = 500,
    econ_limit: float = 0.0,
    seed: Optional[int] = None,
) -> ReservesClassification:
    """Classify PRMS reserves (1P/2P/3P) via DCA-based probabilistic forecasting.

    Uses the existing :func:`~decline_curve.probabilistic_forecast.probabilistic_forecast`
    engine to generate ``n_draws`` forecast realisations, computes EUR for each,
    then maps percentiles to PRMS categories:

    - **P90** (conservative high-probability) → 1P / Proved
    - **P50** (best estimate) → 2P / Proved + Probable
    - **P10** (optimistic) → 3P / Proved + Probable + Possible

    Args:
        series: Historical monthly production with :class:`pandas.DatetimeIndex`.
        model: Base forecasting model (``'arps'`` recommended; Arps variants
            dispatched via ``kind``).
        kind: Decline model variant — ``'modified_hyperbolic'``, ``'hyperbolic'``,
            ``'exponential'``, ``'harmonic'``, ``'duong'``, ``'ple'``, ``'sepd'``.
        horizon: Forecast horizon in months (default 360 = 30 years).
        n_draws: Monte Carlo sample count. 500 is typically sufficient; use
            1000–2000 for regulatory reporting.
        econ_limit: Economic limit rate (BOE/month). Truncates each EUR draw at
            the last month the rate stays at or above this value.
        seed: Random seed for reproducibility.

    Returns:
        :class:`ReservesClassification` with P1/P2/P3 and the full EUR distribution.

    Example:
        >>> import pandas as pd, numpy as np
        >>> import decline_curve as dca
        >>> dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        >>> q = 1000 * np.exp(-0.03 * np.arange(36))
        >>> s = pd.Series(q, index=dates)
        >>> rc = dca.classify_reserves(s, kind='modified_hyperbolic', n_draws=200)
        >>> print(rc.to_series())
    """
    from .probabilistic_forecast import probabilistic_forecast

    # probabilistic_forecast only handles Arps kinds; map shale variants to
    # their closest Arps equivalent for the Monte Carlo draws.
    _kind_map = {
        "modified_hyperbolic": "hyperbolic",
        "mh": "hyperbolic",
        "duong": "hyperbolic",
        "ple": "hyperbolic",
        "sepd": "hyperbolic",
    }
    kind_for_prob = _kind_map.get(kind, kind)

    # probabilistic_forecast returns a ProbabilisticForecast whose .draws
    # attribute is a ForecastDraws with .draws shaped (n_draws, n_periods).
    pf = probabilistic_forecast(
        series,
        model=model,
        kind=kind_for_prob,
        horizon=horizon,
        n_draws=n_draws,
        seed=seed,
    )

    if pf.draws is None:
        raise RuntimeError("probabilistic_forecast returned no draws — cannot classify reserves")

    draws_matrix = pf.draws.draws  # shape (n_draws, n_periods)

    n_actual = draws_matrix.shape[0]

    eur_draws = np.array(
        [_eur_from_forecast_array(draws_matrix[i], econ_limit=econ_limit) for i in range(n_actual)]
    )

    eur_draws = np.sort(eur_draws)

    # PRMS convention: P90 probability of achieving AT LEAST this value
    # → 90% of draws exceed the 1P value → 1P is the 10th percentile.
    # P10 probability of achieving at least P3 → P3 is the 90th percentile.
    p1 = float(np.percentile(eur_draws, 10))   # 1P/Proved — P90 confidence
    p2 = float(np.percentile(eur_draws, 50))   # 2P — P50 best estimate
    p3 = float(np.percentile(eur_draws, 90))   # 3P — P10 optimistic

    return ReservesClassification(
        p1=p1,
        p2=p2,
        p3=p3,
        eur_distribution=eur_draws,
        model=model,
        kind=kind,
        n_draws=n_actual,
    )
