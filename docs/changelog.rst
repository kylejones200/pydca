Changelog
=========

All notable changes to the Decline Curve Analysis library will be documented in this file.

Version 0.6.0 (2026-06-20)
---------------------------

**Reserves classification, type curve normalization, and scenario engine upgrade**

**Added**

- ``decline_curve.prms.ReservesClassification`` dataclass — PRMS 1P/2P/3P reserves
  classification with ``p1`` (P90/Proved), ``p2`` (P50/Probable), ``p3`` (P10/Possible),
  ``eur_distribution`` array, and ``uncertainty_ratio`` property
- ``decline_curve.prms.classify_reserves(series, model, kind, horizon, n_draws, econ_limit, seed)``
  — DCA-based PRMS classification via Monte Carlo EUR draws; P10/P50/P90 of the EUR
  distribution map to 1P/2P/3P per SPE-PRMS 2018 conventions
- ``decline_curve.dca.recommend_model(series, pressure, drive_mechanism)`` — data-driven model
  auto-selection returning one of ``'exponential'``, ``'harmonic'``, ``'hyperbolic'``,
  ``'modified_hyperbolic'``, or ``'duong'``; drive mechanism shortcuts (``'shale'``,
  ``'water_drive'``, ``'solution_gas'``) available for physics-informed routing
- ``decline_curve.type_curve_normalization`` — new module with Fetkovich-style type curve
  normalization:

  - ``TypeCurveMatch`` dataclass (``matched_params``, ``match_error``, ``correlation``,
    ``matched_curve``)
  - ``generate_arps_type_curve(qi_normalized, di_normalized, b, time_normalized)``
  - ``normalize_production_data(time, rate, qi_ref, di_ref)``
  - ``match_type_curve(time, rate, b_values, initial_guess)``
  - ``denormalize_match(match, factors)``

- ``PriceScenario`` upgraded: added ``capex``, ``royalty_rate``, ``working_interest``,
  ``severance_tax_rate``, ``ad_valorem_rate`` fields and ``to_well_economics()`` helper
- ``ScenarioResult`` upgraded: added ``irr``, ``roi``, ``breakeven_price`` fields
- ``run_price_scenarios()`` now uses the full-cycle ``WellEconomics`` / ``cashflow()`` engine
  (compound monthly discounting, royalties, taxes, CAPEX)
- ``run_multi_phase_scenarios()`` now uses compound monthly discounting consistent with ``npv()``
- All new symbols importable directly from ``decline_curve``

**Tests**

- ``tests/test_prms.py`` — 19 tests for PRMS classification
- ``tests/test_type_curve_normalization.py`` — 19 tests for normalization, matching, de-normalization
- ``tests/test_recommend_model.py`` — 12 tests for model auto-selection
- ``tests/test_scenarios.py`` — updated to cover full-cycle scenario columns (IRR, ROI, breakeven)

Version 0.5.0 (2026-06-20)
---------------------------

**Documentation and API polish**

**Added**

- ``WellEconomics``, ``CashflowResult``, ``cashflow``, ``npv``, ``irr``, ``payout``,
  ``roi``, ``breakeven_price`` are now importable directly from ``decline_curve``
  (no submodule path required)
- ``dca.well_economics(production, econ)`` — primary full-cycle economics entry point
  returning NPV, IRR, payout, ROI, breakeven, and the full ``CashflowResult``
- ``docs/cookbook/shale_variants.rst`` — worked example comparing Arps, Duong, and SEPD
  forecasts on a synthetic shale well with 30-year EUR
- ``docs/cookbook/full_economics.rst`` — full cashflow walkthrough: royalties, taxes, CAPEX,
  multi-rate NPV sensitivity, price sweep, and breakeven plot
- ``docs/source/economics.rst`` rewritten to cover the full-cycle engine with WellEconomics
  field reference table and probabilistic economics example

**Changed**

- ``docs/models.rst``: added Duong, PLE, SEPD sections and 6-row model comparison table
- ``docs/quickstart.rst``: added shale variant and full-cycle economics sections
- ``docs/index.rst``: updated feature list; economics page promoted in toctree
- ``README.md``: added 15-row feature comparison matrix vs. petbox-dca and DCApy

Version 0.4.0 (2026-06-19)
---------------------------

**Full-cycle economics engine**

**Added**

- ``decline_curve.economics.WellEconomics`` dataclass covering CAPEX, price, royalty rate,
  working interest, severance tax, ad valorem, fixed and variable OPEX (each with optional
  escalation), annual discount rate, and economic limit
- ``decline_curve.economics.CashflowResult`` dataclass with per-month arrays for every
  revenue and cost line item; ``to_dict()`` method for DataFrame conversion
- ``decline_curve.economics.cashflow(production, econ)`` — builds a monthly cashflow from
  any production array and a ``WellEconomics`` instance
- ``decline_curve.economics.npv(result, discount_rate=None)`` — NPV using compound monthly
  rate ``(1+r)^(1/12) - 1``
- ``decline_curve.economics.irr(result)`` — annual IRR via ``numpy_financial.irr``
- ``decline_curve.economics.payout(result)`` — first month cumulative cashflow turns positive
- ``decline_curve.economics.roi(result)`` — (cumulative NCF + CAPEX) / CAPEX
- ``decline_curve.economics.breakeven_price(production, econ)`` — commodity price at NPV=0
  via ``scipy.optimize.brentq`` over [$0, $500]
- ``decline_curve.risk_report._calculate_npv_samples_from_draws`` — replaced stub that
  returned ``np.array([])`` with full implementation using the new cashflow engine
- ``ProbabilisticForecast`` now stores ``price``, ``opex``, and ``discount_rate`` so
  ``calculate_risk_metrics()`` can compute per-draw NPV distributions
- ``tests/test_economics_full.py`` — 33 tests covering all new economics functions

**Changed**

- ``economic_metrics()`` preserved as a backwards-compatible shim using the original
  simple ``rate/12`` discounting convention

Version 0.3.0 (2026-06-19)
---------------------------

**Shale / unconventional decline variants**

**Added**

- ``DuongParams``, ``PLEParams``, ``SEPDParams`` dataclasses in ``decline_curve.models``
- ``decline_curve.decline_variants.fit_duong``, ``predict_duong``, ``eur_duong``
  — Duong (2011) fracture-matrix transient flow model with m=1 singularity guard
  (L'Hôpital limit) and 4-start optimization to escape local minima
- ``decline_curve.decline_variants.fit_ple``, ``predict_ple``, ``eur_ple``
  — Power-Law Exponential / Ilk (2008) loss-ratio model for tight gas
- ``decline_curve.decline_variants.fit_sepd``, ``predict_sepd``, ``eur_sepd``
  — Stretched Exponential model with closed-form EUR via ``scipy.special.gamma``
- ``decline_curve.decline_variants.forecast_variant(series, kind, horizon)``
  — dispatcher for all three shale variants
- All three models accessible via ``dca.forecast(series, model="arps", kind="duong"|"ple"|"sepd")``
- ``tests/test_decline_variants.py`` — 29 tests covering fit, predict, EUR, and API dispatch
  for all three models

**Changed**

- ``dca.forecast()`` and ``Forecaster.forecast()`` ``kind`` parameter widened from
  ``Literal["exponential","harmonic","hyperbolic"]`` to ``str`` to accommodate new variants

Version 0.1.0 (2025-07-25)
---------------------------

**Initial Release**

**Added**
- Arps decline curve models (exponential, harmonic, hyperbolic)
- ARIMA time series forecasting with automatic parameter selection
- TimesFM foundation model integration (with fallbacks)
- Chronos foundation model integration (with fallbacks)
- Comprehensive evaluation metrics (RMSE, MAE, SMAPE, MAPE, R²)
- Professional Tufte-style plotting capabilities
- Multi-well benchmarking functionality
- Complete test suite with 100+ unit tests
- Comprehensive Sphinx documentation
- Command-line interface
- Type hints throughout codebase

**Features**
- Simple, unified API for all forecasting models
- Automatic fallback mechanisms for advanced models
- Robust error handling and validation
- Support for seasonal ARIMA modeling
- Probabilistic forecasting capabilities
- Professional visualization with customizable plots
- Cross-validation and model comparison tools

**Dependencies**
- numpy>=1.23
- pandas>=2.0
- scipy>=1.10
- matplotlib>=3.7
- statsmodels>=0.14
- transformers>=4.41 (optional)
- torch>=2.0 (optional)

Future Releases
---------------

**Planned for v0.2.0**
- Additional evaluation metrics
- Enhanced plotting customization
- Performance optimizations
- More foundation model integrations
- Improved documentation with more examples

**Planned for v0.3.0**
- Web interface for interactive analysis
- Database integration capabilities
- Enhanced uncertainty quantification
- Multi-variate forecasting models
