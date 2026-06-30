# Changelog

All notable changes to the Decline Curve Analysis library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-06-30

First release of the canonical-kernel contract that external libraries
(`ressmith`) delegate to. Contains everything in the Unreleased section below.

### Added
- `economics.npv_from_cashflow(net_cashflow, annual_discount_rate, periods_per_year=12)`
  — effective-annual NPV over a raw monthly cashflow array, for consumers that hold
  a cashflow series rather than a `CashflowResult`.

## [Unreleased]

### Fixed
- numpy 2.x compatibility: `np.trapz` was removed in numpy>=2.0; restore it as an
  alias for `np.trapezoid` at import time so EUR/reserves work on modern numpy.
- `import decline_curve` no longer hard-fails when `pyyaml` is absent; the YAML
  config path in `panel_analysis_sweep` is now lazy and raises a clear error only
  when a YAML config is actually loaded.

### Changed
- **Public API contract locked.** Added `__all__` to define the supported surface,
  and hoisted `fit_arps`/`predict_arps` to the package root (still importable from
  `decline_curve.models`). External consumers — notably `ressmith`, which now
  delegates its decline/economics math here — depend on this contract.

### Notes
- The economics convention is effective-annual `(1+r)^(m/12)` via `economics.npv`.
  The legacy nominal `r/12` path (`economic_metrics` / `dca.economics`) is retained
  for back-compat but is NOT the canonical entry point.

## [0.2.1] - 2026-01-08

### Changed
- Minor patch release to trigger PyPI update

## [0.2.0] - 2025-11-05

### Added
- **Multi-phase forecasting module** for simultaneous oil, gas, and water forecasting
- `MultiPhaseData` class for managing multi-phase production data
- `MultiPhaseForecaster` with coupled and independent forecasting modes
- **Data processing utilities** module with 11 production-ready functions
- Real Bakken well production data for examples
- 5 comprehensive Jupyter notebook examples
- Complete unit test suite (54 tests, 98% passing)
- API documentation for new modules
- GitHub issue and PR templates
- ROADMAP.md with 6-phase development plan

### Changed
- **Removed pmdarima dependency** to eliminate binary compatibility issues
- Simplified ARIMA forecasting to use statsmodels with default parameters
- Enhanced README with new features and badges
- Updated documentation with multi-phase and data processing APIs

### Fixed
- Numpy/pmdarima binary compatibility issues
- Package import errors

## [0.1.2] - 2025-07-28

### Fixed
- Missing dependencies (requests, xlrd) in pyproject.toml
- CI workflow dependency installation
- Code formatting with black and isort

### Changed
- Updated documentation workflow to use docs/requirements.txt

## [0.1.1] - 2025-07-26

### Fixed
- Package import issues
- Test suite compatibility

## [0.1.0] - 2025-07-25

### Added
- Initial release
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
- Economic analysis (NPV, cash flow, payback period)
- Reserves estimation (EUR calculations)
- Sensitivity analysis with tornado plots
- NDIC data scraper for North Dakota production data

### Features
- Simple, unified API for all forecasting models
- Automatic fallback mechanisms for advanced models
- Robust error handling and validation
- Support for seasonal ARIMA modeling
- Probabilistic forecasting capabilities
- Professional visualization with customizable plots
- Cross-validation and model comparison tools

## [0.0.1] - 2025-07-20

### Added
- Initial project structure
- Basic Arps models implementation
- Core forecasting functionality

---

## Release Types

- **Major** (X.0.0): Breaking changes, major new features
- **Minor** (0.X.0): New features, backwards compatible
- **Patch** (0.0.X): Bug fixes, minor improvements

## Upgrade Guide

### From 0.1.1 to 0.1.2
No breaking changes. Simply upgrade:
```bash
pip install --upgrade decline-curve
```

### From 0.0.x to 0.1.0
Major API changes. See [migration guide](https://decline-analysis.readthedocs.io/migration.html).

## Future Releases

### Planned for v0.2.0
- [ ] Additional evaluation metrics (MASE, WAPE)
- [ ] Enhanced plotting customization
- [ ] Performance optimizations
- [ ] More foundation model integrations
- [ ] Improved documentation with more examples
- [ ] Pre-commit hooks configuration
- [ ] Docker container support

### Planned for v0.3.0
- [ ] Web interface for interactive analysis (Streamlit/Dash)
- [ ] Database integration capabilities
- [ ] Enhanced uncertainty quantification
- [ ] Multi-variate forecasting models
- [ ] Export to Excel/PDF reports
- [ ] Parallel processing for batch analysis

### Planned for v1.0.0
- [ ] Stable API (no breaking changes)
- [ ] Complete documentation
- [ ] 100% test coverage
- [ ] Performance benchmarks
- [ ] Production-ready features
- [ ] Long-term support commitment

---

[Unreleased]: https://github.com/yourusername/decline-curve/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/yourusername/decline-curve/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/yourusername/decline-curve/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yourusername/decline-curve/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/yourusername/decline-curve/releases/tag/v0.0.1
