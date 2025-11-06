# Changelog

All notable changes to the Decline Curve Analysis library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-11-05

### Added
- **Multi-phase forecasting module** for simultaneous oil, gas, and water forecasting
- `MultiPhaseData` class for managing multi-phase production data
- `MultiPhaseForecaster` with coupled and independent forecasting modes
- **Data processing utilities** module with 11 production-ready functions
- Prophet forecasting integration for seasonal patterns
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

[Unreleased]: https://github.com/yourusername/pydca/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/yourusername/pydca/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/yourusername/pydca/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yourusername/pydca/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/yourusername/pydca/releases/tag/v0.0.1
