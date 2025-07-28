Changelog
=========

All notable changes to the Decline Curve Analysis library will be documented in this file.

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
- pmdarima>=2.0
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
