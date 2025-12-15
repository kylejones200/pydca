"""Tests for __main__.py CLI interface."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Import the main function
from decline_curve.__main__ import main


class TestMainCLI:
    """Test CLI interface functionality."""

    def test_main_help(self, capsys):
        """Test that help message works."""
        import sys

        with patch.object(sys, "argv", ["__main__.py", "--help"]):
            try:
                main()
            except SystemExit:
                pass  # argparse calls sys.exit() for --help

        captured = capsys.readouterr()
        assert "Decline curve forecast tool" in captured.out or "usage:" in captured.out

    def test_main_with_csv_and_well(self):
        """Test main function with CSV file and well ID."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "well_id": ["WELL_001"] * 24,
                    "oil_bbl": [1000 - i * 20 for i in range(24)],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            import sys

            with patch.object(
                sys,
                "argv",
                [
                    "__main__.py",
                    csv_path,
                    "--well",
                    "WELL_001",
                    "--model",
                    "arps",
                    "--horizon",
                    "6",
                ],
            ):
                with patch("decline_curve.dca.plot") as mock_plot:
                    with patch("decline_curve.dca.forecast") as mock_forecast:
                        mock_forecast.return_value = pd.Series(
                            [100] * 6,
                            index=pd.date_range("2022-01-01", periods=6, freq="MS"),
                        )
                        main()

                        # Verify forecast was called
                        assert mock_forecast.called
                        # Verify plot was called
                        assert mock_plot.called
        finally:
            os.unlink(csv_path)

    def test_main_benchmark_mode(self):
        """Test main function in benchmark mode."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "well_id": ["WELL_001"] * 12 + ["WELL_002"] * 12,
                    "oil_bbl": [1000 - i * 20 for i in range(24)],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            import sys

            with patch.object(
                sys,
                "argv",
                [
                    "__main__.py",
                    csv_path,
                    "--benchmark",
                    "--model",
                    "arps",
                    "--horizon",
                    "6",
                    "--top_n",
                    "2",
                ],
            ):
                with patch("decline_curve.dca.benchmark") as mock_benchmark:
                    mock_benchmark.return_value = pd.DataFrame(
                        {
                            "well_id": ["WELL_001", "WELL_002"],
                            "rmse": [10.5, 12.3],
                            "mae": [8.2, 9.1],
                            "smape": [0.05, 0.06],
                        }
                    )
                    with patch("decline_curve.__main__.logger.info") as mock_logger:
                        main()

                        # Verify benchmark was called
                        assert mock_benchmark.called
                        # Verify logging was called
                        assert mock_logger.called
        finally:
            os.unlink(csv_path)

    def test_main_missing_well_error(self):
        """Test that main raises error when well is missing in non-benchmark mode."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "well_id": ["WELL_001"] * 24,
                    "oil_bbl": [1000 - i * 20 for i in range(24)],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            import sys

            with patch.object(
                sys, "argv", ["__main__.py", csv_path, "--model", "arps"]
            ):
                with pytest.raises(ValueError, match="Must provide --well"):
                    main()
        finally:
            os.unlink(csv_path)

    def test_main_different_models(self):
        """Test main function with different model options."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "well_id": ["WELL_001"] * 24,
                    "oil_bbl": [1000 - i * 20 for i in range(24)],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            import sys

            for model in ["arps", "timesfm", "chronos"]:
                with patch.object(
                    sys,
                    "argv",
                    [
                        "__main__.py",
                        csv_path,
                        "--well",
                        "WELL_001",
                        "--model",
                        model,
                        "--horizon",
                        "6",
                    ],
                ):
                    with patch("decline_curve.dca.plot") as mock_plot:
                        with patch("decline_curve.dca.forecast") as mock_forecast:
                            mock_forecast.return_value = pd.Series(
                                [100] * 6,
                                index=pd.date_range("2022-01-01", periods=6, freq="MS"),
                            )
                            main()

                            # Verify forecast was called with correct model
                            call_args = mock_forecast.call_args
                            assert call_args[1]["model"] == model
        finally:
            os.unlink(csv_path)

    def test_main_different_kinds(self):
        """Test main function with different Arps kinds."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "well_id": ["WELL_001"] * 24,
                    "oil_bbl": [1000 - i * 20 for i in range(24)],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            import sys

            for kind in ["exponential", "harmonic", "hyperbolic"]:
                with patch.object(
                    sys,
                    "argv",
                    [
                        "__main__.py",
                        csv_path,
                        "--well",
                        "WELL_001",
                        "--model",
                        "arps",
                        "--kind",
                        kind,
                        "--horizon",
                        "6",
                    ],
                ):
                    with patch("decline_curve.dca.plot") as mock_plot:
                        with patch("decline_curve.dca.forecast") as mock_forecast:
                            mock_forecast.return_value = pd.Series(
                                [100] * 6,
                                index=pd.date_range("2022-01-01", periods=6, freq="MS"),
                            )
                            main()

                            # Verify forecast was called with correct kind
                            call_args = mock_forecast.call_args
                            assert call_args[1]["kind"] == kind
        finally:
            os.unlink(csv_path)

    def test_main_verbose_flag(self):
        """Test main function with verbose flag."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "well_id": ["WELL_001"] * 24,
                    "oil_bbl": [1000 - i * 20 for i in range(24)],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            import sys

            with patch.object(
                sys,
                "argv",
                [
                    "__main__.py",
                    csv_path,
                    "--well",
                    "WELL_001",
                    "--model",
                    "arps",
                    "--horizon",
                    "6",
                    "--verbose",
                ],
            ):
                with patch("decline_curve.dca.plot") as mock_plot:
                    with patch("decline_curve.dca.forecast") as mock_forecast:
                        mock_forecast.return_value = pd.Series(
                            [100] * 6,
                            index=pd.date_range("2022-01-01", periods=6, freq="MS"),
                        )
                        main()

                        # Verify forecast was called with verbose=True
                        call_args = mock_forecast.call_args
                        assert call_args[1]["verbose"] is True
        finally:
            os.unlink(csv_path)

    def test_main_invalid_csv(self):
        """Test main function with invalid CSV file."""
        import sys

        with patch.object(
            sys, "argv", ["__main__.py", "nonexistent.csv", "--well", "WELL_001"]
        ):
            with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
                main()
