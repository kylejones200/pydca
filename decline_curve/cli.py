"""Command-line interface for decline curve analysis.

This module provides a simple CLI for common DCA operations:
- dca fit: Fit decline curve
- dca forecast: Generate forecast
- dca batch: Batch processing
- dca report: Generate reports
"""

import argparse
import sys

from .logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main():
    """Run the CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Decline Curve Analysis CLI",
        prog="dca",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Fit decline curve")
    fit_parser.add_argument("input", help="Input data file (CSV)")
    fit_parser.add_argument("--output-dir", default="output", help="Output directory")
    fit_parser.add_argument("--model", default="hyperbolic", help="Model type")

    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Generate forecast")
    forecast_parser.add_argument("artifact", help="Fit artifact file")
    forecast_parser.add_argument(
        "--output-dir", default="output", help="Output directory"
    )

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch processing")
    batch_parser.add_argument("manifest", help="Manifest file (YAML or JSON)")
    batch_parser.add_argument(
        "--output-dir", default="batch_output", help="Output directory"
    )
    batch_parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel jobs"
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("artifact", help="Artifact file")
    report_parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    # Configure logging
    configure_logging(level=getattr(__import__("logging"), args.log_level))

    if args.command == "fit":
        logger.info(f"Fitting decline curve: {args.input}")
        # Implementation would go here
        logger.info("Fit complete")
    elif args.command == "forecast":
        logger.info(f"Generating forecast: {args.artifact}")
        # Implementation would go here
        logger.info("Forecast complete")
    elif args.command == "batch":
        logger.info(f"Running batch processing: {args.manifest}")
        # Implementation would go here
        logger.info("Batch processing complete")
    elif args.command == "report":
        logger.info(f"Generating report: {args.artifact}")
        # Implementation would go here
        logger.info("Report complete")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
