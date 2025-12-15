"""Report generation module for decline curve analysis.

This module provides single-page summary reports for wells and fields,
including HTML and PDF output formats.
"""

from typing import Dict, List, Optional

from .artifacts import FitArtifact
from .fit_diagnostics import DiagnosticsResult
from .fitting import FitResult
from .logging_config import get_logger

logger = get_logger(__name__)


def generate_well_report(
    fit_result: FitResult,
    diagnostics: Optional[DiagnosticsResult] = None,
    artifact: Optional[FitArtifact] = None,
    output_path: Optional[str] = None,
    format: str = "html",
) -> str:
    """Generate single-page well report.

    Args:
        fit_result: FitResult object
        diagnostics: Optional DiagnosticsResult
        artifact: Optional FitArtifact
        output_path: Output file path (auto-generated if None)
        format: Output format ('html' or 'pdf')

    Returns:
        Path to generated report file

    Example:
        >>> report_path = generate_well_report(fit_result, diagnostics)
        >>> print(f"Report saved to {report_path}")
    """
    if output_path is None:
        well_id = artifact.metadata.run_id if artifact else "well_001"
        output_path = f"{well_id}_report.html"

    logger.info(f"Generating well report: {output_path}", extra={"format": format})

    # Generate HTML report
    html_content = _generate_html_report(fit_result, diagnostics, artifact)

    # Write to file
    with open(output_path, "w") as f:
        f.write(html_content)

    # Convert to PDF if requested (would require additional library)
    if format == "pdf":
        logger.warning("PDF generation not yet implemented, saving HTML instead")

    return output_path


def _generate_html_report(
    fit_result: FitResult,
    diagnostics: Optional[DiagnosticsResult],
    artifact: Optional[FitArtifact],
) -> str:
    """Generate HTML content for well report.

    Args:
        fit_result: Fit result object
        diagnostics: Optional diagnostics result
        artifact: Optional fit artifact

    Returns:
        HTML string
    """

    # Extract key information
    params = fit_result.params
    model_name = (
        fit_result.model.name if hasattr(fit_result.model, "name") else "Unknown"
    )

    grade = diagnostics.grade if diagnostics else "N/A"
    quality_score = diagnostics.quality_score if diagnostics else None

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DCA Report - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .grade {{ font-size: 24px; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Decline Curve Analysis Report</h1>

    <div class="section">
        <h2>Model Information</h2>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Fit Status:</strong> {"Success" if fit_result.success else "Failed"}</p>
        {f'<p><strong>Grade:</strong> <span class="grade">{grade}</span></p>' if grade != "N/A" else ""}
        {f'<p><strong>Quality Score:</strong> {quality_score:.2f}</p>' if quality_score else ""}
    </div>

    <div class="section">
        <h2>Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
    """

    for param, value in params.items():
        html += f"<tr><td>{param}</td><td>{value:.4f}</td></tr>"

    html += """
        </table>
    </div>
    """

    if diagnostics:
        html += """
    <div class="section">
        <h2>Diagnostics</h2>
        <div class="metric">
            <strong>RMSE:</strong> {:.4f}
        </div>
        <div class="metric">
            <strong>MAE:</strong> {:.4f}
        </div>
        <div class="metric">
            <strong>R²:</strong> {:.4f}
        </div>
    </div>
    """.format(
            diagnostics.metrics.get("rmse", 0),
            diagnostics.metrics.get("mae", 0),
            diagnostics.metrics.get("r_squared", 0),
        )

    if artifact:
        html += f"""
    <div class="section">
        <h2>Provenance</h2>
        <p><strong>Run ID:</strong> {artifact.metadata.run_id}</p>
        <p><strong>Timestamp:</strong> {artifact.metadata.timestamp}</p>
        <p><strong>Package Version:</strong> {artifact.metadata.package_version}</p>
    </div>
    """

    html += """
</body>
</html>
"""

    return html


def generate_field_summary(
    well_results: List[Dict],
    output_path: Optional[str] = None,
) -> str:
    """Generate field summary table.

    Args:
        well_results: List of well result dictionaries
        output_path: Output file path

    Returns:
        Path to generated summary file
    """
    if output_path is None:
        output_path = "field_summary.csv"

    import pandas as pd

    # Convert to DataFrame
    df = pd.DataFrame(well_results)

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info(
        f"Generated field summary: {len(well_results)} wells",
        extra={"output_path": output_path},
    )

    return output_path
