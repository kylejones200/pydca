"""Data loading and processing utilities for production data."""

from typing import Optional

import pandas as pd


def load_production_csvs(
    paths: list[str],
    date_col: str = "date",
    well_id_col: str = "well_id",
    oil_col: str = "oil_bbl",
) -> pd.DataFrame:
    """Load and stack well-level production CSV files.

    Args:
        paths: List of CSV paths.
        date_col: Name of the date column.
        well_id_col: Name of the well id column.
        oil_col: Name of the oil volume column.

    Returns:
        A DataFrame with [date, well_id, oil_bbl] and a DateTimeIndex.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        _assert_cols(df, [date_col, well_id_col, oil_col])
        df[date_col] = pd.to_datetime(df[date_col])
        frames.append(df[[date_col, well_id_col, oil_col]])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values([well_id_col, date_col])
    out = out.rename(
        columns={date_col: "date", well_id_col: "well_id", oil_col: "oil_bbl"}
    )
    return out.set_index("date")


def to_monthly(
    df: pd.DataFrame, well_id_col: str = "well_id", oil_col: str = "oil_bbl"
) -> pd.DataFrame:
    """Aggregate to monthly frequency.

    Args:
        df: A DataFrame indexed by date.
        well_id_col: Well id column.
        oil_col: Oil volume column.

    Returns:
        A monthly panel by well.
    """
    return (
        df.groupby(well_id_col)
        .resample("M")[oil_col]
        .sum()
        .reset_index()
        .set_index("date")
    )


def make_panel(df: pd.DataFrame, first_n_months: Optional[int] = None) -> pd.DataFrame:
    """Create a relative-time panel for decline fitting.

    Args:
        df: Monthly panel with index date and columns well_id and oil_bbl.
        first_n_months: Truncate each well to N months from first production.

    Returns:
        A panel with cycle (t) per well.
    """
    df = df.sort_values(["well_id", "date"]).copy()
    df["t"] = df.groupby("well_id").cumcount()
    if first_n_months is not None:
        df = df[df["t"] < first_n_months]
    return df


def load_price_csv(
    path: str, date_col: str = "date", price_col: str = "price"
) -> pd.DataFrame:
    """Load oil price CSV.

    Args:
        path: Path to price CSV.
        date_col: Date column name.
        price_col: Price column name.

    Returns:
        A DataFrame indexed by date with a single price column.
    """
    df = pd.read_csv(path)
    _assert_cols(df, [date_col, price_col])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date", price_col: "price"}).set_index("date")
    return df.sort_index()


def _assert_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
