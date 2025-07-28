import argparse

import pandas as pd

from . import dca


def main():
    parser = argparse.ArgumentParser(description="Decline curve forecast tool")
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument(
        "--model", default="arps", choices=["arps", "timesfm", "chronos"]
    )
    parser.add_argument(
        "--kind",
        default="hyperbolic",
        choices=["exponential", "harmonic", "hyperbolic"],
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--well", help="Well ID to forecast")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.benchmark:
        result = dca.benchmark(
            df,
            model=args.model,
            kind=args.kind,
            horizon=args.horizon,
            top_n=args.top_n,
            verbose=args.verbose,
        )
        print(result.to_string(index=False))
    else:
        if args.well is None:
            raise ValueError("Must provide --well when not using --benchmark")
        sub = df[df["well_id"] == args.well].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        y = sub.set_index("date")["oil_bbl"].asfreq("MS")
        yhat = dca.forecast(
            y,
            model=args.model,
            kind=args.kind,
            horizon=args.horizon,
            verbose=args.verbose,
        )
        dca.plot(y, yhat, title=f"{args.well} {args.model}")
