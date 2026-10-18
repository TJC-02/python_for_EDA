"""CSV exploratory analysis. Original observations are never modified."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def profile(data: pd.DataFrame) -> dict:
    """Return JSON-safe dataset metrics without dropping incomplete rows."""
    numeric = data.select_dtypes(include="number")
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_cells": int(data.isna().sum().sum()),
        "numeric_columns": len(numeric.columns),
        "all_missing_columns": data.columns[data.isna().all()].tolist(),
        "constant_columns": data.columns[data.nunique(dropna=True).eq(1)].tolist(),
        "infinite_numeric_cells": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
    }


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def analyze(csv_path: Path, output: Path, target: str | None = None,
            max_features: int = 12) -> dict:
    if max_features < 1:
        raise ValueError("max_features must be at least 1")
    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError("CSV must contain at least one data row")
    if target is not None and target not in data.columns:
        raise ValueError(f"Target column {target!r} is absent from the CSV")
    output.mkdir(parents=True, exist_ok=True)
    summary = profile(data)
    summary["source_file"] = csv_path.name
    (output / "overview.json").write_text(json.dumps(summary, indent=2) + "\n")
    data.describe(include="all").to_csv(output / "descriptive_statistics.csv")
    pd.DataFrame({
        "dtype": data.dtypes.astype(str),
        "missing_count": data.isna().sum(),
        "missing_percent": data.isna().mean().mul(100),
        "unique_nonmissing": data.nunique(dropna=True),
    }).to_csv(output / "column_profile.csv", index_label="column")

    missing = data.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing.gt(0)].head(20)
    if not missing.empty:
        plt.figure(figsize=(10, 6))
        missing.sort_values().plot.barh()
        plt.xlabel("Missing observations (%)")
        plt.title("Columns with the most missing values (up to 20)")
        save_plot(output / "missing_values.png")

    # Exclude the requested outcome and constant/nonfinite-only features.
    numeric = data.select_dtypes(include="number").drop(columns=[target], errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    varying = numeric.loc[:, numeric.nunique().gt(1)]
    selected = varying.columns[:max_features]
    if len(selected):
        varying[selected].hist(figsize=(12, max(4, 3 * ((len(selected) + 2) // 3))),
                               bins=30, layout=(-1, 3))
        plt.suptitle("Numeric distributions (first varying features in file order)")
        save_plot(output / "distributions.png")
    if len(selected) > 1:
        correlation = varying[selected].corr(min_periods=2)
        correlation.to_csv(output / "correlations.csv")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("Pearson correlation (pairwise complete observations)")
        save_plot(output / "correlations.png")

    if target is not None:
        counts = data[target].value_counts(dropna=False)
        counts.to_csv(output / "target_counts.csv", index_label=target)
        if len(counts) <= 30:
            plt.figure(figsize=(8, 5))
            counts.rename(index=lambda value: "<missing>" if pd.isna(value) else str(value)).plot.bar()
            plt.ylabel("Observations")
            plt.title(f"{target} distribution")
            save_plot(output / "target_distribution.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input CSV file")
    parser.add_argument("--output", type=Path, default=Path("outputs"),
                        help="Use a fresh directory for each analysis")
    parser.add_argument("--target", help="Optional outcome column, e.g. Pass/Fail")
    parser.add_argument("--max-features", type=int, default=12)
    args = parser.parse_args()
    try:
        summary = analyze(args.csv, args.output, args.target, args.max_features)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    print(json.dumps(summary, indent=2))
    print(f"Reports saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
