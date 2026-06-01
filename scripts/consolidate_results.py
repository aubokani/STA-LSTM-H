"""scripts/consolidate_results.py — Merge 18 per-animal reports into one table.

Run locally after fetch_results.sh has pulled all results from Ada.

Usage:
    python scripts/consolidate_results.py
    python scripts/consolidate_results.py --results-dir Results/python_pipeline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def consolidate(results_dir: Path) -> pd.DataFrame:
    rows = []
    for animal_dir in sorted(results_dir.glob("animal-*")):
        summary_path = animal_dir / "run_summary.json"
        comp_path    = animal_dir / "comparison_report.csv"

        if not comp_path.exists():
            print(f"  [SKIP] {animal_dir.name}: no comparison_report.csv")
            continue

        comp = pd.read_csv(comp_path)
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        for _, row in comp.iterrows():
            rows.append({
                "animal":         summary.get("animal_id", animal_dir.name.replace("animal-", "")),
                "n_windows":      summary.get("n_windows", None),
                "model":          row.get("model", ""),
                "accuracy_mean":  row.get("accuracy_mean", None),
                "accuracy_std":   row.get("accuracy_std",  None),
                "f1_mean":        row.get("f1_mean",       None),
                "f1_std":         row.get("f1_std",        None),
                "rmse_mean":      row.get("rmse_mean",     None),
                "rmse_std":       row.get("rmse_std",      None),
                "latency_ms_mean":row.get("latency_ms_mean",None),
                "composite_score":row.get("composite_score",None),
                "rank":           row.get("rank",          None),
                "total_time_s":   summary.get("total_time_s", None),
            })

    if not rows:
        raise RuntimeError(f"No completed animal results found under {results_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["animal", "rank"])
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("Results/python_pipeline"))
    p.add_argument("--out", type=Path, default=Path("Results/comparison_report_all_animals.csv"))
    args = p.parse_args()

    print(f"\nConsolidating results from {args.results_dir} …")
    df = consolidate(args.results_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows → {args.out}")

    # Summary stats across animals
    summary = (
        df.groupby("model")[["accuracy_mean", "f1_mean", "rmse_mean", "composite_score"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print("\nAggregate across all animals:")
    print(summary.to_string())

    best_animal = (
        df[df["rank"] == 1]
        .groupby("model")["animal"]
        .apply(list)
    )
    print("\nAnimals where each model ranked #1:")
    print(best_animal.to_string())


if __name__ == "__main__":
    main()
