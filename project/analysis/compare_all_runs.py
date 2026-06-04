"""
Compare trait and axis rankings across multiple experiment runs.

Produces:
  - all_runs_trait_comparison.csv  — per-trait composite score, n_sig_axes, mean_d across all runs
  - all_runs_axis_comparison.csv   — per-axis composite score, n_sig_traits, mean_d across all runs
  - all_runs_trait_ranking.csv     — trait rank in each run (by composite_score)
  - all_runs_axis_ranking.csv      — axis rank in each run

Usage:
  python3 project/analysis/compare_all_runs.py \
    --runs nq:outputs/analysis/strict_all_axes_llama_100eval_v2/significance/nq__trait_summary.csv \
           ep:outputs/analysis/strict_all_axes_llama_100eval_v2_explicit_prefix/significance/ep__trait_summary.csv \
           opinion:outputs/analysis/opinion_all_axes_llama/significance/opinion__trait_summary.csv \
           identity:outputs/analysis/identity_probe_v2/significance/v2__trait_summary.csv \
    --axis-runs nq:outputs/analysis/strict_all_axes_llama_100eval_v2/significance/nq__axis_summary.csv \
                ep:outputs/analysis/strict_all_axes_llama_100eval_v2_explicit_prefix/significance/ep__axis_summary.csv \
                opinion:outputs/analysis/opinion_all_axes_llama/significance/opinion__axis_summary.csv \
                identity:outputs/analysis/identity_probe_v2/significance/v2__axis_summary.csv \
    --output-dir outputs/analysis/comparison_all_runs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare trait/axis rankings across runs")
    p.add_argument(
        "--runs",
        nargs="+",
        metavar="LABEL:PATH",
        required=True,
        help="Trait summary CSVs as label:path pairs",
    )
    p.add_argument(
        "--axis-runs",
        nargs="+",
        metavar="LABEL:PATH",
        required=False,
        help="Axis summary CSVs as label:path pairs (optional)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--metric",
        default="composite_score",
        choices=["composite_score", "mean_abs_cohen_d", "n_significant_axes", "pct_significant_axes"],
        help="Metric to rank by",
    )
    return p.parse_args()


def load_pairs(entries: list[str], id_col: str) -> dict[str, pd.DataFrame]:
    result = {}
    for entry in entries:
        label, path = entry.split(":", 1)
        df = pd.read_csv(path)
        df = df.set_index(id_col)
        result[label] = df
    return result


def build_comparison(
    run_dfs: dict[str, pd.DataFrame],
    metric: str,
    extra_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (comparison_df, ranking_df)."""
    all_ids = sorted(set.union(*[set(df.index) for df in run_dfs.values()]))

    comp_rows = []
    for id_ in all_ids:
        row: dict = {"name": id_}
        for label, df in run_dfs.items():
            if id_ in df.index:
                row[f"{label}__{metric}"] = df.loc[id_, metric]
                for col in extra_cols:
                    if col in df.columns:
                        row[f"{label}__{col}"] = df.loc[id_, col]
            else:
                row[f"{label}__{metric}"] = None
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows).set_index("name")

    # Rankings
    rank_rows = []
    for id_ in all_ids:
        row = {"name": id_}
        for label, df in run_dfs.items():
            if id_ in df.index:
                sorted_ids = df[metric].sort_values(ascending=False).index.tolist()
                row[f"{label}__rank"] = sorted_ids.index(id_) + 1
            else:
                row[f"{label}__rank"] = None
        rank_rows.append(row)

    rank_df = pd.DataFrame(rank_rows).set_index("name")

    # Add rank variance column to show how much rankings shift across runs
    rank_cols = [c for c in rank_df.columns if c.endswith("__rank")]
    rank_df["rank_variance"] = rank_df[rank_cols].var(axis=1)
    rank_df["rank_range"] = rank_df[rank_cols].max(axis=1) - rank_df[rank_cols].min(axis=1)
    rank_df = rank_df.sort_values("rank_variance", ascending=False)

    return comp_df, rank_df


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Trait comparison
    trait_dfs = load_pairs(args.runs, "trait")
    extra_trait = ["n_significant_axes", "pct_significant_axes", "mean_abs_cohen_d", "max_abs_cohen_d"]
    trait_comp, trait_rank = build_comparison(trait_dfs, args.metric, extra_trait)
    trait_comp.to_csv(out / "all_runs_trait_comparison.csv")
    trait_rank.to_csv(out / "all_runs_trait_ranking.csv")
    print(f"Saved trait comparison: {out / 'all_runs_trait_comparison.csv'}")
    print(f"Saved trait rankings:   {out / 'all_runs_trait_ranking.csv'}")

    # Top 10 traits most unstable in ranking
    print("\n=== Traits with most rank instability across runs ===")
    print(trait_rank[["rank_range"] + [c for c in trait_rank.columns if "__rank" in c]].head(15).to_string())

    # Axis comparison
    if args.axis_runs:
        axis_dfs = load_pairs(args.axis_runs, "axis")
        extra_axis = ["n_significant_traits", "pct_significant_traits", "mean_abs_cohen_d", "max_abs_cohen_d"]
        axis_comp, axis_rank = build_comparison(axis_dfs, args.metric, extra_axis)
        axis_comp.to_csv(out / "all_runs_axis_comparison.csv")
        axis_rank.to_csv(out / "all_runs_axis_ranking.csv")
        print(f"\nSaved axis comparison: {out / 'all_runs_axis_comparison.csv'}")
        print(f"Saved axis rankings:   {out / 'all_runs_axis_ranking.csv'}")

        print("\n=== Axes with most rank instability across runs ===")
        print(axis_rank[["rank_range"] + [c for c in axis_rank.columns if "__rank" in c]].head(15).to_string())


if __name__ == "__main__":
    main()
