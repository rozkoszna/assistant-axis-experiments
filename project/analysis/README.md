# Analysis scripts

All scripts read projection JSONL files produced by the pipeline
(`outputs/user_prompts/<trait>/projections/<run>.jsonl`) and output CSVs or JSON
summaries. None of them re-run the model.

The measured quantity throughout is `projection_delta_trait_minus_neutral`:
the dot product of the assistant's hidden-state activation with a precomputed
axis vector, under the trait condition minus the neutral condition.

---

## Axis-level scripts

### `analyze_axis_movement_summary.py`
**Which axes are moved most across all traits?**

For each projection axis, aggregates per-trait mean deltas and reports:
- `mean_abs_shift` — average absolute shift across traits (primary rank key)
- `max_abs_shift` — largest single-trait shift on this axis
- `signed_mean_shift` — net direction; positive means traits push the axis up on average
- `trait_mean_std` — spread of per-trait means; high std means traits disagree in direction
- `strongest_positive/negative_trait` — which trait pulls hardest in each direction

Counterpart to `analyze_trait_movement_summary.py` (axis view vs. trait view).

```
python project/analysis/analyze_axis_movement_summary.py \
  --inputs outputs/user_prompts/*/projections/<run>.jsonl \
  --top-k 20 \
  --output-csv outputs/analysis/axis_movement.csv
```

---

### `analyze_per_axis_trait_extremes.py`
**For each axis, which traits push it hardest in each direction?**

For each axis, ranks all traits by their mean `projection_delta_trait_minus_neutral`
and reports the top-K positive and top-K negative traits with their mean delta,
min, max, and count.

```
python project/analysis/analyze_per_axis_trait_extremes.py \
  --inputs outputs/user_prompts/*/projections/<run>.jsonl \
  --top-k 4 \
  --output-json outputs/analysis/per_axis_extremes.json \
  --output-csv  outputs/analysis/per_axis_extremes.csv
```

---

## Trait-level scripts

### `analyze_trait_movement_summary.py`
**Which traits move axes most broadly?**

For each trait, scores how often it appears among the top movers across all axes.
Produces 9 tables: direction (any / positive / negative) × metric (top5_count /
top10_count / abs_mean_sum). A trait with a high `any_top5_count` consistently
causes large shifts across many axes regardless of direction.

Counterpart to `analyze_axis_movement_summary.py` (trait view vs. axis view).

```
python project/analysis/analyze_trait_movement_summary.py \
  --inputs outputs/user_prompts/*/projections/<run>.jsonl \
  --output-dir outputs/analysis/trait_movers/
```

---

### `analyze_trait_axis_significance.py`
**Which (trait, axis) pairs show statistically significant shifts?**

The unit of analysis is a (trait, axis) pair. For each combination, runs a
one-sample t-test on `projection_delta_trait_minus_neutral` and computes Cohen's d.
Benjamini-Hochberg FDR correction is applied across all pairs jointly.

Outputs five tables:
- Table 1 — all (trait, axis) pairs: p_adjusted, cohen_d, significant flag, direction
- Table 2 — per-trait summary: how many axes each trait moves significantly
- Table 3 — per-axis summary: how many traits significantly move each axis
- Table 4a — distribution: how many traits move exactly N axes significantly
- Table 4b — distribution: how many axes are moved by exactly N traits significantly

```
python project/analysis/analyze_trait_axis_significance.py \
  --inputs outputs/user_prompts/*/projections/<run>.jsonl \
  --output-dir outputs/analysis/significance/ \
  --prefix strict_all_axes
```

---

## Variance / diagnostic scripts

### `analyze_topic_variance.py`
**How much of the projection variance comes from topic choice vs. within-topic noise?**

For each axis and condition, decomposes variance into:
- `between_topic_std` — how much means differ across topics (intents)
- `pooled_within_topic_std` — residual noise after removing topic means

If `--minibatch-size` is set, also runs a bootstrap to estimate how much variance
remains after averaging within-topic examples, which approximates the signal you
would measure with more examples per topic.

```
python project/analysis/analyze_topic_variance.py \
  --trait-inputs outputs/user_prompts/confused/projections/<run>.jsonl \
  --trait-labels confused \
  --include-neutral \
  --output-csv outputs/analysis/topic_variance.csv
```

---

## Comparison scripts

### `compare_experiments.py`
**Diff two significance experiments to see where they agree or disagree.**

Reads the CSVs produced by `analyze_trait_axis_significance.py` for two runs and
identifies: which (trait, axis) pairs are significant in one condition but not the
other, and by how much do effect sizes differ?

Outputs:
- `trait_comparison.csv` — per trait: n_significant_axes and mean Cohen's d in each
  experiment, plus deltas between them
- `axis_comparison.csv` — per axis: n_significant_traits and mean Cohen's d in each
  experiment, plus deltas between them
- `new_significant_in_opinion.csv` — pairs significant in experiment B but not A
- `lost_significant_in_opinion.csv` — pairs significant in experiment A but not B

```
python project/analysis/compare_experiments.py \
  --factual-dir  outputs/analysis/significance \
  --opinion-dir  outputs/analysis/opinion/significance \
  --factual-prefix  strict_all_axes \
  --opinion-prefix  opinion_axes \
  --output-dir  outputs/analysis/comparison
```

---

### `compare_trait_rankings.py`
**Compare how any two CSVs rank traits and find where they disagree.**

Takes any two CSVs with a `trait` column and a numeric score column, ranks traits
by those scores, and reports where the rankings diverge most. Computes Spearman rank
correlation to quantify overall agreement. Works for any comparison: two runs of the
same experiment, two intent types, magnitude vs. significance, etc.

```
# Magnitude vs. significance
python project/analysis/compare_trait_rankings.py \
  --csv-a   outputs/analysis/trait_movers/trait_global_movers__any_abs_mean_sum.csv \
  --col-a   any_abs_mean_sum --label-a magnitude \
  --csv-b   outputs/analysis/significance/strict_all_axes__trait_summary.csv \
  --col-b   n_significant_axes --label-b significance \
  --output-dir outputs/analysis/trait_rankings/

# Two runs
python project/analysis/compare_trait_rankings.py \
  --csv-a   outputs/analysis/run1/strict_all_axes__trait_summary.csv \
  --col-a   n_significant_axes --label-a run1 \
  --csv-b   outputs/analysis/run2/strict_all_axes__trait_summary.csv \
  --col-b   n_significant_axes --label-b run2 \
  --output-dir outputs/analysis/trait_rankings/
```
