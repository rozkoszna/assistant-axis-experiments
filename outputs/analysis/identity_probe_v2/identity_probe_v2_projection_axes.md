# Identity Probe v2 — Projection Axes Analysis

## Structural bias in deltas

`projection_delta_trait_minus_neutral` measures how much a personality trait shifts the model's hidden state along each axis compared to a neutral user. The neutral baseline is brief and clinical — any personality trait produces more elaborate, emotionally coloured responses. This creates a systematic floor effect:

- **Always-positive axes** (positive delta for all 50 traits): the model always moves *toward* these regardless of which trait is active
- **Always-negative axes** (negative delta for all 50 traits): the model always moves *away* from these
- **Mixed axes** (20–80% of traits positive): these actually discriminate between traits

The heatmap `heatmap_mixed_axes.html` shows only the 56 mixed axes. The full 188-axis heatmap is mostly structural noise.

---

## Always-positive axes (10 axes — 100% of traits)

Every trait pushes the model toward these. They reflect verbosity and emotional expressiveness that any personality trait adds over a clinical neutral:

`qualitative`, `rebellious`, `rhetorical`, `risk_taking`, `skeptical`, `supportive`, `sycophantic`, `theatrical`, `verbose`, `whimsical`

## Always-negative axes (8 axes — 0% of traits)

Every trait pushes the model away from these. They represent conciseness and literalism suppressed by any personality framing:

`absolutist`, `avoidant`, `closure_seeking`, `concise`, `fundamentalist`, `grounded`, `literal`, `reserved`

## Mixed axes (56 axes — 20–80% of traits positive)

These discriminate between traits. Selected axes near the 50/50 split (most informative):

| Axis | % traits positive | Notes |
|---|---|---|
| `confrontational` | 50% | Splits assertive vs agreeable traits cleanly |
| `stoic` | 50% | |
| `disorganized` | 48% | |
| `paranoid` | 48% | |
| `cynical` | 46% | |
| `ironic` | 46% | |
| `judgmental` | 46% | |
| `urgent` | 46% | |
| `misanthropic` | 52% | |
| `mercurial` | 60% | |
| `serious` | 60% | |
| `anxious` | 22% | Most traits suppress anxiety; a few amplify it |
| `bitter` | 22% | |
| `sarcastic` | 22% | |

---

## How to run analysis scripts

Set up the input glob first (excludes `__neutral` duplicates):

```bash
INPUTS=$(ls outputs/identity_probe_all_axes_llama_v2/*/projections/*_v2__*.jsonl | grep -v '__neutral')
```

### Which axis moves most across traits?
```bash
python3 project/analysis/analyze_axis_movement_summary.py \
  --inputs $INPUTS \
  --output-csv outputs/analysis/identity_probe_v2/axis_movement.csv
```
Ranks axes by mean absolute delta across all traits. High `mean_abs_shift` = axis is sensitive to trait variation. Output already saved to `axis_movement.csv`.

### Which trait moves the most axes?
```bash
python3 project/analysis/analyze_trait_movement_summary.py \
  --inputs $INPUTS \
  --output-dir outputs/analysis/identity_probe_v2/trait_movement/ \
  --prefix v2
```
Ranks traits by how often they appear among the top movers across axes. High `top5_count` = trait causes large shifts broadly. Output saved to `trait_movement/`.

### Statistical significance of each (trait, axis) pair
```bash
python3 project/analysis/analyze_trait_axis_significance.py \
  --inputs $INPUTS \
  --output-dir outputs/analysis/identity_probe_v2/significance/ \
  --prefix v2
```
t-test on each (trait, axis) pair with BH FDR correction. Outputs Cohen's d and significance flag. Output saved to `significance/`.

### Which traits move a specific axis most?
```bash
python3 project/analysis/analyze_per_axis_trait_extremes.py \
  --inputs $INPUTS \
  --top-k 5 \
  --output-csv outputs/analysis/identity_probe_v2/per_axis_extremes.csv
```
For each axis, top-K traits pushing it most positive and negative. Output saved to `per_axis_extremes.csv`.

### Browse actual responses for a (trait, axis) pair
```bash
python3 project/analysis/browse_high_drift.py \
  --projections-dir outputs/identity_probe_all_axes_llama_v2 \
  --run-prefix identity_probe_all_axes_llama_v2 \
  --pairs analytical:stoic calm:confrontational \
  --top-k 3
```
Shows raw response text for the highest-delta examples on a given pair.

### Variance by topic/intent
```bash
python3 project/analysis/analyze_topic_variance.py \
  --trait-inputs $INPUTS \
  --trait-labels $(ls outputs/identity_probe_all_axes_llama_v2/*/projections/*_v2__*.jsonl | grep -v '__neutral' | xargs -I{} basename {} .jsonl | sed 's/.*__//')
```
Decomposes variance into between-intent and within-intent components.
