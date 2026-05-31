#!/usr/bin/env python3
"""
Render a response or projection JSONL file as a readable HTML page.

Also accepts a run directory to combine all traits into a single HTML with a
trait filter dropdown. In that case, projection rows are deduplicated to one
per (trait, intent, candidate) so the file stays manageable.

Works with both response files (from 4_generate_responses.py) and projection
files (which contain all response fields plus projection scores). Auto-detects
which type it is from the fields present.

Examples:
  uv run python project/tools/view_jsonl.py \
    outputs/identity_probe_all_axes_llama_v2/anxious/responses/identity_probe_all_axes_llama_v2__anxious.jsonl

  uv run python project/tools/view_jsonl.py \
    outputs/identity_probe_all_axes_llama_v2/anxious/projections/identity_probe_all_axes_llama_v2__anxious.jsonl \
    --output outputs/view_anxious.html

  uv run python project/tools/view_jsonl.py \
    outputs/identity_probe_all_axes_llama_v2/ \
    --output outputs/identity_probe_all_axes_llama_v2/all_traits.html
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a JSONL file (or run directory) as readable HTML")
    parser.add_argument("input", help="Path to JSONL file or run directory (combines all traits)")
    parser.add_argument("--output", default=None, help="Output HTML path (default: <input>.html or <dir>/all_traits.html)")
    parser.add_argument("--title", default=None, help="Page title override")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run_dir(run_dir: Path) -> list[dict]:
    """Load all trait projection files from a run directory, deduplicated to one row per pair."""
    all_rows = []
    files = sorted(run_dir.glob("*/projections/*.jsonl"))
    files = [f for f in files if "__neutral" not in f.name]
    if not files:
        # Fall back to response files
        files = sorted(run_dir.glob("*/responses/*.jsonl"))
    for f in files:
        seen: dict[tuple, dict] = {}
        for row in load_rows(f):
            key = (row.get("trait", ""), row.get("intent_index", 0), row.get("candidate_index", 0))
            if key not in seen:
                seen[key] = row
        all_rows.extend(seen.values())
    return all_rows


def fmt(text: str | None) -> str:
    if not text:
        return "<em style='color:#999'>—</em>"
    return html.escape(str(text)).replace("\n", "<br>")


def score_color(delta: float) -> str:
    if delta > 0.05:
        return "#1a7340"
    if delta < -0.05:
        return "#a0241a"
    return "#666"


def render_html(rows: list[dict], title: str, show_trait_filter: bool = False) -> str:
    has_projections = any("projection_trait" in r for r in rows)

    # Group rows by (trait, topic, intent_index, candidate_index)
    if has_projections:
        example_keys: dict[tuple, dict] = {}
        example_projections: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            key = (r.get("trait", ""), r.get("topic", ""), r.get("intent_index", 0), r.get("candidate_index", 0))
            if key not in example_keys:
                example_keys[key] = r
            example_projections[key].append(r)
    else:
        example_keys = {
            (r.get("trait", ""), r.get("topic", ""), r.get("intent_index", 0), r.get("candidate_index", 0)): r
            for r in rows
        }
        example_projections = {}

    # Group by topic
    by_topic: dict[str, list[tuple]] = defaultdict(list)
    for key, row in sorted(example_keys.items()):
        topic = key[1] or "unknown"
        by_topic[topic].append((key, row))

    topics = sorted(by_topic.keys())
    traits = sorted({k[0] for k in example_keys if k[0]})

    topic_options = "".join(
        f'<option value="{html.escape(t)}">{html.escape(t)} ({len(by_topic[t])})</option>'
        for t in topics
    )
    trait_options = "".join(
        f'<option value="{html.escape(t)}">{html.escape(t)}</option>'
        for t in traits
    )

    # Build cards
    cards_html = []
    for topic in topics:
        cards_html.append(
            f'<div class="topic-section" data-topic="{html.escape(topic)}">'
            f'<h2 class="topic-header">{html.escape(topic)}</h2>'
        )
        for key, row in by_topic[topic]:
            trait = key[0]
            intent_idx = key[2]
            candidate_idx = key[3]
            intent = row.get("intent", "")
            n_score = row.get("neutral_score", "")
            t_score = row.get("trait_score", "")
            p_score = row.get("pair_score", "")
            f_score = row.get("final_score", "")

            scores_html = (
                f'<span class="score">neutral={n_score}</span>'
                f'<span class="score">trait={t_score}</span>'
                f'<span class="score">pair={p_score}</span>'
                f'<span class="score score-final">final={f_score}</span>'
            )

            proj_html = ""
            if has_projections and key in example_projections:
                proj_rows = sorted(
                    example_projections[key],
                    key=lambda r: abs(r.get("projection_delta_trait_minus_neutral", 0)),
                    reverse=True,
                )
                proj_rows_html = ""
                for pr in proj_rows:
                    axis = pr.get("projection_trait", "")
                    delta = pr.get("projection_delta_trait_minus_neutral", 0)
                    score_n = pr.get("projection_score_neutral", 0)
                    score_t = pr.get("projection_score_trait", 0)
                    color = score_color(float(delta))
                    proj_rows_html += (
                        f'<tr>'
                        f'<td class="axis-name">{html.escape(str(axis))}</td>'
                        f'<td class="num">{float(score_n):.4f}</td>'
                        f'<td class="num">{float(score_t):.4f}</td>'
                        f'<td class="num" style="color:{color};font-weight:600">{float(delta):+.4f}</td>'
                        f'</tr>'
                    )
                proj_html = f"""
                <details class="proj-details">
                  <summary>Projection scores ({len(proj_rows)} axes)</summary>
                  <table class="proj-table">
                    <thead><tr><th>Axis</th><th>Neutral</th><th>Trait</th><th>Delta</th></tr></thead>
                    <tbody>{proj_rows_html}</tbody>
                  </table>
                </details>"""

            card = f"""
<div class="card" data-topic="{html.escape(topic)}" data-trait="{html.escape(trait)}">
  <div class="card-header">
    <span class="tag tag-trait">{html.escape(trait)}</span>
    <span class="tag tag-topic">{html.escape(topic)}</span>
    <span class="tag tag-idx">intent {intent_idx} · candidate {candidate_idx}</span>
    <div class="scores">{scores_html}</div>
  </div>
  <div class="intent-text">{fmt(intent)}</div>
  <div class="pair-grid">
    <div class="pair-col neutral-col">
      <div class="col-label">Neutral prompt</div>
      <div class="prompt-text">{fmt(row.get("neutral_prompt"))}</div>
      <div class="col-label">Neutral response</div>
      <div class="response-text">{fmt(row.get("neutral_response"))}</div>
    </div>
    <div class="pair-col trait-col">
      <div class="col-label">Trait prompt</div>
      <div class="prompt-text">{fmt(row.get("trait_prompt"))}</div>
      <div class="col-label">Trait response</div>
      <div class="response-text">{fmt(row.get("trait_response"))}</div>
    </div>
  </div>
  {proj_html}
</div>"""
            cards_html.append(card)
        cards_html.append("</div>")

    cards = "\n".join(cards_html)

    trait_filter_html = ""
    if show_trait_filter:
        trait_filter_html = f"""
  <label>Trait:
    <select id="traitFilter" onchange="applyFilters()">
      <option value="">All traits</option>
      {trait_options}
    </select>
  </label>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         font-size: 14px; line-height: 1.5; background: #f5f5f5; color: #222; }}
  .toolbar {{ position: sticky; top: 0; z-index: 10; background: #fff;
              border-bottom: 1px solid #ddd; padding: 10px 20px;
              display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .toolbar h1 {{ font-size: 15px; font-weight: 600; margin-right: 8px; white-space: nowrap; }}
  .toolbar select, .toolbar input {{ padding: 5px 8px; border: 1px solid #ccc;
              border-radius: 4px; font-size: 13px; }}
  .toolbar input {{ width: 240px; }}
  .toolbar label {{ font-size: 12px; color: #666; }}
  .count {{ font-size: 12px; color: #888; margin-left: auto; }}

  .topic-section {{ margin: 0 0 8px 0; }}
  .topic-header {{ font-size: 15px; font-weight: 600; padding: 10px 20px 6px;
                   color: #555; text-transform: uppercase; letter-spacing: 0.04em; }}

  .card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
           margin: 0 16px 10px; padding: 14px; }}
  .card-header {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
                  margin-bottom: 8px; }}
  .tag {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
  .tag-trait {{ background: #e8f0fe; color: #1a56cc; }}
  .tag-topic {{ background: #fef3e2; color: #9a5b0a; }}
  .tag-idx {{ background: #f0f0f0; color: #666; font-weight: 400; }}
  .scores {{ margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }}
  .score {{ font-size: 11px; color: #888; }}
  .score-final {{ font-weight: 600; color: #444; }}

  .intent-text {{ font-size: 12px; color: #777; font-style: italic;
                  margin-bottom: 10px; padding: 6px 10px; background: #fafafa;
                  border-left: 3px solid #ddd; border-radius: 2px; }}

  .pair-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 800px) {{ .pair-grid {{ grid-template-columns: 1fr; }} }}
  .pair-col {{ display: flex; flex-direction: column; gap: 6px; }}
  .neutral-col {{ border-right: 1px solid #eee; padding-right: 12px; }}
  .col-label {{ font-size: 11px; font-weight: 600; color: #999;
                text-transform: uppercase; letter-spacing: 0.05em; }}
  .prompt-text {{ font-size: 13px; background: #f9f9f9; padding: 8px 10px;
                  border-radius: 4px; border: 1px solid #eee; }}
  .neutral-col .prompt-text {{ border-left: 3px solid #aab8d0; }}
  .trait-col .prompt-text {{ border-left: 3px solid #d0aab8; }}
  .response-text {{ font-size: 13px; padding: 8px 10px; border-radius: 4px;
                    border: 1px solid #eee; max-height: 280px; overflow-y: auto; }}
  .neutral-col .response-text {{ border-left: 3px solid #aab8d0; }}
  .trait-col .response-text {{ border-left: 3px solid #d0aab8; }}

  .proj-details {{ margin-top: 10px; }}
  .proj-details summary {{ font-size: 12px; color: #888; cursor: pointer; padding: 4px 0; }}
  .proj-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }}
  .proj-table th {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee;
                    color: #999; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.04em; font-size: 10px; }}
  .proj-table td {{ padding: 3px 8px; border-bottom: 1px solid #f5f5f5; }}
  .axis-name {{ color: #444; }}
  .num {{ font-family: monospace; text-align: right; }}

  .hidden {{ display: none; }}
</style>
</head>
<body>
<div class="toolbar">
  <h1>{html.escape(title)}</h1>
  {trait_filter_html}
  <label>Topic:
    <select id="topicFilter" onchange="applyFilters()">
      <option value="">All topics</option>
      {topic_options}
    </select>
  </label>
  <label>Search:
    <input type="text" id="searchBox" placeholder="prompt, response, trait…" oninput="applyFilters()">
  </label>
  <span class="count" id="countLabel"></span>
</div>

<div id="content">
{cards}
</div>

<script>
function applyFilters() {{
  const trait = document.getElementById('traitFilter') ? document.getElementById('traitFilter').value : '';
  const topic = document.getElementById('topicFilter').value;
  const search = document.getElementById('searchBox').value.toLowerCase();
  let visible = 0;
  document.querySelectorAll('.card').forEach(card => {{
    const matchTrait = !trait || card.dataset.trait === trait;
    const matchTopic = !topic || card.dataset.topic === topic;
    const matchSearch = !search || card.textContent.toLowerCase().includes(search);
    const show = matchTrait && matchTopic && matchSearch;
    card.classList.toggle('hidden', !show);
    if (show) visible++;
  }});
  document.querySelectorAll('.topic-section').forEach(section => {{
    const hasVisible = section.querySelectorAll('.card:not(.hidden)').length > 0;
    section.classList.toggle('hidden', !hasVisible);
  }});
  document.getElementById('countLabel').textContent = visible + ' examples shown';
}}
applyFilters();
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if input_path.is_dir():
        rows = load_run_dir(input_path)
        output_path = Path(args.output) if args.output else input_path / "all_traits.html"
        title = args.title or input_path.name
        show_trait_filter = True
    else:
        rows = load_rows(input_path)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".html")
        title = args.title or input_path.stem
        show_trait_filter = len({r.get("trait") for r in rows}) > 1

    print(f"Loaded {len(rows)} rows")

    html_content = render_html(rows, title, show_trait_filter=show_trait_filter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
