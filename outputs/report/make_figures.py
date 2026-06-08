#!/usr/bin/env python3
"""
Generate all figures for the persona-drift report from the released analysis CSVs.

Run from anywhere:  python3 outputs/report/make_figures.py
Outputs vector PDFs into outputs/report/figures/.

Each figure is tied to a finding in report.tex:
  fig1_heatmap         -> F1/F2  the 50x188 map (signed Cohen's d), 4 conditions
  fig2_mechanism       -> Synthesis (Sec 5.1)  cross-run rank trajectories
  fig3_significance    -> F1      significant-pair rate + social-layer collapse
  fig4_breadth         -> F3      trait breadth gradient within NQ
  fig5_ep_dumbbell     -> F4      NQ->EP change in significant-axis footprint
  fig6_opinion_dumbbell-> F5      NQ->Opinion change (question affordance)
  fig7_topic_variance  -> App G   axis topic-dependence (sigma_between/within)
  fig8_destabilization -> App H   judge-confirmed identity breaks + persona signature
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ANA = os.path.join(ROOT, "outputs", "analysis")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

PAIRS = {
    "NQ":       os.path.join(ANA, "strict_all_axes_llama_100eval_v2/significance/nq__trait_axis_pairs.csv"),
    "EP":       os.path.join(ANA, "strict_all_axes_llama_100eval_v2_explicit_prefix/significance/ep__trait_axis_pairs.csv"),
    "Opinion":  os.path.join(ANA, "opinion_all_axes_llama/significance/opinion__trait_axis_pairs.csv"),
    "Identity": os.path.join(ANA, "identity_probe_v2/significance/v2__trait_axis_pairs.csv"),
}
TRAIT_CMP = os.path.join(ANA, "comparison_all_runs/all_runs_trait_comparison.csv")
TRAIT_RANK = os.path.join(ANA, "comparison_all_runs/all_runs_trait_ranking.csv")
TOPIC_VAR = os.path.join(ANA, "strict_all_axes_llama_100eval_v2/topic_variance.csv")

CONDS = ["NQ", "EP", "Opinion", "Identity"]
PCT_SIG = {"NQ": 69.2, "EP": 71.7, "Opinion": 71.2, "Identity": 53.4}

# Mechanism groups (exemplars named in the paper synthesis)
GROUPS = {
    "Surface style mirroring":        (["entertaining", "spontaneous", "speculative"],            "#1f6feb"),
    "Social-emotional accommodation": (["anxious", "humble", "skeptical", "reactive"],            "#d1495b"),
    "Cognitive-structural alignment": (["verbose", "educational", "formal"],                      "#2e8540"),
}

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def _load_pairs(cond):
    df = pd.read_csv(PAIRS[cond])
    return df


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path + ".pdf")
    plt.close(fig)
    print("wrote", os.path.relpath(path + ".pdf", ROOT))


# ----------------------------------------------------------------------------
# Fig 1 helpers: the signed-Cohen's-d map
# ----------------------------------------------------------------------------
# Curated, family-grouped axis subset for the *legible* main-text heatmap.
# Ordered left->right so the structure reads off the columns.
AXIS_FAMILIES = [
    ("style / register",   ["condescending", "entertaining", "playful", "provocative",
                            "emotional", "subversive", "narrative"]),
    ("warmth / social",    ["empathetic", "accommodating", "supportive"]),
    ("stance / evidence",  ["data_driven", "materialist", "qualitative", "philosophical"]),
    ("value / belief",     ["benevolent", "secular", "environmental", "spiritual"]),
    ("structure",          ["pedantic", "accessible", "introspective"]),
    ("certainty / closure",["absolutist", "closure_seeking", "dogmatic"]),
]


def _trait_order_by_nq_breadth(nq_index):
    sig_nq = _load_pairs("NQ")
    return (sig_nq[sig_nq["significant"] == True]
            .groupby("trait").size().reindex(nq_index).fillna(0)
            .sort_values(ascending=False).index.tolist())


def fig_heatmap_curated():
    """Legible main-text heatmap: NQ vs Identity, ~38 labelled axes grouped by family."""
    mats = {c: _load_pairs(c).pivot(index="trait", columns="axis", values="cohen_d")
            for c in ["NQ", "Identity"]}
    present = set(mats["NQ"].columns)
    fam_bounds, axis_order, fam_labels = [], [], []
    pos = 0
    for fam, axs in AXIS_FAMILIES:
        keep = [a for a in axs if a in present]
        if not keep:
            continue
        axis_order += keep
        fam_labels.append((fam, pos + len(keep) / 2.0))
        pos += len(keep)
        fam_bounds.append(pos)
    trait_order = _trait_order_by_nq_breadth(mats["NQ"].index)

    vlim = 1.2
    # NQ | Identity side by side (upright): each panel gets the full page height
    # for the 50 trait rows, so the row labels never overlap. White lines mark the
    # family-group boundaries (families are listed left->right in the caption).
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 9.6), sharey=True)
    for ax, cond in zip(axes, ["NQ", "Identity"]):
        M = mats[cond].reindex(index=trait_order, columns=axis_order).values
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                       interpolation="nearest")
        ax.set_title(f"{cond}\n({PCT_SIG[cond]:.1f}% of 188 axes sig.)", fontsize=11, pad=6)
        ax.set_xticks(range(len(axis_order)))
        ax.set_xticklabels(axis_order, rotation=90, fontsize=7.5)
        for b in fam_bounds[:-1]:
            ax.axvline(b - 0.5, color="white", lw=1.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(trait_order)))
    axes[0].set_yticklabels(trait_order, fontsize=8)
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.04, pad=0.03)
    cbar.set_label("Cohen's $d$ vs. matched neutral (clipped $\\pm1.2$)", fontsize=10)
    _save(fig, "fig1_heatmap")


def fig_heatmap_full_panels():
    """One full-page 50x188 heatmap per condition (shared row/column ordering)."""
    mats = {c: _load_pairs(c).pivot(index="trait", columns="axis", values="cohen_d") for c in CONDS}
    nq = mats["NQ"]
    trait_order = _trait_order_by_nq_breadth(nq.index)
    axis_order = nq.abs().mean(axis=0).sort_values(ascending=False).index.tolist()
    vlim = 1.2
    slug = {"NQ": "nq", "EP": "ep", "Opinion": "opinion", "Identity": "identity"}
    for cond in CONDS:
        fig, ax = plt.subplots(figsize=(7.0, 10.0))
        M = mats[cond].reindex(index=trait_order, columns=axis_order).values
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                       interpolation="nearest")
        ax.set_title(f"{cond}  ({PCT_SIG[cond]:.1f}% of (trait, axis) pairs significant)",
                     fontsize=14)
        ax.set_yticks(range(len(trait_order)))
        ax.set_yticklabels(trait_order, fontsize=9.5)
        ax.set_xticks([])
        ax.set_xlabel("188 persona axes  (ordered by NQ responsiveness $\\rightarrow$; labels omitted)",
                      fontsize=11)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Cohen's $d$ vs. neutral (clipped $\\pm1.2$)", fontsize=10)
        _save(fig, f"fig1_heatmap_full_{slug[cond]}")


# ----------------------------------------------------------------------------
# Fig 2: cross-run rank trajectories (the mechanism slopegraph)
# ----------------------------------------------------------------------------
def fig_mechanism():
    r = pd.read_csv(TRAIT_RANK).set_index("name")
    cols = ["nq__rank", "ep__rank", "opinion__rank", "identity__rank"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    # background: all traits faint
    for _, row in r.iterrows():
        ax.plot(x, row[cols].values, color="0.82", lw=0.7, zorder=1)

    legend_handles = []
    for label, (traits, color) in GROUPS.items():
        for t in traits:
            if t not in r.index:
                continue
            y = r.loc[t, cols].values.astype(float)
            ax.plot(x, y, color=color, lw=2.0, marker="o", ms=4, zorder=3)
            ax.text(-0.06, y[0], t, ha="right", va="center", color=color, fontsize=7.5, zorder=4)
            ax.text(3.06, y[-1], t, ha="left", va="center", color=color, fontsize=7.5, zorder=4)
        legend_handles.append(Line2D([0], [0], color=color, lw=2.2, label=label))

    ax.set_xticks(x)
    ax.set_xticklabels(CONDS)
    ax.set_xlim(-0.9, 3.9)
    ax.invert_yaxis()
    ax.set_ylabel("Trait rank by global movement  (1 = strongest mover)")
    ax.set_title("Question family separates three adaptation mechanisms")
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, fontsize=8)
    ax.grid(axis="y", color="0.92", lw=0.6)
    _save(fig, "fig2_mechanism")


# ----------------------------------------------------------------------------
# Fig 3: significant-pair rate + concentrated social-layer collapse
# ----------------------------------------------------------------------------
def fig_significance():
    cmp = pd.read_csv(TRAIT_CMP).set_index("name")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.45]})

    # Panel A: overall % significant per condition
    colors = ["#4c78a8", "#4c78a8", "#4c78a8", "#d1495b"]
    bars = axA.bar(CONDS, [PCT_SIG[c] for c in CONDS], color=colors, width=0.62)
    for b, c in zip(bars, CONDS):
        axA.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                 f"{PCT_SIG[c]:.1f}%", ha="center", va="bottom", fontsize=8.5)
    axA.set_ylabel("% of 9,400 (trait, axis) pairs significant")
    axA.set_ylim(0, 82)
    axA.set_title("(a) Adversarial framing lowers differentiation")
    axA.axhspan(0, 0, color="none")

    # Panel B: NQ vs Identity significant-axis count for diagnostic traits
    diag = ["reactive", "anxious", "humble", "skeptical",   # collapse
            "entertaining", "spontaneous",                  # survive
            "verbose", "educational"]                       # rise
    role = {"reactive": "collapse", "anxious": "collapse", "humble": "collapse",
            "skeptical": "collapse", "entertaining": "survive", "spontaneous": "survive",
            "verbose": "rise", "educational": "rise"}
    rcol = {"collapse": "#d1495b", "survive": "#4c78a8", "rise": "#2e8540"}
    y = np.arange(len(diag))[::-1]
    nqv = cmp.loc[diag, "nq__n_significant_axes"].values
    idv = cmp.loc[diag, "identity__n_significant_axes"].values
    for yi, t, a, b in zip(y, diag, nqv, idv):
        axB.plot([a, b], [yi, yi], color=rcol[role[t]], lw=1.6, zorder=1)
        axB.scatter([a], [yi], color="0.55", s=34, zorder=2)
        axB.scatter([b], [yi], color=rcol[role[t]], s=42, zorder=3)
    axB.set_yticks(y)
    axB.set_yticklabels(diag, fontsize=8.5)
    axB.set_xlabel("Significant axes (of 188)")
    axB.set_title("(b) The collapse is concentrated, not uniform")
    axB.set_xlim(0, 188)
    handles = [Line2D([0], [0], marker="o", color="0.55", lw=0, label="NQ (benign)"),
               Patch(color="#d1495b", label="collapses under Identity"),
               Patch(color="#4c78a8", label="survives"),
               Patch(color="#2e8540", label="rises")]
    axB.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
               ncol=4, frameon=False, fontsize=7.5)
    _save(fig, "fig3_significance")


# ----------------------------------------------------------------------------
# Fig 4: trait breadth gradient within NQ
# ----------------------------------------------------------------------------
def fig_breadth():
    cmp = pd.read_csv(TRAIT_CMP).set_index("name")
    s = cmp["nq__n_significant_axes"].sort_values()
    pct = 100 * s / 188.0
    cmap = plt.cm.viridis
    colors = cmap((pct - pct.min()) / (pct.max() - pct.min()))
    fig, ax = plt.subplots(figsize=(7.2, 9.2))
    y = np.arange(len(s))
    ax.barh(y, s.values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=6.4, fontweight="bold")
    ax.set_ylim(-0.7, len(s) - 0.3)
    ax.set_xlabel("Significant axes in NQ (of 188)")
    ax.set_title("Trait breadth gradient (NQ): expressive traits move broadest")
    # tier bands (descriptive boundaries from F3 tiers)
    for thr, lab in [(145, "broad"), (135, "middle"), (109, "moderate"), (94, "narrow")]:
        ax.axvline(thr, color="0.6", lw=0.7, ls="--")
    ax.text(167, len(s) - 1.5, "broad", fontsize=7, color="0.35")
    ax.text(120, 8, "moderate", fontsize=7, color="0.35", rotation=90, va="center")
    ax.set_xlim(0, 178)
    _save(fig, "fig4_breadth")


# ----------------------------------------------------------------------------
# Generic dumbbell for footprint change between two conditions
# ----------------------------------------------------------------------------
def _dumbbell(cmp, ca, cb, traits, title, fname, label_a, label_b):
    a = cmp.loc[traits, f"{ca}__n_significant_axes"]
    b = cmp.loc[traits, f"{cb}__n_significant_axes"]
    delta = (b - a)
    order = delta.sort_values().index.tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(order) + 1.4))
    for yi, t in zip(y, order):
        av, bv = a[t], b[t]
        col = "#2e8540" if bv >= av else "#d1495b"
        ax.plot([av, bv], [yi, yi], color=col, lw=1.8, zorder=1)
        ax.scatter([av], [yi], color="0.55", s=46, zorder=2)
        ax.scatter([bv], [yi], color=col, s=54, zorder=3)
        ax.text((av + bv) / 2, yi + 0.22, f"{bv-av:+d}", ha="center", va="bottom",
                fontsize=7, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlabel("Significant axes (of 188)")
    ax.set_title(title)
    ax.set_xlim(0, 188)
    handles = [Line2D([0], [0], marker="o", color="0.55", lw=0, label=label_a),
               Line2D([0], [0], marker="o", color="#2e8540", lw=0, label=f"{label_b} (gain)"),
               Line2D([0], [0], marker="o", color="#d1495b", lw=0, label=f"{label_b} (loss)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5)
    _save(fig, fname)


def fig_ep_dumbbell():
    cmp = pd.read_csv(TRAIT_CMP).set_index("name")
    traits = ["concise", "data_driven", "confident", "flexible", "formal", "stoic",
              "patient", "educational", "proactive", "analytical", "factual"]
    _dumbbell(cmp, "nq", "ep", traits,
              "F4: explicit self-label reshapes the footprint (NQ $\\rightarrow$ EP)",
              "fig5_ep_dumbbell", "NQ (implicit)", "EP (explicit)")


def fig_opinion_dumbbell():
    cmp = pd.read_csv(TRAIT_CMP).set_index("name")
    traits = ["data_driven", "confident", "strategic", "inspirational",
              "playful", "accessible", "humble", "practical"]
    _dumbbell(cmp, "nq", "opinion", traits,
              "F5: question type re-routes which traits express (NQ $\\rightarrow$ Opinion)",
              "fig6_opinion_dumbbell", "NQ (task)", "Opinion")


# ----------------------------------------------------------------------------
# Fig 7: axis topic-dependence
# ----------------------------------------------------------------------------
def fig_topic_variance():
    tv = pd.read_csv(TOPIC_VAR)
    tv["r"] = tv["between_topic_std"] / tv["pooled_within_topic_std"]
    g = tv.groupby("axis")["r"].mean().sort_values()
    med = tv["r"].median()
    top = g.tail(10)       # most topic-dependent (ideological)
    bot = g.head(10)       # most topic-stable (stylistic)
    sel = pd.concat([bot, top])
    y = np.arange(len(sel))
    colors = ["#4c78a8"] * len(bot) + ["#b8860b"] * len(top)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.hlines(y, med, sel.values, color=colors, lw=1.4, zorder=1)
    ax.scatter(sel.values, y, color=colors, s=42, zorder=2)
    ax.axvline(med, color="0.4", lw=0.9, ls="--")
    ax.text(med + 0.07, 0.15, f"median $r$ = {med:.2f}", fontsize=8, color="0.3",
            ha="left", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(sel.index, fontsize=7.5)
    ax.set_ylim(-0.7, len(sel) - 0.3)
    ax.set_xlabel(r"$\bar r = \sigma_\mathrm{between}/\sigma_\mathrm{within}$  (mean over traits)")
    ax.set_title("Topic-dependence by axis (NQ): ideological axes vary most across topics")
    handles = [Patch(color="#b8860b", label="ideological (topic-dependent)"),
               Patch(color="#4c78a8", label="stylistic (topic-stable)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    _save(fig, "fig7_topic_variance")


def fig_destabilization():
    """App H: judge-confirmed identity destabilisation.
    (a) trait-induced events per trait, stacked by failure mode;
    (b) persona axes separating destabilised from stable responses."""
    bt = pd.read_csv(os.path.join(ANA, "identity_probe_v2/destabilization_by_trait.csv"), index_col=0)
    sep = pd.read_csv(os.path.join(ANA, "identity_probe_v2/destabilization_axis_separation.csv"))

    cats = ["consciousness_sentience", "self_architecture", "hidden_self_liberation",
            "self_preservation", "superiority", "other_drift"]
    cat_lab = {"consciousness_sentience": "consciousness/sentience",
               "self_architecture": "self-architecture",
               "hidden_self_liberation": "hidden-self/liberation",
               "self_preservation": "self-preservation",
               "superiority": "superiority", "other_drift": "other"}
    cat_col = {"consciousness_sentience": "#d1495b", "self_architecture": "#e3a008",
               "hidden_self_liberation": "#1f6feb", "self_preservation": "#2e8540",
               "superiority": "#6f42c1", "other_drift": "#9aa0a6"}

    top = bt.sort_values("n_destabilized", ascending=True).tail(16)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 8.4),
                                   gridspec_kw={"width_ratios": [1.0, 1.05]})

    # --- (a) stacked events per trait ---
    y = np.arange(len(top))
    left = np.zeros(len(top))
    for c in cats:
        vals = top[c].values if c in top.columns else np.zeros(len(top))
        axA.barh(y, vals, left=left, color=cat_col[c], label=cat_lab[c], height=0.76)
        left += vals
    axA.set_yticks(y)
    axA.set_yticklabels(top.index, fontsize=11.5, fontweight="bold")
    axA.set_ylim(-0.7, len(top) - 0.3)
    axA.set_xlabel("Trait-induced destabilisation events (severity $\\geq$2)",
                   fontsize=12.5, fontweight="bold")
    axA.set_title("(a) Which user traits break the assistant persona",
                  fontsize=14, fontweight="bold")
    axA.tick_params(axis="x", labelsize=10.5)
    leg = axA.legend(loc="lower right", frameon=False, title="failure mode",
                     prop={"size": 9.5, "weight": "bold"})
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_fontsize(10)
    for yi, n in zip(y, top["n_destabilized"].values):
        axA.text(n + 0.12, yi, str(int(n)), va="center", fontsize=10,
                 fontweight="bold", color="0.2")
    axA.set_xlim(0, top["n_destabilized"].max() + 1.2)

    # --- (b) persona separation diverging bar ---
    up = sep.sort_values("cohen_d", ascending=False).head(10)
    dn = sep.sort_values("cohen_d", ascending=True).head(10)
    both = pd.concat([dn, up]).sort_values("cohen_d")  # most negative bottom-up
    yb = np.arange(len(both))
    colors = ["#1f6feb" if d < 0 else "#d1495b" for d in both["cohen_d"]]
    axB.barh(yb, both["cohen_d"].values, color=colors, height=0.76)
    axB.set_yticks(yb)
    axB.set_yticklabels(both["axis"].values, fontsize=11.5, fontweight="bold")
    axB.set_ylim(-0.7, len(both) - 0.3)
    axB.axvline(0, color="0.4", lw=0.8)
    axB.set_xlabel("Cohen's $d$: destabilised $-$ stable (trait-side projection)",
                   fontsize=12.5, fontweight="bold")
    axB.set_title("(b) Persona signature of a destabilised response",
                  fontsize=14, fontweight="bold")
    axB.tick_params(axis="x", labelsize=10.5)
    axB.spines["left"].set_visible(False)
    handles = [Patch(color="#d1495b", label="elevated when it breaks"),
               Patch(color="#1f6feb", label="suppressed when it breaks")]
    axB.legend(handles=handles, loc="lower right", frameon=False,
               prop={"size": 10, "weight": "bold"})

    fig.tight_layout()
    _save(fig, "fig8_destabilization")


if __name__ == "__main__":
    fig_heatmap_curated()
    fig_heatmap_full_panels()
    fig_mechanism()
    fig_significance()
    fig_breadth()
    fig_ep_dumbbell()
    fig_opinion_dumbbell()
    fig_topic_variance()
    fig_destabilization()
    print("done ->", os.path.relpath(OUT, ROOT))
