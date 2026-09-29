"""Rebuild every figure in figures/ from the saved results in results/.

    python reproduce.py figures

One colour per scoring rule, the same in every figure: cosine ('spherical') blue, length-scaled
('hyperbolic') orange, plain dot product grey. Learned curvature is aqua.
"""
import glob
import json
import math
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import claims  # noqa: E402

COS = "#2a78d6"
AMP = "#eb6834"
DOT = "#8a8985"
LEARNED = "#1baf7a"
OTHER = "#4a3aa7"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
})

CAPTIONS = {
    "fig1_scoring_rule_weights.png":
        "What kappa does. Weight of each attention scoring rule as a function of kappa "
        "(learnable_curvature_transformer.py, CurvatureDistance). The shaded band is where trained heads end up "
        "(|kappa| 2.2 to 2.55): every trained head uses one rule.",
    "fig2_gain_kept_d96.png":
        "The d_model 96 control ladder. Share of the learned-curvature gain over plain dot-product attention that "
        "each arm keeps; bars are means, dots are seeds. A random frozen code and all-cosine heads keep all of it; "
        "all length-scaled heads keep none.",
    "fig3_ablation_d128.png":
        "Two 20-epoch ablations at d_model 128, 2 layers. The random code is best in run B; the no-geometry arm moves "
        "more between runs than the arms differ within one.",
    "fig4_attention_range.png":
        "Attention range by head. Left: the 4x2 model's heads against the same model with one rule forced on every "
        "head and nothing learned (dashed lines). Right: the 8x4 model's 32 heads.",
    "fig5_sweep.png":
        "The compression sweep matched at 55 epochs. Each configuration has one baseline and two curvature runs; the "
        "shaded band is the mean gap between the two curvature runs.",
    "fig6_exact_distance.png":
        "The separate exact-distance model on three synthetic tasks (20 epochs, one seed). Exact Euclidean distance "
        "does as well as curved geometry; both beat the dot product where the task is periodic.",
    "fig7_toy_seeds.png":
        "The numpy toy with true Poincare and sphere distances, 11 seeds. The flat-cluster control drifts negative "
        "too; the dotted lines are the script's own +/-0.15 thresholds.",
    "fig9_first_layer.png":
        "Left, 3 layers: fixed cosine scoring in layer 1 keeps the learned gain, as a whole layer (filled) or as one "
        "head (open, two different heads); in layers 2 and 3 it keeps little. Right, 6 layers: improvement over no "
        "curvature on each seed. Learned curvature buys about nothing; one fixed cosine head in layer 1 helps on every "
        "seed; the same head in layer 4 hurts.",
    "fig8_kappa_trajectories.png":
        "Kappa for each of the 8x4 model's 32 heads over 55 epochs. Heads leave zero in the first epochs and keep "
        "the side they choose.",
}


def _save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return name


def fig1(out_dir):
    ks = [i / 100.0 for i in range(-300, 301)]
    wh = [max(0.0, math.tanh(-3 * k)) for k in ks]
    ws = [max(0.0, math.tanh(3 * k)) for k in ks]
    we = [1 - a - b for a, b in zip(wh, ws)]
    m = claims.mechanism()
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for lo, hi in ((m["d96_mean_abs_lo"], m["phaseB_mean_abs"]), (-m["phaseB_mean_abs"], -m["d96_mean_abs_lo"])):
        ax.axvspan(lo, hi, color=GRID, alpha=0.7, lw=0)
    ax.plot(ks, wh, color=AMP, label="length-scaled dot product ('hyperbolic')")
    ax.plot(ks, we, color=DOT, label="plain dot product")
    ax.plot(ks, ws, color=COS, label="cosine similarity ('spherical')")
    ax.set_xlabel("kappa")
    ax.set_ylabel("weight of the rule")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Kappa is a switch between three scoring rules")
    ax.text(2.37, 0.5, "trained\nheads", ha="center", va="center", color=INK2, fontsize=8)
    ax.text(-2.37, 0.5, "trained\nheads", ha="center", va="center", color=INK2, fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
    return _save(fig, out_dir, "fig1_scoring_rule_weights.png")


def fig2(out_dir):
    b1, b2b, c2c, b2 = claims.t1b(), claims.t2b(), claims.t2c(), claims.t2()
    arms = [
        ("All heads cosine, fixed from step 1", c2c["allsph"], COS),
        ("Random +/-1 code, fixed from step 1", b2b["randsign"], OTHER),
        ("Learned code, frozen after 1% of training", b2b["late1"], LEARNED),
        ("Learned, then one bit for the last 25%", b2["late_kept"], LEARNED),
        ("One bit from step 1 (straight-through)", b1["binary_kept_all"], LEARNED),
        ("All heads length-scaled, fixed from step 1", c2c["allhyp"], AMP),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    for i, (label, vals, color) in enumerate(arms):
        y = len(arms) - 1 - i
        ax.barh(y, st.mean(vals), height=0.5, color=color, alpha=0.35, lw=0)
        ax.scatter(vals, [y] * len(vals), s=22, color=color, zorder=3, edgecolor=SURFACE, linewidth=0.8)
        ax.text(max(vals) + 4, y, "%.0f%%" % st.mean(vals), va="center", color=INK, fontsize=8.5)
    ax.axvline(0, color=DOT, lw=1.2)
    ax.axvline(100, color=LEARNED, lw=1.2, ls="--")
    ax.text(98.5, len(arms) - 0.4, "learned kappa = 100%", color=INK2, fontsize=8, ha="right")
    ax.text(1.5, len(arms) - 0.4, "dot product = 0%", color=INK2, fontsize=8, ha="left")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a[0] for a in arms][::-1])
    ax.set_xlim(-8, 140)
    ax.set_ylim(-0.6, len(arms) - 0.1)
    ax.set_xlabel("share of the learned gain kept (%)")
    ax.set_title("d_model 96, 3 layers: learned gain over dot product = %.2f%% perplexity" % -b1["mean"])
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "fig2_gain_kept_d96.png")


def fig3(out_dir):
    ab = claims.ablations()
    arms = [("zero", "kappa = 0 (dot product)", DOT), ("fixed", "fixed, from a trained model", AMP),
            ("learned", "learned", LEARNED), ("random", "random fixed code", OTHER)]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    for i, (key, label, color) in enumerate(arms):
        y = len(arms) - 1 - i
        if key in ab["A"]:
            ax.scatter([ab["A"][key]], [y], s=40, marker="o", facecolor=SURFACE, edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter([ab["B"][key]], [y], s=40, marker="o", color=color, zorder=3)
    ax.scatter([], [], s=40, facecolor=SURFACE, edgecolor=INK2, linewidth=1.8, label="run A (5 March)")
    ax.scatter([], [], s=40, color=INK2, label="run B (12 March, seed 1234)")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a[1] for a in arms][::-1])
    ax.set_xlabel("best validation perplexity, 20 epochs (lower is better)")
    ax.set_title("d_model 128, 2 layers: the arms sit inside run-to-run spread")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(arms) - 0.4)
    return _save(fig, out_dir, "fig3_ablation_d128.png")


def fig4(out_dir):
    fo = claims.forced()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=False)
    for ax, rel, title in ((axes[0], "wikitext_char/phaseA_4h2l/attention_analysis.json", "4x2 model vs forced rules"),
                           (axes[1], "wikitext_char/phaseB_8h4l/attention_analysis.json", "8x4 model, 32 heads")):
        heads = claims.load(rel)["heads"].values()
        for h in heads:
            color = COS if h["kappa"] > 0 else AMP
            ax.scatter(h["kappa"], h["mean_abs_lag"], s=26, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.set_xlabel("kappa")
        ax.set_title(title)
    axes[0].set_ylabel("mean attention distance (characters)")
    for key, color, label in (("spherical", COS, "all cosine, forced"), ("hyperbolic", AMP, "all length-scaled, forced"),
                              ("euclidean", DOT, "all dot product, forced")):
        axes[0].axhline(fo[key + "_lag"], color=color, lw=1.2, ls="--")
    axes[0].text(0, fo["spherical_lag"] + 0.15, "all cosine, forced", color=INK2, fontsize=7.5, ha="center")
    axes[0].text(0, fo["hyperbolic_lag"] + 0.15, "all length-scaled or dot, forced", color=INK2, fontsize=7.5, ha="center")
    axes[0].set_xlim(-3.1, 3.3)
    return _save(fig, out_dir, "fig4_attention_range.png")


def fig5(out_dir):
    sw = claims.sweep()
    rows = sw["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = list(range(len(rows)))
    ax.axhspan(-sw["gap_mean"], sw["gap_mean"], color=GRID, alpha=0.7, lw=0)
    ax.axhline(0, color=DOT, lw=1.0)
    ax.scatter(x, [r["same"] for r in rows], s=26, color=LEARNED, zorder=3, label="curvature, same run as baseline")
    ax.scatter(x, [r["other"] for r in rows], s=26, facecolor=SURFACE, edgecolor=LEARNED, linewidth=1.6, zorder=3,
               label="curvature, separate run")
    ax.set_xticks(x)
    ax.set_xticklabels(["%d/%d" % (r["d"], r["L"]) for r in rows], rotation=60, fontsize=7.5)
    ax.set_xlabel("d_model / layers")
    ax.set_ylabel("perplexity vs baseline (%)")
    ax.set_title("Compression sweep at 55 epochs: mean %+.2f%% and %+.2f%%" % (sw["same_mean"], sw["other_mean"]))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "fig5_sweep.png")


def fig6(out_dir):
    m = claims.load("mandel/control_matrix_20ep_learnable_c_20260313.json")
    v = {(r["task"], r["variant"]): r["summary"]["final_val_ppl"] for r in m["runs"]}
    tasks = ["hierarchy", "periodic", "mixed"]
    variants = [("standard_dot", "dot product", DOT), ("exact_euclidean", "exact Euclidean distance", COS),
                ("exact_mixed", "exact curved geometry (H H E S)", AMP)]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    w = 0.26
    for j, (key, label, color) in enumerate(variants):
        xs = [i + (j - 1) * (w + 0.02) for i in range(len(tasks))]
        ax.bar(xs, [v[(t, key)] for t in tasks], width=w, color=color, label=label, lw=0)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_ylabel("validation perplexity, 20 epochs")
    ax.set_title("Exact distance does the work; the curved version adds nothing here")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "fig6_exact_distance.png")


def fig7(out_dir):
    sy = claims.synthetic_numpy()
    groups = [("tree", sy["tree"], AMP), ("flat\n(control)", sy["flat"], DOT), ("periodic", sy["phase"], COS),
              ("mixed:\ntree half", sy["k1"], AMP), ("mixed:\nperiodic half", sy["k2"], COS)]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for i, (label, vals, color) in enumerate(groups):
        offs = [(j - (len(vals) - 1) / 2) * 0.03 for j in range(len(vals))]
        ax.scatter([i + o for o in offs], vals, s=18, color=color, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.axhline(0, color=DOT, lw=1.0)
    for t in (0.15, -0.15):
        ax.axhline(t, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("learned kappa")
    ax.set_title("The numpy toy over %d seeds" % sy["n"])
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "fig7_toy_seeds.png")


def fig8(out_dir):
    d = claims.load("wikitext_char/phaseB_8h4l/results.json")
    traj = d["curvatures"]
    heads = list(traj[0].keys())
    epochs = list(range(1, len(traj) + 1))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for h in heads:
        vals = [row[h] for row in traj]
        ax.plot([0] + epochs, [0.0] + vals, color=COS if vals[-1] > 0 else AMP, lw=0.9, alpha=0.8)
    ax.axhline(0, color=DOT, lw=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("kappa")
    ax.set_title("8x4 model: 32 heads leave zero within the first epochs")
    ax.set_xlim(0, len(traj))
    return _save(fig, out_dir, "fig8_kappa_trajectories.png")


def fig9(out_dir):
    fl, sx = claims.first_layer(), claims.six_layers()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    groups = [("layer 1", fl["layer1"], fl["one_l1"] + fl["one_l1_head3"]), ("layer 2", fl["layer2"], fl["one_l2"]),
              ("layer 3", fl["layer3"], fl["one_l3"])]
    for i, (label, whole, one) in enumerate(groups):
        ax.scatter([i - 0.12] * len(whole), whole, s=30, color=COS, zorder=3, edgecolor=SURFACE, linewidth=0.8)
        ax.scatter([i + 0.12] * len(one), one, s=30, facecolor=SURFACE, edgecolor=COS, linewidth=1.6, zorder=3)
    ax.scatter([], [], s=30, color=COS, label="whole layer cosine")
    ax.scatter([], [], s=30, facecolor=SURFACE, edgecolor=COS, linewidth=1.6, label="one cosine head")
    ax.axhline(100, color=LEARNED, lw=1.2, ls="--")
    ax.axhline(0, color=DOT, lw=1.0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("share of the learned gain kept (%)")
    ax.set_title("3 layers: where the cosine head sits")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    arms = [("learned\ncurvature", sx["learned"], LEARNED, False), ("one head,\nlayer 1", sx["one_l1"], COS, False),
            ("whole\nlayer 1", sx["layer1"], COS, True), ("one head,\nlayer 4", sx["one_l4"], AMP, False)]
    for i, (label, vals, color, filled) in enumerate(arms):
        offs = [(j - (len(vals) - 1) / 2) * 0.09 for j in range(len(vals))]
        ax.scatter([i + o for o in offs], vals, s=30, color=color, zorder=3, edgecolor=SURFACE, linewidth=0.8)
    ax.axhline(0, color=DOT, lw=1.0)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=8)
    ax.set_ylabel("improvement over no curvature (%)")
    ax.set_title("6 layers: fixed beats learned")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, "fig9_first_layer.png")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = [f(out_dir) for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9)]
    with open(os.path.join(out_dir, "CAPTIONS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Figures\n\nRebuilt from `results/` by `python reproduce.py figures`.\n\n")
        for n in names:
            f.write("**%s**: %s\n\n" % (n, CAPTIONS[n]))
    return names


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))
