"""Every number in CLAIMS.md, computed from the saved result files under results/.

Each claim is a hypothesis with a verdict in the lab's vocabulary:
    POSITIVE          the effect is there and survived the named control
    ANTI              the control removed it: the evidence points the other way
    INCONCLUSIVE      too few seeds, or the gap is inside run-to-run spread
    DEFINITION        a fact about the code, not an experiment

`python reproduce.py claims` writes CLAIMS.md from this file; `python reproduce.py check` rebuilds
it in memory and fails if the committed CLAIMS.md differs by a single character.
"""
import glob
import json
import math
import os
import statistics as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, "results")


def load(rel):
    with open(os.path.join(R, rel), encoding="utf-8") as f:
        return json.load(f)


def pct(a, b):
    """Relative change of a against b, in percent."""
    return 100.0 * (a / b - 1.0)


def kept(zero, cont, arm):
    """Share of the learned gain an arm keeps: (zero - arm) / (zero - continuous), in percent."""
    return 100.0 * (zero - arm) / (zero - cont)


def ppls(rows):
    return [r["ppl"] for r in rows if r.get("ppl") is not None]


def f2(x):
    return "%+.2f%%" % x


# ---------------------------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------------------------

def mechanism():
    t0 = load("ternary_lane/RESULT_T0_20260920.json")
    t0b = load("ternary_lane/RESULT_T0B_20260920.json")["results"]
    t2 = load("ternary_lane/RESULT_T2_20260921.json")["rows"]
    mean_abs = [t2[s]["mean_abs_kappa"] for s in sorted(t2)]
    return {
        "dominant_weight_at_1": math.tanh(3.0),
        "phaseB_mean_abs": t0["kappa"]["mean_abs"],
        "d96_mean_abs_lo": min(mean_abs),
        "d96_mean_abs_hi": max(mean_abs),
        "blend": t0b["blended_learned"]["ppl"],
        "hard": t0b["hard_learned"]["ppl"],
    }


def t1b():
    d = load("ternary_lane/RESULT_T1B_20260921.json")
    p = d["ppl"]
    seeds = range(4)
    kept_binary = [kept(p["zero_seed%d" % s], p["continuous_seed%d" % s], p["binary_seed%d" % s]) for s in seeds]
    return {
        "diffs": d["continuous_vs_zero"]["diffs"],
        "mean": d["continuous_vs_zero"]["mean"],
        "sd": d["continuous_vs_zero"]["sd"],
        "binary_vs_zero": d["binary_vs_zero"]["mean"],
        "binary_kept": st.mean(kept_binary),
        "binary_kept_all": kept_binary,
    }


def t2b():
    d = load("ternary_lane/RESULT_T2B_20260921.json")
    rows = d["rows"]
    return {
        "randsign": d["randsign_kept"]["per_seed"],
        "randsign_mean": d["randsign_kept"]["mean"],
        "oracle": d["oracle_kept"]["per_seed"],
        "xseed": d["xseed_kept"]["per_seed"],
        "late1": d["late1_kept"]["per_seed"],
        "late1_mean": d["late1_kept"]["mean"],
        "settled": [rows[s]["code_settled_by_step"] for s in sorted(rows)],
    }


def t2c():
    rows = load("ternary_lane/RESULT_T2C_20260922.json")["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r["kept"])
    mixed = [r["kept"] for r in rows if r["arm"] not in ("allsph", "allhyp")]
    return {"allsph": by["allsph"], "allhyp": by["allhyp"], "mixed": mixed}


def fixed_codes_all():
    """Every run in which a mixed +/- code was frozen from the first steps (T2b and T2c)."""
    b, c = t2b(), t2c()
    runs = b["randsign"] + b["oracle"] + b["xseed"] + c["mixed"]
    return {"n": len(runs), "below60": sorted(x for x in runs if x < 60), "ge90": sum(x >= 90 for x in runs)}


def t2():
    d = load("ternary_lane/RESULT_T2_20260921.json")
    rows = d["rows"]
    late = [kept(rows[s]["zero_t1b"], rows[s]["cont_t2"], rows[s]["late_binary"]) for s in sorted(rows)]
    return {"posthoc": d["posthoc_sign1_vs_cont_t2"]["diffs"], "late_kept": late}


def first_layer():
    """T2d and T2e (3 layers): the same fixed cosine code placed in each layer, and one cosine head per layer."""
    d = load("ternary_lane/RESULT_T2D_20260923.json")["rows"]
    e = load("ternary_lane/RESULT_T2E_20260924.json")["rows"]
    rows = d + e
    pick = lambda arm: [r["kept"] for r in sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"])]
    return {
        "layer1": pick("layer1"), "layer2": pick("layer2"), "layer3": pick("layer3"),
        "one_l1": pick("one_l1"), "one_l1_head3": pick("k1"), "one_l2": pick("one_l2"), "one_l3": pick("one_l3"),
    }


def six_layers():
    """T2f and T2g (6 layers): improvement over the same seed's zero-curvature model, in percent."""
    f = load("ternary_lane/RESULT_T2F_20260924.json")
    g = load("ternary_lane/RESULT_T2G_20260925.json")["rows"]
    ref = f["references"]
    learned = {s: 100.0 * (ref["zero_seed%d" % s] - ref["continuous_seed%d" % s]) / ref["zero_seed%d" % s] for s in (0, 1)}
    out = {"learned": dict(learned), "one_l1": {}, "layer1": {}, "one_l4": {}, "lrk3": {}}
    for r in f["rows"]:
        out[r["arm"]][r["seed"]] = r["improvement_vs_zero_pct"]
    for r in g:
        key = {"continuous": "learned", "one_l1": "one_l1", "layer1": "layer1", "continuous_lrk3": "lrk3"}.get(r["arm"])
        if key:
            out[key][r["seed"]] = r["improvement_vs_zero_pct"]
        elif r["arm"] == "one_l1_40k":
            out["one_l1_40k"] = r["improvement_vs_zero_pct"]
        elif r["arm"] == "continuous_40k":
            out["learned_40k"] = r["improvement_vs_zero_pct"]
    for k in ("learned", "one_l1", "layer1", "one_l4", "lrk3"):
        out[k] = [out[k][seed] for seed in sorted(out[k])]
    return out


def claims_t2_rows():
    rows = load("ternary_lane/RESULT_T2_20260921.json")["rows"]
    return [rows[k] for k in sorted(rows)]


def t0():
    d = load("ternary_lane/RESULT_T0_20260920.json")
    rel = d["relative_to_learned_pct"]
    shuffled = [rel["shuffled_seed%d" % i] for i in range(10)]
    return {
        "learned": d["results"]["learned"]["ppl"],
        "sign": d["results"]["sign_1.0"]["ppl"],
        "sign_rel": rel["sign_1.0"],
        "shuffle_mean": pct(d["results"]["shuffled_summary"]["mean_ppl"], d["results"]["learned"]["ppl"]),
        "shuffle_min": min(shuffled),
        "shuffle_worse": sum(x > 0 for x in shuffled),
        "zero_rel": rel["zero"],
        "all_sph_rel": rel["all_sph"],
        "neg": d["kappa"]["neg"],
        "pos": d["kappa"]["pos"],
    }


def t1():
    s = load("ternary_lane/RESULT_T1_20260921.json")["summary"]
    diffs = [pct(c, z) for c, z in zip(s["continuous"]["ppls"], s["zero"]["ppls"])]
    return {"cont": s["continuous"]["mean_ppl"], "zero": s["zero"]["mean_ppl"], "randsign": s["randsign"]["mean_ppl"],
            "diffs": diffs, "mean": st.mean(diffs), "sd": st.stdev(diffs)}


def t3():
    rows = load("ternary_lane/RESULT_T3_20260921.json")["rows"]
    diffs = [pct(rows[s]["cont_t"], rows[s]["zero_t"]) for s in sorted(rows)]
    return {"diffs": diffs, "mean": st.mean(diffs), "sd": st.stdev(diffs), "neg": sum(x < 0 for x in diffs)}


def ablations():
    a = load("wikitext_char/ablation_3arm_20260305/results.json")
    b = load("wikitext_char/ablation_4arm_random_20260312/results.json")
    A = {k: min(ppls(a[k + "_kappa"])) for k in ("learned", "fixed", "zero")}
    B = {k: min(ppls(b[k + "_kappa"])) for k in ("learned", "fixed", "zero", "random")}
    return {"A": A, "B": B}


def forced():
    d = load("wikitext_char/forced_geometry_20260312/results.json")
    out = {k: min(ppls(d[k + "_forced"])) for k in ("spherical", "hyperbolic", "euclidean")}
    for k in ("spherical", "hyperbolic", "euclidean"):
        s = d["attention_analysis"][k + "_forced"]["summary"]
        out[k + "_lag"] = s["mean_mean_abs_lag"]
        out[k + "_long"] = s["mean_long16_mass"]
    return out


def attention(rel):
    d = load(rel)
    heads = d["heads"].values()
    cos = [h for h in heads if h["kappa"] > 0]
    amp = [h for h in heads if h["kappa"] < 0]
    return {
        "r_long": d["correlations"]["kappa_vs_long16_mass"],
        "r_lag": d["correlations"]["kappa_vs_mean_abs_lag"],
        "r_local": d["correlations"]["kappa_vs_local4_mass"],
        "n": len(d["heads"]),
        "cos_lag": st.mean(h["mean_abs_lag"] for h in cos),
        "amp_lag": st.mean(h["mean_abs_lag"] for h in amp),
    }


def sweep():
    """Compression sweep: each 'std' run trained a curvature model and a baseline side by side for 55
    epochs; each 'kappa' run trained a second curvature model on its own. d128 L1 (std) crashed and
    d128 L3 stopped at 5 epochs, so 18 of the 20 configurations have a matched pair."""
    rows = []
    for path in sorted(glob.glob(os.path.join(R, "wikitext_char", "sweep", "sweep_std_*.json"))):
        name = os.path.basename(path)[len("sweep_std_"):-5]
        std = load(os.path.relpath(path, R))
        kap = load("wikitext_char/sweep/sweep_kappa_%s.json" % name)
        base, curv, curv2 = ppls(std["baseline"]), ppls(std["curvature_model"]), ppls(kap["curvature_model"])
        if len(base) < 55 or len(curv) < 55 or len(curv2) < 55:
            continue
        d, L = name.split("_")
        rows.append({"d": int(d[1:]), "L": int(L[1:]), "same": pct(curv[54], base[54]), "other": pct(curv2[54], base[54]),
                     "curv_gap": abs(pct(curv2[54], curv[54])), "base10": base[9], "base55": base[54]})
    rows.sort(key=lambda r: (r["d"], r["L"]))
    same = [r["same"] for r in rows]
    other = [r["other"] for r in rows]
    d96l3 = next(r for r in rows if r["d"] == 96 and r["L"] == 3)
    summary = load("wikitext_char/sweep/summary_20260310.json")
    return {
        "rows": rows, "n": len(rows),
        "same_mean": st.mean(same), "same_wins": sum(x < 0 for x in same),
        "other_mean": st.mean(other), "other_wins": sum(x < 0 for x in other),
        "gap_mean": st.mean(r["curv_gap"] for r in rows), "gap_max": max(r["curv_gap"] for r in rows),
        "d96l3_same": d96l3["same"], "d96l3_other": d96l3["other"],
        "summary_json": summary,
    }


def mandel():
    m = load("mandel/control_matrix_20ep_learnable_c_20260313.json")
    v = {(r["task"], r["variant"]): r["summary"]["final_val_ppl"] for r in m["runs"]}
    out = {}
    for task in ("hierarchy", "periodic", "mixed"):
        out[task + "_euc_vs_dot"] = pct(v[(task, "exact_euclidean")], v[(task, "standard_dot")])
        out[task + "_mixed_vs_euc"] = pct(v[(task, "exact_mixed")], v[(task, "exact_euclidean")])
    dot = load("mandel/wikitext_standard_dot.json")["summary"]["best_val_ppl"]
    euc = load("mandel/wikitext_exact_euclidean.json")["summary"]["best_val_ppl"]
    tau = load("mandel/wikitext_temperature_tau.json")["summary"]["best_val_ppl"]
    out.update({"wt_dot": dot, "wt_euc": euc, "wt_tau": tau, "wt_euc_vs_dot": pct(euc, dot), "wt_tau_vs_dot": pct(tau, dot)})
    return out


def synthetic_numpy():
    files = sorted(glob.glob(os.path.join(R, "synthetic_numpy", "seeds", "seed*.json")),
                   key=lambda p: int(os.path.basename(p)[4:-5]))
    runs = [json.load(open(p, encoding="utf-8")) for p in files]
    tree = [r["tree"]["kappa"] for r in runs]
    phase = [r["phase"]["kappa"] for r in runs]
    flat = [r["flat"]["kappa"] for r in runs]
    k1 = [r["mixed"]["k1"] for r in runs]
    k2 = [r["mixed"]["k2"] for r in runs]
    orig = load("synthetic_numpy/autograd_geometry_results_seed42.json")
    seed42 = json.load(open(os.path.join(R, "synthetic_numpy", "seeds", "seed42.json"), encoding="utf-8"))
    return {
        "n": len(runs), "tree": tree, "phase": phase, "flat": flat, "k1": k1, "k2": k2,
        "tree_below_flat": sum(t < f for t, f in zip(tree, flat)),
        "phase_pos_flat_neg": sum(p > 0 and f < 0 for p, f in zip(phase, flat)),
        "phase_over_015": sum(p > 0.15 for p in phase),
        "k2_minus_k1": [b - a for a, b in zip(k1, k2)],
        "orig_phase": orig["phase"]["kappa"],
        "orig_phase_label": {"ℍ": "hyperbolic", "S": "spherical", "ℝ": "Euclidean"}[orig["phase"]["regime"]],
        "seed42_identical": seed42 == orig,
    }


def synthetic_lm():
    d = load("synthetic_lm/curvature_transformer_results.json")
    c, b = ppls(d["curvature_model"]), ppls(d["baseline"])
    reach = next(i + 1 for i, x in enumerate(c) if x <= b[-1])
    return {"curv": c[-1], "base": b[-1], "rel": pct(c[-1], b[-1]), "reach": reach, "epochs": len(b)}


def pilot():
    d = load("wikitext_char/pilot_4h2l/results.json")
    c, b = ppls(d["curvature_model"]), ppls(d["baseline"])
    return {"curv": c[-1], "base": b[-1], "rel": pct(c[-1], b[-1])}


def trajectories():
    t = load("wikitext_char/kappa_timing_20260312.json")["runs"]
    a = load("wikitext_char/phaseA_4h2l/results.json")["curvatures"][-1]
    b = load("wikitext_char/phaseB_8h4l/results.json")["curvatures"][-1]
    return {
        "A_neg": sum(v < 0 for v in a.values()), "A_pos": sum(v > 0 for v in a.values()),
        "A_range": max(a.values()) - min(a.values()),
        "B_neg": sum(v < 0 for v in b.values()), "B_pos": sum(v > 0 for v in b.values()),
        "B_range": max(b.values()) - min(b.values()),
        "epoch80": {r["label"]: r["epoch_80pct_final_range"] for r in t},
    }


def clamp():
    c = load("wikitext_char/clamp_20260305/results.json")
    a = load("wikitext_char/phaseA_4h2l/results.json")
    return {"clamp20": ppls(c["curvature_model"])[-1], "phaseA20": ppls(a["curvature_model"])[19]}


# ---------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------

def rng(xs, fmt="%.1f"):
    return (fmt + " to " + fmt) % (min(xs), max(xs))


def build():
    m, b1, b2b, c2c, fc, b2, z, one, three = mechanism(), t1b(), t2b(), t2c(), fixed_codes_all(), t2(), t0(), t1(), t3()
    ab, fo, sw, ma, sy, sl, pi, tr, cl = ablations(), forced(), sweep(), mandel(), synthetic_numpy(), synthetic_lm(), pilot(), trajectories(), clamp()
    atB = attention("wikitext_char/phaseB_8h4l/attention_analysis.json")
    atA = attention("wikitext_char/phaseA_4h2l/attention_analysis.json")

    kept_rows = [
        # (id, hypothesis, number, source, control, verdict)
        ("M1", "The parameter called curvature (kappa) is a per-head switch between three attention scoring rules: "
               "plain dot product (kappa near 0), dot product scaled up by the vectors' lengths (kappa < 0, called "
               "'hyperbolic') and cosine similarity times sqrt(d_k) (kappa > 0, called 'spherical'). No curved space "
               "is computed.",
         "At |kappa| = 1 the winning rule already has weight %.4f. Trained heads sit far past that: mean |kappa| %.2f "
         "(8x4 model), %.1f to %.1f (d96 models). Swapping the blend for a hard switch changes perplexity from %.9f "
         "to %.9f." % (m["dominant_weight_at_1"], m["phaseB_mean_abs"], m["d96_mean_abs_lo"], m["d96_mean_abs_hi"],
                       m["blend"], m["hard"]),
         "src/learnable_curvature_transformer.py (CurvatureDistance); ternary_lane/RESULT_T0B_20260920.json, RESULT_T2_20260921.json",
         "Hard-switch replacement on a trained model",
         "DEFINITION"),
        ("C1", "At d_model 96, 3 layers, a per-head scoring rule beats plain dot-product attention.",
         "Learned kappa vs kappa fixed at 0: %s mean (sd %.2f), per seed %s; 4 of 4 seeds better."
         % (f2(b1["mean"]), b1["sd"], ", ".join("%+.2f" % x for x in b1["diffs"])),
         "ternary_lane/RESULT_T1B_20260921.json",
         "Paired seeds: same initial weights per seed, only kappa differs",
         "POSITIVE (one scale)"),
        ("C2", "The gain comes from cosine scoring being present: fixing every head to the cosine rule from step 1, "
               "with nothing learned, keeps it; fixing every head to the length-scaled rule does not.",
         "Share of the learned gain kept: all cosine %.1f%% (seeds: %s); all length-scaled %.1f%% (seeds: %s)."
         % (st.mean(c2c["allsph"]), ", ".join("%.1f" % x for x in c2c["allsph"]),
            st.mean(c2c["allhyp"]), ", ".join("%.1f" % x for x in c2c["allhyp"])),
         "ternary_lane/RESULT_T2C_20260922.json",
         "Mirror arm (all length-scaled) and the zero arm, same seeds",
         "POSITIVE (one scale, 2 seeds)"),
        ("C3", "Learning which heads get which rule matters (a learned assignment beats a random one).",
         "Random frozen +/-1 code keeps %.1f%% of the learned gain (seeds %s). At d_model 128, 2 layers: random %.4f, "
         "learned %.4f, fixed-discovered %.4f (one seed). Frozen mixed codes overall: %d of %d runs keep >= 90%%; the "
         "two below 60%% (%s) share one code, which kept 98%% on a third seed."
         % (b2b["randsign_mean"], rng(b2b["randsign"]), ab["B"]["random"], ab["B"]["learned"], ab["B"]["fixed"],
            fc["ge90"], fc["n"], ", ".join("%.1f%%" % x for x in fc["below60"])),
         "ternary_lane/RESULT_T2B_20260921.json, RESULT_T2C_20260922.json; wikitext_char/ablation_4arm_random_20260312/results.json",
         "Random fixed code, same seeds (this is the control)",
         "ANTI"),
        ("C4", "In a trained model, the sign of kappa carries all of it: rounding every kappa to +/-1 costs nothing, "
               "and the signs are tied to the weights around them.",
         "8x4 model: %.4f -> %.4f (%+.4f%%). d96, 4 seeds: |change| <= %.4f%%. Shuffling the same signs across heads: "
         "%+.2f%% mean over 10 shuffles, every one worse (smallest %+.2f%%)."
         % (z["learned"], z["sign"], z["sign_rel"], max(abs(x) for x in b2["posthoc"]), z["shuffle_mean"], z["shuffle_min"]),
         "ternary_lane/RESULT_T0_20260920.json, RESULT_T2_20260921.json",
         "Sign permutation across heads (10 shuffles)",
         "POSITIVE (compression of a trained model)"),
        ("C5", "Committing to one bit late in training keeps the gain; training in one bit from the first step does not.",
         "Continuous for 75%% of training, then sign: keeps %s%% (4 seeds). Code frozen after 1%% of training: %.1f%% mean. "
         "One bit from step 1 (straight-through): keeps %.0f%% (%s)."
         % (rng(b2["late_kept"]), b2b["late1_mean"], b1["binary_kept"], ", ".join("%.0f%%" % x for x in b1["binary_kept_all"])),
         "ternary_lane/RESULT_T2_20260921.json, RESULT_T2B_20260921.json, RESULT_T1B_20260921.json",
         "Same seeds and zero arm as C1",
         "POSITIVE (why step-1 binary fails is untested)"),
        ("C6", "Heads with the cosine rule spread attention over longer distances.",
         "8x4 model (%d heads): r(kappa, long-range mass) = %+.3f, r(kappa, mean lag) = %+.3f, r(kappa, local mass) = %+.3f. "
         "4x2 model (%d heads): %+.3f, %+.3f, %+.3f."
         % (atB["n"], atB["r_long"], atB["r_lag"], atB["r_local"], atA["n"], atA["r_long"], atA["r_lag"], atA["r_local"]),
         "wikitext_char/phaseB_8h4l/attention_analysis.json, phaseA_4h2l/attention_analysis.json",
         "Forced-rule models (C7)",
         "POSITIVE (attention-range proxy)"),
        ("C7", "The kappa-behaviour link in C6 is something the model discovers, beyond what the rule does by itself.",
         "Forcing one rule on every head, nothing learned (4x2, 20 epochs): mean lag %.2f cosine, %.2f length-scaled, "
         "%.2f dot. In the trained 4x2 model: %.2f cosine heads, %.2f length-scaled heads. The forced rule gives %.0f%% "
         "of the trained gap."
         % (fo["spherical_lag"], fo["hyperbolic_lag"], fo["euclidean_lag"], atA["cos_lag"], atA["amp_lag"],
            100 * (fo["spherical_lag"] - fo["hyperbolic_lag"]) / (atA["cos_lag"] - atA["amp_lag"])),
         "wikitext_char/forced_geometry_20260312/results.json, phaseA_4h2l/attention_analysis.json",
         "Forced-rule models, one seed",
         "INCONCLUSIVE (most of it is the rule)"),
        ("C8", "Heads leave kappa = 0 early and split between both rules.",
         "4x2: %d length-scaled / %d cosine, range %.3f; 8x4: %d / %d, range %.3f. 80%% of the final range by epoch %d "
         "(4x2) and %d (8x4). d96 codes final by step %d on all 4 seeds."
         % (tr["A_neg"], tr["A_pos"], tr["A_range"], tr["B_neg"], tr["B_pos"], tr["B_range"],
            tr["epoch80"]["resume_long55"], tr["epoch80"]["run"], max(b2b["settled"])),
         "wikitext_char/phaseA_4h2l/results.json, phaseB_8h4l/results.json, kappa_timing_20260312.json; ternary_lane/RESULT_T2B_20260921.json",
         "None needed for the description; C3 shows the split is not what helps",
         "POSITIVE (descriptive)"),
        ("C9", "At small scale (d_model 16 to 96, 1 to 4 layers, 55 epochs, one baseline per config) learnable kappa lowers perplexity.",
         "%d matched configs. Same-run curvature vs baseline: %s mean, better in %d of %d. Separately trained curvature vs "
         "the same baseline: %s mean, better in %d of %d. Two curvature runs of one config differ by %.2f%% on average "
         "(largest %.2f%%)."
         % (sw["n"], f2(sw["same_mean"]), sw["same_wins"], sw["n"], f2(sw["other_mean"]), sw["other_wins"], sw["n"],
            sw["gap_mean"], sw["gap_max"]),
         "wikitext_char/sweep/sweep_std_*.json, sweep_kappa_*.json",
         "A second curvature run per config (run-to-run spread)",
         "INCONCLUSIVE (gain smaller than spread)"),
        ("C10", "At d_model 48, 2 layers, geometry helps.",
         "Learned %.3f vs zero %.3f vs random code %.3f (3 seeds, 6,000 steps). Paired differences %s, mean %s, sd %.2f."
         % (one["cont"], one["zero"], one["randsign"], ", ".join("%+.2f" % x for x in one["diffs"]), f2(one["mean"]), one["sd"]),
         "ternary_lane/RESULT_T1_20260921.json",
         "Paired seeds",
         "INCONCLUSIVE (no gain seen; 3 seeds cannot rule out 2%)"),
        ("C11", "Replacing the dot product with an exact distance helps, and adding curved geometry on top adds nothing "
                "(separate exact-distance model, synthetic tasks).",
         "Exact Euclidean vs dot product: periodic %s, mixed %s, hierarchy %s. Mixed curved geometry vs exact Euclidean: "
         "%s, %s, %s (20 epochs, one seed)."
         % (f2(ma["periodic_euc_vs_dot"]), f2(ma["mixed_euc_vs_dot"]), f2(ma["hierarchy_euc_vs_dot"]),
            f2(ma["periodic_mixed_vs_euc"]), f2(ma["mixed_mixed_vs_euc"]), f2(ma["hierarchy_mixed_vs_euc"])),
         "mandel/control_matrix_20ep_learnable_c_20260313.json",
         "Exact-Euclidean arm on the same code path",
         "POSITIVE (synthetic, one seed)"),
        ("C12", "Exact-distance scoring also beats the dot product on WikiText-2 (no curved arm was run here).",
         "Exact Euclidean %.4f vs dot product %.4f (%s); a learned per-head temperature on the dot product %.4f (%s). "
         "55 epochs, one seed."
         % (ma["wt_euc"], ma["wt_dot"], f2(ma["wt_euc_vs_dot"]), ma["wt_tau"], f2(ma["wt_tau_vs_dot"])),
         "mandel/wikitext_*.json",
         "Per-head temperature arm (one extra scalar per head, no distance)",
         "INCONCLUSIVE (one seed)"),
        ("C13", "True curvature (Poincare-ball and sphere distances, one kappa, contrastive loss, 6 dimensions) moves "
                "negative on tree-structured data.",
         "Tree: kappa %s over %d seeds. Flat-cluster control: %s. Tree below the control in %d of %d seeds."
         % (rng(sy["tree"], "%+.3f"), sy["n"], rng(sy["flat"], "%+.3f"), sy["tree_below_flat"], sy["n"]),
         "synthetic_numpy/seeds/seed*.json",
         "Flat-cluster task, same seeds",
         "POSITIVE (relative to the control only)"),
        ("C14", "The same toy moves positive on periodic data.",
         "Periodic: kappa %s; positive while the control is negative in %d of %d seeds. Above the script's own 0.15 "
         "threshold for 'spherical' in %d of %d."
         % (rng(sy["phase"], "%+.3f"), sy["phase_pos_flat_neg"], sy["n"], sy["phase_over_015"], sy["n"]),
         "synthetic_numpy/seeds/seed*.json",
         "Flat-cluster task, same seeds",
         "POSITIVE (direction only; small)"),
        ("C15", "On mixed data the toy gives each half of the vector its own curvature.",
         "Periodic half kappa %s; tree half %s (it does not go negative). Difference %s in every seed."
         % (rng(sy["k2"], "%+.3f"), rng(sy["k1"], "%+.3f"), rng(sy["k2_minus_k1"], "%+.3f")),
         "synthetic_numpy/seeds/seed*.json",
         "Seed replication",
         "POSITIVE (different, not hyperbolic x spherical)"),
        ("C16", "On the synthetic sequence task the transformer with kappa trains faster.",
         "Reaches the baseline's final perplexity (%.4f, epoch %d) at epoch %d; final %.4f vs %.4f (%s). One seed."
         % (sl["base"], sl["epochs"], sl["reach"], sl["curv"], sl["base"], f2(sl["rel"])),
         "synthetic_lm/curvature_transformer_results.json",
         "None (no random-code or cosine-only arm, one seed)",
         "INCONCLUSIVE"),
        ("C17", "With ternary (BitNet b1.58) block weights, the geometry keeps its gain.",
         "Learned vs zero with ternary weights: %s mean, sd %.2f, better on %d of 4 seeds (float weights: %s)."
         % (f2(three["mean"]), three["sd"], three["neg"], f2(b1["mean"])),
         "ternary_lane/RESULT_T3_20260921.json",
         "Paired seeds, same zero arm design as C1",
         "INCONCLUSIVE (about half, unreliably)"),
    ]

    t2rows = claims_t2_rows()
    kept_rows.append(
        ("R1", "Retraining a d_model 96 run on CPU reproduces it exactly.",
         "The lab retrained the continuous arm in a separate run for T2: %d of %d seeds match T1b's perplexity to every "
         "stored digit (e.g. %.15f)." % (sum(r["cont_t2"] == r["cont_t1b"] for r in t2rows), len(t2rows), t2rows[0]["cont_t2"]),
         "ternary_lane/RESULT_T2_20260921.json",
         "Independent retrain, same seed and code",
         "POSITIVE"))

    fl, sx = first_layer(), six_layers()
    kept_rows.extend([
        ("C18", "At 3 layers the gain lives in the first layer: fixed cosine scoring there keeps it, and the same "
                "scoring in a later layer does not.",
         "Share of the learned gain kept, one value per seed. Whole layer cosine: layer 1 %s, layer 2 %s, layer 3 %s. "
         "One cosine head: in layer 1 %s (head 0) and %s (head 3); in layer 2 %s; in layer 3 %s."
         % (", ".join("%.0f%%" % x for x in fl["layer1"]), ", ".join("%.0f%%" % x for x in fl["layer2"]),
            ", ".join("%.0f%%" % x for x in fl["layer3"]), ", ".join("%.0f%%" % x for x in fl["one_l1"]),
            ", ".join("%.0f%%" % x for x in fl["one_l1_head3"]), ", ".join("%.0f%%" % x for x in fl["one_l2"]),
            ", ".join("%.0f%%" % x for x in fl["one_l3"])),
         "ternary_lane/RESULT_T2D_20260923.json, RESULT_T2E_20260924.json",
         "The same code moved to layers 2 and 3, same seeds",
         "POSITIVE (3 layers, 2 seeds, 2 different heads)"),
        ("C19", "At 6 layers, learned curvature improves on no curvature.",
         "Improvement over the same seed's zero-curvature model: %s (4 seeds, mean %s). With a 3x larger curvature "
         "step: %s. At 40,000 steps: %s (one seed)."
         % (", ".join("%+.2f%%" % x for x in sx["learned"]), f2(st.mean(sx["learned"])),
            ", ".join("%+.2f%%" % x for x in sx["lrk3"]), f2(sx["learned_40k"])),
         "ternary_lane/RESULT_T2F_20260924.json, RESULT_T2G_20260925.json",
         "Paired zero-curvature model per seed",
         "INCONCLUSIVE (no gain seen at 20,000 steps)"),
        ("C20", "At 6 layers, one fixed cosine head in the first layer improves on no curvature, where learned "
                "curvature does not.",
         "One fixed layer-1 head: %s (better than zero on %d of %d seeds, mean %s), better than learned curvature "
         "on %d of %d seeds. Whole layer 1 cosine: %s. The same head in layer 4: %s (worse than zero on %d of %d). "
         "At 40,000 steps: %s against learned %s (one seed)."
         % (", ".join("%+.2f%%" % x for x in sx["one_l1"]), sum(x > 0 for x in sx["one_l1"]), len(sx["one_l1"]),
            f2(st.mean(sx["one_l1"])), sum(a > b for a, b in zip(sx["one_l1"], sx["learned"])), len(sx["one_l1"]),
            ", ".join("%+.2f%%" % x for x in sx["layer1"]), ", ".join("%+.2f%%" % x for x in sx["one_l4"]),
            sum(x < 0 for x in sx["one_l4"]), len(sx["one_l4"]), f2(sx["one_l1_40k"]), f2(sx["learned_40k"])),
         "ternary_lane/RESULT_T2F_20260924.json, RESULT_T2G_20260925.json",
         "Zero-curvature and learned-curvature models on the same seeds; the head moved to layer 4",
         "POSITIVE (6 layers, 4 seeds, preregistered)"),
    ])

    withdrawn = [
        ("W1", "'Learnable curvature': the model finds its own geometry, and that is the source of the gain.",
         "C3 and C2: a random frozen code keeps %.0f%%, all-cosine keeps %.0f%%." % (b2b["randsign_mean"], st.mean(c2c["allsph"]))),
        ("W2", "'-7% at d_model 96, 3 layers' and 'up to 38% parameter savings' (March sweep summary).",
         "The summary compared a standard model stopped at epoch 10 with a curvature model at epoch 55. Matched at 55 "
         "epochs: %s (same run), %s (separate run)." % (f2(sw["d96l3_same"]), f2(sw["d96l3_other"]))),
        ("W3", "Curvature is behaviourally meaningful: heads discover their function (r = 0.88).",
         "C7: forcing the rule with nothing learned reproduces most of the pattern."),
        ("W4", "The toy 'discovers spherical geometry' for periodic data (kappa = +0.15).",
         "The script's own threshold labels the original run '%s' (kappa %+.3f); %d of %d seeds cross it (C14)."
         % (sy["orig_phase_label"], sy["orig_phase"], sy["phase_over_015"], sy["n"])),
        ("W5", "The toy 'discovers product geometry'.",
         "The tree half stays at %s, not negative (C15)." % rng(sy["k1"], "%+.3f")),
        ("W6", "Zeroing kappa costs %.0f%%, so the geometry is worth that much." % z["zero_rel"],
         "That is removing the rule a trained model's weights were built around. Retrained with kappa = 0 the cost is "
         "%s (C1)." % f2(-b1["mean"])),
        ("W7", "Fixed discovered kappa beats learned kappa (4x2, 20 epochs).",
         "Run A: fixed %.4f vs learned %.4f; run B reverses it: %.4f vs %.4f."
         % (ab["A"]["fixed"], ab["A"]["learned"], ab["B"]["fixed"], ab["B"]["learned"])),
        ("W8", "Geometry beats none at d_model 128, 2 layers, 20 epochs.",
         "Run A: learned %.4f vs zero %.4f (%s); run B: %.4f vs %.4f (%s). The two runs disagree."
         % (ab["A"]["learned"], ab["A"]["zero"], f2(pct(ab["A"]["learned"], ab["A"]["zero"])),
            ab["B"]["learned"], ab["B"]["zero"], f2(pct(ab["B"]["learned"], ab["B"]["zero"])))),
        ("W9", "Bounded kappa in [-2, 2] is slightly better.",
         "%.4f vs %.4f at 20 epochs (%s), one run each; smaller than the gap between the two zero arms in W8 (%s)."
         % (cl["clamp20"], cl["phaseA20"], f2(pct(cl["clamp20"], cl["phaseA20"])), f2(pct(ab["B"]["zero"], ab["A"]["zero"])))),
        ("W10", "WikiText-2 pilot: curvature beats baseline by 1.15%.",
         "%.4f vs %.4f (%s) after 5 epochs, one seed; inside the run-to-run spread of C9." % (pi["curv"], pi["base"], f2(pi["rel"]))),
        ("W11", "Only the sign of kappa matters at inference (the lab's working line).",
         "True of a trained model (C4). Trained from scratch, which heads get which sign does not matter either (C3); "
         "what matters is cosine scoring in the first layer (C2, C18, C20)."),
    ]

    extra = {
        "forced": fo, "sweep_rows": sw["rows"], "seed42_identical": sy["seed42_identical"], "t0": z,
    }
    return kept_rows, withdrawn, extra


LIMITATIONS = [
    "Every language result is character-level WikiText-2 (256 byte values, 64-character windows). Nothing here is "
    "tested on subword tokens, another corpus, or any task beyond next-character prediction.",
    "The largest trained model (8 heads x 4 layers, d_model 128) has under 0.9M parameters. Clear gains appear only at d_model 96 (C1, C2 at 3 "
    "layers; C18, C20 at 3 and 6 layers). At d_model 48 no gain was seen (C10) and the sweep's gains are inside "
    "run-to-run spread (C9). Nothing here says how this behaves in larger models.",
    "Seeds are few: 4 for C1, 2 per uniform arm for C2, 1 for most March runs. The March runs (C6-C9, C16, W7-W10) "
    "mostly have a single seed and no confidence interval.",
    "'Spherical' and 'hyperbolic' name scoring rules, not geometries (M1). The spherical rule is cosine-similarity "
    "attention at a fixed scale, close to query-key normalisation, which is published prior work; the related-work "
    "section has to say so.",
    "The attention-range analysis (C6, C7) is a proxy: attention mass by distance on character sequences. It is not a "
    "syntactic probe and says nothing about linguistic function.",
    "No speed, memory or deployment claim. The sign-only model (C4) still computes every branch; nothing was timed.",
    "The d_model 128 L1 standard run crashed and d_model 128 L3 stopped at 5 epochs, so the sweep has 18 of 20 "
    "configurations (results/wikitext_char/sweep/crash_log_std_d128_L1.txt).",
    "The exact-distance runs (C11, C12) use a separate implementation with one seed each; they point the same way as "
    "C2 but were not run under the d96 protocol.",
    "The March language runs loaded WikiText-2 through Hugging Face with the model file's own split; the d96 runs use "
    "a 90/10 split of the concatenated file with 2,500 validation sequences. Perplexities from the two families are "
    "not comparable with each other.",
    "The d96 experiments (C1-C5, C10, C17-C20) were preregistered by the lab before they ran; the March experiments "
    "were not. One reading in C19's source (learned runs do better when they happen to end with one cosine head in "
    "layer 1) was noticed after the runs and is not claimed.",
    "Running on the lab's lane when this was written, and not included: T2h (curvature started at the recipe and "
    "then learned; the recipe at 12 layers) and T2i (cosine on every head at 6 layers, which tests whether the "
    "first-layer recipe is more than cosine attention everywhere). Nothing above depends on them.",
]


def markdown():
    kept_rows, withdrawn, extra = build()
    out = []
    out.append("# Claims\n")
    out.append("Generated by `python reproduce.py claims` from the files in `results/`. Do not edit by hand: "
               "`python reproduce.py check` rebuilds this file and fails if it differs.\n")
    out.append("Each row is a hypothesis, the number that tests it, the file the number comes from, the control it "
               "faced, and the verdict. POSITIVE: survived the control. ANTI: the control points the other way. "
               "INCONCLUSIVE: too few seeds, or the effect is inside run-to-run spread.\n")
    out.append("## What the paper can claim\n")
    out.append("| ID | Hypothesis | Number | Source (under `results/`) | Control | Verdict |")
    out.append("|---|---|---|---|---|---|")
    for row in kept_rows:
        out.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    out.append("\n## What the drafts claimed and the controls took away\n")
    out.append("| ID | Draft claim | Why it goes |")
    out.append("|---|---|---|")
    for row in withdrawn:
        out.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    out.append("\n## Compression sweep, matched at 55 epochs\n")
    out.append("Relative perplexity change of the curvature model against the baseline trained in the same run, and of "
               "a separately trained curvature model against that same baseline. Negative favours curvature.\n")
    out.append("| d_model | layers | baseline PPL | same run | separate run |")
    out.append("|---:|---:|---:|---:|---:|")
    for r in extra["sweep_rows"]:
        out.append("| %d | %d | %.4f | %+.2f%% | %+.2f%% |" % (r["d"], r["L"], r["base55"], r["same"], r["other"]))
    out.append("| 128 | 1 | crashed | | |")
    out.append("| 128 | 3 | stopped at epoch 5 | | |")
    out.append("\n## Limitations\n")
    for i, text in enumerate(LIMITATIONS, 1):
        out.append("%d. %s" % (i, text))
    out.append("")
    return "\n".join(out)
