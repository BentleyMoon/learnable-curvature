"""Copy the saved result files this paper cites into results/, and record where each came from.

Maintainer tool. The source paths are the author's machine; a clone never needs to run this.
Every copied file gets its source path and SHA-256 in results/MANIFEST.json, so any number in
CLAIMS.md can be traced back to the file the original run wrote.

    python tools/collect_results.py
"""
import glob
import hashlib
import json
import os
import shutil

FRACTAL = r"C:\gptcodex\fractaltest\files"
RUNS = os.path.join(FRACTAL, "runs")
MANDEL = r"C:\mandel\runs"
TERNARY = r"C:\engine\projects\ternary_native"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")

# (destination under results/, source path or glob)
SOURCES = [
    ("synthetic_numpy/autograd_geometry_results_seed42.json", os.path.join(FRACTAL, "autograd_geometry_results.json")),
    ("synthetic_lm/curvature_transformer_results.json", os.path.join(FRACTAL, "curvature_transformer_results.json")),
    ("synthetic_lm/kappa_trajectory.json", os.path.join(FRACTAL, "kappa_trajectory.json")),
    ("wikitext_char/pilot_4h2l/results.json", os.path.join(RUNS, "language_20260305_cpu_pilot", "curvature_transformer_results.json")),
    ("wikitext_char/phaseA_4h2l/results.json", os.path.join(RUNS, "language_20260305_cpu_pilot", "resume_long55_results.json")),
    ("wikitext_char/phaseA_4h2l/kappa_trajectory.json", os.path.join(RUNS, "language_20260305_cpu_pilot", "resume_long55_kappa_trajectory.json")),
    ("wikitext_char/phaseA_4h2l/attention_analysis.json", os.path.join(RUNS, "language_20260305_cpu_pilot", "attention_geometry_analysis_55ep.json")),
    ("wikitext_char/phaseB_8h4l/results.json", os.path.join(RUNS, "language_20260305_8h4l", "run_results.json")),
    ("wikitext_char/phaseB_8h4l/kappa_trajectory.json", os.path.join(RUNS, "language_20260305_8h4l", "run_kappa_trajectory.json")),
    ("wikitext_char/phaseB_8h4l/attention_analysis.json", os.path.join(RUNS, "language_20260305_8h4l", "attention_geometry_analysis_55ep.json")),
    ("wikitext_char/phaseB_8h4l/sign_only_eval_20260318.json", os.path.join(RUNS, "ternary_kappa_quantization_eval_20260318.json")),
    ("wikitext_char/ablation_3arm_20260305/results.json", os.path.join(RUNS, "language_20260305_ablation", "run_ablation_results.json")),
    ("wikitext_char/ablation_4arm_random_20260312/results.json", os.path.join(RUNS, "language_20260312_ablation_random", "run_ablation_results.json")),
    ("wikitext_char/forced_geometry_20260312/results.json", os.path.join(RUNS, "language_20260312_forced_geometry", "run_results.json")),
    ("wikitext_char/clamp_20260305/results.json", os.path.join(RUNS, "language_20260305_clamp", "run_results.json")),
    ("wikitext_char/kappa_timing_20260312.json", os.path.join(RUNS, "kappa_trajectory_timing_20260312.json")),
    ("wikitext_char/sweep/summary_20260310.json", os.path.join(RUNS, "compression_sweep_summary_20260307.json")),
    ("wikitext_char/sweep/crash_log_std_d128_L1.txt", os.path.join(RUNS, "run_compression_restart2_worker0_20260311.err.log")),
    ("mandel/control_matrix_5ep_20260313.json", os.path.join(MANDEL, "control_matrix_5ep_20260313.json")),
    ("mandel/control_matrix_5ep_learnable_c_20260313.json", os.path.join(MANDEL, "control_matrix_5ep_learnable_c_20260313.json")),
    ("mandel/control_matrix_20ep_learnable_c_20260313.json", os.path.join(MANDEL, "control_matrix_20ep_learnable_c_20260313.json")),
    ("mandel/wikitext_standard_dot.json", os.path.join(MANDEL, "wikitext_standard_dot", "run_results.json")),
    ("mandel/wikitext_exact_euclidean.json", os.path.join(MANDEL, "wikitext_exact_euc", "run_results.json")),
    ("mandel/wikitext_temperature_tau.json", os.path.join(MANDEL, "wikitext_temperature_tau", "run_results.json")),
]
for d in sorted(glob.glob(os.path.join(RUNS, "sweep_*"))):
    SOURCES.append(("wikitext_char/sweep/%s.json" % os.path.basename(d), os.path.join(d, "run_results.json")))
# Experiments still running on the lab's lane are left out until their result file is final.
PENDING = ("T2H",)
for p in sorted(glob.glob(os.path.join(TERNARY, "RESULT_*.json"))):
    if not any(("RESULT_%s_" % tag) in os.path.basename(p) for tag in PENDING):
        SOURCES.append(("ternary_lane/%s" % os.path.basename(p), p))
for p in sorted(glob.glob(os.path.join(TERNARY, "t*_runs", "**", "*.json"), recursive=True)):
    rel = os.path.relpath(p, TERNARY).replace(os.sep, "/")
    if not any(rel.startswith("%s_runs/" % tag.lower()) for tag in PENDING):
        SOURCES.append(("ternary_lane/%s" % rel, p))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {}
    for dest, src in SOURCES:
        if not os.path.exists(src):
            print("MISSING", src)
            continue
        target = os.path.join(OUT, dest)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(src, target)
        manifest[dest] = {"source": src, "sha256": sha256(target), "bytes": os.path.getsize(target)}
    # Files this repository generated itself (the synthetic toy's fresh seeds) are listed too.
    for root, _, files in os.walk(OUT):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), OUT).replace(os.sep, "/")
            if rel != "MANIFEST.json" and rel not in manifest:
                manifest[rel] = {"source": "generated in this repository by src/run_synthetic_seeds.py",
                                 "sha256": sha256(os.path.join(root, name)),
                                 "bytes": os.path.getsize(os.path.join(root, name))}
    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("copied %d files into %s" % (len(manifest), OUT))


if __name__ == "__main__":
    main()
