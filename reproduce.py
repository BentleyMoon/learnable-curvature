"""Reproduce the numbers and figures in this repository.

    python reproduce.py check        # default. Seconds, no downloads, no training:
                                     #   1. every result file matches the SHA-256 in results/MANIFEST.json
                                     #   2. CLAIMS.md rebuilt from results/ is identical to the committed one
                                     #   3. every figure rebuilds from results/
    python reproduce.py claims       # rewrite CLAIMS.md from results/
    python reproduce.py figures      # rewrite figures/*.png from results/
    python reproduce.py synthetic    # re-run the numpy toy on seeds 42 and 0-9 (about 6 min on 4 cores)
                                     # and confirm the fresh outputs equal the committed ones
    python reproduce.py rerun <arm>  # retrain one d_model-96 run from scratch on CPU (1 to 1.5 h each)
                                     # and compare its perplexity with the committed result
    python reproduce.py rerun --list # the arms that can be retrained this way

Training reruns need the corpus first: python scripts/get_data.py
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

# Retrainable d_model-96 arms: name -> (arguments to src/ternary/t1_native_ternary.py, committed result file).
# Every one is 20,000 steps at d_model 96, 4 heads, 3 layers, as in the lab's T1b/T2c runs.
D96 = ["--steps", "20000", "--d_model", "96", "--n_heads", "4", "--n_layers", "3", "--threads", "4"]
RERUNS = {
    "zero_seed0": (["--condition", "zero", "--seed", "0"] + D96, "ternary_lane/t1b_runs/zero_seed0.json"),
    "continuous_seed0": (["--condition", "continuous", "--seed", "0"] + D96, "ternary_lane/t1b_runs/continuous_seed0.json"),
    "allsph_seed0": (["--condition", "fixedcode", "--code=++++++++++++", "--seed", "0"] + D96,
                     "ternary_lane/t2c_runs/allsph/fixedcode_seed0.json"),
    "allhyp_seed0": (["--condition", "fixedcode", "--code=------------", "--seed", "0"] + D96,
                     "ternary_lane/t2c_runs/allhyp/fixedcode_seed0.json"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest():
    with open(os.path.join(REPO, "results", "MANIFEST.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    bad = [rel for rel, meta in manifest.items() if sha256(os.path.join(REPO, "results", rel)) != meta["sha256"]]
    print("result files: %d checked, %d changed" % (len(manifest), len(bad)))
    for rel in bad:
        print("  CHANGED", rel)
    return not bad


def check_claims():
    import claims
    fresh = claims.markdown()
    with open(os.path.join(REPO, "CLAIMS.md"), encoding="utf-8") as f:
        committed = f.read()
    same = fresh == committed
    print("CLAIMS.md: %s" % ("every number reproduces from results/" if same else "DIFFERS from results/"))
    if not same:
        import difflib
        for line in list(difflib.unified_diff(committed.splitlines(), fresh.splitlines(), "committed", "rebuilt", lineterm=""))[:40]:
            print("  " + line)
    return same


def write_claims():
    import claims
    with open(os.path.join(REPO, "CLAIMS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(claims.markdown())
    print("wrote CLAIMS.md")


def figures(out_dir):
    import make_figures
    names = make_figures.main(out_dir)
    print("figures: %d written to %s" % (len(names), out_dir))
    return names


def check_figures():
    with tempfile.TemporaryDirectory() as tmp:
        names = figures(tmp)
    missing = [n for n in names if not os.path.exists(os.path.join(REPO, "figures", n))]
    for n in missing:
        print("  MISSING from figures/:", n)
    return not missing


def synthetic():
    committed = {}
    seeds_dir = os.path.join(REPO, "results", "synthetic_numpy", "seeds")
    for name in os.listdir(seeds_dir):
        if name.endswith(".json"):
            with open(os.path.join(seeds_dir, name), encoding="utf-8") as f:
                committed[name] = json.load(f)
    code = subprocess.call([sys.executable, os.path.join(SRC, "run_synthetic_seeds.py"), "--seeds", "42,0-9"])
    if code:
        return False
    same = True
    for name, old in sorted(committed.items()):
        with open(os.path.join(seeds_dir, name), encoding="utf-8") as f:
            new = json.load(f)
        if new != old:
            same = False
            print("  DIFFERS", name)
    print("synthetic toy: %d seeds re-run, %s" % (len(committed), "identical to the committed results" if same else "DIFFERENT"))
    return same


def rerun(arm):
    args, rel = RERUNS[arm]
    data = os.environ.get("LC_DATA", os.path.join(REPO, "data", "wikitext2_all.txt"))
    if not os.path.exists(data):
        print("missing %s: run python scripts/get_data.py first" % data)
        return False
    out_dir = os.path.join(REPO, "reruns", arm)
    os.makedirs(out_dir, exist_ok=True)
    condition = args[args.index("--condition") + 1]
    seed = args[args.index("--seed") + 1]
    out = os.path.join(out_dir, "%s_seed%s.json" % (condition, seed))
    flags = 0x00000040 if os.name == "nt" else 0  # IDLE_PRIORITY_CLASS on Windows
    cmd = [sys.executable, os.path.join(SRC, "ternary", "t1_native_ternary.py")] + args + ["--out_dir", out_dir]
    print(" ".join(cmd))
    code = subprocess.call(cmd, creationflags=flags)
    if code:
        return False
    with open(out, encoding="utf-8") as f:
        new = json.load(f)["val_ppl"]
    with open(os.path.join(REPO, "results", rel), encoding="utf-8") as f:
        old = json.load(f)["val_ppl"]
    print("%s: rerun %.10f, committed %.10f, difference %.2e" % (arm, new, old, new - old))
    return abs(new - old) < 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="check", choices=["check", "claims", "figures", "synthetic", "rerun"])
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.command == "check":
        ok = check_manifest() & check_claims() & check_figures()
        print("OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    if args.command == "claims":
        write_claims()
    elif args.command == "figures":
        figures(os.path.join(REPO, "figures"))
    elif args.command == "synthetic":
        sys.exit(0 if synthetic() else 1)
    elif args.command == "rerun":
        if args.list or not args.arm:
            for name, (a, rel) in RERUNS.items():
                print("%-18s -> results/%s" % (name, rel))
            return
        sys.exit(0 if rerun(args.arm) else 1)


if __name__ == "__main__":
    main()
