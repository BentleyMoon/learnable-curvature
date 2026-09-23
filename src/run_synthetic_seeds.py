"""Re-run the numpy synthetic geometry test (learnable_geometry.py) on several seeds.

The original run used one seed (42). Each seed takes about two minutes on one CPU core. Runs go
four at a time at idle priority, and each writes results/synthetic_numpy/seeds/seed<N>.json.

    python src/run_synthetic_seeds.py --seeds 0-9
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "results", "synthetic_numpy", "seeds")


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run(seed):
    out = os.path.join(OUT, "seed%d.json" % seed)
    env = dict(os.environ, LG_SEED=str(seed), LG_OUT=out, PYTHONIOENCODING="utf-8")
    flags = 0x00000040 if os.name == "nt" else 0  # IDLE_PRIORITY_CLASS on Windows
    log = open(os.path.join(OUT, "seed%d.log" % seed), "w", encoding="utf-8")
    proc = subprocess.run([sys.executable, os.path.join(HERE, "learnable_geometry.py")], env=env,
                          stdout=log, stderr=subprocess.STDOUT, creationflags=flags, cwd=OUT)
    log.close()
    return seed, proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for seed, code in pool.map(run, parse_seeds(args.seeds)):
            print("seed %d exit %d" % (seed, code), flush=True)


if __name__ == "__main__":
    main()
