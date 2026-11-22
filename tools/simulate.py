"""Compare strategies and tune brain weights offline.

    python tools/simulate.py --games 1000
    python tools/simulate.py --tune 40 --games 300      # random search over weights
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.brain import DEFAULT_W
from bot.sim import POLICIES, policy_brain, win_rate

ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=500)
ap.add_argument("--tune", type=int, default=0, help="number of random weight sets to try")
args = ap.parse_args()

if args.tune:
    rng = random.Random(1)
    best_w, best = dict(DEFAULT_W), win_rate(policy_brain, args.games, DEFAULT_W)
    print("start", best_w, best)
    for t in range(args.tune):
        w = {"len": 100.0, **{k: max(0.0, v * rng.uniform(0.3, 2.5)) for k, v in DEFAULT_W.items() if k != "len"}}
        r = win_rate(policy_brain, args.games, w)
        if r > best:
            best_w, best = w, r
            print(f"[{t}] better: {({k: round(v, 1) for k, v in w.items()})} win={r[0]:.1%} cleared={r[1]:.1f}")
    print("BEST", {k: round(v, 1) for k, v in best_w.items()}, best)
    # Validate on unseen games so the tuned weights aren't just overfit
    print("held-out:", win_rate(policy_brain, 2000, best_w, seed0=100000))
else:
    for name, pol in POLICIES.items():
        wr, cl = win_rate(pol, args.games, DEFAULT_W, seed0=100000)
        print(f"{name:14s} win rate {wr:6.1%}   avg cards cleared {cl:5.1f}/28")
