"""Command line.

    python -m disconnection compare data.csv --blocks dem=age,months,volume ll=ll_ disc=disc_ conn=conn_ \
        --targets naming reading --models ridge bagged_trees --k 10 --repeats 1 --out results/
    python -m disconnection benchmark --datasets 3 --out benchmark/ [--quick]

compare: each --blocks entry is name=spec, where spec is a comma-separated list of
column names, or a prefix ending in '_' (every column starting with it). The
feature sets are the standard ones whose blocks are all available (see
features.STANDARD_SETS), plus stack(ll,conn) when both exist. Writes scores.csv
(task x model x set), comparisons.csv (each set vs lesion load, paired tests
with Holm correction over tasks) and predictions.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _columns(df, spec):
    if spec.endswith("_") and "," not in spec:
        cols = [c for c in df.columns if c.startswith(spec)]
    else:
        cols = [c.strip() for c in spec.split(",") if c.strip()]
    missing = [c for c in cols if c not in df.columns]
    if missing or not cols:
        raise SystemExit(f"block spec '{spec}': columns not found: {missing or '(none match)'}")
    return cols


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m disconnection", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("csv")
    c.add_argument("--blocks", nargs="+", required=True)
    c.add_argument("--targets", nargs="+", required=True)
    c.add_argument("--models", nargs="+", default=["ridge"])
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--repeats", type=int, default=1)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--jobs", type=int, default=1)
    c.add_argument("--out", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--datasets", type=int, default=3)
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--quick", action="store_true")
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.cmd == "compare":
        from .compare import compare_feature_sets
        from .features import STANDARD_SETS
        df = pd.read_csv(a.csv)
        blocks = {}
        for entry in a.blocks:
            name, spec = entry.split("=", 1)
            blocks[name] = df[_columns(df, spec)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        sets = {n: s for n, s in STANDARD_SETS.items() if set(s) <= set(blocks)}
        if {"ll", "conn"} <= set(sets):
            sets["stack(ll,conn)"] = ("stack", "ll", "conn")
        if not sets:
            raise SystemExit("no standard feature set can be built: provide at least dem and ll blocks")
        Y = df[a.targets].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        res = compare_feature_sets(blocks, Y, sets, models=a.models, k=a.k, repeats=a.repeats, seed=a.seed,
                                   task_names=a.targets, n_jobs=a.jobs, verbose=True)
        res.table.to_csv(out / "scores.csv", index=False)
        if "ll" in sets:
            comps = pd.concat([res.compare("ll", s) for s in sets if s != "ll"], ignore_index=True)
            comps.to_csv(out / "comparisons.csv", index=False)
            print(comps[["task", "model", "other", "relative", "p_t_holm", "p_pitman_morgan_holm"]]
                  .round(3).to_string(index=False))
        rows = []
        for (task, model, st), (idx, pred) in res.predictions.items():
            rows += [{"row": int(i), "task": task, "model": model, "set": st, "prediction": float(p)}
                     for i, p in zip(idx, pred)]
        pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    else:
        from . import benchmark
        scores, comps = benchmark.representation_benchmark(datasets=1 if a.quick else a.datasets, n_jobs=a.jobs,
                                                           models=("ridge", "bagged_trees") if a.quick else
                                                           benchmark.DEFAULT_MODELS)
        scores.to_csv(out / "scores.csv", index=False)
        comps.to_csv(out / "comparisons.csv", index=False)
        benchmark.summarise(scores).to_csv(out / "summary.csv")
        benchmark.summarise_comparisons(comps).to_csv(out / "comparison_summary.csv")
        benchmark.figure(scores, out / "benchmark.png")
        pits = benchmark.pitfalls(quick=a.quick)
        (out / "pitfalls.json").write_text(json.dumps(pits, indent=1))
        print(benchmark.summarise(scores).to_string())
        print(benchmark.summarise_comparisons(comps).to_string())
        print(json.dumps(pits, indent=1))


if __name__ == "__main__":
    main()
