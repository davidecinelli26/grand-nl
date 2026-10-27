"""Experiment grid: variants x contamination levels x seeds.

Every (variant, contamination, seed) triple is one run. Runs that share a
contamination level and a seed see the same training set and start from the
same initial weights, so variants can be compared pair by pair. Results are
appended to `runs.csv` as soon as each run ends, and runs already there are
skipped, so an interrupted grid resumes where it stopped.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from grand.config import ModelConfig, variant_config
from grand.data import Dataset, prepare
from grand.evaluate import alarm_metrics, ranking_metrics, score
from grand.model import GRAnD
from grand.thresholds import ThresholdPolicy
from grand.trainer import fit, set_seed

KEYS = ["variant", "contamination", "seed"]


def load_dataset(raw_dir: Path, cache_dir: Path, contamination: float, seed: int) -> Dataset:
    """`data.prepare`, cached as .npz (preprocessing takes longer than an epoch)."""
    path = cache_dir / f"nslkdd_c{contamination:.2f}_s{seed}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return Dataset(z["X_train"], z["y_train"], z["X_test"], z["y_test"], list(z["features"]))
    data = prepare(raw_dir, contamination, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X_train=data.X_train, y_train=data.y_train, X_test=data.X_test, y_test=data.y_test,
        features=np.array(data.feature_names),
    )  # fmt: skip
    return data


def run(
    variant: str,
    contamination: float,
    seed: int,
    *,
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path = "results",
    model_cfg: ModelConfig | None = None,
    overrides: dict | None = None,
    n_score_samples: int = 16,
    threads: int | None = None,
    save_model: bool = False,
) -> dict:
    """Train and evaluate one configuration; returns one row of `runs.csv`."""
    if threads:
        torch.set_num_threads(threads)
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    cfg = variant_config(variant, **(overrides or {}))
    data = load_dataset(raw_dir, raw_dir.parent / "cache", contamination, seed)

    set_seed(seed)
    model = GRAnD(data.X_train.shape[1], model_cfg)
    hist, _ = fit(model, data.X_train, cfg, y=data.y_train)

    kw = {"kl_weight": cfg.kl_weight, "n_samples": n_score_samples, "seed": seed}
    s_test = score(model, data.X_test, **kw)
    s_train = score(model, data.X_train, **kw)
    # Alarm threshold chosen without labels: u = min(Q3 + alpha IQR, Q_{1-q}) of the
    # training scores, the boundary of the nominal set L (t_q, at q = 0.1%, flags
    # almost nothing on a test set that is 57% attacks).
    alarm = ThresholdPolicy(cfg)(s_train).u

    row = {
        "variant": variant,
        "contamination": contamination,
        "seed": seed,
        "epochs": hist.epochs,
        "stop_reason": hist.stop_reason,
        "train_time_s": round(hist.train_time, 2),
        **ranking_metrics(data.y_test, s_test),
        **{f"alarm_{k}": v for k, v in alarm_metrics(data.y_test, s_test, alarm).items()},
        "frac_u": hist.n_u[-1] / len(data.X_train),
        "frac_s": hist.n_s[-1] / len(data.X_train),
    }
    # always the same columns, NaN when there is no contamination to diagnose
    for key in ("s_precision", "s_recall", "us_recall", "l_contamination"):
        values = getattr(hist, key)
        row[key] = values[-1] if values else float("nan")

    name = f"{variant}__c{contamination:.2f}__s{seed}"
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)
    (out_dir / "histories" / f"{name}.json").write_text(json.dumps(asdict(hist)))
    if save_model:
        (out_dir / "models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "models" / f"{name}.pt")
    return row


def _job(args: tuple) -> dict:
    variant, c, seed, kwargs = args
    return run(variant, c, seed, **kwargs)


def run_grid(
    variants: list[str],
    contaminations: list[float],
    seeds: list[int],
    *,
    workers: int = 1,
    out_dir: str | Path = "results",
    **kwargs,
) -> pd.DataFrame:
    """Run every missing triple of the grid and return the full results table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "runs.csv"
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=KEYS)
    have = {(v, round(c, 4), int(s)) for v, c, s in done[KEYS].itertuples(index=False)}
    todo = [
        (v, c, s, {**kwargs, "out_dir": out_dir})
        for c in contaminations
        for s in seeds
        for v in variants
        if (v, round(c, 4), s) not in have
    ]
    print(f"{len(todo)} runs to do, {len(have)} already in {csv}")
    # build the dataset caches first, so parallel workers only ever read them
    raw_dir = Path(kwargs.get("raw_dir", "data/raw"))
    for c, s in sorted({(job[1], job[2]) for job in todo}):
        load_dataset(raw_dir, raw_dir.parent / "cache", c, s)

    def append(row: dict) -> None:
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        print(f"done {row['variant']:<13} c={row['contamination']:.2f} seed={row['seed']}  "
              f"AUROC {row['auroc']:.4f}  epochs {row['epochs']}  {row['train_time_s']:.0f}s")  # fmt: skip

    if workers <= 1:
        for job in todo:
            append(_job(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_job, job) for job in todo]):
                append(fut.result())
    return pd.read_csv(csv)
