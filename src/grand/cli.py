"""Command line: `grand download | train | experiment | report | decompose | explain`."""

from __future__ import annotations

import argparse
from pathlib import Path

from grand.config import VARIANTS


def _download(args) -> None:
    from grand.data import download

    print(f"NSL-KDD ready in {download(args.raw_dir)}")


def _train(args) -> None:
    from grand.experiments import run

    overrides = {"max_epochs": args.epochs} if args.epochs else {}
    row = run(args.variant, args.contamination, args.seed, raw_dir=args.raw_dir, out_dir=args.out_dir,
              overrides=overrides, save_model=True)  # fmt: skip
    width = max(map(len, row))
    for k, v in row.items():
        print(f"{k:<{width}}  {v:.4f}" if isinstance(v, float) else f"{k:<{width}}  {v}")


def _experiment(args) -> None:
    from grand.experiments import run_grid

    run_grid(args.variants, args.contaminations, args.seeds, workers=args.workers, out_dir=args.out_dir,
             raw_dir=args.raw_dir, threads=args.threads, save_model=args.save_models)  # fmt: skip


def _report(args) -> None:
    from grand.report import write_report

    print(f"written {write_report(args.out_dir)}")


def _decompose(args) -> None:
    from grand.report import decompose, write_report

    decompose(args.out_dir, args.raw_dir)
    print(f"written {write_report(args.out_dir)}")


def _explain(args) -> None:
    import numpy as np
    import torch

    from grand.config import ModelConfig, variant_config
    from grand.evaluate import score
    from grand.experiments import load_dataset, run
    from grand.explain import explain, group_one_hot
    from grand.model import GRAnD

    name = f"{args.variant}__c{args.contamination:.2f}__s{args.seed}"
    weights = Path(args.out_dir) / "models" / f"{name}.pt"
    if not weights.exists():
        run(args.variant, args.contamination, args.seed, raw_dir=args.raw_dir, out_dir=args.out_dir, save_model=True)
    raw = Path(args.raw_dir)
    data = load_dataset(raw, raw.parent / "cache", args.contamination, args.seed)
    model = GRAnD(data.X_test.shape[1], ModelConfig())
    model.load_state_dict(torch.load(weights))
    kl_weight = variant_config(args.variant).kl_weight

    scores = score(model, data.X_test, kl_weight=kl_weight, seed=args.seed)
    for i in np.argsort(scores)[::-1][: args.top]:
        ex = explain(model, data.X_test[i], kl_weight=kl_weight, seed=args.seed)
        print(f"\ntest sample {i}  label={'attack' if data.y_test[i] else 'normal'}  F(x)={ex.energy:.2f}  "
              f"reconstruction share={ex.reconstruction_share:.2f}  IG completeness gap={ex.completeness_gap:.3f}")  # fmt: skip
        main = ex.squared_error if ex.reconstruction_share > 0.5 else ex.integrated_gradients
        label = "squared error" if ex.reconstruction_share > 0.5 else "integrated gradients"
        for feat, v in list(group_one_hot(data.feature_names, main).items())[:8]:
            print(f"  {feat:<28} {label}: {v:+.3f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="grand", description=__doc__)
    p.add_argument("--raw-dir", default="data/raw", help="where KDDTrain+.txt and KDDTest+.txt live")
    p.add_argument("--out-dir", default="results", help="where runs, histories and figures go")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="download and verify NSL-KDD").set_defaults(fn=_download)

    t = sub.add_parser("train", help="train and evaluate one configuration")
    t.add_argument("--variant", default="grand", choices=sorted(VARIANTS))
    t.add_argument("--contamination", type=float, default=0.10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, help="override max_epochs")
    t.set_defaults(fn=_train)

    e = sub.add_parser("experiment", help="run the ablation grid (resumable)")
    e.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=sorted(VARIANTS))
    e.add_argument("--contaminations", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15])
    e.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    e.add_argument("--workers", type=int, default=1, help="parallel processes")
    e.add_argument("--threads", type=int, default=None, help="torch threads per process")
    e.add_argument("--save-models", action="store_true")
    e.set_defaults(fn=_experiment)

    sub.add_parser("report", help="tables and figures from results/runs.csv").set_defaults(fn=_report)
    sub.add_parser("decompose", help="test AUROC of each term of the score, from the saved models").set_defaults(
        fn=_decompose
    )

    x = sub.add_parser("explain", help="explain the top-scored test samples of one model")
    x.add_argument("--variant", default="grand", choices=sorted(VARIANTS))
    x.add_argument("--contamination", type=float, default=0.10)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--top", type=int, default=3)
    x.set_defaults(fn=_explain)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
