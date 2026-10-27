"""Tables and figures from `results/runs.csv` and the per-run histories."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ORDER = ["vanilla", "grand", "grand+es", "grand+potcap", "grand+aesad", "grand+all", "grand-kl0.05"]
REFERENCE = "grand"


def _order(df: pd.DataFrame) -> list[str]:
    present = list(df["variant"].unique())
    return [v for v in ORDER if v in present] + sorted(v for v in present if v not in ORDER)


def _md(table: pd.DataFrame) -> str:
    return table.to_markdown(disable_numparse=True)


def _pct_columns(table: pd.DataFrame) -> pd.DataFrame:
    table.columns = [f"{c:.0%}" for c in table.columns]
    return table


def mean_std_table(df: pd.DataFrame, metric: str, digits: int = 4) -> pd.DataFrame:
    """Variant x contamination table of 'mean ± std' over seeds."""
    g = df.groupby(["variant", "contamination"])[metric].agg(["mean", "std"])
    cell = g.apply(lambda r: f"{r['mean']:.{digits}f} ± {r['std']:.{digits}f}", axis=1).unstack()
    return _pct_columns(cell).loc[_order(df)]


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def _p(p: float) -> str:
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def paired_comparison(df: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """Each variant against the reference on the same (contamination, seed) pairs.

    ΔAUROC is variant minus reference; p is the two-sided Wilcoxon signed-rank
    test over all pairs, Holm-adjusted across the variants compared; time ratio
    is the median of variant time / reference time.
    """
    ref = df[df["variant"] == reference].set_index(["contamination", "seed"])
    rows, pvalues = [], []
    for v in _order(df):
        if v == reference:
            continue
        both = df[df["variant"] == v].set_index(["contamination", "seed"]).join(ref, rsuffix="_ref", how="inner")
        if both.empty:
            continue
        d_roc = both["auroc"] - both["auroc_ref"]
        d_pr = both["auprc"] - both["auprc_ref"]
        pvalues.append(wilcoxon(d_roc).pvalue if (d_roc != 0).any() else 1.0)
        rows.append({
            "variant": v,
            "pairs": str(len(both)),
            "ΔAUROC mean": f"{d_roc.mean():+.4f}",
            "ΔAUROC min, max": f"{d_roc.min():+.4f}, {d_roc.max():+.4f}",
            "better": f"{(d_roc > 0).sum()}/{len(both)}",
            "ΔAUPRC mean": f"{d_pr.mean():+.4f}",
            "time ratio": f"{(both['train_time_s'] / both['train_time_s_ref']).median():.2f}",
        })  # fmt: skip
    for row, p, p_adj in zip(rows, pvalues, holm(pvalues)):
        row["Wilcoxon p"] = _p(p)
        row["Holm p"] = _p(p_adj)
    return pd.DataFrame(rows).set_index("variant")


def rejection_quality(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Variant x contamination mean of a rejection diagnostic (c > 0 only)."""
    d = df[(df["contamination"] > 0) & df[metric].notna()]
    table = d.pivot_table(index="variant", columns="contamination", values=metric, aggfunc="mean")
    return _pct_columns(table.map(lambda x: f"{x:.3f}")).loc[_order(d)]


def figure_auroc(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for v in _order(df):
        g = df[df["variant"] == v].groupby("contamination")["auroc"].agg(["mean", "std"])
        style = {"ls": "--", "color": "black"} if v == "vanilla" else {}
        ax.errorbar(g.index * 100, g["mean"], yerr=g["std"], marker="o", capsize=3, label=v, **style)
    ax.set_xlabel("training contamination (%)")
    ax.set_ylabel("test AUROC (mean ± std, 5 seeds)")
    ax.set_xticks([0, 5, 10, 15])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_rejection(df: pd.DataFrame, histories: Path, path: Path) -> None:
    """Left: attacks caught in U ∪ S along training (15%). Right: attack share of L at the end."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for v in ("vanilla", "grand", "grand+potcap"):
        runs = [json.loads(p.read_text()) for p in histories.glob(f"{v}__c0.15__s*.json")]
        if runs:
            n = min(len(r["us_recall"]) for r in runs)
            ax1.plot(np.arange(1, n + 1), np.mean([r["us_recall"][:n] for r in runs], axis=0), label=v)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("injected attacks in U ∪ S")
    ax1.set_ylim(0, 1)
    ax1.set_title("15% contamination, mean of 5 seeds", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    d = df[df["contamination"] > 0]
    for v in ("vanilla", "grand", "grand+potcap"):
        g = d[d["variant"] == v].groupby("contamination")["l_contamination"].mean()
        ax2.plot(g.index * 100, g.values, marker="o", label=v)
    ax2.plot([5, 15], [0.05, 0.15], color="grey", ls=":", label="injected share (no filtering)")
    ax2.set_xlabel("training contamination (%)")
    ax2.set_ylabel("attack share of L at the end of training")
    ax2.set_xticks([5, 10, 15])
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


DECOMPOSED = ("vanilla", "grand", "grand-kl0.05")


def decompose(results_dir: str | Path = "results", raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Test AUROC of each term of the free energy, for the saved models of DECOMPOSED.

    Exploratory: it reuses the test set, so it suggests directions, it does not
    select a model. Needs `grand experiment --save-models`.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    from grand.evaluate import score_components
    from grand.experiments import load_dataset
    from grand.model import GRAnD

    results_dir, raw_dir = Path(results_dir), Path(raw_dir)
    rows = []
    for path in sorted((results_dir / "models").glob("*.pt")):
        variant, c, seed = path.stem.split("__")
        if variant not in DECOMPOSED:
            continue
        c, seed = float(c[1:]), int(seed[1:])
        data = load_dataset(raw_dir, raw_dir.parent / "cache", c, seed)
        model = GRAnD(data.X_test.shape[1])
        model.load_state_dict(torch.load(path))
        recon, kl = score_components(model, data.X_test, seed=seed)
        y = data.y_test
        rows.append({
            "variant": variant, "contamination": c, "seed": seed,
            "reconstruction": roc_auc_score(y, recon),
            "KL": roc_auc_score(y, kl),
            "reconstruction + KL": roc_auc_score(y, recon + kl),
            "reconstruction + 0.05 KL": roc_auc_score(y, recon + 0.05 * kl),
        })  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "decomposition.csv", index=False)
    return df


def decomposition_table(df: pd.DataFrame) -> pd.DataFrame:
    terms = ["reconstruction", "KL", "reconstruction + KL", "reconstruction + 0.05 KL"]
    g = df.groupby(["variant", "contamination"])[terms].agg(["mean", "std"])
    table = pd.DataFrame(
        {t: [f"{m:.4f} ± {sd:.4f}" for m, sd in zip(g[(t, "mean")], g[(t, "std")])] for t in terms}, index=g.index
    )
    table.index = [f"{v}, {c:.0%}" for v, c in table.index]
    order = [f"{v}, {c:.0%}" for v in DECOMPOSED for c in sorted(df["contamination"].unique())]
    return table.loc[[i for i in order if i in table.index]]


def write_report(results_dir: str | Path = "results") -> Path:
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "runs.csv")
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    figure_auroc(df, fig_dir / "auroc_vs_contamination.png")
    figure_rejection(df, results_dir / "histories", fig_dir / "rejection_quality.png")

    seeds = df.groupby(["variant", "contamination"]).size()
    n_seeds = f"{seeds.min()}" if seeds.min() == seeds.max() else f"{seeds.min()}–{seeds.max()}"
    parts = [
        "# Results\n",
        (f"NSL-KDD; test set KDDTest+, attack rate {df['auprc_chance'].iloc[0]:.3f} (the chance level of AUPRC). "
         f"{len(df)} runs, {n_seeds} seeds per cell, mean ± standard deviation over seeds. "
         "Generated by `grand report` from `runs.csv`.\n"),
        "![AUROC versus contamination](figures/auroc_vs_contamination.png)\n",
        "## Test AUROC\n", _md(mean_std_table(df, "auroc")), "",
        "## Test AUPRC\n", _md(mean_std_table(df, "auprc")), "",
        f"## Each variant against `{REFERENCE}`\n",
        ("Paired by contamination and seed (same training set, same initial weights), 4 contaminations × 5 seeds. "
         "Two-sided Wilcoxon signed-rank test on ΔAUROC, Holm-adjusted over the six comparisons.\n"),
        _md(paired_comparison(df)), "",
        "## Training cost\n", "Epochs:\n", _md(mean_std_table(df, "epochs", 1)), "",
        "Wall-clock seconds (7 runs in parallel on a laptop CPU, 2 threads each):\n",
        _md(mean_std_table(df, "train_time_s", 0)), "",
        "## Alarm threshold chosen without labels\n",
        ("Threshold `u = min(Q3 + 1.5·IQR, Q_0.999)` of the training scores, i.e. the boundary of the nominal set. "
         "F1 on the test set:\n"),
        _md(mean_std_table(df, "alarm_f1", 3)), "",
        "## How well the split isolates the injected attacks\n",
        ("Measured with the training labels at the end of training (they are never used by the model). "
         "For `vanilla` the split is computed but not used.\n"),
        "![Rejection quality](figures/rejection_quality.png)\n",
        "Fraction of the injected attacks that end up in U ∪ S:\n", _md(rejection_quality(df, "us_recall")), "",
        "Attack share of L (what the model is trained to reconstruct well):\n",
        _md(rejection_quality(df, "l_contamination")), "",
    ]  # fmt: skip
    if (results_dir / "decomposition.csv").exists():
        parts += [
            "## Exploratory: which term of the score detects the attacks\n",
            ("Test AUROC of each term of the free energy, scored separately, for the saved models "
             "(`grand decompose`). This reuses the test set, so it points at directions to validate, "
             "it does not select a model. Rows: training variant and contamination.\n"),
            _md(decomposition_table(pd.read_csv(results_dir / "decomposition.csv"))), "",
        ]  # fmt: skip
    out = results_dir / "summary.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
