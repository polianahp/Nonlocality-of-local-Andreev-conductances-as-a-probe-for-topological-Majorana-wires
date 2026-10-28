#!/usr/bin/env python3
r"""Step-by-step passing-area figures for the top-N ranked protocol instances.

For each of the top-N instances of `Outputs/Protocols/` (ranked exactly as in the master leaderboard:
Mean FDR ascending, ties by Mean TPR descending) this script re-evaluates the protocol in memory
(no step arrays are read from disk, so it works for `keep_data: false` instances) on one fixed
realization per disorder strength and draws every FPE/DPE step in order, so the refinement of the
passing area is visible.

Panel conventions (project invariants): x = V_z, y = mu; only the ground-truth Pfaffian boundary
(Q = -1) is drawn as the black dashed contour; green = true positive, red = false positive,
blue = false negative, white = true negative. Rank-3 (per-cutter) steps are shown as the number of
cutters passing (0..N_cutters).

The bottom panel of every figure shows the ENSEMBLE-mean survival / FDR / TPR after each step
(from `config.yaml`, all realizations and temperatures pooled), since a single realization per
disorder strength can be unrepresentative.

Realization choice is fixed (`--realization-index`, default 10, the first realization in every
V0 folder) and is not selected on outcome.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.Protocol_Elements import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    RealizationFeatureCache,
    compute_step_diagnostic,
    load_protocol_class_from_file,
)
from src.Protocol_Elements.leaderboard import (  # noqa: E402
    RANK_RULE,
    build_ranked_rows,
    load_instance_records,
    v0_label_to_mev,
)

REPO_ROOT = Path(__file__).resolve().parent
PROTOCOLS_ROOT = REPO_ROOT / "Protocols"

TP_RGB = [0.13, 0.65, 0.28]
FP_RGB = [0.86, 0.20, 0.18]
FN_RGB = [0.12, 0.53, 0.90]


def confusion_rgb(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    rgb = np.ones(pred.shape + (3,), dtype=float)
    rgb[pred & gt] = TP_RGB
    rgb[pred & ~gt] = FP_RGB
    rgb[~pred & gt] = FN_RGB
    return rgb


def short_step_title(name: str) -> str:
    parts = name.split("_")
    if len(parts) > 2 and parts[0] == "step" and parts[1].isdigit():
        return f"{parts[1]}: " + " ".join(parts[2:])
    return name.replace("_", " ")


def find_realization_dir(data_root: Path, v0_label: str, index: int) -> Path:
    d = data_root / v0_label / f"disorder_realization_{index}"
    if not d.exists():
        raise FileNotFoundError(f"Realization directory not found: {d}")
    return d


def render_instance_figure(
    protocol_name: str,
    instance_name: str,
    rank: int,
    outputs_root: Path,
    data_root: Path,
    temperature_mK: float,
    realization_index: int,
    v0_labels: List[str],
) -> Path:
    inst_dir = outputs_root / protocol_name / instance_name
    with open(inst_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = cfg["free_parameters"]
    summary = cfg["summary_metrics"]

    proto_cls = load_protocol_class_from_file(PROTOCOLS_ROOT / protocol_name / "protocol.py")
    proto = proto_cls()

    per_v0: List[Dict[str, Any]] = []
    for lbl in v0_labels:
        cache = RealizationFeatureCache(
            find_realization_dir(data_root, lbl, realization_index),
            temperature_mK=temperature_mK,
            v0_label=lbl,
        )
        steps = proto.evaluate_steps(cache, params)
        per_v0.append({"label": lbl, "cache": cache, "steps": steps})

    step_names = list(per_v0[0]["steps"].keys())
    n_steps = len(step_names)
    n_rows = len(per_v0)

    col_w, row_h = 3.1, 2.7
    fig = plt.figure(figsize=(col_w * n_steps + 1.0, row_h * n_rows + 3.6))
    gs = GridSpec(
        n_rows + 1,
        n_steps,
        figure=fig,
        height_ratios=[1.0] * n_rows + [1.15],
        hspace=0.55,
        wspace=0.12,
    )

    for r, item in enumerate(per_v0):
        cache: RealizationFeatureCache = item["cache"]
        gt = cache.pfaffian_mask
        mu, vz = cache.mu_vals, cache.vz_vals
        extent = [vz[0], vz[-1], mu[0], mu[-1]]
        for c, sname in enumerate(step_names):
            ax = fig.add_subplot(gs[r, c])
            arr = np.asarray(item["steps"][sname], dtype=bool)
            diag = compute_step_diagnostic(arr, gt)
            if arr.ndim == 3:
                ax.imshow(
                    arr.sum(axis=0), origin="lower", extent=extent, aspect="auto",
                    cmap="viridis", vmin=0, vmax=arr.shape[0], interpolation="nearest",
                )
                tag = f"cutters passing (0-{arr.shape[0]})\n"
            else:
                ax.imshow(
                    confusion_rgb(arr, gt), origin="lower", extent=extent, aspect="auto",
                    interpolation="nearest",
                )
                tag = ""
            ax.contour(vz, mu, gt.astype(float), levels=[0.5], colors=["black"],
                       linewidths=[1.2], linestyles=["--"])
            ax.set_title(
                textwrap.fill(short_step_title(sname), 26) + "\n" + tag
                + f"surv {diag['survival_pct']:.1f}% | FDR {diag['fdr_pct']:.1f}% | TPR {diag['tpr_pct']:.1f}%",
                fontsize=7.2,
            )
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(
                    f"$V_0={v0_label_to_mev(item['label'])}$ meV\n({cache.realization_name.replace('disorder_realization_', 'real. ')})\n$\\mu$ (meV)",
                    fontsize=8,
                )
            else:
                ax.set_yticklabels([])
            if r == n_rows - 1:
                ax.set_xlabel("$V_z$ (meV)", fontsize=7)
            else:
                ax.set_xticklabels([])

    # Ensemble waterfall panel
    ax_w = fig.add_subplot(gs[n_rows, :])
    wf = summary["step_waterfall_metrics"]
    xs = np.arange(len(step_names))
    surv = [wf[s]["mean_survival_pct"] for s in step_names]
    fdr = [wf[s]["mean_FDR_pct"] for s in step_names]
    tpr = [wf[s]["mean_TPR_pct"] for s in step_names]
    ax_w.bar(xs, surv, color="0.80", label="mean surviving area (%)", width=0.6)
    ax_w.plot(xs, fdr, "o-", color=FP_RGB, label="mean FDR (%)")
    ax_w.plot(xs, tpr, "s-", color=TP_RGB, label="mean TPR (%)")
    ax_w.set_xticks(xs)
    ax_w.set_xticklabels([short_step_title(s) for s in step_names], rotation=15, ha="right", fontsize=7)
    ax_w.set_ylim(0, 100)
    ax_w.set_ylabel("%", fontsize=8)
    ax_w.tick_params(labelsize=7)
    ax_w.legend(fontsize=7, ncol=3, loc="upper right")
    ax_w.set_title(
        f"Ensemble mean after each step (N={summary['n_realizations']} realization-temperature states, all $V_0$ pooled). "
        "Rank-3 steps: per-cutter average.",
        fontsize=8,
    )

    fig.suptitle(
        f"#{rank}  {protocol_name} / {instance_name}\n"
        f"Maps: {int(temperature_mK)} mK, one fixed realization per $V_0$ (index {realization_index}); "
        f"final ensemble: Mean FDR {summary['mean_FDR_pct']:.2f}%, Mean TPR {summary['mean_TPR_pct']:.2f}%, "
        f"pass {summary['realization_pass_rate_pct']:.1f}%",
        fontsize=10, y=0.995,
    )
    fig.legend(
        handles=[
            Patch(color=TP_RGB, label="true positive"),
            Patch(color=FP_RGB, label="false positive"),
            Patch(color=FN_RGB, label="false negative"),
            Patch(facecolor="white", edgecolor="0.6", label="true negative"),
            plt.Line2D([0], [0], color="black", ls="--", label="Pfaffian boundary ($\\mathcal{Q}=-1$)"),
        ],
        loc="lower center", ncol=5, fontsize=8, frameon=False,
    )
    out_path = inst_dir / f"step_by_step_{int(temperature_mK)}mK.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Step-by-step passing-area figures for the top-N protocols.")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=30.0)
    ap.add_argument("--realization-index", type=int, default=10, help="Realization index to plot (default: 10).")
    ap.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional folder name under Outputs/ (e.g., 'Microsoft_Hybrids_Post_Processing'). Overrides --outputs-root.",
    )
    ap.add_argument("--outputs-root", type=str, default=str(DEFAULT_OUTPUTS_ROOT))
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    args = ap.parse_args()

    outputs_root = (
        REPO_ROOT / "Outputs" / args.output_name
        if args.output_name
        else Path(args.outputs_root)
    )
    data_root = Path(args.data_root)
    instances = load_instance_records(outputs_root)
    ranked = build_ranked_rows(instances)[: args.top_n]
    v0_labels = sorted(
        {r.get("v0_label") for i in instances for r in i["records"] if r.get("v0_label")},
        key=lambda s: float(v0_label_to_mev(s)),
    )

    index_lines = [
        f"# Step-by-step figures: top {args.top_n} protocols",
        "",
        f"Ranking: {RANK_RULE}. "
        f"Maps at {int(args.temperature)} mK, realization index {args.realization_index} of each $V_0$ folder.",
        "",
        "| Rank | Protocol | Instance | Mean FDR (%) | Mean TPR (%) | Pass (%) | Figure |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]
    for row in ranked:
        path = render_instance_figure(
            row["protocol_name"], row["instance_name"], row["rank"], outputs_root, data_root,
            args.temperature, args.realization_index, v0_labels,
        )
        rel = path.relative_to(outputs_root)
        index_lines.append(
            f"| {row['rank']} | `{row['protocol_name']}` | `{row['instance_name']}` | "
            f"{row['mean_FDR_pct']:.2f} | {row['mean_TPR_pct']:.2f} | {row['pass_rate_pct']:.1f} | [{rel.name}]({rel.as_posix()}) |"
        )
        print(f"[#{row['rank']}] wrote {path}")
    (outputs_root / "top_protocols_step_by_step.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
