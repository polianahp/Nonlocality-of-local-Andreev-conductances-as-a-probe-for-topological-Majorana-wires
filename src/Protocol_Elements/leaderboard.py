r"""Master / per-protocol leaderboard builder for `Outputs/Protocols/`.

The leaderboards are rebuilt from the results on disk (every
`<Protocol>/<Instance>/realization_metrics.json` + `config.yaml`) rather than from the specs
of the most recent run, so a partial run (a single protocol, a YAML filter, a realization cap)
never drops previously evaluated instances from the tables. `run_protocol_specs` calls
`rebuild_leaderboards` after every protocol test, so the tables are always current.

Ranking rule (user directive, 2026-10-23): rows are ranked first by Mean FDR (ascending),
ties broken by Mean TPR (descending).
Mean FDR is the per-realization mean (an empty prediction contributes FDR = 0, see
`metrics.py`), so `Pass (%)` must be read alongside it: a protocol that predicts nothing in
most realizations will have a low Mean FDR and a near-zero TPR.

Layout of `master_protocols_leaderboard.md`:
  1. Overall (all disorder strengths and temperatures pooled)
  2. One section per disorder strength V0 (temperatures pooled)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from .metrics import aggregate_ensemble_metrics

RANK_RULE = "Area >= 10 px, ranked by Mean FDR (ascending) then Mean TPR (descending); Area < 10 px placed last"

CSV_FIELDS = [
    "scope",
    "rank",
    "protocol_name",
    "instance_name",
    "keep_data",
    "n_states",
    "pass_rate_pct",
    "mean_FDR_pct",
    "median_FDR_pct",
    "mean_FDR_active_only_pct",
    "mean_TPR_pct",
    "median_TPR_pct",
    "mean_MZM_confinement_C",
    "mean_MZM_separability_S",
    "mean_island_count",
    "mean_island_area_px",
]


def _rank_key(row: Dict[str, Any]):
    is_disqualified = bool(row["mean_island_area_px"] < 10.0)
    return (
        is_disqualified,
        row["mean_FDR_pct"],
        -row["mean_TPR_pct"],
        row["protocol_name"],
        row["instance_name"],
    )


def v0_label_to_mev(label: str) -> str:
    """`V0_0_378` -> `0.378`, `V0_1_2` -> `1.2`."""
    body = label[len("V0_"):] if label.startswith("V0_") else label
    return body.replace("_", ".")


def _v0_sort_value(label: str) -> float:
    try:
        return float(v0_label_to_mev(label))
    except ValueError:
        return float("inf")


def load_instance_records(outputs_root: Union[str, Path]) -> List[Dict[str, Any]]:
    """Collect every evaluated protocol instance under `outputs_root` from disk."""
    root = Path(outputs_root)
    instances: List[Dict[str, Any]] = []
    if not root.exists():
        return instances
    for proto_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for inst_dir in sorted(p for p in proto_dir.iterdir() if p.is_dir()):
            metrics_path = inst_dir / "realization_metrics.json"
            config_path = inst_dir / "config.yaml"
            if not metrics_path.exists() or not config_path.exists():
                continue
            with open(metrics_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if not records:
                continue
            instances.append({
                "protocol_name": proto_dir.name,
                "instance_name": inst_dir.name,
                "keep_data": bool(cfg.get("keep_data", False)),
                "records": records,
            })
    return instances


def _row_from_summary(inst: Dict[str, Any], scope: str, sm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scope": scope,
        "protocol_name": inst["protocol_name"],
        "instance_name": inst["instance_name"],
        "keep_data": inst["keep_data"],
        "n_states": sm.get("n_realizations", 0),
        "pass_rate_pct": sm.get("realization_pass_rate_pct", 0.0),
        "mean_FDR_pct": sm.get("mean_FDR_pct", 0.0),
        "median_FDR_pct": sm.get("median_FDR_pct", 0.0),
        "mean_FDR_active_only_pct": sm.get("mean_FDR_active_only_pct", 0.0),
        "mean_TPR_pct": sm.get("mean_TPR_pct", 0.0),
        "median_TPR_pct": sm.get("median_TPR_pct", 0.0),
        "mean_MZM_confinement_C": sm.get("mean_MZM_confinement_C_active", 0.0),
        "mean_MZM_separability_S": sm.get("mean_MZM_separability_S_active", 0.0),
        "mean_island_count": sm.get("mean_island_count", 0.0),
        "mean_island_area_px": sm.get("mean_island_area_px", 0.0),
    }


def build_ranked_rows(
    instances: Sequence[Dict[str, Any]],
    v0_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Aggregate and rank all instances (optionally restricted to one disorder strength)."""
    scope = "overall" if v0_label is None else v0_label
    rows: List[Dict[str, Any]] = []
    for inst in instances:
        recs = (
            inst["records"]
            if v0_label is None
            else [r for r in inst["records"] if r.get("v0_label") == v0_label]
        )
        if not recs:
            continue
        rows.append(_row_from_summary(inst, scope, aggregate_ensemble_metrics(recs)))
    rows.sort(key=_rank_key)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def _md_table(rows: Sequence[Dict[str, Any]], include_protocol: bool = True) -> List[str]:
    head_cols = ["Rank"] + (["Protocol"] if include_protocol else []) + [
        "Instance", "N", "Pass (%)", "Mean FDR (%)", "Active FDR (%)", "Mean TPR (%)",
        r"Confinement $\mathcal{C}$", r"Separability $\mathcal{S}$", "Islands", "Area (px)",
        "`keep_data`",
    ]
    lines = [
        "| " + " | ".join(head_cols) + " |",
        "| " + " | ".join([":---:"] + [":---"] * (2 if include_protocol else 1) + [":---:"] * (len(head_cols) - (3 if include_protocol else 2))) + " |",
    ]
    for r in rows:
        cells = [str(r["rank"])]
        if include_protocol:
            cells.append(f"`{r['protocol_name']}`")
        cells += [
            f"`{r['instance_name']}`",
            str(r["n_states"]),
            f"{r['pass_rate_pct']:.1f}",
            f"**{r['mean_FDR_pct']:.2f}**",
            f"{r['mean_FDR_active_only_pct']:.2f}",
            f"**{r['mean_TPR_pct']:.2f}**",
            f"{r['mean_MZM_confinement_C']:.3f}",
            f"{r['mean_MZM_separability_S']:.3f}",
            f"{r['mean_island_count']:.2f}",
            f"{r['mean_island_area_px']:.1f}",
            f"`{r['keep_data']}`",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rebuild_leaderboards(outputs_root: Union[str, Path]) -> Optional[Path]:
    """Rebuild the master leaderboard (overall + per-V0) and every per-protocol leaderboard.

    Returns the path of `master_protocols_leaderboard.md`, or `None` if there is nothing to rank.
    """
    root = Path(outputs_root)
    instances = load_instance_records(root)
    if not instances:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    v0_labels = sorted(
        {r.get("v0_label") for inst in instances for r in inst["records"] if r.get("v0_label")},
        key=_v0_sort_value,
    )

    overall = build_ranked_rows(instances)
    per_v0 = {lbl: build_ranked_rows(instances, v0_label=lbl) for lbl in v0_labels}

    md: List[str] = [
        "# Master Cross-Protocol Leaderboard (`Outputs/Protocols/`)",
        "",
        f"_Rebuilt {stamp} from {len(instances)} evaluated protocol instances. "
        f"Ranking: {RANK_RULE}._",
        "",
        "`N` = number of realization-temperature states aggregated (temperatures pooled). "
        "`Mean FDR` is the per-realization mean with empty predictions counted as FDR = 0; "
        "`Active FDR` averages only realizations with at least one passing pixel; read both "
        "alongside `Pass (%)` and `Mean TPR`.",
        "",
        "## Overall (all disorder strengths pooled)",
        "",
    ] + _md_table(overall, include_protocol=True) + [""]

    for lbl in v0_labels:
        rows = per_v0[lbl]
        if not rows:
            continue
        md += [
            f"## Disorder strength $V_0 = {v0_label_to_mev(lbl)}$ meV",
            "",
        ] + _md_table(rows, include_protocol=True) + [""]

    master_md = root / "master_protocols_leaderboard.md"
    master_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    csv_rows: List[Dict[str, Any]] = list(overall)
    for lbl in v0_labels:
        csv_rows.extend(per_v0[lbl])
    _write_csv(root / "master_protocols_leaderboard.csv", csv_rows)

    # Per-protocol leaderboards (same ranking, overall + per-V0 in one file)
    for proto in sorted({i["protocol_name"] for i in instances}):
        p_inst = [i for i in instances if i["protocol_name"] == proto]
        p_overall = build_ranked_rows(p_inst)
        p_md = [
            f"# Leaderboard Summary: `{proto}`",
            "",
            f"_Rebuilt {stamp}. Ranking: {RANK_RULE}._",
            "",
            "## Overall (all disorder strengths pooled)",
            "",
        ] + _md_table(p_overall, include_protocol=False) + [""]
        p_csv = list(p_overall)
        for lbl in v0_labels:
            rows = build_ranked_rows(p_inst, v0_label=lbl)
            if not rows:
                continue
            p_md += [f"## Disorder strength $V_0 = {v0_label_to_mev(lbl)}$ meV", ""]
            p_md += _md_table(rows, include_protocol=False) + [""]
            p_csv.extend(rows)
        (root / proto / "leaderboard_summary.md").write_text("\n".join(p_md) + "\n", encoding="utf-8")
        _write_csv(root / proto / "leaderboard_summary.csv", p_csv)

    return master_md
