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
    "temperature_mK",
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


def _row_from_summary(
    inst: Dict[str, Any],
    scope: str,
    sm: Dict[str, Any],
    temperature_mK: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "scope": scope,
        "temperature_mK": f"{temperature_mK:.1f}" if temperature_mK is not None else "pooled",
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
    temperature_mK: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Aggregate and rank all instances (optionally restricted to one disorder strength and/or temperature)."""
    scope_parts = []
    if v0_label is not None:
        scope_parts.append(v0_label)
    if temperature_mK is not None:
        scope_parts.append(f"{int(round(temperature_mK))}mK")
    scope = "_".join(scope_parts) if scope_parts else "overall"

    rows: List[Dict[str, Any]] = []
    for inst in instances:
        recs = inst["records"]
        if v0_label is not None:
            recs = [r for r in recs if r.get("v0_label") == v0_label]
        if temperature_mK is not None:
            recs = [r for r in recs if abs(float(r.get("temperature_mK", 30.0)) - temperature_mK) < 1e-3]
        if not recs:
            continue
        rows.append(_row_from_summary(inst, scope, aggregate_ensemble_metrics(recs), temperature_mK=temperature_mK))
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
    """Rebuild the master leaderboard (overall + per-V0 + per-temperature) and every per-protocol leaderboard.

    Generates:
    - master_protocols_leaderboard.md & master_protocols_leaderboard.csv (all temperatures pooled)
    - master_protocols_leaderboard_30mK.md & master_protocols_leaderboard_30mK.csv
    - master_protocols_leaderboard_40mK.md & master_protocols_leaderboard_40mK.csv
    - Per-protocol summary files for pooled, 30 mK, and 40 mK inside each protocol folder.

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
    temperatures = sorted(
        {round(float(r.get("temperature_mK", 30.0)), 1) for inst in instances for r in inst["records"]},
    )

    overall = build_ranked_rows(instances)
    per_v0 = {lbl: build_ranked_rows(instances, v0_label=lbl) for lbl in v0_labels}

    # Navigation links for temperature splits
    temp_links = [
        f"[{int(round(t))} mK Leaderboard](master_protocols_leaderboard_{int(round(t))}mK.md)"
        for t in temperatures
    ]
    temp_nav_banner = (
        "> **Temperature Leaderboards**: " + " | ".join(temp_links)
        if len(temperatures) > 1 or (len(temperatures) == 1 and temperatures[0] in (30.0, 40.0))
        else ""
    )

    md: List[str] = [
        "# Master Cross-Protocol Leaderboard (`Outputs/Protocols/`)",
        "",
        f"_Rebuilt {stamp} from {len(instances)} evaluated protocol instances. "
        f"Ranking: {RANK_RULE}._",
        "",
    ]
    if temp_nav_banner:
        md.extend([temp_nav_banner, ""])

    md.extend([
        "`N` = number of realization-temperature states aggregated (temperatures pooled). "
        "`Mean FDR` is the per-realization mean with empty predictions counted as FDR = 0; "
        "`Active FDR` averages only realizations with at least one passing pixel; read both "
        "alongside `Pass (%)` and `Mean TPR`.",
        "",
        "## Overall (all disorder strengths and temperatures pooled)",
        "",
    ] + _md_table(overall, include_protocol=True) + [""])

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

    # Dedicated Per-Temperature Master Leaderboards (e.g. 30 mK and 40 mK)
    for t_mK in temperatures:
        t_tag = f"{int(round(t_mK))}mK"
        t_overall = build_ranked_rows(instances, temperature_mK=t_mK)
        t_per_v0 = {lbl: build_ranked_rows(instances, v0_label=lbl, temperature_mK=t_mK) for lbl in v0_labels}

        other_links = [
            f"[{int(round(ot))} mK Leaderboard](master_protocols_leaderboard_{int(round(ot))}mK.md)"
            for ot in temperatures if ot != t_mK
        ] + ["[Pooled Master Leaderboard](master_protocols_leaderboard.md)"]
        t_nav_banner = "> **Related Leaderboards**: " + " | ".join(other_links)

        t_md: List[str] = [
            f"# Master Cross-Protocol Leaderboard — {t_tag} (`Outputs/Protocols/`)",
            "",
            f"_Rebuilt {stamp} from {len(instances)} evaluated protocol instances at $T = {int(round(t_mK))}$ mK. "
            f"Ranking: {RANK_RULE}._",
            "",
            t_nav_banner,
            "",
            f"`N` = number of realization states aggregated at {t_tag}. "
            "`Mean FDR` is the per-realization mean with empty predictions counted as FDR = 0; "
            "`Active FDR` averages only realizations with at least one passing pixel; read both "
            "alongside `Pass (%)` and `Mean TPR`.",
            "",
            f"## Overall (all disorder strengths pooled at {t_tag})",
            "",
        ] + _md_table(t_overall, include_protocol=True) + [""]

        for lbl in v0_labels:
            rows = t_per_v0[lbl]
            if not rows:
                continue
            t_md += [
                f"## Disorder strength $V_0 = {v0_label_to_mev(lbl)}$ meV ({t_tag})",
                "",
            ] + _md_table(rows, include_protocol=True) + [""]

        t_master_md = root / f"master_protocols_leaderboard_{t_tag}.md"
        t_master_md.write_text("\n".join(t_md) + "\n", encoding="utf-8")

        # Also write lowercase (e.g. 30mk) for robust link/CLI resolution
        t_lower_tag = f"{int(round(t_mK))}mk"
        if t_lower_tag != t_tag:
            try:
                (root / f"master_protocols_leaderboard_{t_lower_tag}.md").write_text("\n".join(t_md) + "\n", encoding="utf-8")
            except Exception:
                pass

        t_csv_rows: List[Dict[str, Any]] = list(t_overall)
        for lbl in v0_labels:
            t_csv_rows.extend(t_per_v0[lbl])
        _write_csv(root / f"master_protocols_leaderboard_{t_tag}.csv", t_csv_rows)
        if t_lower_tag != t_tag:
            try:
                _write_csv(root / f"master_protocols_leaderboard_{t_lower_tag}.csv", t_csv_rows)
            except Exception:
                pass

    # Per-protocol leaderboards (pooled + per-temperature)
    for proto in sorted({i["protocol_name"] for i in instances}):
        p_inst = [i for i in instances if i["protocol_name"] == proto]
        p_overall = build_ranked_rows(p_inst)
        p_md = [
            f"# Leaderboard Summary: `{proto}`",
            "",
            f"_Rebuilt {stamp}. Ranking: {RANK_RULE}._",
            "",
        ]
        if temp_nav_banner:
            p_temp_links = [
                f"[{int(round(t))} mK Summary](leaderboard_summary_{int(round(t))}mK.md)"
                for t in temperatures
            ]
            p_md.extend(["> **Temperature Summaries**: " + " | ".join(p_temp_links), ""])

        p_md.extend([
            "## Overall (all disorder strengths pooled)",
            "",
        ] + _md_table(p_overall, include_protocol=False) + [""])
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

        # Per-temperature summary files inside each protocol directory
        for t_mK in temperatures:
            t_tag = f"{int(round(t_mK))}mK"
            p_t_overall = build_ranked_rows(p_inst, temperature_mK=t_mK)
            p_t_md = [
                f"# Leaderboard Summary: `{proto}` — {t_tag}",
                "",
                f"_Rebuilt {stamp} at $T = {int(round(t_mK))}$ mK. Ranking: {RANK_RULE}._",
                "",
                f"## Overall (all disorder strengths pooled at {t_tag})",
                "",
            ] + _md_table(p_t_overall, include_protocol=False) + [""]
            p_t_csv = list(p_t_overall)
            for lbl in v0_labels:
                rows = build_ranked_rows(p_inst, v0_label=lbl, temperature_mK=t_mK)
                if not rows:
                    continue
                p_t_md += [f"## Disorder strength $V_0 = {v0_label_to_mev(lbl)}$ meV ({t_tag})", ""]
                p_t_md += _md_table(rows, include_protocol=False) + [""]
                p_t_csv.extend(rows)
            (root / proto / f"leaderboard_summary_{t_tag}.md").write_text("\n".join(p_t_md) + "\n", encoding="utf-8")
            _write_csv(root / proto / f"leaderboard_summary_{t_tag}.csv", p_t_csv)
            t_lower_tag = f"{int(round(t_mK))}mk"
            if t_lower_tag != t_tag:
                try:
                    (root / proto / f"leaderboard_summary_{t_lower_tag}.md").write_text("\n".join(p_t_md) + "\n", encoding="utf-8")
                    _write_csv(root / proto / f"leaderboard_summary_{t_lower_tag}.csv", p_t_csv)
                except Exception:
                    pass

    return master_md
