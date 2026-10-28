#!/usr/bin/env python3
"""High-Throughput CLI & Multi-Core Runner for `/home/pseudonym/Documents/Code/NonlocalProtocol/Protocols/`.

Discovers protocols and their `Parameters/*.yaml` configurations, expands parameter sweeps,
executes Realization-Major evaluation (shared `RealizationFeatureCache` per realization across
all protocol instances, scaling up to 256 cores on Harpers Ferry), and writes structured
outputs to `/home/pseudonym/Documents/Code/NonlocalProtocol/Outputs/Protocols/<Protocol_Name>/<Instance_Name>/`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

from src.Protocol_Elements import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    ProtocolInstanceSpec,
    expand_yaml_to_instance_specs,
    run_protocol_specs,
)

REPO_ROOT = Path(__file__).resolve().parent
PROTOCOLS_ROOT = REPO_ROOT / "Protocols"


def collect_protocol_specs(
    protocols_root: Path = PROTOCOLS_ROOT,
    protocol_names: Optional[List[str]] = None,
    yaml_filter: Optional[str] = None,
    keep_data_override: Optional[bool] = None,
) -> List[ProtocolInstanceSpec]:
    """Collect and expand all matching `Protocols/<Protocol_Name>/Parameters/*.yaml` files."""
    specs: List[ProtocolInstanceSpec] = []
    p_dirs = sorted([d for d in protocols_root.iterdir() if d.is_dir()])
    for p_dir in p_dirs:
        if protocol_names and p_dir.name not in protocol_names:
            continue
        proto_py = p_dir / "protocol.py"
        params_dir = p_dir / "Parameters"
        if not proto_py.exists() or not params_dir.exists():
            continue
        yaml_files = sorted(list(params_dir.glob("*.yaml")) + list(params_dir.glob("*.yml")))
        for y_path in yaml_files:
            if yaml_filter and yaml_filter not in y_path.name:
                continue
            specs.extend(
                expand_yaml_to_instance_specs(
                    proto_py, y_path, keep_data_override=keep_data_override
                )
            )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute Non-Local Protocol FPE/DPE suite across disorder realizations."
    )
    parser.add_argument(
        "--protocols",
        nargs="*",
        default=None,
        help="Optional list of protocol folder names inside Protocols/ (default: all).",
    )
    parser.add_argument(
        "--yaml-filter",
        type=str,
        default=None,
        help="Optional substring filter for YAML filenames inside Parameters/.",
    )
    parser.add_argument(
        "--v0",
        nargs="*",
        default=None,
        help="Optional V0 folder filter, e.g., V0_0_645 V0_0_872.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[30.0],
        help="Temperatures in mK to evaluate (e.g., 30.0 40.0).",
    )
    parser.add_argument(
        "--max-realizations-per-v0",
        type=int,
        default=None,
        help="Cap number of realizations per V0 for rapid local benchmarking.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (use 256 on Harpers Ferry).",
    )
    parser.add_argument(
        "--keep-data",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override keep_data flag across all YAML files ('auto' uses each YAML's flag).",
    )
    parser.add_argument(
        "--keep-instances",
        nargs="*",
        default=None,
        help="Explicit list of instance names that should save step_data (all others keep_data=False).",
    )
    parser.add_argument(
        "--keep-instances-file",
        type=str,
        default=None,
        help="Path to text/yaml file containing instance names that should save step_data.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Path to Disorder_Realizations root directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional folder name under Outputs/ (e.g., 'Microsoft_Hybrids_Post_Processing'). Overrides --outputs-root.",
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=str(DEFAULT_OUTPUTS_ROOT),
        help="Path to Outputs directory (default: Outputs/Protocols).",
    )
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        default=False,
        help="Force re-evaluation of all realization metrics and replace on-disk realization_metrics.json files.",
    )
    args = parser.parse_args()

    outputs_root_path = (
        REPO_ROOT / "Outputs" / args.output_name
        if args.output_name
        else Path(args.outputs_root)
    )

    keep_override: Optional[bool] = None
    if args.keep_data == "true":
        keep_override = True
    elif args.keep_data == "false":
        keep_override = False

    t0 = time.perf_counter()
    specs = collect_protocol_specs(
        protocols_root=PROTOCOLS_ROOT,
        protocol_names=args.protocols,
        yaml_filter=args.yaml_filter,
        keep_data_override=keep_override,
    )

    # Apply selective keep_data whitelist if provided
    whitelist: Optional[set] = None
    if args.keep_instances:
        whitelist = set(args.keep_instances)
    elif args.keep_instances_file:
        p = Path(args.keep_instances_file)
        if p.exists():
            with open(p, "r") as f:
                whitelist = {line.strip() for line in f if line.strip() and not line.strip().startswith("#")}
        else:
            raise FileNotFoundError(f"Keep instances file not found: {p}")

    if whitelist is not None:
        for s in specs:
            s.keep_data = s.instance_name in whitelist

    print(
        f"[Protocol Suite] Expanded {len(specs)} protocol instances across YAML configs. "
        f"Saving step arrays (keep_data=True) for {sum(1 for s in specs if s.keep_data)} instances."
    )

    summary_map = run_protocol_specs(
        specs,
        data_root=args.data_root,
        outputs_root=outputs_root_path,
        v0_filter=args.v0,
        temperatures_mK=args.temperatures,
        max_realizations_per_v0=args.max_realizations_per_v0,
        max_workers=args.workers,
        force_recalculate=args.force_recalculate,
    )
    elapsed = time.perf_counter() - t0

    # The master + per-protocol leaderboards are rebuilt from all on-disk results inside
    # run_protocol_specs (src/Protocol_Elements/leaderboard.py) after every run.
    out_root = outputs_root_path
    t_lines = []
    for t_mK in args.temperatures:
        t_tag = f"{int(round(t_mK))}mK"
        t_path = out_root / f"master_protocols_leaderboard_{t_tag}.md"
        t_lines.append(f"  - {t_tag} Leaderboard:   {t_path}")
    t_str = ("\n" + "\n".join(t_lines)) if t_lines else ""

    print(
        f"[Protocol Suite] Completed {len(specs)} instances in {elapsed:.2f}s.\n"
        f"  - Master (Pooled):    {out_root / 'master_protocols_leaderboard.md'}"
        f"{t_str}"
    )


if __name__ == "__main__":
    main()
