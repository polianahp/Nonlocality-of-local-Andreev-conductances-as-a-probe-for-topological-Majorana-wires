"""High-Throughput Protocol Execution Engine, YAML Sweep Expander & Output Manager.

Implements:
- `BaseProtocol`: Base class for all protocols in `/home/pseudonym/Documents/Code/NonlocalProtocol/Protocols/<Protocol_Name>/protocol.py`.
- Strict recursive YAML parameter validation with `difflib.get_close_matches` typo detection.
- Cartesian `sweep_grid` and explicit `sweep_instances` YAML parameter expansion.
- Realization-Major multi-core execution (`ProcessPoolExecutor` scaling up to 256 cores on
  Harpers Ferry `#SBATCH -c 256 -p day_3gb`), computing continuous features once per
  `(realization, T)` in `RealizationFeatureCache` and evaluating all $M$ swept protocol
  instances in memory.
- Output management in `/home/pseudonym/Documents/Code/NonlocalProtocol/Outputs/Protocols/<Protocol_Name>/<Instance_Name>/`:
  - Always writes `config.yaml` (containing all resolved free parameters, `keep_data` flag,
    per-step waterfall metrics, and ensemble summary metrics).
  - Writes `step_data/ensemble_steps.npz` (compressed boolean arrays for every step in the
    protocol) **if and only if** `keep_data: true`.
"""

from __future__ import annotations

import copy
import csv
import difflib
import hashlib
import importlib.util
import itertools
import json
import subprocess
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import yaml

from .cache import RealizationFeatureCache
from .leaderboard import rebuild_leaderboards
from .metrics import aggregate_ensemble_metrics, evaluate_realization_metrics

import os
import re

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_default_data_root() -> Path:
    """Dynamically resolve default Disorder_Realizations path across local and cluster environments."""
    candidates = [
        REPO_ROOT / "Data" / "Disorder_Realizations",
        REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations",
    ]
    scratch_env = os.environ.get("SCRATCH", "/scratch/nos00002")
    if scratch_env:
        scratch = Path(scratch_env)
        candidates.extend([
            scratch / "nonlocalcond" / "Data" / "Disorder_Realizations",
            scratch / "nonlocalcond" / "Outputs" / "Data" / "Disorder_Realizations",
            scratch / "Data" / "Disorder_Realizations",
        ])
    for c in candidates:
        if c.exists() and any(c.glob("V0_*")):
            return c
    return (
        REPO_ROOT / "Data" / "Disorder_Realizations"
        if (REPO_ROOT / "Data" / "Disorder_Realizations").exists()
        else REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"
    )


DEFAULT_DATA_ROOT = _resolve_default_data_root()
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "Outputs" / "Protocols"


def _get_git_commit_hash() -> str:
    """Return current git commit hash of NonlocalProtocol repo for provenance."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _flatten_dict_keys(d: Dict[str, Any], prefix: str = "") -> List[str]:
    """Return dotted leaf keys of a nested dictionary."""
    keys: List[str] = []
    for k, v in d.items():
        full_k = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            keys.extend(_flatten_dict_keys(v, full_k))
        else:
            keys.append(full_k)
    return keys


def _set_dotted_key(
    target: Dict[str, Any],
    dotted_key: str,
    value: Any,
    valid_leaf_keys: Sequence[str],
) -> None:
    """Set a dotted key (e.g. `'fpe_corr.corr_threshold'`) on `target` with typo validation."""
    if dotted_key not in valid_leaf_keys:
        matches = difflib.get_close_matches(dotted_key, valid_leaf_keys, n=3, cutoff=0.5)
        hint = f" Did you mean: {matches}?" if matches else f" Valid keys: {list(valid_leaf_keys)}"
        raise ValueError(f"Unknown parameter key '{dotted_key}'.{hint}")
    parts = dotted_key.split(".")
    curr = target
    for p in parts[:-1]:
        curr = curr[p]
    curr[parts[-1]] = value


def _merge_and_validate_nested(
    defaults: Dict[str, Any],
    overrides: Dict[str, Any],
    valid_leaf_keys: Sequence[str],
    prefix: str = "",
) -> Dict[str, Any]:
    """Recursively merge `overrides` into a deep copy of `defaults`, rejecting unknown keys."""
    result = copy.deepcopy(defaults)
    for k, v in (overrides or {}).items():
        full_k = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _merge_and_validate_nested(
                result[k], v, valid_leaf_keys, prefix=full_k
            )
        elif "." in str(k):
            _set_dotted_key(result, str(k), v, valid_leaf_keys)
        else:
            if full_k not in valid_leaf_keys:
                matches = difflib.get_close_matches(full_k, valid_leaf_keys, n=3, cutoff=0.5)
                hint = (
                    f" Did you mean: {matches}?"
                    if matches
                    else f" Valid keys: {list(valid_leaf_keys)}"
                )
                raise ValueError(f"Unknown parameter key '{full_k}'.{hint}")
            result[k] = v
    return result


def _format_val_slug(val: Any) -> str:
    """Format a parameter value into a filesystem-safe slug token."""
    if isinstance(val, bool):
        return "T" if val else "F"
    if isinstance(val, float):
        s = f"{val:.4g}".replace("-", "m").replace(".", "p")
        return s
    if isinstance(val, (list, tuple)):
        return "x".join(_format_val_slug(x) for x in val)
    return str(val).replace(" ", "_").replace("/", "_")


def build_instance_slug(
    yaml_stem: str,
    resolved_params: Dict[str, Any],
    swept_overrides: Optional[Dict[str, Any]] = None,
    explicit_name: Optional[str] = None,
) -> str:
    """Construct a deterministic, human-readable, collision-proof instance directory name."""
    canonical_json = json.dumps(resolved_params, sort_keys=True, default=str)
    short_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:8]

    if explicit_name:
        clean_explicit = explicit_name.strip().replace(" ", "_").replace("/", "_")
        return clean_explicit

    if not swept_overrides:
        return f"{yaml_stem}__{short_hash}"

    tokens: List[str] = [yaml_stem]
    for dotted_k, val in swept_overrides.items():
        short_k = dotted_k.split(".")[-1]
        # Abbreviate common long parameter names for clean directory slugs
        abbrev_map = {
            "corr_threshold": "corr",
            "curvature_threshold": "curv3w",
            "max_abs_2w": "max2w",
            "min_island_size": "area",
            "use_core_filter": "core",
            "core_mode": "cmode",
            "tau": "tau",
            "k_min": "kmin",
            "min_gap_mev": "gap",
            "radius": "rad",
            "t_N": "tN",
            "wiring_mode": "wire",
            "mode": "mode",
            "lead": "lead",
        }
        label = abbrev_map.get(short_k, short_k[:10])
        tokens.append(f"{label}_{_format_val_slug(val)}")
    tokens.append(f"h{short_hash[:6]}")
    slug = "__".join(tokens)
    return slug[:220]


@dataclass
class ProtocolInstanceSpec:
    """Fully resolved specification for a single protocol parameter instance."""

    protocol_name: str
    protocol_file: str
    instance_name: str
    yaml_source_path: str
    keep_data: bool
    free_parameters: Dict[str, Any]
    swept_overrides: Dict[str, Any]


class BaseProtocol:
    """Abstract base class for all protocols in `/home/pseudonym/Documents/Code/NonlocalProtocol/Protocols/`."""

    name: str = "Base_Protocol"
    description: str = "Abstract base protocol."
    default_parameters: Dict[str, Any] = {}

    @classmethod
    def get_valid_leaf_keys(cls) -> List[str]:
        return _flatten_dict_keys(cls.default_parameters)

    @classmethod
    def resolve_and_validate_params(
        cls,
        user_params: Optional[Dict[str, Any]] = None,
        swept_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate `user_params` and `swept_overrides` against `cls.default_parameters`."""
        valid_keys = cls.get_valid_leaf_keys()
        merged = _merge_and_validate_nested(
            cls.default_parameters, user_params or {}, valid_keys
        )
        if swept_overrides:
            for dotted_k, val in swept_overrides.items():
                _set_dotted_key(merged, dotted_k, val, valid_keys)
        return merged

    def evaluate_steps(
        self,
        cache: RealizationFeatureCache,
        params: Dict[str, Any],
    ) -> OrderedDict[str, np.ndarray]:
        """Execute the ordered FPE and DPE steps and return `OrderedDict[step_name, mask_array]`.

        Must end with the canonical terminal post-processing step $C_4$ (`dpe_canonical_post_processor`)
        returning a Rank-2 `(N_mu, N_Vz)` boolean array.
        """
        raise NotImplementedError("Subclasses must implement evaluate_steps(cache, params).")


def load_protocol_class_from_file(protocol_py_path: Union[str, Path]) -> Type[BaseProtocol]:
    """Dynamically load the `Protocol` class from `Protocols/<Name>/protocol.py`."""
    p = Path(protocol_py_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Protocol script not found: {p}")
    spec = importlib.util.spec_from_file_location(f"dynamic_protocol_{p.parent.name}", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "Protocol") and issubclass(module.Protocol, BaseProtocol):
        return module.Protocol
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, BaseProtocol) and obj is not BaseProtocol:
            return obj
    raise AttributeError(f"No BaseProtocol subclass found in {p}")


def expand_yaml_to_instance_specs(
    protocol_py_path: Union[str, Path],
    yaml_path: Union[str, Path],
    keep_data_override: Optional[bool] = None,
) -> List[ProtocolInstanceSpec]:
    """Expand a single YAML file in `Protocols/<Name>/Parameters/*.yaml` into `ProtocolInstanceSpec`s."""
    p_cls = load_protocol_class_from_file(protocol_py_path)
    y_path = Path(yaml_path).resolve()
    with open(y_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    keep_data = (
        bool(keep_data_override)
        if keep_data_override is not None
        else bool(raw_cfg.get("keep_data", False))
    )
    base_params = raw_cfg.get("parameters", {})
    sweep_grid = raw_cfg.get("sweep_grid", {})
    sweep_instances = raw_cfg.get("sweep_instances", [])
    explicit_instance_name = raw_cfg.get("instance_name", None)
    yaml_stem = y_path.stem

    specs: List[ProtocolInstanceSpec] = []

    if sweep_instances:
        for idx, item in enumerate(sweep_instances):
            inst_overrides = item.get("parameters", item)
            inst_name = item.get("instance_name", None)
            inst_keep = (
                bool(keep_data_override)
                if keep_data_override is not None
                else bool(item.get("keep_data", keep_data))
            )
            resolved = p_cls.resolve_and_validate_params(base_params, inst_overrides)
            slug = build_instance_slug(
                yaml_stem,
                resolved,
                swept_overrides=inst_overrides,
                explicit_name=inst_name,
            )
            specs.append(
                ProtocolInstanceSpec(
                    protocol_name=p_cls.name,
                    protocol_file=str(Path(protocol_py_path).resolve()),
                    instance_name=slug,
                    yaml_source_path=str(y_path),
                    keep_data=inst_keep,
                    free_parameters=resolved,
                    swept_overrides=dict(inst_overrides),
                )
            )
        return specs

    if sweep_grid:
        grid_keys = list(sweep_grid.keys())
        grid_values = [
            v if isinstance(v, (list, tuple)) else [v] for v in sweep_grid.values()
        ]
        for combo in itertools.product(*grid_values):
            swept_dict = dict(zip(grid_keys, combo))
            resolved = p_cls.resolve_and_validate_params(base_params, swept_dict)
            slug = build_instance_slug(yaml_stem, resolved, swept_overrides=swept_dict)
            specs.append(
                ProtocolInstanceSpec(
                    protocol_name=p_cls.name,
                    protocol_file=str(Path(protocol_py_path).resolve()),
                    instance_name=slug,
                    yaml_source_path=str(y_path),
                    keep_data=keep_data,
                    free_parameters=resolved,
                    swept_overrides=swept_dict,
                )
            )
        return specs

    # Single parameter instance
    resolved = p_cls.resolve_and_validate_params(base_params, None)
    slug = build_instance_slug(
        yaml_stem, resolved, swept_overrides=None, explicit_name=explicit_instance_name
    )
    specs.append(
        ProtocolInstanceSpec(
            protocol_name=p_cls.name,
            protocol_file=str(Path(protocol_py_path).resolve()),
            instance_name=slug,
            yaml_source_path=str(y_path),
            keep_data=keep_data,
            free_parameters=resolved,
            swept_overrides={},
        )
    )
    return specs


def _realization_sort_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.name)
    return int(m.group(1)) if m else 0


def discover_realization_tasks(
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    v0_filter: Optional[Sequence[str]] = None,
    temperatures_mK: Sequence[float] = (30.0,),
    max_realizations_per_v0: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """Discover `(realization_dir, v0_label, temperature_mK)` tasks under `data_root`."""
    root = Path(data_root)
    if not root.exists():
        fallback = _resolve_default_data_root()
        if fallback.exists():
            root = fallback
        else:
            raise FileNotFoundError(f"Data root not found: {root} (fallback checked: {fallback})")

    tasks: List[Tuple[str, str, float]] = []
    if root.is_dir() and root.name.startswith("V0_"):
        v0_dirs = [root]
    else:
        v0_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("V0_")])
    for v0_dir in v0_dirs:
        if v0_filter and v0_dir.name not in v0_filter:
            continue
        r_dirs = sorted(
            [
                d
                for d in v0_dir.iterdir()
                if d.is_dir() and "realization" in d.name and (d / "all_params.npz").exists()
            ],
            key=_realization_sort_key,
        )
        if max_realizations_per_v0 is not None:
            r_dirs = r_dirs[: int(max_realizations_per_v0)]
        for t_mK in temperatures_mK:
            for r_dir in r_dirs:
                tasks.append((str(r_dir), v0_dir.name, float(t_mK)))
    return tasks


def _evaluate_single_realization_worker(
    task_args: Tuple[str, str, float, List[ProtocolInstanceSpec]],
) -> Dict[str, Any]:
    """Worker function for Realization-Major parallel execution across up to 256 cores.

    Loads `RealizationFeatureCache` ONCE for `(realization_dir, v0_label, T_mK)` and evaluates
    all `ProtocolInstanceSpec`s in memory, sharing memoized continuous observables across all
    parameter instances and protocols.
    """
    r_dir_str, v0_label, t_mK, specs = task_args
    cache = RealizationFeatureCache(r_dir_str, temperature_mK=t_mK, v0_label=v0_label)

    # Cache loaded protocol instances by protocol_file
    proto_objs: Dict[str, BaseProtocol] = {}
    per_instance_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for spec in specs:
        if spec.protocol_file not in proto_objs:
            p_cls = load_protocol_class_from_file(spec.protocol_file)
            proto_objs[spec.protocol_file] = p_cls()
        proto = proto_objs[spec.protocol_file]

        steps = proto.evaluate_steps(cache, spec.free_parameters)
        metrics_rec = evaluate_realization_metrics(cache, steps)

        step_arrays_to_keep: Optional[Dict[str, np.ndarray]] = None
        if spec.keep_data:
            step_arrays_to_keep = {
                k: np.asarray(v, dtype=bool) for k, v in steps.items()
            }

        per_instance_results[(spec.protocol_name, spec.instance_name)] = {
            "metrics": metrics_rec,
            "step_arrays": step_arrays_to_keep,
        }

    return {
        "realization_key": f"{int(round(t_mK))}mK__{v0_label}__{cache.realization_name}",
        "mu_vals": cache.mu_vals,
        "vz_vals": cache.vz_vals,
        "pfaffian_mask": cache.pfaffian_mask if any(s.keep_data for s in specs) else None,
        "results": per_instance_results,
    }


def run_protocol_specs(
    specs: Sequence[ProtocolInstanceSpec],
    *,
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    outputs_root: Union[str, Path] = DEFAULT_OUTPUTS_ROOT,
    v0_filter: Optional[Sequence[str]] = None,
    temperatures_mK: Sequence[float] = (30.0,),
    max_realizations_per_v0: Optional[int] = None,
    max_workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Execute a collection of `ProtocolInstanceSpec`s and write outputs to `Outputs/Protocols/`."""
    if not specs:
        return {}

    tasks = discover_realization_tasks(
        data_root=data_root,
        v0_filter=v0_filter,
        temperatures_mK=temperatures_mK,
        max_realizations_per_v0=max_realizations_per_v0,
    )
    if not tasks:
        raise RuntimeError(
            f"No realization tasks found in {data_root} for v0_filter={v0_filter}, T={temperatures_mK}"
        )

    specs_list = list(specs)
    worker_inputs = [(r_dir, v0_lbl, t_mK, specs_list) for (r_dir, v0_lbl, t_mK) in tasks]

    eff_workers = max(1, min(int(max_workers), len(worker_inputs)))
    if eff_workers == 1:
        raw_outputs = [_evaluate_single_realization_worker(inp) for inp in worker_inputs]
    else:
        with ProcessPoolExecutor(max_workers=eff_workers) as pool:
            raw_outputs = list(pool.map(_evaluate_single_realization_worker, worker_inputs))

    git_hash = _get_git_commit_hash()
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_root = Path(outputs_root)

    summary_by_instance: Dict[str, Dict[str, Any]] = {}
    specs_by_protocol: Dict[str, List[ProtocolInstanceSpec]] = {}
    for s in specs_list:
        specs_by_protocol.setdefault(s.protocol_name, []).append(s)

    for spec in specs_list:
        key = (spec.protocol_name, spec.instance_name)
        new_metrics = [ro["results"][key]["metrics"] for ro in raw_outputs]

        inst_dir = out_root / spec.protocol_name / spec.instance_name
        inst_dir.mkdir(parents=True, exist_ok=True)
        metrics_json_path = inst_dir / "realization_metrics.json"

        # Merge with existing metrics on disk so multi-V0 or sequential runs accumulate rather than overwrite
        merged_records: Dict[Tuple[float, str, str], Dict[str, Any]] = OrderedDict()
        if metrics_json_path.exists():
            try:
                with open(metrics_json_path, "r", encoding="utf-8") as f:
                    old_recs = json.load(f)
                for rec in old_recs:
                    t_k = round(float(rec.get("temperature_mK", 30.0)), 4)
                    v_k = str(rec.get("v0_label", ""))
                    r_k = str(rec.get("realization_name", ""))
                    merged_records[(t_k, v_k, r_k)] = rec
            except Exception:
                pass

        for rec in new_metrics:
            t_k = round(float(rec.get("temperature_mK", 30.0)), 4)
            v_k = str(rec.get("v0_label", ""))
            r_k = str(rec.get("realization_name", ""))
            merged_records[(t_k, v_k, r_k)] = rec

        r_metrics = list(merged_records.values())
        ensemble_summary = aggregate_ensemble_metrics(r_metrics)

        step_names = list(r_metrics[0]["step_diagnostics"].keys()) if r_metrics else []

        saved_step_data_files: List[str] = []
        if spec.keep_data:
            step_data_dir = inst_dir / "step_data"
            step_data_dir.mkdir(parents=True, exist_ok=True)

            # Group raw_outputs by grid shape / cohort (since V0_0_378 is 200x70 and V0_0_645+ is 225x70)
            cohort_groups: Dict[str, List[Dict[str, Any]]] = {}
            for ro in raw_outputs:
                parts = ro["realization_key"].split("__")
                cohort_key = "__".join(parts[:2]) if len(parts) >= 2 else "default"
                cohort_groups.setdefault(cohort_key, []).append(ro)

            for cohort_key, c_outputs in sorted(cohort_groups.items()):
                npz_path = step_data_dir / f"ensemble_{cohort_key}_steps.npz"
                r_keys = np.array([ro["realization_key"] for ro in c_outputs])
                mu_vals = c_outputs[0]["mu_vals"]
                vz_vals = c_outputs[0]["vz_vals"]
                pf_masks = np.stack([ro["pfaffian_mask"] for ro in c_outputs], axis=0)

                npz_dict: Dict[str, np.ndarray] = {
                    "realization_keys": r_keys,
                    "mu_vals": mu_vals,
                    "vz_vals": vz_vals,
                    "pfaffian_masks": pf_masks,
                }
                for sname in step_names:
                    npz_dict[sname] = np.stack(
                        [ro["results"][key]["step_arrays"][sname] for ro in c_outputs], axis=0
                    )
                np.savez_compressed(npz_path, **npz_dict)
                saved_step_data_files.append(str(npz_path.relative_to(inst_dir)))

            # If all cohorts share the same shape, also save unified ensemble_steps.npz
            shapes = {ro["pfaffian_mask"].shape for ro in raw_outputs}
            if len(shapes) == 1:
                unified_path = step_data_dir / "ensemble_steps.npz"
                r_keys = np.array([ro["realization_key"] for ro in raw_outputs])
                npz_dict = {
                    "realization_keys": r_keys,
                    "mu_vals": raw_outputs[0]["mu_vals"],
                    "vz_vals": raw_outputs[0]["vz_vals"],
                    "pfaffian_masks": np.stack([ro["pfaffian_mask"] for ro in raw_outputs], axis=0),
                }
                for sname in step_names:
                    npz_dict[sname] = np.stack(
                        [ro["results"][key]["step_arrays"][sname] for ro in raw_outputs], axis=0
                    )
                np.savez_compressed(unified_path, **npz_dict)
                saved_step_data_files.insert(0, str(unified_path.relative_to(inst_dir)))

        config_payload = {
            "protocol_name": spec.protocol_name,
            "instance_name": spec.instance_name,
            "yaml_source_path": spec.yaml_source_path,
            "keep_data": bool(spec.keep_data),
            "step_data_archives": saved_step_data_files,
            "timestamp_utc": timestamp_utc,
            "git_commit_hash": git_hash,
            "swept_overrides": spec.swept_overrides,
            "free_parameters": spec.free_parameters,
            "step_names": step_names,
            "summary_metrics": ensemble_summary,
        }

        # Write canonical config.yaml (and parameters_and_summary.yaml alias)
        config_yaml_path = inst_dir / "config.yaml"
        with open(config_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_payload, f, sort_keys=False, default_flow_style=False)

        with open(inst_dir / "realization_metrics.json", "w", encoding="utf-8") as f:
            json.dump(r_metrics, f, indent=2)

        summary_by_instance[f"{spec.protocol_name}/{spec.instance_name}"] = config_payload

    # Rebuild master + per-protocol leaderboards from ALL results on disk after every run
    # (overall + per-disorder-strength sections; ranked by Mean FDR then Mean TPR).
    rebuild_leaderboards(out_root)

    return summary_by_instance
