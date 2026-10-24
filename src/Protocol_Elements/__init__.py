"""Non-Local Protocol Elements Package (`FPE`, `DPE`, `RealizationFeatureCache`, `BaseProtocol`).

Provides the canonical mathematical algebra and high-throughput execution engine for
Majorana topological gap and conductance asymmetry protocols.
"""

from . import DPE, FPE
from .cache import RealizationFeatureCache, get_linear_broaden_and_spline_operators
from .DPE import DPE_REGISTRY
from .engine import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    BaseProtocol,
    ProtocolInstanceSpec,
    discover_realization_tasks,
    expand_yaml_to_instance_specs,
    load_protocol_class_from_file,
    run_protocol_specs,
)
from .FPE import FPE_REGISTRY
from .leaderboard import rebuild_leaderboards
from .metrics import (
    aggregate_ensemble_metrics,
    compute_step_diagnostic,
    evaluate_realization_metrics,
    extract_island_medoids_and_areas,
)

__all__ = [
    "FPE",
    "DPE",
    "FPE_REGISTRY",
    "DPE_REGISTRY",
    "RealizationFeatureCache",
    "get_linear_broaden_and_spline_operators",
    "BaseProtocol",
    "ProtocolInstanceSpec",
    "load_protocol_class_from_file",
    "expand_yaml_to_instance_specs",
    "discover_realization_tasks",
    "run_protocol_specs",
    "rebuild_leaderboards",
    "evaluate_realization_metrics",
    "aggregate_ensemble_metrics",
    "compute_step_diagnostic",
    "extract_island_medoids_and_areas",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUTS_ROOT",
]
