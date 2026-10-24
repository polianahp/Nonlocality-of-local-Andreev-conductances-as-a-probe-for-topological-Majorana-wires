r"""Logical Combinators & Multi-Lead / Multi-Cutter Consensus DPEs.

Derived Protocol Elements (DPEs) map binary spaces $\{0, 1\}^k \times \Theta \to \{0, 1\}^{N_\mu \times N_{V_z}}$.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import numpy as np


def _as_bool_mask(arr: np.ndarray, name: str = "mask") -> np.ndarray:
    """Convert an input array to a canonical boolean numpy array without NaNs."""
    a = np.asarray(arr)
    if a.dtype == bool:
        return a
    return np.nan_to_num(a, nan=0.0) > 0


def dpe_and(*masks: np.ndarray) -> np.ndarray:
    r"""Pointwise boolean conjunction $\bigwedge_{i=1}^k D_i$.

    Parameters
    ----------
    *masks : np.ndarray
        One or more boolean arrays of identical or broadcastable shape `(N_mu, N_Vz)`
        (or `(N_cutters, N_mu, N_Vz)`).

    Returns
    -------
    np.ndarray
        Boolean array with the logical AND of all inputs.
    """
    if not masks:
        raise ValueError("dpe_and requires at least one input mask.")
    out = _as_bool_mask(masks[0], "masks[0]").copy()
    for idx, m in enumerate(masks[1:], start=1):
        out = np.logical_and(out, _as_bool_mask(m, f"masks[{idx}]"))
    return out


def dpe_or(*masks: np.ndarray) -> np.ndarray:
    r"""Pointwise boolean disjunction $\bigvee_{i=1}^k D_i$.

    Parameters
    ----------
    *masks : np.ndarray
        One or more boolean arrays of identical or broadcastable shape.

    Returns
    -------
    np.ndarray
        Boolean array with the logical OR of all inputs.
    """
    if not masks:
        raise ValueError("dpe_or requires at least one input mask.")
    out = _as_bool_mask(masks[0], "masks[0]").copy()
    for idx, m in enumerate(masks[1:], start=1):
        out = np.logical_or(out, _as_bool_mask(m, f"masks[{idx}]"))
    return out


def dpe_not(mask: np.ndarray) -> np.ndarray:
    r"""Pointwise boolean negation $\neg D$."""
    return np.logical_not(_as_bool_mask(mask, "mask"))


def dpe_xor(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    r"""Pointwise boolean exclusive-OR $D_a \oplus D_b$ (Diagnostic One-Lead-Only gate)."""
    return np.logical_xor(_as_bool_mask(mask_a, "mask_a"), _as_bool_mask(mask_b, "mask_b"))


def dpe_lead_consensus(
    mask_L: np.ndarray,
    mask_R: np.ndarray,
    *,
    mode: Literal["both", "either", "xor", "L_only", "R_only", "neither"] = "both",
) -> np.ndarray:
    r"""Combine Left-lead and Right-lead binary masks $D_L, D_R \in \{0, 1\}^{d}$.

    Parameters
    ----------
    mask_L, mask_R : np.ndarray
        Binary masks for the Left and Right leads, shape `(N_mu, N_Vz)` or
        `(N_cutters, N_mu, N_Vz)`.
    mode : {"both", "either", "xor", "L_only", "R_only", "neither"}
        - `"both"`: $D_L \land D_R$ (Strict Both Leads consensus)
        - `"either"`: $D_L \lor D_R$ (Relaxed Either Lead consensus)
        - `"xor"`: $D_L \oplus D_R$ (Diagnostic asymmetric one-lead-only gate)
        - `"L_only"`: $D_L$
        - `"R_only"`: $D_R$
        - `"neither"`: $\neg(D_L \lor D_R)$

    Returns
    -------
    np.ndarray
        Combined boolean mask of the same shape as `mask_L` and `mask_R`.
    """
    mL = _as_bool_mask(mask_L, "mask_L")
    mR = _as_bool_mask(mask_R, "mask_R")
    if mL.shape != mR.shape:
        raise ValueError(f"dpe_lead_consensus shape mismatch: mask_L {mL.shape} vs mask_R {mR.shape}")
    if mode == "both":
        return np.logical_and(mL, mR)
    if mode == "either":
        return np.logical_or(mL, mR)
    if mode == "xor":
        return np.logical_xor(mL, mR)
    if mode == "L_only":
        return mL.copy()
    if mode == "R_only":
        return mR.copy()
    if mode == "neither":
        return np.logical_not(np.logical_or(mL, mR))
    raise ValueError(
        f"Unknown dpe_lead_consensus mode '{mode}'. "
        "Valid modes: 'both', 'either', 'xor', 'L_only', 'R_only', 'neither'."
    )


def dpe_k_of_n_cutter_consensus(
    cutter_mask_3d: np.ndarray,
    *,
    k_min: Optional[int] = 2,
    min_fraction: Optional[float] = None,
    cutter_axis: int = 0,
) -> np.ndarray:
    r"""Rank-reducing contraction DPE mapping `{0, 1}^{N_cutters x N_mu x N_Vz} -> {0, 1}^{N_mu x N_Vz}`.

    Evaluates $\sum_{c=1}^{N_c} D^{(c)}(\mu, V_z) \ge k_{\min}$ (or fraction $\ge f_{\min}$).

    Parameters
    ----------
    cutter_mask_3d : np.ndarray
        Boolean array of shape `(N_cutters, N_mu, N_Vz)` (or Rank-2 `(N_mu, N_Vz)` if
        already contracted, in which case `k_min <= 1` passes unchanged).
    k_min : int, optional
        Minimum integer number of passing cutters required (`1 <= k_min <= N_cutters`).
        Ignored if `min_fraction` is explicitly provided.
    min_fraction : float, optional
        If provided (in `(0.0, 1.0]`), requires `(sum_c D^(c)) / N_cutters >= min_fraction`.
    cutter_axis : int, default 0
        Axis corresponding to the cutter index in a 3D mask.

    Returns
    -------
    np.ndarray
        Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    """
    m = _as_bool_mask(cutter_mask_3d, "cutter_mask_3d")
    if m.ndim == 2:
        if min_fraction is not None:
            return m.copy() if min_fraction <= 1.0 else np.zeros_like(m, dtype=bool)
        if k_min is None or k_min <= 1:
            return m.copy()
        return np.zeros_like(m, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"dpe_k_of_n_cutter_consensus expects a 2D or 3D mask, got shape {m.shape}.")

    n_cutters = m.shape[cutter_axis]
    pass_count = np.sum(m, axis=cutter_axis)
    if min_fraction is not None:
        return (pass_count / float(max(n_cutters, 1))) >= (float(min_fraction) - 1e-12)
    if k_min is None:
        k_min = 1
    return pass_count >= int(k_min)


def evaluate_bipartite_lead_criterion(
    cache: Any,
    fpe_fn: Callable[..., np.ndarray],
    fpe_params: Dict[str, Any],
    dpe_lead_params: Dict[str, Any],
    *,
    lead_kwarg: str = "lead",
    prefix: str = "crit",
    steps_out: Optional[OrderedDict] = None,
) -> np.ndarray:
    """Helper unifying binary lead consensus (`dpe_lead_consensus`) and continuous lead pooling.

    Prevents keyword collisions on `lead` / `lead_mode` / `lead_pooling` and allows a YAML
    configuration to seamlessly switch a bipartite criterion between:
    - Binary consensus modes: `"both"`, `"either"`, `"xor"`, `"L_only"`, `"R_only"`, `"neither"`
    - Continuous FPE pooling mode: `"continuous_pool"` (calls `fpe_fn(cache, **fpe_params)` directly).
    """
    mode = dpe_lead_params.get("mode", "both")
    base_kwargs = {
        k: v for k, v in fpe_params.items() if k not in ("lead", "lead_mode", "lead_pooling")
    }

    mask_L = fpe_fn(cache, **{lead_kwarg: "L"}, **base_kwargs)
    mask_R = fpe_fn(cache, **{lead_kwarg: "R"}, **base_kwargs)

    if mode == "continuous_pool":
        combined = fpe_fn(cache, **fpe_params)
    else:
        combined = dpe_lead_consensus(mask_L, mask_R, mode=mode)

    if steps_out is not None:
        steps_out[f"{prefix}_fpe_L"] = mask_L
        steps_out[f"{prefix}_fpe_R"] = mask_R
        steps_out[f"{prefix}_lead_consensus"] = combined

    return combined
