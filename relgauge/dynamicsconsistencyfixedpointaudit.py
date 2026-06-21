"""
dynamicsconsistencyfixedpointaudit.py

Dynamics consistency fixed-point audit.

This module implements the inside-out fixed-point proposal:

    T -> boundary proliferation topology -> same-fiber chart atlas
      -> temporal chart face -> effective dynamics T_eff

and then iterates T <- T_eff on a finite sampled state universe.

It deliberately does *not* select dynamics by rewarding desired downstream
signals such as C2, worldlines, charge conservation, or Standard Model-like
numbers.  The only operation is self-consistency: the dynamics must reproduce
under the boundary-chart atlas it itself generates.  Random-start modes now include
generic full maps, global permutations, and local reversible shear programs so the
audit can distinguish irreversible randomness, global information conservation,
and boundary-factorable reversible dynamics.

Operational definition of T_eff
-------------------------------
For a current finite transition map on sampled states, the audit:

1. runs the boundary proliferation operator using that transition map;
2. builds a finite boundary-chart atlas over live fibers;
3. assigns each sampled microstate an atlas signature from generated domain
   labels and partial chart labels;
4. reads the temporal face as the relation

       atlas_signature(x) -> atlas_signature(Tx)

5. replaces each signature class by its deterministic majority target class,
   then lifts each target class to a canonical representative microstate.

This yields a new sampled transition map T_eff.  If T_eff == T, or if the
sequence of effective maps enters a short cycle, the dynamics/topology pair is
self-consistent at this sampled resolution.

Important caveat
----------------
The audit is finite and sampled.  T_eff is a sampled quotient-lift, not yet a
local rule table on the full q^k microscopic arena.  The point is to test the
principle "dynamics is the temporal face of the generated chart atlas" without
inserting an external scoring function.

Example
-------
Random-start derivation attempt, with no named rule family:

python -m relgauge.dynamicsconsistencyfixedpointaudit 2 ^
  --vertices 7 ^
  --rule-modes random_full_map,random_full_permutation,random_local_reversible,random_affine_bijection ^
  --instances 20 ^
  --proliferation-iterations 4 ^
  --fixedpoint-iterations 8 ^
  --horizon 3 ^
  --max-state-samples 512 ^
  --effective-lift-mode bijective ^
  --out example_results/dynamics_consistency_fp_q2_random.csv ^
  --plot example_results/fig_dynamics_consistency_fp_q2_random.png

Reference check against named rule families:

python -m relgauge.dynamicsconsistencyfixedpointaudit 2 ^
  --vertices 7 ^
  --rule-modes random_full_map,affine_mix,random_table,shift_xor ^
  --instances 5 ^
  --proliferation-iterations 4 ^
  --fixedpoint-iterations 6 ^
  --horizon 3 ^
  --max-state-samples 512 ^
  --out example_results/dynamics_consistency_fp_q2.csv ^
  --plot example_results/fig_dynamics_consistency_fp_q2.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import boundaryproliferationaudit as BPA
    from . import fiberchartconnectionaudit as FCA
except Exception:  # pragma: no cover
    import boundaryproliferationaudit as BPA  # type: ignore
    import fiberchartconnectionaudit as FCA  # type: ignore

State = Tuple[int, ...]

RANDOM_FULL_MAP_MODES = {
    "random_full_map",
    "random_state_map",
    "random_transition_map",
    "random_function",
    "random_global_map",
}
RANDOM_FULL_PERMUTATION_MODES = {
    "random_full_permutation",
    "random_permutation_map",
    "random_bijection",
    "random_global_permutation",
}
RANDOM_LOCAL_REVERSIBLE_MODES = {
    "random_local_reversible",
    "random_local_permutation",
    "random_reversible_local",
    "random_shear_reversible",
    "random_boundary_factorable_reversible",
    "random_local_shear",
}
RANDOM_AFFINE_BIJECTION_MODES = {
    "random_affine_bijection",
    "random_affine_permutation",
    "random_invertible_affine",
    "random_linear_bijection",
    "random_affine_reversible",
}
RANDOM_START_MODES = (
    RANDOM_FULL_MAP_MODES
    | RANDOM_FULL_PERMUTATION_MODES
    | RANDOM_LOCAL_REVERSIBLE_MODES
    | RANDOM_AFFINE_BIJECTION_MODES
)


def is_random_start_mode(mode: str) -> bool:
    return str(mode).strip().lower() in RANDOM_START_MODES


def random_start_kind(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in RANDOM_LOCAL_REVERSIBLE_MODES:
        return "random_local_reversible"
    if m in RANDOM_AFFINE_BIJECTION_MODES:
        return "random_affine_bijection"
    if m in RANDOM_FULL_PERMUTATION_MODES:
        return "random_full_permutation"
    if m in RANDOM_FULL_MAP_MODES:
        return "random_full_map"
    return "local_rule_mode"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)


def _parse_modes(text: str) -> List[str]:
    return [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]


def _hash_int_sequence(vals: Sequence[int]) -> str:
    h = hashlib.sha256()
    arr = np.asarray(vals, dtype=np.int64)
    h.update(arr.tobytes())
    return h.hexdigest()[:20]


def _hash_signature_map(src_to_tgt: Dict[int, int]) -> str:
    items = sorted((int(k), int(v)) for k, v in src_to_tgt.items())
    raw = json.dumps(items, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def transition_statistics(next_idx: Sequence[int], n_states: Optional[int] = None, prefix: str = "transition") -> Dict[str, object]:
    """Finite-map information-conservation diagnostics on the sampled universe.

    For a sampled transition map T: X -> X, bijectivity is the finite analogue of
    information conservation on X.  The audit uses these values only as diagnostics
    unless the user explicitly starts from random_full_permutation; it does not reward
    gauge, C2, charge, or matter.
    """
    arr = np.asarray(next_idx, dtype=np.int64)
    n = int(len(arr) if n_states is None else n_states)
    if n <= 0 or len(arr) == 0:
        return {
            f"{prefix}_image_size": 0,
            f"{prefix}_image_fraction": 0.0,
            f"{prefix}_injective": False,
            f"{prefix}_surjective": False,
            f"{prefix}_bijective": False,
            f"{prefix}_collision_fraction": 1.0,
            f"{prefix}_max_preimage_size": 0,
            f"{prefix}_mean_preimage_size_on_image": 0.0,
        }
    valid = arr[(arr >= 0) & (arr < n)]
    counts = Counter(int(x) for x in valid)
    image_size = int(len(counts))
    injective = bool(len(valid) == len(arr) and image_size == len(arr))
    surjective = bool(image_size == n)
    bijective = bool(len(arr) == n and injective and surjective)
    max_pre = int(max(counts.values(), default=0))
    mean_pre = float(np.mean(list(counts.values()))) if counts else 0.0
    return {
        f"{prefix}_image_size": int(image_size),
        f"{prefix}_image_fraction": float(image_size / max(1, n)),
        f"{prefix}_injective": bool(injective),
        f"{prefix}_surjective": bool(surjective),
        f"{prefix}_bijective": bool(bijective),
        f"{prefix}_collision_fraction": float(1.0 - image_size / max(1, len(arr))),
        f"{prefix}_max_preimage_size": int(max_pre),
        f"{prefix}_mean_preimage_size_on_image": float(mean_pre),
    }


def entropy_from_labels(labels: Sequence[int]) -> float:
    return float(BPA.entropy_of_labels(np.asarray(labels, dtype=np.int64))) if len(labels) else 0.0


def canonical_signature_labels(components: Sequence[np.ndarray]) -> np.ndarray:
    """Canonical labels for tuples of component labels."""
    if not components:
        return np.zeros(0, dtype=np.int32)
    n = len(components[0])
    tuples: List[Tuple[int, ...]] = []
    for i in range(n):
        tuples.append(tuple(int(comp[i]) for comp in components))
    return BPA.canonical_relabel(tuples)


@dataclass
class AtlasExtraction:
    domains_current: List[BPA.BoundaryDomain]
    domains_all: List[BPA.BoundaryDomain]
    domain_rows: List[Dict[str, object]]
    dependency_rows: List[Dict[str, object]]
    chart_rows: List[Dict[str, object]]
    chart_cycle_rows: List[Dict[str, object]]
    signature_labels: np.ndarray
    n_signature_classes: int
    signature_entropy_bits: float
    n_charts: int
    n_chart_cycles: int
    n_chart_valid_cycles: int
    n_chart_nontrivial: int
    n_chart_c2: int
    n_chart_c3: int
    max_chart_order: int
    n_dependency_edges: int
    dependency_beta1: int
    nontriviality_score: float
    architecture_hash: str


def _bounded_total_states(q: int, vertices: int, stop_above: int) -> Tuple[int, bool]:
    total = 1
    for _ in range(int(vertices)):
        total *= int(q)
        if total > int(stop_above):
            return int(total), False
    return int(total), True


def _all_states(q: int, vertices: int) -> List[State]:
    return [tuple(int(x) for x in vals) for vals in itertools.product(range(int(q)), repeat=int(vertices))]


def _sample_state_universe(q: int, vertices: int, n_samples: int, rng: np.random.Generator) -> List[State]:
    states: List[State] = []
    seen = set()
    attempts = 0
    target = int(max(1, n_samples))
    # Rejection sampling is fine at the tested scales; cap attempts for large spaces.
    while len(states) < target and attempts < target * 50:
        st = tuple(int(x) for x in rng.integers(0, int(q), size=int(vertices)))
        attempts += 1
        if st not in seen:
            seen.add(st); states.append(st)
    return states


def _random_context(k: int, target: int, rng: np.random.Generator, max_pred: int, local: bool = True) -> Tuple[int, ...]:
    """Choose a small context excluding the updated coordinate.

    For local reversible starts, contexts are biased toward ring-neighborhood
    coordinates.  The exclusion of target is essential: the elementary map

        x_target <- x_target + f(context)  (mod q)

    is then exactly invertible by subtracting f(context).
    """
    k = int(k); target = int(target)
    max_pred = max(1, min(int(max_pred), max(1, k - 1)))
    if k <= 1:
        return tuple()
    if local:
        candidates: List[int] = []
        radius = max(1, min(k // 2, max_pred + 1))
        for d in range(1, radius + 1):
            candidates.append((target - d) % k)
            candidates.append((target + d) % k)
        candidates = [c for c in dict.fromkeys(candidates) if c != target]
        if len(candidates) < max_pred:
            candidates += [i for i in range(k) if i != target and i not in candidates]
    else:
        candidates = [i for i in range(k) if i != target]
    if not candidates:
        return tuple()
    size = int(rng.integers(1, min(max_pred, len(candidates)) + 1))
    return tuple(sorted(int(x) for x in rng.choice(candidates, size=size, replace=False)))


def _make_shear_program(
    q: int,
    k: int,
    rng: np.random.Generator,
    max_pred: int,
    layers: Optional[int] = None,
    affine_only: bool = False,
    local: bool = True,
) -> List[Tuple[int, Tuple[int, ...], np.ndarray]]:
    """Build a reversible boundary-factorable transition program.

    Each elementary shear is a reversible local update:

        x_i <- x_i + table(x_context)  mod q,

    where context excludes i.  A composition of such shears is a permutation of
    the full state space for any finite q.  If affine_only is true, the table is
    generated from an affine form over Z_q; otherwise it is an arbitrary context
    table and can be nonlinear while remaining reversible.
    """
    q = int(q); k = int(k)
    if layers is None:
        layers = max(2 * k, 4)
    program: List[Tuple[int, Tuple[int, ...], np.ndarray]] = []
    for _ in range(int(layers)):
        order = list(range(k))
        rng.shuffle(order)
        for target in order:
            ctx = _random_context(k, target, rng, max_pred=max_pred, local=local)
            n_entries = q ** len(ctx)
            if n_entries <= 0:
                table = np.zeros(1, dtype=np.int16)
            elif affine_only:
                coeffs = rng.integers(0, q, size=len(ctx), dtype=np.int64)
                # Avoid all-zero affine table too often; a constant translation is
                # still reversible, but less boundary-informative.
                if len(ctx) and not np.any(coeffs % q):
                    coeffs[int(rng.integers(0, len(ctx)))] = int(rng.integers(1, q)) if q > 1 else 0
                bias = int(rng.integers(0, q))
                vals: List[int] = []
                for assn in itertools.product(range(q), repeat=len(ctx)):
                    vals.append(int((bias + sum(int(c) * int(a) for c, a in zip(coeffs, assn))) % q))
                table = np.asarray(vals, dtype=np.int16)
            else:
                table = rng.integers(0, q, size=n_entries, dtype=np.int16)
                # Ensure the shear is not identically zero.  Identically-zero
                # shears are allowed but uninformative; resample one entry.
                if n_entries and not np.any(table % q) and q > 1:
                    table[int(rng.integers(0, n_entries))] = int(rng.integers(1, q))
            program.append((int(target), tuple(ctx), table))
    return program


def _program_step_state(state: Sequence[int], q: int, program: Sequence[Tuple[int, Tuple[int, ...], np.ndarray]]) -> State:
    st = [int(x) % int(q) for x in state]
    q = int(q)
    for target, ctx, table in program:
        code = 0
        for c in ctx:
            code = code * q + int(st[int(c)])
        delta = int(table[int(code)]) if len(table) else 0
        st[int(target)] = int((st[int(target)] + delta) % q)
    return tuple(int(x) for x in st)


def _sampled_transition_from_state_function(states: Sequence[State], step_fn) -> Tuple[np.ndarray, bool, float]:
    """Build a sampled transition from a state map.

    Returns next_idx, all_targets_in_universe, missing_fraction.  For a full
    universe this is exact.  For a sampled universe, a target can fall outside the
    sample; those targets are projected to a deterministic nearest available state
    as a conservative fallback, and the missing fraction records the limitation.
    """
    state_to_idx = {tuple(st): i for i, st in enumerate(states)}
    next_idx: List[int] = []
    missing = 0
    for st in states:
        nxt = tuple(step_fn(st))
        j = state_to_idx.get(nxt)
        if j is None:
            missing += 1
            # Deterministic fallback: hash target to an available sampled state.
            raw = hashlib.sha256(repr(nxt).encode('utf-8')).digest()
            j = int.from_bytes(raw[:8], 'little') % max(1, len(states))
        next_idx.append(int(j))
    return np.asarray(next_idx, dtype=np.int64), bool(missing == 0), float(missing / max(1, len(states)))


def _shear_program_digest(program: Sequence[Tuple[int, Tuple[int, ...], np.ndarray]]) -> str:
    h = hashlib.sha256()
    for target, ctx, table in program:
        h.update(str(int(target)).encode())
        h.update(str(tuple(int(x) for x in ctx)).encode())
        h.update(np.asarray(table, dtype=np.int16).tobytes())
    return h.hexdigest()[:20]


def initialize_sampled_transition(
    q: int,
    vertices: int,
    mode: str,
    rng: np.random.Generator,
    max_state_samples: int,
    max_total_states: int,
    max_pred: int,
    proliferation_iterations: int,
    horizon: int,
) -> Tuple[List[State], np.ndarray, Dict[str, object]]:
    """Create the initial sampled dynamics.

    Named rule modes use the historical local-rule families.  The new random-start
    modes create an arbitrary transition map directly on the sampled/full state
    universe, so iteration starts from no structured rule family.
    """
    mode_s = str(mode).strip()
    mode_l = mode_s.lower()
    cap = int(max(1, max_total_states))
    if mode_l in RANDOM_START_MODES:
        total, is_small = _bounded_total_states(q, vertices, cap)
        if is_small:
            states = _all_states(q, vertices)
            universe_mode = "full"
            full_universe = True
        else:
            n = int(min(max(1, max_state_samples), cap))
            states = _sample_state_universe(q, vertices, n, rng)
            universe_mode = "sampled"
            full_universe = False
        n_states = int(len(states))
        if n_states <= 0:
            raise RuntimeError("random-start initialization produced no states")
        family = "unstructured_sampled_transition"
        missing_target_fraction = 0.0
        exact_state_function_on_universe = True
        program_hash = ""
        if mode_l in RANDOM_LOCAL_REVERSIBLE_MODES:
            program = _make_shear_program(q, vertices, rng, max_pred=max_pred, layers=max(2 * int(vertices), 6), affine_only=False, local=True)
            program_hash = _shear_program_digest(program)
            next_idx, exact_state_function_on_universe, missing_target_fraction = _sampled_transition_from_state_function(
                states, lambda st: _program_step_state(st, q, program)
            )
            family = "local_reversible_shear_program"
        elif mode_l in RANDOM_AFFINE_BIJECTION_MODES:
            program = _make_shear_program(q, vertices, rng, max_pred=max(vertices - 1, max_pred), layers=max(2 * int(vertices), 6), affine_only=True, local=False)
            program_hash = _shear_program_digest(program)
            next_idx, exact_state_function_on_universe, missing_target_fraction = _sampled_transition_from_state_function(
                states, lambda st: _program_step_state(st, q, program)
            )
            family = "affine_reversible_shear_program"
        elif mode_l in RANDOM_FULL_PERMUTATION_MODES:
            next_idx = np.asarray(rng.permutation(n_states), dtype=np.int64)
            family = "unstructured_sampled_permutation"
        else:
            next_idx = np.asarray(rng.integers(0, n_states, size=n_states), dtype=np.int64)
            family = "unstructured_sampled_transition"
        meta = {
            "initial_dynamics_kind": random_start_kind(mode_l),
            "initial_dynamics_family": family,
            "state_universe_mode": universe_mode,
            "state_universe_full": bool(full_universe),
            "state_universe_total_size_known_or_lower_bound": int(total),
            "initial_transition_hash": _hash_int_sequence(next_idx),
            "random_start_unstructured": True,
            "boundary_factorable_reversible_start": bool(mode_l in (RANDOM_LOCAL_REVERSIBLE_MODES | RANDOM_AFFINE_BIJECTION_MODES)),
            "local_reversible_start": bool(mode_l in RANDOM_LOCAL_REVERSIBLE_MODES),
            "affine_reversible_start": bool(mode_l in RANDOM_AFFINE_BIJECTION_MODES),
            "initial_program_hash": str(program_hash),
            "exact_state_function_on_sampled_universe": bool(exact_state_function_on_universe),
            "sampled_target_missing_fraction": float(missing_target_fraction),
        }
        meta.update(transition_statistics(next_idx, n_states=n_states, prefix="initial_transition"))
        return states, next_idx, meta

    dyn = BPA.make_dynamics(q, vertices, mode_s, rng, max_pred=int(max_pred))
    closure_steps = max(1, int(proliferation_iterations) * (int(horizon) + 1) + 2)
    initial_states = BPA.sample_initial_states(q, vertices, max_state_samples, rng)
    states, _state_to_idx, next_idx0 = BPA.build_state_closure(dyn, initial_states, closure_steps, max_total_states)
    n_states = len(states)
    current_next = np.asarray(next_idx0[:n_states], dtype=np.int64)
    total, is_small = _bounded_total_states(q, vertices, int(max_total_states))
    meta = {
        "initial_dynamics_kind": "local_rule_mode",
        "initial_dynamics_family": mode_s,
        "state_universe_mode": "closure",
        "state_universe_full": bool(is_small and n_states == total),
        "state_universe_total_size_known_or_lower_bound": int(total),
        "initial_transition_hash": _hash_int_sequence(current_next),
        "random_start_unstructured": False,
    }
    meta.update(transition_statistics(current_next, n_states=n_states, prefix="initial_transition"))
    return list(states), current_next, meta


# ---------------------------------------------------------------------------
# Proliferation replay on an arbitrary sampled transition map
# ---------------------------------------------------------------------------
def build_domains_from_transition(
    states: Sequence[State],
    next_idx: Sequence[int],
    q: int,
    rng: np.random.Generator,
    proliferation_iterations: int = 4,
    horizon: int = 3,
    initial_boundary: str = "sum_mod_q",
    initial_boundary_q: Optional[int] = None,
    max_domains_per_depth: int = 32,
    min_live_classes: int = 2,
    min_fiber_size: int = 2,
    min_entropy_bits: float = 0.05,
    synergy_threshold: float = 0.01,
) -> Tuple[List[BPA.BoundaryDomain], List[BPA.BoundaryDomain], List[Dict[str, object]], List[Dict[str, object]], str, int, int, float]:
    """Run the proliferation operator using a provided index transition map."""
    labels0 = BPA.initial_boundary_labels(states, int(q), initial_boundary, rng, boundary_q=initial_boundary_q)
    root = BPA.BoundaryDomain(0, 0, -1, "root", labels0, int(len(set(labels0))), BPA.entropy_of_labels(labels0), active=True)
    domains: List[BPA.BoundaryDomain] = [root]
    current: List[BPA.BoundaryDomain] = [root]
    domain_rows: List[Dict[str, object]] = []
    dep_rows: List[Dict[str, object]] = []
    last_hash = ""
    last_beta1 = 0
    last_edges = 0
    last_nontriv = 0.0

    for it in range(int(proliferation_iterations) + 1):
        root_future = BPA.canonical_relabel(BPA.future_label_history(root.labels, next_idx, horizon=horizon, include_current=False))
        dep_edges, mean_syn, beta1 = BPA.dependency_edges_for_domains(current, root_future, synergy_threshold)
        sig = BPA.architecture_signature(domains, current, dep_edges, beta1)
        last_hash = BPA.architecture_hash(sig)
        last_beta1 = int(beta1)
        last_edges = int(len(dep_edges))
        last_nontriv = float(BPA.compute_nontriviality(current, dep_edges, beta1))
        for d in current:
            domain_rows.append({
                "iteration": int(it),
                "domain_id": int(d.domain_id),
                "depth": int(d.depth),
                "parent_id": int(d.parent_id),
                "name": str(d.name),
                "n_labels": int(d.n_labels),
                "entropy_bits": float(d.entropy_bits),
                "live_fiber_count": int(d.live_fiber_count),
                "mean_predictive_classes_per_fiber": float(d.mean_predictive_classes_per_fiber),
                "max_predictive_classes_per_fiber": int(d.max_predictive_classes_per_fiber),
                "mean_fiber_entropy_bits": float(d.mean_fiber_entropy_bits),
                "active": bool(d.active),
            })
        for e in dep_edges:
            er = dict(e)
            er.update({"iteration": int(it), "dependency_mean_synergy_bits": float(mean_syn)})
            dep_rows.append(er)
        if it >= int(proliferation_iterations):
            break
        children: List[BPA.BoundaryDomain] = []
        for parent in current:
            for child in BPA.proliferate_children(
                parent, next_idx, horizon=horizon,
                min_live_classes=min_live_classes, min_fiber_size=min_fiber_size,
            ):
                if child.active and float(child.entropy_bits) >= float(min_entropy_bits):
                    child.domain_id = len(domains) + len(children)
                    children.append(child)
        children.sort(key=lambda d: (d.entropy_bits, d.live_fiber_count, d.n_labels), reverse=True)
        children = children[: int(max_domains_per_depth)]
        for c in children:
            c.domain_id = len(domains)
            domains.append(c)
        current = children
        if not current:
            break
    return current, domains, domain_rows, dep_rows, last_hash, last_edges, last_beta1, last_nontriv


# ---------------------------------------------------------------------------
# Chart atlas signature and temporal face
# ---------------------------------------------------------------------------
def _top_domains(domains: Sequence[BPA.BoundaryDomain], max_domains: int) -> List[BPA.BoundaryDomain]:
    ds = [d for d in domains if int(d.n_labels) >= 2 and float(d.entropy_bits) > 0]
    ds.sort(key=lambda d: (float(d.entropy_bits), int(d.live_fiber_count), int(d.n_labels)), reverse=True)
    return ds[: int(max_domains)]


def build_atlas_signature(
    domains_current: Sequence[BPA.BoundaryDomain],
    domains_all: Sequence[BPA.BoundaryDomain],
    states: Sequence[State],
    next_idx: Sequence[int],
    q: int,
    horizon: int,
    max_signature_domains: int = 16,
    max_parent_domains: int = 8,
    max_fibers_per_parent: int = 6,
    max_charts_per_fiber: int = 16,
    max_signature_charts: int = 48,
    max_chart_coords: int = 5,
    max_support_coords: int = 4,
    min_fiber_states: int = 8,
    min_support_states: int = 4,
    min_overlap_states: int = 4,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.05,
    max_cycle_len: int = 4,
    max_cycles_per_fiber: int = 500,
) -> Tuple[np.ndarray, List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    """Build an atlas signature over sampled states from generated domains/charts."""
    n = int(len(states))
    components: List[np.ndarray] = []
    chart_rows: List[Dict[str, object]] = []
    cycle_rows: List[Dict[str, object]] = []

    # Domain-face of the topology.
    sig_domains = _top_domains(list(domains_current) or list(domains_all), max_signature_domains)
    for d in sig_domains:
        components.append(np.asarray(d.labels, dtype=np.int64))

    # Chart-face: same-fiber boundary charts from high-entropy parent domains.
    charts_for_signature: List[FCA.FiberChart] = []
    parent_domains = _top_domains(list(domains_current) or list(domains_all), max_parent_domains)
    chart_global_id = 0
    n_cycles = n_valid = n_nontriv = n_c2 = n_c3 = max_order = 0
    for d in parent_domains:
        labs = np.asarray(d.labels, dtype=np.int64)
        counts = Counter(int(x) for x in labs)
        fiber_labs = [lab for lab, cnt in counts.most_common() if int(cnt) >= int(min_fiber_states)]
        for flab in fiber_labs[: int(max_fibers_per_parent)]:
            charts = FCA.build_charts_for_domain_fiber(
                d, int(flab), states, next_idx, int(q), int(horizon),
                max_chart_coords=max_chart_coords, max_support_coords=max_support_coords,
                max_charts_per_fiber=max_charts_per_fiber,
                min_chart_classes=min_chart_classes, min_chart_entropy=min_chart_entropy,
                min_support_states=min_support_states,
            )
            # Ensure chart ids are local for cycle computation but globally unique in output.
            local_charts: List[FCA.FiberChart] = []
            for ci, ch in enumerate(charts):
                ch2 = FCA.FiberChart(
                    chart_id=int(ci), parent_domain_id=ch.parent_domain_id, fiber_label=ch.fiber_label,
                    chart_type=ch.chart_type, support_desc=ch.support_desc, label_desc=ch.label_desc,
                    support_mask=ch.support_mask, labels_full=ch.labels_full, n_support=ch.n_support,
                    n_labels=ch.n_labels, entropy_bits=ch.entropy_bits,
                )
                local_charts.append(ch2)
            edge_rows, edge_maps = FCA.build_chart_transports(local_charts, min_overlap_states=min_overlap_states)
            crows = FCA.analyze_chart_cycles(local_charts, edge_maps, max_cycle_len=max_cycle_len, max_cycles=max_cycles_per_fiber)
            n_cycles += int(len(crows))
            n_valid += int(sum(1 for r in crows if bool(r.get("chart_loop_valid", False))))
            n_nontriv += int(sum(1 for r in crows if bool(r.get("chart_nontrivial_holonomy", False))))
            n_c2 += int(sum(1 for r in crows if bool(r.get("chart_c2_holonomy", False))))
            n_c3 += int(sum(1 for r in crows if bool(r.get("chart_c3_holonomy", False))))
            max_order = max(max_order, max([_safe_int(r.get("chart_loop_order"), 0) for r in crows], default=0))
            for ch in local_charts:
                row = {
                    "parent_domain_id": int(d.domain_id),
                    "fiber_label": int(flab),
                    "chart_id_local": int(ch.chart_id),
                    "chart_id_global": int(chart_global_id),
                    "chart_type": ch.chart_type,
                    "support_desc": ch.support_desc,
                    "label_desc": ch.label_desc,
                    "n_support": int(ch.n_support),
                    "n_labels": int(ch.n_labels),
                    "entropy_bits": float(ch.entropy_bits),
                }
                chart_rows.append(row)
                if len(charts_for_signature) < int(max_signature_charts):
                    # Copy with global id for debug; signature only needs labels.
                    charts_for_signature.append(ch)
                chart_global_id += 1
            for cr in crows:
                rr = dict(cr)
                rr.update({"parent_domain_id": int(d.domain_id), "fiber_label": int(flab)})
                cycle_rows.append(rr)

    # Add chart labels as partial components.  Sort by entropy/support for stable signatures.
    charts_for_signature.sort(key=lambda ch: (float(ch.entropy_bits), int(ch.n_support), int(ch.n_labels)), reverse=True)
    for ch in charts_for_signature[: int(max_signature_charts)]:
        components.append(np.asarray(ch.labels_full, dtype=np.int64))

    if not components:
        # Degenerate fallback: every state has same signature.
        sig_labels = np.zeros(n, dtype=np.int32)
    else:
        # Make sure all components have length n.
        components = [np.asarray(c[:n], dtype=np.int64) for c in components if len(c) >= n]
        sig_labels = canonical_signature_labels(components)

    stats = {
        "n_charts": int(len(chart_rows)),
        "n_chart_cycles": int(n_cycles),
        "n_chart_valid_cycles": int(n_valid),
        "n_chart_nontrivial": int(n_nontriv),
        "n_chart_c2": int(n_c2),
        "n_chart_c3": int(n_c3),
        "max_chart_order": int(max_order),
    }
    return sig_labels, chart_rows, cycle_rows, stats


def extract_effective_dynamics(
    signature_labels: Sequence[int],
    current_next_idx: Sequence[int],
    lift_mode: str = "representative",
) -> Tuple[np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    """Temporal face of atlas signature, lifted back to sampled microstates.

    lift_mode="representative" keeps the original quotient-lift behavior: every
    source signature class maps to one canonical representative of its majority
    target signature class.  This is the minimal observer-effective map, but it is
    generally many-to-one.

    lift_mode="bijective" or "permutation" keeps the same inferred signature map
    but lifts it to a permutation of the sampled microstate universe whenever
    possible.  It assigns source states to unused target states in the preferred
    target signature class, falling back to remaining states only when signature
    class sizes make exact conservative transport impossible.  This is the
    information-conserving temporal-face test.
    """
    sig = np.asarray(signature_labels, dtype=np.int64)
    nxt = np.asarray(current_next_idx, dtype=np.int64)
    n = int(min(len(sig), len(nxt)))
    sig = sig[:n]; nxt = nxt[:n]
    reps: Dict[int, int] = {}
    class_members: Dict[int, List[int]] = defaultdict(list)
    for i, s in enumerate(sig):
        reps.setdefault(int(s), int(i))
        class_members[int(s)].append(int(i))
    target_counts: Dict[int, Counter] = defaultdict(Counter)
    total_by_src: Counter = Counter()
    for i in range(n):
        s = int(sig[i])
        t = int(sig[int(nxt[i])]) if 0 <= int(nxt[i]) < n else int(sig[i])
        target_counts[s][t] += 1
        total_by_src[s] += 1
    src_to_tgt: Dict[int, int] = {}
    majority_good = 0
    relation_pairs = 0
    map_rows: List[Dict[str, object]] = []
    for s, cnt in sorted(target_counts.items()):
        relation_pairs += int(len(cnt))
        tgt, good = cnt.most_common(1)[0]
        src_to_tgt[int(s)] = int(tgt)
        majority_good += int(good)
        map_rows.append({
            "source_signature": int(s),
            "target_signature": int(tgt),
            "source_count": int(total_by_src[s]),
            "majority_count": int(good),
            "determinism_fraction": float(good / max(1, total_by_src[s])),
            "target_support": int(cnt.get(tgt, 0)),
            "n_possible_targets": int(len(cnt)),
        })

    mode = str(lift_mode or "representative").strip().lower()
    if mode in {"bijective", "permutation", "conservative", "information_conserving"}:
        eff = np.full(n, -1, dtype=np.int64)
        used: Set[int] = set()
        # Available targets by signature class, in canonical order.
        avail_by_sig: Dict[int, List[int]] = {int(k): list(v) for k, v in class_members.items()}
        preferred_hits = 0
        for i in range(n):
            desired_sig = int(src_to_tgt.get(int(sig[i]), int(sig[i])))
            bucket = avail_by_sig.get(desired_sig, [])
            chosen = None
            while bucket:
                cand = int(bucket.pop(0))
                if cand not in used:
                    chosen = cand
                    break
            if chosen is not None:
                preferred_hits += 1
            else:
                # Signature-class capacity mismatch.  Fill with the first unused state.
                for cand in range(n):
                    if cand not in used:
                        chosen = int(cand)
                        break
            if chosen is None:
                chosen = int(i)
            eff[i] = int(chosen)
            used.add(int(chosen))
        preferred_fraction = float(preferred_hits / max(1, n))
        effective_lift_bijective = True
    else:
        eff = np.zeros(n, dtype=np.int64)
        preferred_hits = 0
        for i in range(n):
            s = int(sig[i])
            tgt_sig = src_to_tgt.get(s, s)
            eff[i] = int(reps.get(int(tgt_sig), i))
            if int(sig[int(eff[i])]) == int(tgt_sig):
                preferred_hits += 1
        preferred_fraction = float(preferred_hits / max(1, n))
        effective_lift_bijective = bool(transition_statistics(eff, n_states=n, prefix="tmp")["tmp_bijective"])

    determinism = float(majority_good / max(1, n))
    micro_agreement = float(np.mean(eff == nxt)) if n else 0.0
    sig_agreement = float(np.mean(sig[eff] == sig[nxt])) if n else 0.0
    image_classes = len(set(int(src_to_tgt[s]) for s in src_to_tgt))
    stats = {
        "effective_lift_mode": str(mode),
        "effective_lift_bijective": bool(effective_lift_bijective),
        "effective_lift_preferred_signature_fraction": float(preferred_fraction),
        "effective_lift_exact_signature_flow": bool(preferred_fraction >= 1.0 - 1e-12),
        "signature_temporal_determinism": float(determinism),
        "micro_transition_agreement": float(micro_agreement),
        "signature_transition_agreement": float(sig_agreement),
        "effective_transition_changed_fraction": float(1.0 - micro_agreement),
        "n_signature_classes": int(len(reps)),
        "n_signature_image_classes": int(image_classes),
        "signature_relation_pair_count": int(relation_pairs),
        "signature_map_hash": _hash_signature_map(src_to_tgt),
        "effective_next_hash": _hash_int_sequence(eff),
    }
    return eff, stats, map_rows

def analyze_one_effective_step(
    states: Sequence[State],
    current_next_idx: Sequence[int],
    q: int,
    rng: np.random.Generator,
    proliferation_iterations: int,
    horizon: int,
    initial_boundary: str,
    initial_boundary_q: Optional[int],
    max_domains_per_depth: int,
    min_live_classes: int,
    min_fiber_size: int,
    min_entropy_bits: float,
    synergy_threshold: float,
    max_signature_domains: int,
    max_parent_domains: int,
    max_fibers_per_parent: int,
    max_charts_per_fiber: int,
    max_signature_charts: int,
    min_fiber_states: int,
    min_support_states: int,
    min_overlap_states: int,
    min_chart_classes: int,
    min_chart_entropy: float,
    max_chart_coords: int,
    max_support_coords: int,
    max_cycle_len: int,
    max_cycles_per_fiber: int,
    effective_lift_mode: str = "representative",
) -> Tuple[AtlasExtraction, np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    current, all_domains, domain_rows, dep_rows, arch_hash, n_edges, beta1, nontriv = build_domains_from_transition(
        states, current_next_idx, q, rng,
        proliferation_iterations=proliferation_iterations, horizon=horizon,
        initial_boundary=initial_boundary, initial_boundary_q=initial_boundary_q,
        max_domains_per_depth=max_domains_per_depth,
        min_live_classes=min_live_classes, min_fiber_size=min_fiber_size,
        min_entropy_bits=min_entropy_bits, synergy_threshold=synergy_threshold,
    )
    sig_labels, chart_rows, cycle_rows, chart_stats = build_atlas_signature(
        current, all_domains, states, current_next_idx, q, horizon,
        max_signature_domains=max_signature_domains, max_parent_domains=max_parent_domains,
        max_fibers_per_parent=max_fibers_per_parent, max_charts_per_fiber=max_charts_per_fiber,
        max_signature_charts=max_signature_charts, max_chart_coords=max_chart_coords,
        max_support_coords=max_support_coords, min_fiber_states=min_fiber_states,
        min_support_states=min_support_states, min_overlap_states=min_overlap_states,
        min_chart_classes=min_chart_classes, min_chart_entropy=min_chart_entropy,
        max_cycle_len=max_cycle_len, max_cycles_per_fiber=max_cycles_per_fiber,
    )
    eff, eff_stats, map_rows = extract_effective_dynamics(sig_labels, current_next_idx, lift_mode=effective_lift_mode)
    atlas = AtlasExtraction(
        domains_current=list(current), domains_all=list(all_domains), domain_rows=domain_rows,
        dependency_rows=dep_rows, chart_rows=chart_rows, chart_cycle_rows=cycle_rows,
        signature_labels=sig_labels,
        n_signature_classes=int(len(set(int(x) for x in sig_labels))),
        signature_entropy_bits=float(entropy_from_labels(sig_labels)),
        n_charts=int(chart_stats["n_charts"]),
        n_chart_cycles=int(chart_stats["n_chart_cycles"]),
        n_chart_valid_cycles=int(chart_stats["n_chart_valid_cycles"]),
        n_chart_nontrivial=int(chart_stats["n_chart_nontrivial"]),
        n_chart_c2=int(chart_stats["n_chart_c2"]),
        n_chart_c3=int(chart_stats["n_chart_c3"]),
        max_chart_order=int(chart_stats["max_chart_order"]),
        n_dependency_edges=int(n_edges), dependency_beta1=int(beta1),
        nontriviality_score=float(nontriv), architecture_hash=str(arch_hash),
    )
    return atlas, eff, eff_stats, map_rows


# ---------------------------------------------------------------------------
# Audit driver
# ---------------------------------------------------------------------------
def run_dynamics_consistency_fixedpoint_audit(
    q: int,
    vertices: int = 7,
    rule_modes: Sequence[str] = ("random_full_map",),
    instances: int = 3,
    fixedpoint_iterations: int = 6,
    proliferation_iterations: int = 4,
    horizon: int = 3,
    max_state_samples: int = 512,
    max_total_states: int = 4096,
    initial_boundary: str = "sum_mod_q",
    initial_boundary_q: Optional[int] = None,
    max_domains_per_depth: int = 32,
    min_live_classes: int = 2,
    min_fiber_size: int = 2,
    min_entropy_bits: float = 0.05,
    synergy_threshold: float = 0.01,
    max_pred: int = 3,
    max_signature_domains: int = 16,
    max_parent_domains: int = 8,
    max_fibers_per_parent: int = 6,
    max_charts_per_fiber: int = 16,
    max_signature_charts: int = 48,
    min_fiber_states: int = 8,
    min_support_states: int = 4,
    min_overlap_states: int = 4,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.05,
    max_chart_coords: int = 5,
    max_support_coords: int = 4,
    max_cycle_len: int = 4,
    max_cycles_per_fiber: int = 500,
    min_nontrivial_compression: float = 0.05,
    effective_lift_mode: str = "representative",
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[object, object, object, object, Dict[str, object]]:
    if pd is None:
        raise RuntimeError("pandas is required")
    q = int(q); vertices = int(vertices)
    rule_modes = list(rule_modes)
    rows: List[Dict[str, object]] = []
    domain_rows_all: List[Dict[str, object]] = []
    map_rows_all: List[Dict[str, object]] = []
    chart_cycle_rows_all: List[Dict[str, object]] = []

    for mi, mode in enumerate(rule_modes):
        for inst in range(int(instances)):
            seed_val = int(seed) + 1000003 * (mi + 1) + 7919 * inst
            rng = np.random.default_rng(seed_val)
            states, current_next, init_meta = initialize_sampled_transition(
                q=q, vertices=vertices, mode=str(mode), rng=rng,
                max_state_samples=max_state_samples, max_total_states=max_total_states,
                max_pred=max_pred, proliferation_iterations=proliferation_iterations, horizon=horizon,
            )
            n_states = len(states)
            seen: Dict[str, int] = {}
            best_det = 0.0
            min_change = 1.0
            fixed_found = False
            cycle_found = False
            for fp_it in range(int(fixedpoint_iterations) + 1):
                step_rng = np.random.default_rng(seed_val + 9176 * (fp_it + 1))
                atlas, eff_next, eff_stats, map_rows = analyze_one_effective_step(
                    states=states, current_next_idx=current_next, q=q, rng=step_rng,
                    proliferation_iterations=proliferation_iterations, horizon=horizon,
                    initial_boundary=initial_boundary, initial_boundary_q=initial_boundary_q,
                    max_domains_per_depth=max_domains_per_depth,
                    min_live_classes=min_live_classes, min_fiber_size=min_fiber_size,
                    min_entropy_bits=min_entropy_bits, synergy_threshold=synergy_threshold,
                    max_signature_domains=max_signature_domains, max_parent_domains=max_parent_domains,
                    max_fibers_per_parent=max_fibers_per_parent,
                    max_charts_per_fiber=max_charts_per_fiber,
                    max_signature_charts=max_signature_charts,
                    min_fiber_states=min_fiber_states, min_support_states=min_support_states,
                    min_overlap_states=min_overlap_states, min_chart_classes=min_chart_classes,
                    min_chart_entropy=min_chart_entropy, max_chart_coords=max_chart_coords,
                    max_support_coords=max_support_coords, max_cycle_len=max_cycle_len,
                    max_cycles_per_fiber=max_cycles_per_fiber,
                    effective_lift_mode=effective_lift_mode,
                )
                cur_hash = _hash_int_sequence(current_next)
                eff_hash = str(eff_stats.get("effective_next_hash", ""))
                repeated = cur_hash in seen
                repeat_period = int(fp_it - seen[cur_hash]) if repeated else 0
                seen.setdefault(cur_hash, fp_it)
                exact_fixed = bool(np.array_equal(eff_next, current_next))
                # Fixed at the observer/effective level if the temporal face is deterministic and no micro-lift changes.
                det = _safe_float(eff_stats.get("signature_temporal_determinism"), 0.0)
                change = _safe_float(eff_stats.get("effective_transition_changed_fraction"), 1.0)
                compression = float(1.0 - (float(atlas.n_signature_classes) / max(1.0, float(n_states))))
                nontrivial_atlas = bool(atlas.n_signature_classes > 1 and compression >= float(min_nontrivial_compression))
                best_det = max(best_det, det)
                min_change = min(min_change, change)
                fixed = bool(nontrivial_atlas and (exact_fixed or (det >= 0.999 and change <= 1e-12)))
                cyc = bool(nontrivial_atlas and repeated and repeat_period > 0)
                fixed_found = fixed_found or fixed
                cycle_found = cycle_found or (cyc and not fixed)
                current_info = transition_statistics(current_next, n_states=n_states, prefix="current_transition")
                effective_info = transition_statistics(eff_next, n_states=n_states, prefix="effective_transition")
                row = {
                    "q": int(q),
                    "vertices": int(vertices),
                    "rule_mode": str(mode),
                    "instance": int(inst),
                    "seed": int(seed_val),
                    "initial_dynamics_kind": str(init_meta.get("initial_dynamics_kind", "")),
                    "initial_dynamics_family": str(init_meta.get("initial_dynamics_family", str(mode))),
                    "state_universe_mode": str(init_meta.get("state_universe_mode", "")),
                    "state_universe_full": bool(init_meta.get("state_universe_full", False)),
                    "state_universe_total_size_known_or_lower_bound": int(init_meta.get("state_universe_total_size_known_or_lower_bound", 0)),
                    "initial_transition_hash": str(init_meta.get("initial_transition_hash", "")),
                    "random_start_unstructured": bool(init_meta.get("random_start_unstructured", False)),
                    "initial_transition_image_fraction": float(init_meta.get("initial_transition_image_fraction", 0.0)),
                    "initial_transition_injective": bool(init_meta.get("initial_transition_injective", False)),
                    "initial_transition_surjective": bool(init_meta.get("initial_transition_surjective", False)),
                    "initial_transition_bijective": bool(init_meta.get("initial_transition_bijective", False)),
                    "initial_transition_collision_fraction": float(init_meta.get("initial_transition_collision_fraction", 1.0)),
                    "boundary_factorable_reversible_start": bool(init_meta.get("boundary_factorable_reversible_start", False)),
                    "local_reversible_start": bool(init_meta.get("local_reversible_start", False)),
                    "affine_reversible_start": bool(init_meta.get("affine_reversible_start", False)),
                    "initial_program_hash": str(init_meta.get("initial_program_hash", "")),
                    "exact_state_function_on_sampled_universe": bool(init_meta.get("exact_state_function_on_sampled_universe", True)),
                    "sampled_target_missing_fraction": float(init_meta.get("sampled_target_missing_fraction", 0.0)),
                    "fixedpoint_iteration": int(fp_it),
                    "n_states": int(n_states),
                    "current_transition_hash": str(cur_hash),
                    "effective_transition_hash": str(eff_hash),
                    "architecture_hash": str(atlas.architecture_hash),
                    "repeated_transition_hash": bool(repeated),
                    "transition_repeat_period": int(repeat_period),
                    "exact_dynamics_fixed_point": bool(exact_fixed),
                    "dynamics_fixed_point_candidate": bool(fixed),
                    "dynamics_limit_cycle_candidate": bool(cyc and not fixed),
                    "n_domains_current": int(len(atlas.domains_current)),
                    "n_domains_all": int(len(atlas.domains_all)),
                    "n_dependency_edges": int(atlas.n_dependency_edges),
                    "dependency_beta1": int(atlas.dependency_beta1),
                    "topology_nontriviality_score": float(atlas.nontriviality_score),
                    "n_charts": int(atlas.n_charts),
                    "n_chart_cycles": int(atlas.n_chart_cycles),
                    "n_chart_valid_cycles": int(atlas.n_chart_valid_cycles),
                    "n_chart_nontrivial_holonomy": int(atlas.n_chart_nontrivial),
                    "n_chart_c2_holonomy": int(atlas.n_chart_c2),
                    "n_chart_c3_holonomy": int(atlas.n_chart_c3),
                    "max_chart_holonomy_order": int(atlas.max_chart_order),
                    "n_signature_classes": int(atlas.n_signature_classes),
                    "signature_entropy_bits": float(atlas.signature_entropy_bits),
                    "signature_compression_fraction": float(compression),
                    "nontrivial_atlas_for_dynamics": bool(nontrivial_atlas),
                    **current_info,
                    **effective_info,
                    **eff_stats,
                }
                rows.append(row)
                for dr in atlas.domain_rows:
                    dd = dict(dr)
                    dd.update({"q": q, "vertices": vertices, "rule_mode": mode, "instance": inst, "seed": seed_val, "fixedpoint_iteration": fp_it})
                    domain_rows_all.append(dd)
                for mr in map_rows:
                    mm = dict(mr)
                    mm.update({"q": q, "vertices": vertices, "rule_mode": mode, "instance": inst, "seed": seed_val, "fixedpoint_iteration": fp_it})
                    map_rows_all.append(mm)
                for cr in atlas.chart_cycle_rows:
                    cc = dict(cr)
                    cc.update({"q": q, "vertices": vertices, "rule_mode": mode, "instance": inst, "seed": seed_val, "fixedpoint_iteration": fp_it})
                    chart_cycle_rows_all.append(cc)
                if verbose:
                    print(
                        f"dyn-fp mode={mode} inst={inst+1}/{instances} fp={fp_it}/{fixedpoint_iterations} "
                        f"sig={atlas.n_signature_classes} det={det:.3f} change={change:.3f} "
                        f"fixed={int(fixed)} cyc={int(cyc)} charts={atlas.n_charts} c2={atlas.n_chart_c2}"
                    )
                if fp_it >= int(fixedpoint_iterations):
                    break
                current_next = np.asarray(eff_next, dtype=np.int64)
            # continue to next instance

    df = pd.DataFrame(rows)
    domain_df = pd.DataFrame(domain_rows_all)
    map_df = pd.DataFrame(map_rows_all)
    cycle_df = pd.DataFrame(chart_cycle_rows_all)
    any_fixed = bool(df.get("dynamics_fixed_point_candidate", pd.Series(dtype=bool)).any()) if not df.empty else False
    any_cycle = bool(df.get("dynamics_limit_cycle_candidate", pd.Series(dtype=bool)).any()) if not df.empty else False
    any_chart = bool((df.get("n_chart_nontrivial_holonomy", pd.Series(dtype=int)) > 0).any()) if not df.empty else False
    final = df.sort_values("fixedpoint_iteration").groupby(["rule_mode", "instance"]).tail(1) if not df.empty else df
    random_final = final[final.get("random_start_unstructured", False) == True] if not final.empty and "random_start_unstructured" in final.columns else final.iloc[0:0]
    random_all = df[df.get("random_start_unstructured", False) == True] if not df.empty and "random_start_unstructured" in df.columns else df.iloc[0:0]
    any_random_fixed = bool(random_all.get("dynamics_fixed_point_candidate", pd.Series(dtype=bool)).any()) if not random_all.empty else False
    any_random_cycle = bool(random_all.get("dynamics_limit_cycle_candidate", pd.Series(dtype=bool)).any()) if not random_all.empty else False
    any_random_chart = bool((random_all.get("n_chart_nontrivial_holonomy", pd.Series(dtype=int)) > 0).any()) if not random_all.empty else False
    any_random_gauge_fixed = bool(((random_all.get("dynamics_fixed_point_candidate", pd.Series(dtype=bool)) == True) & (random_all.get("n_chart_nontrivial_holonomy", pd.Series(dtype=int)) > 0)).any()) if not random_all.empty else False
    by_mode: List[Dict[str, object]] = []
    if not final.empty:
        for mode, g in final.groupby("rule_mode"):
            allg = df[df["rule_mode"] == mode]
            by_mode.append({
                "rule_mode": str(mode),
                "n": int(g["instance"].nunique()),
                "fixed_point_fraction": float(g["dynamics_fixed_point_candidate"].mean()),
                "limit_cycle_fraction": float(g["dynamics_limit_cycle_candidate"].mean()),
                "mean_final_signature_determinism": float(g["signature_temporal_determinism"].mean()),
                "mean_final_change_fraction": float(g["effective_transition_changed_fraction"].mean()),
                "min_change_fraction": float(allg["effective_transition_changed_fraction"].min()),
                "max_signature_determinism": float(allg["signature_temporal_determinism"].max()),
                "max_chart_c2_holonomy": int(allg["n_chart_c2_holonomy"].max()),
                "mean_final_signature_classes": float(g["n_signature_classes"].mean()),
                "mean_final_compression_fraction": float(g["signature_compression_fraction"].mean()),
                "random_start_unstructured_fraction": float(g["random_start_unstructured"].mean()) if "random_start_unstructured" in g.columns else 0.0,
                "initial_bijective_fraction": float(g["initial_transition_bijective"].mean()) if "initial_transition_bijective" in g.columns else 0.0,
                "boundary_factorable_reversible_fraction": float(g["boundary_factorable_reversible_start"].mean()) if "boundary_factorable_reversible_start" in g.columns else 0.0,
                "local_reversible_fraction": float(g["local_reversible_start"].mean()) if "local_reversible_start" in g.columns else 0.0,
                "affine_reversible_fraction": float(g["affine_reversible_start"].mean()) if "affine_reversible_start" in g.columns else 0.0,
                "mean_sampled_target_missing_fraction": float(g["sampled_target_missing_fraction"].mean()) if "sampled_target_missing_fraction" in g.columns else 0.0,
                "final_current_bijective_fraction": float(g["current_transition_bijective"].mean()) if "current_transition_bijective" in g.columns else 0.0,
                "final_effective_bijective_fraction": float(g["effective_transition_bijective"].mean()) if "effective_transition_bijective" in g.columns else 0.0,
                "mean_final_current_collision_fraction": float(g["current_transition_collision_fraction"].mean()) if "current_transition_collision_fraction" in g.columns else 0.0,
                "mean_final_effective_collision_fraction": float(g["effective_transition_collision_fraction"].mean()) if "effective_transition_collision_fraction" in g.columns else 0.0,
                "final_gauge_bearing_fraction": float((g["n_chart_nontrivial_holonomy"] > 0).mean()),
                "final_c2_bearing_fraction": float((g["n_chart_c2_holonomy"] > 0).mean()),
            })
    def _subset_summary(g):
        if g is None or g.empty:
            return {
                "instances": 0,
                "fixed_point_fraction": 0.0,
                "limit_cycle_fraction": 0.0,
                "final_gauge_bearing_fraction": 0.0,
                "final_c2_bearing_fraction": 0.0,
                "gauge_bearing_fixed_point_fraction": 0.0,
                "flat_fixed_point_fraction": 0.0,
                "unique_final_effective_attractors": 0,
                "mean_final_signature_classes": 0.0,
                "mean_final_compression_fraction": 0.0,
                "mean_final_current_collision_fraction": 0.0,
                "mean_final_effective_collision_fraction": 0.0,
            }
        return {
            "instances": int(g[["rule_mode", "instance"]].drop_duplicates().shape[0]),
            "fixed_point_fraction": float(g["dynamics_fixed_point_candidate"].mean()),
            "limit_cycle_fraction": float(g["dynamics_limit_cycle_candidate"].mean()),
            "final_gauge_bearing_fraction": float((g["n_chart_nontrivial_holonomy"] > 0).mean()),
            "final_c2_bearing_fraction": float((g["n_chart_c2_holonomy"] > 0).mean()),
            "gauge_bearing_fixed_point_fraction": float(((g["dynamics_fixed_point_candidate"] == True) & (g["n_chart_nontrivial_holonomy"] > 0)).mean()),
            "flat_fixed_point_fraction": float(((g["dynamics_fixed_point_candidate"] == True) & (g["n_chart_nontrivial_holonomy"] <= 0)).mean()),
            "unique_final_effective_attractors": int(g["effective_transition_hash"].nunique()),
            "mean_final_signature_classes": float(g["n_signature_classes"].mean()),
            "mean_final_compression_fraction": float(g["signature_compression_fraction"].mean()),
            "mean_final_current_collision_fraction": float(g["current_transition_collision_fraction"].mean()) if "current_transition_collision_fraction" in g.columns else 0.0,
            "mean_final_effective_collision_fraction": float(g["effective_transition_collision_fraction"].mean()) if "effective_transition_collision_fraction" in g.columns else 0.0,
        }

    invertible_random_final = random_final[random_final.get("initial_transition_bijective", False) == True] if not random_final.empty and "initial_transition_bijective" in random_final.columns else random_final.iloc[0:0]
    noninvertible_random_final = random_final[random_final.get("initial_transition_bijective", False) == False] if not random_final.empty and "initial_transition_bijective" in random_final.columns else random_final.iloc[0:0]
    factorable_random_final = random_final[random_final.get("boundary_factorable_reversible_start", False) == True] if not random_final.empty and "boundary_factorable_reversible_start" in random_final.columns else random_final.iloc[0:0]
    local_reversible_random_final = random_final[random_final.get("local_reversible_start", False) == True] if not random_final.empty and "local_reversible_start" in random_final.columns else random_final.iloc[0:0]
    affine_reversible_random_final = random_final[random_final.get("affine_reversible_start", False) == True] if not random_final.empty and "affine_reversible_start" in random_final.columns else random_final.iloc[0:0]
    invertible_summary = _subset_summary(invertible_random_final)
    noninvertible_summary = _subset_summary(noninvertible_random_final)
    factorable_summary = _subset_summary(factorable_random_final)
    local_reversible_summary = _subset_summary(local_reversible_random_final)
    affine_reversible_summary = _subset_summary(affine_reversible_random_final)
    invertible_gauge_fixed = bool(invertible_summary["gauge_bearing_fixed_point_fraction"] > 0)
    noninvertible_gauge_fixed = bool(noninvertible_summary["gauge_bearing_fixed_point_fraction"] > 0)
    factorable_gauge_fixed = bool(factorable_summary["gauge_bearing_fixed_point_fraction"] > 0)
    factorable_gauge_final = bool(factorable_summary["final_gauge_bearing_fraction"] > 0)

    if factorable_gauge_fixed:
        verdict = "BOUNDARY-FACTORABLE REVERSIBLE DYNAMICS DERIVATION SIGNAL: local reversible starts converged to gauge-bearing chart fixed points"
    elif factorable_gauge_final:
        verdict = "BOUNDARY-FACTORABLE REVERSIBLE DYNAMICS GAUGE SIGNAL: local reversible starts generated chart holonomy during the fixed-point flow"
    elif not factorable_random_final.empty and bool(factorable_summary["fixed_point_fraction"] > 0):
        verdict = "BOUNDARY-FACTORABLE REVERSIBLE DYNAMICS FLAT FIXED-POINT SIGNAL: local reversible starts converged, but gauge-bearing structure not established"
    elif not invertible_random_final.empty and invertible_gauge_fixed:
        verdict = "INVERTIBLE RANDOM-START DYNAMICS DERIVATION SIGNAL: information-preserving unstructured starts converged to gauge-bearing chart fixed points"
    elif not invertible_random_final.empty and bool(invertible_summary["fixed_point_fraction"] > 0) and not invertible_gauge_fixed:
        verdict = "INVERTIBLE RANDOM-START DYNAMICS FLAT FIXED-POINT SIGNAL: information-preserving starts converged, but gauge-bearing structure not established"
    elif not random_all.empty and any_random_gauge_fixed:
        verdict = "RANDOM-START DYNAMICS DERIVATION SIGNAL: unstructured dynamics converged to gauge-bearing chart fixed point"
    elif not random_all.empty and any_random_fixed:
        verdict = "RANDOM-START DYNAMICS FLAT FIXED-POINT SIGNAL: unstructured dynamics converged, but gauge-bearing structure not established"
    elif not random_all.empty and any_random_cycle:
        verdict = "RANDOM-START DYNAMICS LIMIT-CYCLE SIGNAL: unstructured dynamics entered recurrent chart-dynamics cycle"
    elif not random_all.empty and any_random_chart:
        verdict = "RANDOM-START DYNAMICS ITERATION SIGNAL: chart-atlas temporal face became nontrivial, fixed point not established"
    elif any_fixed:
        verdict = "DYNAMICS CONSISTENCY FIXED-POINT SIGNAL: temporal chart face reproduces the sampled dynamics"
    elif any_cycle:
        verdict = "DYNAMICS CONSISTENCY LIMIT-CYCLE SIGNAL: temporal chart face enters a recurrent dynamics cycle"
    elif any_chart:
        verdict = "DYNAMICS CONSISTENCY ITERATION SIGNAL: chart-atlas temporal face is nontrivial but fixed point not established"
    else:
        verdict = "DYNAMICS CONSISTENCY WEAK/INERT: effective temporal face did not produce nontrivial self-reproduction"
    summary: Dict[str, object] = {
        "verdict": verdict,
        "audit_version": "dynamics_consistency_fixedpoint_v4_local_reversible_random_start",
        "q": int(q),
        "vertices": int(vertices),
        "rule_modes": list(rule_modes),
        "instances": int(instances),
        "fixedpoint_iterations": int(fixedpoint_iterations),
        "proliferation_iterations": int(proliferation_iterations),
        "horizon": int(horizon),
        "max_state_samples": int(max_state_samples),
        "n_rows": int(len(df)),
        "n_domain_rows": int(len(domain_df)),
        "n_effective_map_rows": int(len(map_df)),
        "n_chart_cycle_rows": int(len(cycle_df)),
        "any_dynamics_fixed_point": bool(any_fixed),
        "any_dynamics_limit_cycle": bool(any_cycle),
        "any_chart_holonomy_during_iteration": bool(any_chart),
        "random_start_modes_present": bool(not random_all.empty),
        "random_start_instances": int(random_final[["rule_mode", "instance"]].drop_duplicates().shape[0]) if not random_final.empty else 0,
        "random_start_any_fixed_point": bool(any_random_fixed),
        "random_start_any_limit_cycle": bool(any_random_cycle),
        "random_start_any_chart_holonomy": bool(any_random_chart),
        "random_start_any_gauge_bearing_fixed_point": bool(any_random_gauge_fixed),
        "random_start_fixed_point_fraction": float(random_final["dynamics_fixed_point_candidate"].mean()) if not random_final.empty else 0.0,
        "random_start_final_gauge_bearing_fraction": float((random_final["n_chart_nontrivial_holonomy"] > 0).mean()) if not random_final.empty else 0.0,
        "random_start_final_c2_bearing_fraction": float((random_final["n_chart_c2_holonomy"] > 0).mean()) if not random_final.empty else 0.0,
        "random_start_gauge_bearing_fixed_point_fraction": float(((random_final["dynamics_fixed_point_candidate"] == True) & (random_final["n_chart_nontrivial_holonomy"] > 0)).mean()) if not random_final.empty else 0.0,
        "random_start_flat_fixed_point_fraction": float(((random_final["dynamics_fixed_point_candidate"] == True) & (random_final["n_chart_nontrivial_holonomy"] <= 0)).mean()) if not random_final.empty else 0.0,
        "random_start_unique_final_effective_attractors": int(random_final["effective_transition_hash"].nunique()) if not random_final.empty else 0,
        "random_start_final_effective_attractor_counts": (random_final["effective_transition_hash"].value_counts().to_dict() if not random_final.empty else {}),
        "boundary_factorable_random_start_summary": factorable_summary,
        "local_reversible_random_start_summary": local_reversible_summary,
        "affine_reversible_random_start_summary": affine_reversible_summary,
        "boundary_factorable_random_start_instances": int(factorable_summary["instances"]),
        "boundary_factorable_random_start_fixed_point_fraction": float(factorable_summary["fixed_point_fraction"]),
        "boundary_factorable_random_start_final_gauge_bearing_fraction": float(factorable_summary["final_gauge_bearing_fraction"]),
        "boundary_factorable_random_start_final_c2_bearing_fraction": float(factorable_summary["final_c2_bearing_fraction"]),
        "boundary_factorable_random_start_gauge_bearing_fixed_point_fraction": float(factorable_summary["gauge_bearing_fixed_point_fraction"]),
        "boundary_factorable_random_start_flat_fixed_point_fraction": float(factorable_summary["flat_fixed_point_fraction"]),
        "boundary_factorable_random_start_unique_final_effective_attractors": int(factorable_summary["unique_final_effective_attractors"]),
        "local_reversible_random_start_instances": int(local_reversible_summary["instances"]),
        "local_reversible_random_start_gauge_bearing_fixed_point_fraction": float(local_reversible_summary["gauge_bearing_fixed_point_fraction"]),
        "affine_reversible_random_start_instances": int(affine_reversible_summary["instances"]),
        "affine_reversible_random_start_gauge_bearing_fixed_point_fraction": float(affine_reversible_summary["gauge_bearing_fixed_point_fraction"]),
        "invertible_random_start_summary": invertible_summary,
        "noninvertible_random_start_summary": noninvertible_summary,
        "invertible_random_start_instances": int(invertible_summary["instances"]),
        "noninvertible_random_start_instances": int(noninvertible_summary["instances"]),
        "invertible_random_start_fixed_point_fraction": float(invertible_summary["fixed_point_fraction"]),
        "invertible_random_start_final_gauge_bearing_fraction": float(invertible_summary["final_gauge_bearing_fraction"]),
        "invertible_random_start_final_c2_bearing_fraction": float(invertible_summary["final_c2_bearing_fraction"]),
        "invertible_random_start_gauge_bearing_fixed_point_fraction": float(invertible_summary["gauge_bearing_fixed_point_fraction"]),
        "invertible_random_start_flat_fixed_point_fraction": float(invertible_summary["flat_fixed_point_fraction"]),
        "invertible_random_start_unique_final_effective_attractors": int(invertible_summary["unique_final_effective_attractors"]),
        "noninvertible_random_start_fixed_point_fraction": float(noninvertible_summary["fixed_point_fraction"]),
        "noninvertible_random_start_final_gauge_bearing_fraction": float(noninvertible_summary["final_gauge_bearing_fraction"]),
        "noninvertible_random_start_final_c2_bearing_fraction": float(noninvertible_summary["final_c2_bearing_fraction"]),
        "noninvertible_random_start_gauge_bearing_fixed_point_fraction": float(noninvertible_summary["gauge_bearing_fixed_point_fraction"]),
        "noninvertible_random_start_flat_fixed_point_fraction": float(noninvertible_summary["flat_fixed_point_fraction"]),
        "noninvertible_random_start_unique_final_effective_attractors": int(noninvertible_summary["unique_final_effective_attractors"]),
        "invertibility_gap_gauge_bearing_fixed_point_fraction": float(invertible_summary["gauge_bearing_fixed_point_fraction"] - noninvertible_summary["gauge_bearing_fixed_point_fraction"]),
        "invertibility_gap_final_c2_bearing_fraction": float(invertible_summary["final_c2_bearing_fraction"] - noninvertible_summary["final_c2_bearing_fraction"]),
        "max_signature_temporal_determinism": float(df["signature_temporal_determinism"].max()) if not df.empty else 0.0,
        "min_effective_transition_changed_fraction": float(df["effective_transition_changed_fraction"].min()) if not df.empty else 1.0,
        "max_chart_c2_holonomy": int(df["n_chart_c2_holonomy"].max()) if not df.empty else 0,
        "max_signature_compression_fraction": float(df["signature_compression_fraction"].max()) if not df.empty else 0.0,
        "min_nontrivial_compression": float(min_nontrivial_compression),
        "effective_lift_mode": str(effective_lift_mode),
        "by_mode": by_mode,
    }
    return df, domain_df, map_df, cycle_df, summary


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def maybe_write_csv(df, path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path}")


def write_summary(summary: Dict[str, object], path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {path}")


def plot_summary(df, path: str) -> None:
    if not path or df.empty:
        return
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    for mode, g in df.groupby("rule_mode"):
        gg = g.groupby("fixedpoint_iteration").agg({
            "signature_temporal_determinism": "mean",
            "effective_transition_changed_fraction": "mean",
            "n_signature_classes": "mean",
        }).reset_index()
        ax1.plot(gg["fixedpoint_iteration"], gg["signature_temporal_determinism"], marker="o", label=f"{mode}: determinism")
        ax1.plot(gg["fixedpoint_iteration"], gg["effective_transition_changed_fraction"], marker="s", linestyle="--", label=f"{mode}: change frac")
    ax1.set_xlabel("Fixed-point iteration T -> T_eff")
    ax1.set_ylabel("fraction")
    ax2 = ax1.twinx()
    for mode, g in df.groupby("rule_mode"):
        gg = g.groupby("fixedpoint_iteration").agg({"n_signature_classes": "mean"}).reset_index()
        ax2.plot(gg["fixedpoint_iteration"], gg["n_signature_classes"], marker="^", alpha=0.7, label=f"{mode}: classes")
    ax2.set_ylabel("signature classes")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.set_title("Dynamics consistency fixed point: T -> topology -> temporal chart face -> T_eff")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dynamics consistency fixed-point audit: random T -> generated chart atlas -> T_eff iteration.")
    p.add_argument("q", type=int)
    p.add_argument("--vertices", type=int, default=7)
    p.add_argument("--rule-modes", default="random_full_map", help="Comma modes. Use random_full_map for irreversible random T, random_full_permutation for global information-preserving random T, random_local_reversible for local reversible shear programs, and random_affine_bijection for reversible affine shear programs; named modes are reference checks.")
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--fixedpoint-iterations", type=int, default=6)
    p.add_argument("--proliferation-iterations", type=int, default=4)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--max-state-samples", type=int, default=512)
    p.add_argument("--max-total-states", type=int, default=4096)
    p.add_argument("--initial-boundary", default="sum_mod_q")
    p.add_argument("--initial-boundary-q", type=int, default=0)
    p.add_argument("--max-domains-per-depth", type=int, default=32)
    p.add_argument("--min-live-classes", type=int, default=2)
    p.add_argument("--min-fiber-size", type=int, default=2)
    p.add_argument("--min-entropy-bits", type=float, default=0.05)
    p.add_argument("--synergy-threshold", type=float, default=0.01)
    p.add_argument("--max-pred", type=int, default=3)
    p.add_argument("--max-signature-domains", type=int, default=16)
    p.add_argument("--max-parent-domains", type=int, default=8)
    p.add_argument("--max-fibers-per-parent", type=int, default=6)
    p.add_argument("--max-charts-per-fiber", type=int, default=16)
    p.add_argument("--max-signature-charts", type=int, default=48)
    p.add_argument("--min-fiber-states", type=int, default=8)
    p.add_argument("--min-support-states", type=int, default=4)
    p.add_argument("--min-overlap-states", type=int, default=4)
    p.add_argument("--min-chart-classes", type=int, default=2)
    p.add_argument("--min-chart-entropy", type=float, default=0.05)
    p.add_argument("--max-chart-coords", type=int, default=5)
    p.add_argument("--max-support-coords", type=int, default=4)
    p.add_argument("--max-cycle-len", type=int, default=4)
    p.add_argument("--max-cycles-per-fiber", type=int, default=500)
    p.add_argument("--min-nontrivial-compression", type=float, default=0.05)
    p.add_argument("--effective-lift-mode", choices=["representative", "bijective", "permutation", "conservative"], default="representative", help="Lift signature dynamics back to sampled states. representative is the observer quotient-lift; bijective/permutation is the information-conserving lift.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/dynamics_consistency_fixedpoint.csv")
    p.add_argument("--plot", default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    df, domain_df, map_df, cycle_df, summary = run_dynamics_consistency_fixedpoint_audit(
        q=args.q, vertices=args.vertices, rule_modes=_parse_modes(args.rule_modes), instances=args.instances,
        fixedpoint_iterations=args.fixedpoint_iterations, proliferation_iterations=args.proliferation_iterations,
        horizon=args.horizon, max_state_samples=args.max_state_samples, max_total_states=args.max_total_states,
        initial_boundary=args.initial_boundary, initial_boundary_q=(args.initial_boundary_q or None),
        max_domains_per_depth=args.max_domains_per_depth, min_live_classes=args.min_live_classes,
        min_fiber_size=args.min_fiber_size, min_entropy_bits=args.min_entropy_bits,
        synergy_threshold=args.synergy_threshold, max_pred=args.max_pred,
        max_signature_domains=args.max_signature_domains, max_parent_domains=args.max_parent_domains,
        max_fibers_per_parent=args.max_fibers_per_parent, max_charts_per_fiber=args.max_charts_per_fiber,
        max_signature_charts=args.max_signature_charts, min_fiber_states=args.min_fiber_states,
        min_support_states=args.min_support_states, min_overlap_states=args.min_overlap_states,
        min_chart_classes=args.min_chart_classes, min_chart_entropy=args.min_chart_entropy,
        max_chart_coords=args.max_chart_coords, max_support_coords=args.max_support_coords,
        max_cycle_len=args.max_cycle_len, max_cycles_per_fiber=args.max_cycles_per_fiber,
        min_nontrivial_compression=args.min_nontrivial_compression,
        effective_lift_mode=args.effective_lift_mode,
        seed=args.seed, verbose=not bool(args.quiet),
    )
    stem = os.path.splitext(args.out)[0]
    maybe_write_csv(df, args.out)
    maybe_write_csv(domain_df, stem + "_domains.csv")
    maybe_write_csv(map_df, stem + "_effective_maps.csv")
    maybe_write_csv(cycle_df, stem + "_chart_cycles.csv")
    write_summary(summary, stem + "_summary.json")
    if args.plot:
        plot_summary(df, args.plot)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
