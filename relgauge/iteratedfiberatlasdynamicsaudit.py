"""
iteratedfiberatlasdynamicsaudit.py

Iterated fiber-atlas dynamics audit.

This module performs the decisive test that sits between
``dynamicsconsistencyfixedpointaudit`` and ``fiberatlasfixedpointaudit``:

    T_0
      -> live-fiber proliferation
      -> finite boundary-chart atlas A(T_0)
      -> temporal face of a *fiber-preserving* atlas quotient
      -> effective dynamics T_1
      -> repeat

The older dynamics-consistency audit did iterate, but the effective readout was
a boundary/signature lift that tends to erase the unresolved fiber structure in
which chart holonomy lives.  The fiber-atlas audit preserved that structure, but
was primarily a diagnostic pass over profiles/capacities.  This audit combines
both: it iterates while explicitly tracking whether gauge/C2 appears, persists,
or is erased under fiber-preserving/full-atlas reconstruction.

No C2, C3, matter, charge, or Standard-Model-like target is optimized.  Holonomy
orders are reported post hoc from the same-fiber chart atlas.

Example
-------
python -m relgauge.iteratedfiberatlasdynamicsaudit 2 ^
  --vertices 7 ^
  --rule-modes random_full_map,random_full_permutation,random_local_reversible,random_affine_bijection,affine_mix ^
  --instances 10 ^
  --atlas-capacities 8,16,32,64 ^
  --profiles fiber_preserving,full_atlas ^
  --atlas-iterations 8 ^
  --proliferation-iterations 4 ^
  --horizon 3 ^
  --max-state-samples 512 ^
  --out example_results/iterated_fiber_atlas_dynamics_q2.csv ^
  --plot example_results/fig_iterated_fiber_atlas_dynamics_q2.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import boundaryproliferationaudit as BPA
    from . import dynamicsconsistencyfixedpointaudit as DCFP
    from . import fiberatlasfixedpointaudit as FAF
except Exception:  # pragma: no cover
    import boundaryproliferationaudit as BPA  # type: ignore
    import dynamicsconsistencyfixedpointaudit as DCFP  # type: ignore
    import fiberatlasfixedpointaudit as FAF  # type: ignore


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


def _parse_csv_ints(text: str, default: Sequence[int]) -> List[int]:
    vals: List[int] = []
    for p in str(text or "").replace(";", ",").split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(float(p)))
    return vals or [int(x) for x in default]


def _parse_csv_text(text: str, default: Sequence[str]) -> List[str]:
    vals = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    return vals or [str(x) for x in default]


def _hash_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _hash_arr(vals: Sequence[int]) -> str:
    arr = np.asarray(vals, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:20]


def _top_domains(domains, max_domains: int):
    ds = [d for d in domains if int(getattr(d, "n_labels", 0)) >= 2 and float(getattr(d, "entropy_bits", 0.0)) > 0]
    ds.sort(key=lambda d: (float(d.entropy_bits), int(getattr(d, "live_fiber_count", 0)), int(d.n_labels)), reverse=True)
    return ds[: int(max_domains)]




def transition_target_labels(labels: Sequence[int], next_idx: Sequence[int]) -> np.ndarray:
    """Return labels(Tx) for each sampled state index x."""
    lab = np.asarray(labels, dtype=np.int64)
    nxt = np.asarray(next_idx, dtype=np.int64)
    n = int(min(len(lab), len(nxt)))
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        j = int(nxt[i])
        out[i] = int(lab[j]) if 0 <= j < len(lab) else int(lab[i])
    return out


def profile_signature_labels(atlas: DCFP.AtlasExtraction, next_idx: Sequence[int], profile: str, max_domains: int = 16) -> np.ndarray:
    """Public helper returning unbounded canonical profile labels for tests/use.

    ``boundary_only`` uses generated domain labels only. ``fiber_preserving``
    returns the same-fiber chart atlas signature. ``full_atlas`` appends the
    one-step temporal face labels(Tx).
    """
    prof = str(profile or "fiber_preserving").strip().lower()
    n = int(len(atlas.signature_labels))
    if prof in {"boundary", "boundary_only", "domain", "domains"}:
        comps: List[np.ndarray] = []
        for d in _top_domains(list(atlas.domains_current) or list(atlas.domains_all), int(max_domains)):
            labs = np.asarray(getattr(d, "labels"), dtype=np.int64)
            if len(labs) >= n:
                comps.append(labs[:n])
        return DCFP.canonical_signature_labels(comps) if comps else np.zeros(n, dtype=np.int32)
    base = np.asarray(atlas.signature_labels, dtype=np.int64)
    if prof in {"full", "full_atlas", "temporal", "temporal_atlas"}:
        return DCFP.canonical_signature_labels([base, transition_target_labels(base, next_idx)])
    if prof in {"fiber", "fiber_preserving", "atlas"}:
        return DCFP.canonical_signature_labels([base])
    raise ValueError(f"unknown profile {profile!r}")

def _labels_for_profile(atlas: DCFP.AtlasExtraction, current_next: Sequence[int], profile: str, capacity: int,
                        max_signature_domains: int) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return bounded labels for a profile.

    boundary_only uses domain labels only.  fiber_preserving uses the full
    same-fiber chart signature from DCFP (domains + chart labels).  full_atlas
    adds the one-step temporal face label as an additional component before
    applying the finite capacity cap.
    """
    profile_l = str(profile or "fiber_preserving").strip().lower()
    n = int(len(atlas.signature_labels))
    if profile_l in {"boundary", "boundary_only", "domain", "domains"}:
        comps: List[np.ndarray] = []
        ds = _top_domains(list(atlas.domains_current) or list(atlas.domains_all), int(max_signature_domains))
        for d in ds:
            labs = np.asarray(getattr(d, "labels"), dtype=np.int64)
            if len(labs) >= n:
                comps.append(labs[:n])
        if comps:
            raw = DCFP.canonical_signature_labels(comps)
        else:
            raw = np.zeros(n, dtype=np.int32)
        source_profile = "boundary_only"
    elif profile_l in {"full", "full_atlas", "temporal", "temporal_atlas"}:
        base = np.asarray(atlas.signature_labels, dtype=np.int64)
        nxt = np.asarray(current_next, dtype=np.int64)
        target = np.zeros(n, dtype=np.int64)
        for i in range(n):
            j = int(nxt[i]) if i < len(nxt) else i
            target[i] = int(base[j]) if 0 <= j < n else int(base[i])
        raw = DCFP.canonical_signature_labels([base, target])
        source_profile = "full_atlas"
    else:
        raw = np.asarray(atlas.signature_labels, dtype=np.int64)
        source_profile = "fiber_preserving"

    bounded = FAF.compress_labels_by_capacity(raw, int(capacity))
    stats = {
        "profile_canonical": source_profile,
        "raw_profile_classes": int(len(set(int(x) for x in raw))),
        "raw_profile_entropy_bits": float(DCFP.entropy_from_labels(raw)),
        "bounded_atlas_classes": int(len(set(int(x) for x in bounded))),
        "bounded_atlas_entropy_bits": float(DCFP.entropy_from_labels(bounded)),
        "bounded_atlas_fiber_entropy_bits": float(max(0.0, math.log2(max(1, n)) - DCFP.entropy_from_labels(bounded))),
        "bounded_atlas_compression_fraction": float(1.0 - len(set(int(x) for x in bounded)) / max(1, n)),
    }
    return np.asarray(bounded, dtype=np.int64), stats


def _one_atlas_pass(states, current_next: np.ndarray, q: int, rng: np.random.Generator, args) -> DCFP.AtlasExtraction:
    atlas, _eff, _eff_stats, _map_rows = DCFP.analyze_one_effective_step(
        states=states,
        current_next_idx=current_next,
        q=int(q),
        rng=rng,
        proliferation_iterations=int(args.proliferation_iterations),
        horizon=int(args.horizon),
        initial_boundary=str(args.initial_boundary),
        initial_boundary_q=args.initial_boundary_q,
        max_domains_per_depth=int(args.max_domains_per_depth),
        min_live_classes=int(args.min_live_classes),
        min_fiber_size=int(args.min_fiber_size),
        min_entropy_bits=float(args.min_entropy_bits),
        synergy_threshold=float(args.synergy_threshold),
        max_signature_domains=int(args.max_signature_domains),
        max_parent_domains=int(args.max_parent_domains),
        max_fibers_per_parent=int(args.max_fibers_per_parent),
        max_charts_per_fiber=int(args.max_charts_per_fiber),
        max_signature_charts=int(args.max_signature_charts),
        min_fiber_states=int(args.min_fiber_states),
        min_support_states=int(args.min_support_states),
        min_overlap_states=int(args.min_overlap_states),
        min_chart_classes=int(args.min_chart_classes),
        min_chart_entropy=float(args.min_chart_entropy),
        max_chart_coords=int(args.max_chart_coords),
        max_support_coords=int(args.max_support_coords),
        max_cycle_len=int(args.max_cycle_len),
        max_cycles_per_fiber=int(args.max_cycles_per_fiber),
        effective_lift_mode="representative",
    )
    return atlas


def run_iterated_fiber_atlas_dynamics_audit(
    q: int,
    vertices: int = 7,
    rule_modes: Sequence[str] = ("random_full_map",),
    instances: int = 3,
    atlas_capacities: Sequence[int] = (8, 16, 32, 64),
    profiles: Sequence[str] = ("fiber_preserving", "full_atlas"),
    atlas_iterations: int = 8,
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
    atlas_lift_mode: str = "bijective",
    min_compression_fraction: float = 0.05,
    min_fiber_entropy_bits: float = 0.25,
    min_temporal_determinism: float = 0.95,
    seed: int = 0,
    save_transition_at: int = -1,
    out: str = "example_results/iterated_fiber_atlas_dynamics.csv",
    verbose: bool = True,
):
    if pd is None:
        raise RuntimeError("pandas is required")
    q = int(q); vertices = int(vertices)
    rows: List[Dict[str, object]] = []
    domain_rows_all: List[Dict[str, object]] = []
    map_rows_all: List[Dict[str, object]] = []
    chart_cycle_rows_all: List[Dict[str, object]] = []
    attractor_rows: List[Dict[str, object]] = []

    class Args:
        pass
    args = Args()
    for k, v in dict(
        proliferation_iterations=proliferation_iterations, horizon=horizon,
        initial_boundary=initial_boundary, initial_boundary_q=initial_boundary_q,
        max_domains_per_depth=max_domains_per_depth, min_live_classes=min_live_classes,
        min_fiber_size=min_fiber_size, min_entropy_bits=min_entropy_bits,
        synergy_threshold=synergy_threshold, max_signature_domains=max_signature_domains,
        max_parent_domains=max_parent_domains, max_fibers_per_parent=max_fibers_per_parent,
        max_charts_per_fiber=max_charts_per_fiber, max_signature_charts=max_signature_charts,
        min_fiber_states=min_fiber_states, min_support_states=min_support_states,
        min_overlap_states=min_overlap_states, min_chart_classes=min_chart_classes,
        min_chart_entropy=min_chart_entropy, max_chart_coords=max_chart_coords,
        max_support_coords=max_support_coords, max_cycle_len=max_cycle_len,
        max_cycles_per_fiber=max_cycles_per_fiber,
    ).items():
        setattr(args, k, v)

    for mi, mode in enumerate(list(rule_modes)):
        mode_s = str(mode)
        for inst in range(int(instances)):
            init_seed = int(seed) + 1000003 * (mi + 1) + 7919 * int(inst)
            init_rng = np.random.default_rng(init_seed)
            states, initial_next, init_meta = DCFP.initialize_sampled_transition(
                q=q, vertices=vertices, mode=mode_s, rng=init_rng,
                max_state_samples=int(max_state_samples), max_total_states=int(max_total_states),
                max_pred=int(max_pred), proliferation_iterations=int(proliferation_iterations), horizon=int(horizon),
            )
            n_states = int(len(states))
            for profile in list(profiles):
                for cap in list(atlas_capacities):
                    current_next = np.asarray(initial_next, dtype=np.int64).copy()
                    seen_effective: Dict[str, int] = {}
                    first_c2_iter: Optional[int] = None
                    first_gauge_iter: Optional[int] = None
                    initial_c2_count: Optional[int] = None
                    initial_gauge_count: Optional[int] = None
                    c2_after_initial = 0
                    gauge_after_initial = 0
                    fixed_or_cycle_iter: Optional[int] = None
                    fixed_candidate = False
                    limit_cycle_candidate = False
                    final_hash = ""
                    for it in range(int(atlas_iterations) + 1):
                        profile_hash = int(hashlib.sha1(str(profile).encode("utf-8")).hexdigest()[:8], 16)
                        iter_seed = int(init_seed) + 1299709 * int(cap) + 15485863 * int(it) + 104729 * (profile_hash % 1000)
                        rng = np.random.default_rng(iter_seed)
                        if int(save_transition_at) >= 0 and int(it) == int(save_transition_at):
                            stem = os.path.splitext(str(out))[0] if str(out or "") else "transition"
                            safe_mode = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(mode_s))
                            safe_profile = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(profile))
                            path = f"{stem}_transition_iter{int(it)}_{safe_mode}_{safe_profile}_cap{int(cap)}_inst{int(inst)}_seed{int(init_seed)}.npy"
                            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                            np.save(path, np.asarray(current_next, dtype=np.int64))
                        atlas = _one_atlas_pass(states, current_next, q, rng, args)
                        bounded, pstats = _labels_for_profile(atlas, current_next, str(profile), int(cap), int(max_signature_domains))
                        rel_stats, rel_rows = FAF.temporal_relation_stats(bounded, current_next)
                        eff, lift_stats, eff_rows = DCFP.extract_effective_dynamics(bounded, current_next, lift_mode=str(atlas_lift_mode))

                        bounded_nontrivial = bool(
                            _safe_float(pstats.get("bounded_atlas_compression_fraction"), 0.0) >= float(min_compression_fraction)
                            and _safe_float(pstats.get("bounded_atlas_fiber_entropy_bits"), 0.0) >= float(min_fiber_entropy_bits)
                            and _safe_int(pstats.get("bounded_atlas_classes"), 0) > 1
                        )
                        temporal_ok = bool(_safe_float(rel_stats.get("temporal_relation_determinism"), 0.0) >= float(min_temporal_determinism))
                        gauge_present = bool(int(atlas.n_chart_nontrivial) > 0)
                        c2_present = bool(int(atlas.n_chart_c2) > 0)
                        if it == 0:
                            initial_c2_count = int(atlas.n_chart_c2)
                            initial_gauge_count = int(atlas.n_chart_nontrivial)
                        else:
                            if c2_present:
                                c2_after_initial += 1
                            if gauge_present:
                                gauge_after_initial += 1
                        if c2_present and first_c2_iter is None:
                            first_c2_iter = int(it)
                        if gauge_present and first_gauge_iter is None:
                            first_gauge_iter = int(it)

                        eff_hash = _hash_arr(eff)
                        final_hash = eff_hash
                        atlas_shape_hash = _hash_obj({
                            "profile": str(pstats.get("profile_canonical")),
                            "cap": int(cap),
                            "classes": _safe_int(pstats.get("bounded_atlas_classes")),
                            "rel_hash": rel_stats.get("temporal_relation_hash"),
                            "c2": int(atlas.n_chart_c2),
                            "nontriv": int(atlas.n_chart_nontrivial),
                            "order": int(atlas.max_chart_order),
                            "arch": str(atlas.architecture_hash),
                        })
                        effective_fixed = bool(np.array_equal(eff, current_next))
                        if effective_fixed and fixed_or_cycle_iter is None:
                            fixed_candidate = True
                            fixed_or_cycle_iter = int(it)
                        if eff_hash in seen_effective and fixed_or_cycle_iter is None:
                            limit_cycle_candidate = True
                            fixed_or_cycle_iter = int(it)
                        seen_effective.setdefault(eff_hash, int(it))
                        atlas_dynamics_candidate = bool(bounded_nontrivial and temporal_ok)
                        gauge_bearing_atlas_candidate = bool(atlas_dynamics_candidate and gauge_present)
                        c2_bearing_atlas_candidate = bool(atlas_dynamics_candidate and c2_present)
                        c2_generated_after_start = bool((initial_c2_count or 0) == 0 and it > 0 and c2_present)
                        gauge_generated_after_start = bool((initial_gauge_count or 0) == 0 and it > 0 and gauge_present)
                        c2_erased_from_start = bool((initial_c2_count or 0) > 0 and it > 0 and not c2_present)
                        row = {
                            "rule_mode": mode_s,
                            "initial_dynamics_kind": init_meta.get("initial_dynamics_kind", ""),
                            "initial_dynamics_family": init_meta.get("initial_dynamics_family", ""),
                            "random_start_unstructured": bool(init_meta.get("random_start_unstructured", False)),
                            "boundary_factorable_reversible_start": bool(init_meta.get("boundary_factorable_reversible_start", False)),
                            "local_reversible_start": bool(init_meta.get("local_reversible_start", False)),
                            "affine_reversible_start": bool(init_meta.get("affine_reversible_start", False)),
                            "instance": int(inst),
                            "initial_seed": int(init_seed),
                            "q": int(q),
                            "vertices": int(vertices),
                            "n_states": int(n_states),
                            "profile": str(pstats.get("profile_canonical")),
                            "atlas_capacity": int(cap),
                            "atlas_iteration": int(it),
                            "initial_c2_count": int(initial_c2_count or 0),
                            "initial_gauge_count": int(initial_gauge_count or 0),
                            "n_chart_nontrivial": int(atlas.n_chart_nontrivial),
                            "n_chart_c2": int(atlas.n_chart_c2),
                            "n_chart_c3": int(atlas.n_chart_c3),
                            "max_chart_order": int(atlas.max_chart_order),
                            "n_chart_cycles": int(atlas.n_chart_cycles),
                            "n_chart_valid_cycles": int(atlas.n_chart_valid_cycles),
                            "chart_gauge_present": bool(gauge_present),
                            "chart_c2_present": bool(c2_present),
                            "c2_generated_after_start": bool(c2_generated_after_start),
                            "gauge_generated_after_start": bool(gauge_generated_after_start),
                            "c2_erased_from_start": bool(c2_erased_from_start),
                            "first_c2_iteration_so_far": int(first_c2_iter) if first_c2_iter is not None else -1,
                            "first_gauge_iteration_so_far": int(first_gauge_iter) if first_gauge_iter is not None else -1,
                            "bounded_nontrivial_atlas": bool(bounded_nontrivial),
                            "atlas_dynamics_candidate": bool(atlas_dynamics_candidate),
                            "gauge_bearing_atlas_candidate": bool(gauge_bearing_atlas_candidate),
                            "c2_bearing_atlas_candidate": bool(c2_bearing_atlas_candidate),
                            "effective_fixed_point": bool(effective_fixed),
                            "effective_limit_cycle_so_far": bool(eff_hash in seen_effective and not effective_fixed and it != seen_effective.get(eff_hash)),
                            "effective_next_hash": str(eff_hash),
                            "atlas_shape_hash": str(atlas_shape_hash),
                            "architecture_hash": str(atlas.architecture_hash),
                            "dependency_edges": int(atlas.n_dependency_edges),
                            "dependency_beta1": int(atlas.dependency_beta1),
                            "proliferation_nontriviality_score": float(atlas.nontriviality_score),
                            "full_signature_classes": int(atlas.n_signature_classes),
                            "full_signature_entropy_bits": float(atlas.signature_entropy_bits),
                            **pstats,
                            **rel_stats,
                            **{f"lift_{k}": v for k, v in lift_stats.items()},
                        }
                        rows.append(row)
                        for dr in atlas.domain_rows:
                            rr = dict(dr)
                            rr.update({
                                "rule_mode": mode_s, "instance": int(inst), "profile": str(pstats.get("profile_canonical")),
                                "atlas_capacity": int(cap), "atlas_iteration": int(it), "initial_seed": int(init_seed),
                            })
                            domain_rows_all.append(rr)
                        for mr in rel_rows:
                            rr = dict(mr)
                            rr.update({
                                "rule_mode": mode_s, "instance": int(inst), "profile": str(pstats.get("profile_canonical")),
                                "atlas_capacity": int(cap), "atlas_iteration": int(it), "initial_seed": int(init_seed),
                                "row_kind": "temporal_relation",
                            })
                            map_rows_all.append(rr)
                        for mr in eff_rows:
                            rr = dict(mr)
                            rr.update({
                                "rule_mode": mode_s, "instance": int(inst), "profile": str(pstats.get("profile_canonical")),
                                "atlas_capacity": int(cap), "atlas_iteration": int(it), "initial_seed": int(init_seed),
                                "row_kind": "effective_lift",
                            })
                            map_rows_all.append(rr)
                        for cr in atlas.chart_cycle_rows:
                            rr = dict(cr)
                            rr.update({
                                "rule_mode": mode_s, "instance": int(inst), "profile": str(pstats.get("profile_canonical")),
                                "atlas_capacity": int(cap), "atlas_iteration": int(it), "initial_seed": int(init_seed),
                            })
                            chart_cycle_rows_all.append(rr)
                        if verbose:
                            print(
                                f"iter-atlas mode={mode_s} inst={inst+1}/{int(instances)} prof={pstats.get('profile_canonical')} cap={cap} "
                                f"it={it}/{int(atlas_iterations)} classes={pstats.get('bounded_atlas_classes')} "
                                f"det={_safe_float(rel_stats.get('temporal_relation_determinism')):.3f} "
                                f"c2={int(atlas.n_chart_c2)} gen={int(c2_generated_after_start)} fixed={int(effective_fixed)}"
                            )
                        current_next = np.asarray(eff, dtype=np.int64)
                    # end iteration
                    final_c2_count = int(rows[-1]["n_chart_c2"]) if rows else 0
                    final_gauge_count = int(rows[-1]["n_chart_nontrivial"]) if rows else 0
                    attractor_rows.append({
                        "rule_mode": mode_s,
                        "instance": int(inst),
                        "profile": str(pstats.get("profile_canonical")),
                        "atlas_capacity": int(cap),
                        "initial_seed": int(init_seed),
                        "initial_c2_count": int(initial_c2_count or 0),
                        "final_c2_count": int(final_c2_count),
                        "initial_gauge_count": int(initial_gauge_count or 0),
                        "final_gauge_count": int(final_gauge_count),
                        "c2_generated_after_start_any": bool((initial_c2_count or 0) == 0 and c2_after_initial > 0),
                        "gauge_generated_after_start_any": bool((initial_gauge_count or 0) == 0 and gauge_after_initial > 0),
                        "first_c2_iteration": int(first_c2_iter) if first_c2_iter is not None else -1,
                        "first_gauge_iteration": int(first_gauge_iter) if first_gauge_iter is not None else -1,
                        "c2_persistence_iterations_after_initial": int(c2_after_initial),
                        "gauge_persistence_iterations_after_initial": int(gauge_after_initial),
                        "c2_persistent_to_final": bool(final_c2_count > 0),
                        "gauge_persistent_to_final": bool(final_gauge_count > 0),
                        "effective_fixed_point_found": bool(fixed_candidate),
                        "effective_limit_cycle_found": bool(limit_cycle_candidate),
                        "first_fixed_or_cycle_iteration": int(fixed_or_cycle_iter) if fixed_or_cycle_iter is not None else -1,
                        "final_effective_hash": str(final_hash),
                    })

    df = pd.DataFrame(rows)
    ddf = pd.DataFrame(domain_rows_all)
    mdf = pd.DataFrame(map_rows_all)
    cdf = pd.DataFrame(chart_cycle_rows_all)
    adf = pd.DataFrame(attractor_rows)

    def frac(series) -> float:
        if series is None or len(series) == 0:
            return 0.0
        return float(np.mean(np.asarray(series, dtype=bool)))

    summary: Dict[str, object] = {
        "verdict": "ITERATED FIBER-ATLAS DYNAMICS COMPLETE",
        "audit_version": "iterated_fiber_atlas_dynamics_v1_fiber_preserving_self_observation",
        "q": int(q),
        "vertices": int(vertices),
        "rule_modes": list(rule_modes),
        "instances": int(instances),
        "atlas_capacities": [int(x) for x in atlas_capacities],
        "profiles": list(profiles),
        "atlas_iterations": int(atlas_iterations),
        "proliferation_iterations": int(proliferation_iterations),
        "horizon": int(horizon),
        "atlas_lift_mode": str(atlas_lift_mode),
        "n_rows": int(len(df)),
        "n_domain_rows": int(len(ddf)),
        "n_map_rows": int(len(mdf)),
        "n_chart_cycle_rows": int(len(cdf)),
        "n_attractor_rows": int(len(adf)),
        "any_c2_during_iteration": bool(len(df) and df["chart_c2_present"].astype(bool).any()),
        "any_c2_generated_after_start": bool(len(adf) and adf["c2_generated_after_start_any"].astype(bool).any()),
        "any_gauge_generated_after_start": bool(len(adf) and adf["gauge_generated_after_start_any"].astype(bool).any()),
        "any_c2_persistent_to_final": bool(len(adf) and adf["c2_persistent_to_final"].astype(bool).any()),
        "any_gauge_persistent_to_final": bool(len(adf) and adf["gauge_persistent_to_final"].astype(bool).any()),
        "max_chart_c2_holonomy": int(df["n_chart_c2"].max()) if len(df) else 0,
        "max_chart_nontrivial_holonomy": int(df["n_chart_nontrivial"].max()) if len(df) else 0,
        "c2_generation_fraction": frac(adf["c2_generated_after_start_any"]) if len(adf) else 0.0,
        "gauge_generation_fraction": frac(adf["gauge_generated_after_start_any"]) if len(adf) else 0.0,
        "c2_final_fraction": frac(adf["c2_persistent_to_final"]) if len(adf) else 0.0,
        "gauge_final_fraction": frac(adf["gauge_persistent_to_final"]) if len(adf) else 0.0,
    }
    # Verdict refinement.
    if summary["any_c2_generated_after_start"] and summary["any_c2_persistent_to_final"]:
        summary["verdict"] = "ITERATED FIBER-ATLAS GAUGE-GENERATION SIGNAL: C2 appears from initially non-C2 atlas dynamics and persists"
    elif summary["any_c2_generated_after_start"]:
        summary["verdict"] = "ITERATED FIBER-ATLAS TRANSIENT GAUGE-GENERATION SIGNAL: C2 appears after start but is not always retained"
    elif summary["any_c2_during_iteration"]:
        summary["verdict"] = "ITERATED FIBER-ATLAS GAUGE-RETENTION/ERASURE SIGNAL: C2 exists only for initially structured atlas dynamics"
    else:
        summary["verdict"] = "ITERATED FIBER-ATLAS FLAT SIGNAL: no C2 generated in tested iterations"

    if len(adf):
        by_mode_profile = []
        for (mode, prof), g in adf.groupby(["rule_mode", "profile"], dropna=False):
            by_mode_profile.append({
                "rule_mode": str(mode),
                "profile": str(prof),
                "n": int(len(g)),
                "c2_generation_fraction": frac(g["c2_generated_after_start_any"]),
                "gauge_generation_fraction": frac(g["gauge_generated_after_start_any"]),
                "c2_final_fraction": frac(g["c2_persistent_to_final"]),
                "gauge_final_fraction": frac(g["gauge_persistent_to_final"]),
                "mean_first_c2_iteration": float(g.loc[g["first_c2_iteration"] >= 0, "first_c2_iteration"].mean()) if (g["first_c2_iteration"] >= 0).any() else -1.0,
                "max_final_c2_count": int(g["final_c2_count"].max()) if len(g) else 0,
                "fixed_point_fraction": frac(g["effective_fixed_point_found"]),
                "limit_cycle_fraction": frac(g["effective_limit_cycle_found"]),
            })
        by_capacity = []
        for cap, g in adf.groupby("atlas_capacity", dropna=False):
            by_capacity.append({
                "atlas_capacity": int(cap),
                "n": int(len(g)),
                "c2_generation_fraction": frac(g["c2_generated_after_start_any"]),
                "gauge_generation_fraction": frac(g["gauge_generated_after_start_any"]),
                "c2_final_fraction": frac(g["c2_persistent_to_final"]),
                "gauge_final_fraction": frac(g["gauge_persistent_to_final"]),
            })
        summary["by_mode_profile"] = by_mode_profile
        summary["by_capacity"] = by_capacity
    return df, ddf, mdf, cdf, adf, pd.DataFrame(summary.get("by_capacity", [])), summary


def write_outputs(df, ddf, mdf, cdf, adf, bycap, summary, out: str, plot: Optional[str] = None) -> None:
    base, ext = os.path.splitext(out)
    if not ext:
        out = base + ".csv"; base, ext = os.path.splitext(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    ddf.to_csv(base + "_domains.csv", index=False)
    mdf.to_csv(base + "_atlas_maps.csv", index=False)
    cdf.to_csv(base + "_chart_cycles.csv", index=False)
    adf.to_csv(base + "_attractors.csv", index=False)
    bycap.to_csv(base + "_by_capacity.csv", index=False)
    with open(base + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(15, 6))
            if len(df):
                grouped = df.groupby(["rule_mode", "profile", "atlas_iteration"], as_index=False).agg({
                    "n_chart_c2": "max",
                    "n_chart_nontrivial": "max",
                    "temporal_relation_determinism": "mean",
                    "bounded_atlas_classes": "mean",
                    "c2_generated_after_start": "mean",
                })
                ax2 = ax1.twinx()
                # Keep plot readable: use final/profile label aggregates by mode.
                for (mode, prof), g in grouped.groupby(["rule_mode", "profile"]):
                    g = g.sort_values("atlas_iteration")
                    label = f"{mode}:{prof}"
                    ax1.plot(g["atlas_iteration"], g["temporal_relation_determinism"], marker="o", label=f"{label} det")
                    ax2.plot(g["atlas_iteration"], g["n_chart_c2"], marker="^", linestyle="--", label=f"{label} C2")
                ax1.set_xlabel("Atlas iteration: T -> fiber atlas temporal face -> T_eff")
                ax1.set_ylabel("temporal determinism")
                ax2.set_ylabel("max C2 chart holonomy count")
                ax1.set_title(str(summary.get("verdict", "Iterated fiber-atlas dynamics")))
                h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=7)
            else:
                ax1.set_title("Iterated fiber-atlas dynamics: no rows")
            fig.tight_layout()
            os.makedirs(os.path.dirname(plot) or ".", exist_ok=True)
            fig.savefig(plot, dpi=150)
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"plot failed: {e}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(f"wrote {base + '_domains.csv'}")
    print(f"wrote {base + '_atlas_maps.csv'}")
    print(f"wrote {base + '_chart_cycles.csv'}")
    print(f"wrote {base + '_attractors.csv'}")
    print(f"wrote {base + '_by_capacity.csv'}")
    print(f"wrote {base + '_summary.json'}")
    if plot:
        print(f"wrote {plot}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Iterated fiber-atlas dynamics audit")
    p.add_argument("q", type=int)
    p.add_argument("--vertices", type=int, default=7)
    p.add_argument("--rule-modes", default="random_full_map")
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--atlas-capacities", default="8,16,32,64")
    p.add_argument("--profiles", default="fiber_preserving,full_atlas")
    p.add_argument("--atlas-iterations", type=int, default=8)
    p.add_argument("--proliferation-iterations", type=int, default=4)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--max-state-samples", type=int, default=512)
    p.add_argument("--max-total-states", type=int, default=4096)
    p.add_argument("--initial-boundary", default="sum_mod_q")
    p.add_argument("--initial-boundary-q", type=int, default=None)
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
    p.add_argument("--atlas-lift-mode", default="bijective", choices=["representative", "bijective", "permutation", "conservative", "information_conserving"])
    p.add_argument("--min-compression-fraction", type=float, default=0.05)
    p.add_argument("--min-fiber-entropy-bits", type=float, default=0.25)
    p.add_argument("--min-temporal-determinism", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-transition-at", type=int, default=-1, help="Save current_next transition table before atlas pass at this atlas iteration; -1 disables.")
    p.add_argument("--out", default="example_results/iterated_fiber_atlas_dynamics.csv")
    p.add_argument("--plot", default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    modes = _parse_csv_text(args.rule_modes, default=("random_full_map",))
    capacities = _parse_csv_ints(args.atlas_capacities, default=(8, 16, 32, 64))
    profiles = _parse_csv_text(args.profiles, default=("fiber_preserving", "full_atlas"))
    df, ddf, mdf, cdf, adf, bycap, summary = run_iterated_fiber_atlas_dynamics_audit(
        q=int(args.q), vertices=int(args.vertices), rule_modes=modes, instances=int(args.instances),
        atlas_capacities=capacities, profiles=profiles, atlas_iterations=int(args.atlas_iterations),
        proliferation_iterations=int(args.proliferation_iterations), horizon=int(args.horizon),
        max_state_samples=int(args.max_state_samples), max_total_states=int(args.max_total_states),
        initial_boundary=str(args.initial_boundary), initial_boundary_q=args.initial_boundary_q,
        max_domains_per_depth=int(args.max_domains_per_depth), min_live_classes=int(args.min_live_classes),
        min_fiber_size=int(args.min_fiber_size), min_entropy_bits=float(args.min_entropy_bits),
        synergy_threshold=float(args.synergy_threshold), max_pred=int(args.max_pred),
        max_signature_domains=int(args.max_signature_domains), max_parent_domains=int(args.max_parent_domains),
        max_fibers_per_parent=int(args.max_fibers_per_parent), max_charts_per_fiber=int(args.max_charts_per_fiber),
        max_signature_charts=int(args.max_signature_charts), min_fiber_states=int(args.min_fiber_states),
        min_support_states=int(args.min_support_states), min_overlap_states=int(args.min_overlap_states),
        min_chart_classes=int(args.min_chart_classes), min_chart_entropy=float(args.min_chart_entropy),
        max_chart_coords=int(args.max_chart_coords), max_support_coords=int(args.max_support_coords),
        max_cycle_len=int(args.max_cycle_len), max_cycles_per_fiber=int(args.max_cycles_per_fiber),
        atlas_lift_mode=str(args.atlas_lift_mode), min_compression_fraction=float(args.min_compression_fraction),
        min_fiber_entropy_bits=float(args.min_fiber_entropy_bits), min_temporal_determinism=float(args.min_temporal_determinism),
        seed=int(args.seed), save_transition_at=int(args.save_transition_at), out=str(args.out), verbose=not bool(args.quiet),
    )
    write_outputs(df, ddf, mdf, cdf, adf, bycap, summary, out=str(args.out), plot=(str(args.plot) or None))


if __name__ == "__main__":  # pragma: no cover
    main()
