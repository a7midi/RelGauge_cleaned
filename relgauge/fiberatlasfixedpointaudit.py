"""
fiberatlasfixedpointaudit.py

Fiber-atlas fixed-point audit.

This module tests the correction to dynamicsconsistencyfixedpointaudit.py:

dynamics should not be reconstructed only as a boundary-label/state map.  Gauge
lives in unresolved fibers and in the chart transports between different local
boundary charts over those fibers.  The effective object should therefore be a
finite *atlas morphism* rather than a full microscopic transition table.

Operationally, the audit:

    T
      -> boundary proliferation
      -> same-fiber chart atlas
      -> bounded finite atlas quotient
      -> temporal atlas face on quotient classes
      -> fiber-preserving lift for iteration

The key difference from the older dynamics fixed-point audit is the explicit
finite capacity bound.  Without such a bound, the atlas can refine toward the
full microstate space and erase the hidden fibers that make gauge possible.
Here, the observer is forced to remain finite: atlas classes are capped to a
chosen capacity, and fixed points are detected at the atlas-morphism level.

No target group, matter worldline, charge conservation, or Standard-Model-like
quantity is optimized.  C2/C3/etc. are reported post hoc from the chart atlas.

Example
-------
python -m relgauge.fiberatlasfixedpointaudit 2 ^
  --vertices 7 ^
  --rule-modes random_full_map,random_full_permutation,random_local_reversible,random_affine_bijection ^
  --instances 10 ^
  --atlas-capacities 8,16,32,64 ^
  --fixedpoint-iterations 8 ^
  --proliferation-iterations 4 ^
  --horizon 3 ^
  --max-state-samples 512 ^
  --max-total-states 4096 ^
  --atlas-lift-mode bijective ^
  --out example_results/fiber_atlas_fixedpoint_q2.csv ^
  --plot example_results/fig_fiber_atlas_fixedpoint_q2.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import dynamicsconsistencyfixedpointaudit as DCFP
except Exception:  # pragma: no cover
    import dynamicsconsistencyfixedpointaudit as DCFP  # type: ignore


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


def _parse_modes(text: str) -> List[str]:
    return [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]


def _entropy_from_counts(counts: Sequence[int]) -> float:
    total = float(sum(int(c) for c in counts))
    if total <= 0:
        return 0.0
    out = 0.0
    for c in counts:
        if int(c) <= 0:
            continue
        p = float(c) / total
        out -= p * math.log2(p)
    return float(out)


def _hash_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _hash_int_sequence(vals: Sequence[int]) -> str:
    arr = np.asarray(vals, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:20]


def compress_labels_by_capacity(labels: Sequence[int], capacity: int) -> np.ndarray:
    """Deterministically bound atlas labels to a finite observer capacity.

    The largest capacity-1 classes are preserved and all remaining rare classes
    are merged into a single residual class.  This is a conservative finite-
    observer operation: it prevents full microstate resolution while retaining
    the dominant live distinctions.

    If capacity is <= 1, all states are mapped to one label.  If the number of
    classes is already <= capacity, labels are just canonicalized.
    """
    arr = np.asarray(labels, dtype=np.int64)
    n = int(len(arr))
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    cap = int(max(1, capacity))
    vals = [int(x) for x in arr]
    counts = Counter(vals)
    if len(counts) <= cap:
        # canonicalize by first occurrence, via DCFP helper
        return DCFP.canonical_signature_labels([arr])
    if cap <= 1:
        return np.zeros(n, dtype=np.int32)
    # Keep the largest cap-1 classes, tie-broken by label id for reproducibility.
    keep = {lab for lab, _cnt in sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[: cap - 1]}
    residual_marker = -10**18
    coarse = np.asarray([int(x) if int(x) in keep else residual_marker for x in arr], dtype=np.int64)
    return DCFP.canonical_signature_labels([coarse])


def temporal_relation_stats(labels: Sequence[int], next_idx: Sequence[int]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Relation label(x) -> label(Tx) for a bounded atlas quotient."""
    lab = np.asarray(labels, dtype=np.int64)
    nxt = np.asarray(next_idx, dtype=np.int64)
    n = int(min(len(lab), len(nxt)))
    lab = lab[:n]
    nxt = nxt[:n]
    relation: Dict[int, Counter] = defaultdict(Counter)
    source_counts: Counter = Counter()
    for i in range(n):
        s = int(lab[i])
        j = int(nxt[i])
        t = int(lab[j]) if 0 <= j < n else s
        relation[s][t] += 1
        source_counts[s] += 1
    good = 0
    image = set()
    rows: List[Dict[str, object]] = []
    rel_for_hash: List[Tuple[int, int, int]] = []
    moving_edges = 0
    for s in sorted(relation):
        cnt = relation[s]
        tgt, c = cnt.most_common(1)[0]
        good += int(c)
        image.add(int(tgt))
        if int(tgt) != int(s):
            moving_edges += 1
        for t, cc in sorted(cnt.items()):
            rel_for_hash.append((int(s), int(t), int(cc)))
        rows.append({
            "source_atlas_class": int(s),
            "target_atlas_class": int(tgt),
            "source_count": int(source_counts[s]),
            "majority_count": int(c),
            "determinism_fraction": float(c / max(1, source_counts[s])),
            "n_possible_targets": int(len(cnt)),
            "moving_majority_edge": bool(int(tgt) != int(s)),
        })
    counts = Counter(int(x) for x in lab)
    h_labels = _entropy_from_counts(counts.values())
    cond_entropy = 0.0
    for s, cnt in relation.items():
        sc = float(source_counts[s])
        cond_entropy += (sc / max(1.0, float(n))) * _entropy_from_counts(cnt.values())
    stats = {
        "atlas_class_count": int(len(counts)),
        "atlas_entropy_bits": float(h_labels),
        "atlas_fiber_entropy_bits": float(max(0.0, math.log2(max(1, n)) - h_labels)),
        "temporal_relation_pair_count": int(sum(len(c) for c in relation.values())),
        "temporal_relation_determinism": float(good / max(1, n)),
        "temporal_relation_conditional_entropy_bits": float(cond_entropy),
        "temporal_relation_image_classes": int(len(image)),
        "temporal_relation_moving_majority_edges": int(moving_edges),
        "temporal_relation_hash": _hash_obj(rel_for_hash),
    }
    return stats, rows


def atlas_shape_signature(row_base: Dict[str, object], rel_stats: Dict[str, object]) -> str:
    """Canonical-ish hash of atlas architecture, not microstate transition map."""
    payload = {
        "capacity": _safe_int(row_base.get("atlas_capacity")),
        "classes": _safe_int(rel_stats.get("atlas_class_count")),
        "image": _safe_int(rel_stats.get("temporal_relation_image_classes")),
        "rel_hash": rel_stats.get("temporal_relation_hash"),
        "domain_depth_classes": row_base.get("domain_depth_class_signature"),
        "chart_nontrivial": _safe_int(row_base.get("n_chart_nontrivial")),
        "chart_c2": _safe_int(row_base.get("n_chart_c2")),
        "chart_c3": _safe_int(row_base.get("n_chart_c3")),
        "chart_order": _safe_int(row_base.get("max_chart_order")),
        "dependency_beta1": _safe_int(row_base.get("dependency_beta1")),
    }
    return _hash_obj(payload)


def domain_depth_class_signature(domain_rows: Sequence[Dict[str, object]], fp_iteration: int) -> str:
    """Compact canonical summary of generated domains by depth/n_labels."""
    ctr: Counter = Counter()
    for r in domain_rows:
        if _safe_int(r.get("fixedpoint_iteration"), fp_iteration) != int(fp_iteration):
            continue
        depth = _safe_int(r.get("depth"), 0)
        labs = _safe_int(r.get("n_labels"), 0)
        ent_bin = round(_safe_float(r.get("entropy_bits"), 0.0), 2)
        ctr[(depth, labs, ent_bin)] += 1
    return _hash_obj(sorted((str(k), int(v)) for k, v in ctr.items()))


def analyze_capacity_iteration(
    states,
    current_next: np.ndarray,
    q: int,
    rng: np.random.Generator,
    capacity: int,
    args,
) -> Tuple[object, np.ndarray, Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    """One bounded-atlas fixed-point step."""
    atlas, _old_eff, _old_eff_stats, _old_map_rows = DCFP.analyze_one_effective_step(
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
    bounded = compress_labels_by_capacity(atlas.signature_labels, int(capacity))
    rel_stats, rel_rows = temporal_relation_stats(bounded, current_next)
    eff, lift_stats, map_rows = DCFP.extract_effective_dynamics(bounded, current_next, lift_mode=str(args.atlas_lift_mode))
    # Replace map rows with bounded atlas relation rows plus lift diagnostics.
    for mr in map_rows:
        mr["bounded_atlas_capacity"] = int(capacity)
        mr["lift_mode"] = str(args.atlas_lift_mode)
    for rr in rel_rows:
        rr["bounded_atlas_capacity"] = int(capacity)
        rr["relation_type"] = "bounded_atlas_temporal_relation"
    map_rows = rel_rows + map_rows
    stats = dict(rel_stats)
    stats.update({f"lift_{k}": v for k, v in lift_stats.items()})
    return atlas, eff, stats, map_rows, list(atlas.chart_cycle_rows)


def run_fiber_atlas_fixedpoint_audit(
    q: int,
    vertices: int = 7,
    rule_modes: Sequence[str] = ("random_full_map",),
    instances: int = 3,
    atlas_capacities: Sequence[int] = (8, 16, 32, 64),
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
    atlas_lift_mode: str = "bijective",
    min_compression_fraction: float = 0.05,
    min_fiber_entropy_bits: float = 0.25,
    min_temporal_determinism: float = 0.95,
    seed: int = 0,
    verbose: bool = True,
):
    if pd is None:
        raise RuntimeError("pandas is required")
    q = int(q); vertices = int(vertices)
    rows: List[Dict[str, object]] = []
    domain_rows_all: List[Dict[str, object]] = []
    map_rows_all: List[Dict[str, object]] = []
    chart_cycle_rows_all: List[Dict[str, object]] = []

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
        max_cycles_per_fiber=max_cycles_per_fiber, atlas_lift_mode=atlas_lift_mode,
    ).items():
        setattr(args, k, v)

    for mi, mode in enumerate(list(rule_modes)):
        for inst in range(int(instances)):
            init_seed = int(seed) + 1000003 * (mi + 1) + 7919 * inst
            init_rng = np.random.default_rng(init_seed)
            states, initial_next, init_meta = DCFP.initialize_sampled_transition(
                q=q, vertices=vertices, mode=str(mode), rng=init_rng,
                max_state_samples=int(max_state_samples), max_total_states=int(max_total_states),
                max_pred=int(max_pred), proliferation_iterations=int(proliferation_iterations), horizon=int(horizon),
            )
            n_states = int(len(states))
            for cap in atlas_capacities:
                current_next = np.asarray(initial_next, dtype=np.int64).copy()
                seen_hashes: Dict[str, int] = {}
                fixed_found = False
                cycle_found = False
                gauge_fixed_found = False
                for fp in range(int(fixedpoint_iterations) + 1):
                    step_seed = int(init_seed) + 1299709 * int(cap) + 15485863 * int(fp)
                    rng = np.random.default_rng(step_seed)
                    atlas, eff, stats, map_rows, cycle_rows = analyze_capacity_iteration(
                        states, current_next, q, rng, int(cap), args
                    )
                    # Domain rows are useful for debugging, but avoid duplicating all domain rows for every capacity too much.
                    domain_sig = []
                    for dr in atlas.domain_rows:
                        rr = dict(dr)
                        rr.update({
                            "rule_mode": str(mode), "instance": int(inst), "atlas_capacity": int(cap),
                            "fixedpoint_iteration": int(fp), "initial_seed": int(init_seed),
                        })
                        domain_rows_all.append(rr)
                        domain_sig.append((_safe_int(dr.get("depth")), _safe_int(dr.get("n_labels")), round(_safe_float(dr.get("entropy_bits")), 2)))
                    depth_sig = _hash_obj(sorted(Counter(domain_sig).items(), key=lambda x: str(x[0])))
                    compression = float(1.0 - _safe_int(stats.get("atlas_class_count"), 0) / max(1, n_states))
                    bounded_nontrivial = bool(
                        compression >= float(min_compression_fraction)
                        and _safe_float(stats.get("atlas_fiber_entropy_bits"), 0.0) >= float(min_fiber_entropy_bits)
                        and _safe_int(stats.get("atlas_class_count"), 0) > 1
                    )
                    temporal_ok = bool(_safe_float(stats.get("temporal_relation_determinism"), 0.0) >= float(min_temporal_determinism))
                    gauge_present = bool(int(atlas.n_chart_nontrivial) > 0)
                    c2_present = bool(int(atlas.n_chart_c2) > 0)
                    row_base = {
                        "rule_mode": str(mode),
                        "instance": int(inst),
                        "initial_seed": int(init_seed),
                        "atlas_capacity": int(cap),
                        "fixedpoint_iteration": int(fp),
                        "q": int(q),
                        "vertices": int(vertices),
                        "n_states": int(n_states),
                        "domain_depth_class_signature": depth_sig,
                        "architecture_hash": str(atlas.architecture_hash),
                        "dependency_edges": int(atlas.n_dependency_edges),
                        "dependency_beta1": int(atlas.dependency_beta1),
                        "proliferation_nontriviality_score": float(atlas.nontriviality_score),
                        "full_signature_classes": int(atlas.n_signature_classes),
                        "full_signature_entropy_bits": float(atlas.signature_entropy_bits),
                        "bounded_atlas_classes": int(stats.get("atlas_class_count", 0)),
                        "bounded_atlas_entropy_bits": float(stats.get("atlas_entropy_bits", 0.0)),
                        "bounded_atlas_fiber_entropy_bits": float(stats.get("atlas_fiber_entropy_bits", 0.0)),
                        "bounded_atlas_compression_fraction": float(compression),
                        "bounded_nontrivial_atlas": bool(bounded_nontrivial),
                        "temporal_relation_determinism": float(stats.get("temporal_relation_determinism", 0.0)),
                        "temporal_relation_conditional_entropy_bits": float(stats.get("temporal_relation_conditional_entropy_bits", 0.0)),
                        "temporal_relation_image_classes": int(stats.get("temporal_relation_image_classes", 0)),
                        "temporal_relation_pair_count": int(stats.get("temporal_relation_pair_count", 0)),
                        "temporal_relation_moving_majority_edges": int(stats.get("temporal_relation_moving_majority_edges", 0)),
                        "temporal_relation_hash": str(stats.get("temporal_relation_hash", "")),
                        "temporal_relation_deterministic_enough": bool(temporal_ok),
                        "n_charts": int(atlas.n_charts),
                        "n_chart_cycles": int(atlas.n_chart_cycles),
                        "n_chart_valid_cycles": int(atlas.n_chart_valid_cycles),
                        "n_chart_nontrivial": int(atlas.n_chart_nontrivial),
                        "n_chart_c2": int(atlas.n_chart_c2),
                        "n_chart_c3": int(atlas.n_chart_c3),
                        "max_chart_order": int(atlas.max_chart_order),
                        "chart_holonomy_present": bool(gauge_present),
                        "chart_c2_present": bool(c2_present),
                        "atlas_lift_mode": str(atlas_lift_mode),
                    }
                    atlas_hash = atlas_shape_signature(row_base, stats)
                    row_base["atlas_morphism_hash"] = str(atlas_hash)
                    row_base["atlas_fixed_point"] = bool(atlas_hash in seen_hashes and seen_hashes[atlas_hash] == int(fp) - 1)
                    row_base["atlas_limit_cycle"] = bool(atlas_hash in seen_hashes and seen_hashes[atlas_hash] < int(fp) - 1)
                    row_base["atlas_cycle_length"] = int(fp - seen_hashes[atlas_hash]) if atlas_hash in seen_hashes else 0
                    row_base["gauge_bearing_atlas_fixed_point"] = bool(row_base["atlas_fixed_point"] and bounded_nontrivial and temporal_ok and gauge_present)
                    row_base["c2_bearing_atlas_fixed_point"] = bool(row_base["atlas_fixed_point"] and bounded_nontrivial and temporal_ok and c2_present)
                    row_base["capacity_window_candidate"] = bool(bounded_nontrivial and temporal_ok and gauge_present)
                    row_base.update({k: v for k, v in init_meta.items() if k not in row_base})
                    row_base.update(DCFP.transition_statistics(current_next, n_states=n_states, prefix="current_transition"))
                    row_base.update({f"atlas_lift_{k}": v for k, v in stats.items() if str(k).startswith("lift_")})
                    rows.append(row_base)
                    for mr in map_rows:
                        mm = dict(mr)
                        mm.update({"rule_mode": str(mode), "instance": int(inst), "atlas_capacity": int(cap), "fixedpoint_iteration": int(fp)})
                        map_rows_all.append(mm)
                    for cr in cycle_rows:
                        cc = dict(cr)
                        cc.update({"rule_mode": str(mode), "instance": int(inst), "atlas_capacity": int(cap), "fixedpoint_iteration": int(fp)})
                        chart_cycle_rows_all.append(cc)

                    if verbose:
                        print(
                            f"fiber-atlas mode={mode} inst={inst+1}/{instances} cap={cap} "
                            f"fp={fp}/{fixedpoint_iterations} cls={row_base['bounded_atlas_classes']} "
                            f"det={row_base['temporal_relation_determinism']:.3f} "
                            f"comp={row_base['bounded_atlas_compression_fraction']:.3f} "
                            f"c2={row_base['n_chart_c2']} afp={int(row_base['atlas_fixed_point'])} "
                            f"gfp={int(row_base['gauge_bearing_atlas_fixed_point'])}"
                        )
                    if row_base["atlas_fixed_point"]:
                        fixed_found = True
                    if row_base["atlas_limit_cycle"]:
                        cycle_found = True
                    if row_base["gauge_bearing_atlas_fixed_point"]:
                        gauge_fixed_found = True
                    seen_hashes.setdefault(atlas_hash, int(fp))
                    current_next = np.asarray(eff, dtype=np.int64)

    df = pd.DataFrame(rows)
    ddf = pd.DataFrame(domain_rows_all)
    mdf = pd.DataFrame(map_rows_all)
    cdf = pd.DataFrame(chart_cycle_rows_all)

    def frac(mask) -> float:
        try:
            return float(np.mean(mask)) if len(mask) else 0.0
        except Exception:
            return 0.0

    final = df.sort_values("fixedpoint_iteration").groupby(["rule_mode", "instance", "atlas_capacity"], as_index=False).tail(1) if len(df) else df
    cap_rows = []
    if len(final):
        for (mode, cap), g in final.groupby(["rule_mode", "atlas_capacity"]):
            cap_rows.append({
                "rule_mode": mode,
                "atlas_capacity": int(cap),
                "n": int(len(g)),
                "final_gauge_fraction": frac(g["chart_holonomy_present"].astype(bool)),
                "final_c2_fraction": frac(g["chart_c2_present"].astype(bool)),
                "final_capacity_window_fraction": frac(g["capacity_window_candidate"].astype(bool)),
                "final_gauge_bearing_atlas_fixed_point_fraction": frac(g["gauge_bearing_atlas_fixed_point"].astype(bool)),
                "mean_final_bounded_classes": float(g["bounded_atlas_classes"].mean()),
                "mean_final_fiber_entropy_bits": float(g["bounded_atlas_fiber_entropy_bits"].mean()),
                "mean_final_temporal_determinism": float(g["temporal_relation_determinism"].mean()),
                "mean_final_c2_count": float(g["n_chart_c2"].mean()),
            })
    by_capacity = pd.DataFrame(cap_rows)

    any_fixed = bool(len(df) and df["atlas_fixed_point"].astype(bool).any())
    any_gauge = bool(len(df) and df["chart_holonomy_present"].astype(bool).any())
    any_gauge_fp = bool(len(df) and df["gauge_bearing_atlas_fixed_point"].astype(bool).any())
    any_c2_fp = bool(len(df) and df["c2_bearing_atlas_fixed_point"].astype(bool).any())
    any_capacity_window = bool(len(df) and df["capacity_window_candidate"].astype(bool).any())
    if any_gauge_fp:
        verdict = "FIBER-ATLAS FIXED-POINT SIGNAL: bounded atlas morphism retains gauge-bearing fixed points"
    elif any_capacity_window:
        verdict = "FIBER-ATLAS CAPACITY-WINDOW SIGNAL: bounded atlases retain gauge during iteration, fixed point not established"
    elif any_gauge:
        verdict = "FIBER-ATLAS TRANSIENT GAUGE SIGNAL: full atlas sees gauge, bounded fixed point not established"
    else:
        verdict = "FIBER-ATLAS FLAT/WEAK SIGNAL: bounded atlas fixed points found without gauge"
    summary = {
        "verdict": verdict,
        "audit_version": "fiber_atlas_fixedpoint_v1_bounded_chart_atlas_morphism",
        "q": int(q),
        "vertices": int(vertices),
        "rule_modes": list(rule_modes),
        "instances": int(instances),
        "atlas_capacities": [int(c) for c in atlas_capacities],
        "fixedpoint_iterations": int(fixedpoint_iterations),
        "proliferation_iterations": int(proliferation_iterations),
        "horizon": int(horizon),
        "atlas_lift_mode": str(atlas_lift_mode),
        "n_rows": int(len(df)),
        "n_domain_rows": int(len(ddf)),
        "n_map_rows": int(len(mdf)),
        "n_chart_cycle_rows": int(len(cdf)),
        "any_atlas_fixed_point": bool(any_fixed),
        "any_atlas_limit_cycle": bool(len(df) and df["atlas_limit_cycle"].astype(bool).any()),
        "any_chart_holonomy_during_iteration": bool(any_gauge),
        "any_c2_during_iteration": bool(len(df) and df["chart_c2_present"].astype(bool).any()),
        "any_capacity_window_candidate": bool(any_capacity_window),
        "any_gauge_bearing_atlas_fixed_point": bool(any_gauge_fp),
        "any_c2_bearing_atlas_fixed_point": bool(any_c2_fp),
        "max_chart_c2_holonomy": int(df["n_chart_c2"].max()) if len(df) else 0,
        "max_chart_nontrivial_holonomy": int(df["n_chart_nontrivial"].max()) if len(df) else 0,
        "mean_bounded_atlas_compression_fraction": float(df["bounded_atlas_compression_fraction"].mean()) if len(df) else 0.0,
        "mean_bounded_atlas_fiber_entropy_bits": float(df["bounded_atlas_fiber_entropy_bits"].mean()) if len(df) else 0.0,
        "mean_temporal_relation_determinism": float(df["temporal_relation_determinism"].mean()) if len(df) else 0.0,
        "final_gauge_bearing_atlas_fixed_point_fraction": frac(final["gauge_bearing_atlas_fixed_point"].astype(bool)) if len(final) else 0.0,
        "final_capacity_window_candidate_fraction": frac(final["capacity_window_candidate"].astype(bool)) if len(final) else 0.0,
        "final_c2_fraction": frac(final["chart_c2_present"].astype(bool)) if len(final) else 0.0,
        "by_capacity": cap_rows,
    }
    return df, ddf, mdf, cdf, by_capacity, summary


def write_outputs(df, ddf, mdf, cdf, bycap, summary, out: str, plot: Optional[str] = None) -> None:
    base, ext = os.path.splitext(out)
    if not ext:
        out = base + ".csv"
        base, ext = os.path.splitext(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    ddf.to_csv(base + "_domains.csv", index=False)
    mdf.to_csv(base + "_atlas_maps.csv", index=False)
    cdf.to_csv(base + "_chart_cycles.csv", index=False)
    bycap.to_csv(base + "_by_capacity.csv", index=False)
    with open(base + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(14, 6))
            if len(df):
                grouped = df.groupby(["atlas_capacity", "fixedpoint_iteration"], as_index=False).agg({
                    "temporal_relation_determinism": "mean",
                    "bounded_atlas_compression_fraction": "mean",
                    "bounded_atlas_fiber_entropy_bits": "mean",
                    "n_chart_c2": "max",
                    "gauge_bearing_atlas_fixed_point": "mean",
                    "capacity_window_candidate": "mean",
                })
                # Plot one line per capacity for determinism/compression; c2 max on second axis.
                ax2 = ax1.twinx()
                for cap, g in grouped.groupby("atlas_capacity"):
                    g = g.sort_values("fixedpoint_iteration")
                    ax1.plot(g["fixedpoint_iteration"], g["temporal_relation_determinism"], marker="o", label=f"cap {cap}: determinism")
                    ax1.plot(g["fixedpoint_iteration"], g["bounded_atlas_compression_fraction"], marker="s", linestyle="--", label=f"cap {cap}: compression")
                    ax2.plot(g["fixedpoint_iteration"], g["n_chart_c2"], marker="^", linestyle=":", label=f"cap {cap}: max C2")
                ax1.set_xlabel("Fixed-point iteration: T -> bounded atlas temporal face -> T_eff")
                ax1.set_ylabel("fraction")
                ax2.set_ylabel("C2 chart holonomy count")
                ax1.set_title("Fiber-atlas fixed point: bounded finite chart atlas morphism")
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
            else:
                ax1.set_title("Fiber-atlas fixed point: no rows")
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
    print(f"wrote {base + '_by_capacity.csv'}")
    print(f"wrote {base + '_summary.json'}")
    if plot:
        print(f"wrote {plot}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fiber-atlas bounded fixed-point audit")
    p.add_argument("q", type=int)
    p.add_argument("--vertices", type=int, default=7)
    p.add_argument("--rule-modes", default="random_full_map")
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--atlas-capacities", default="8,16,32,64")
    p.add_argument("--fixedpoint-iterations", type=int, default=6)
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
    p.add_argument("--out", default="example_results/fiber_atlas_fixedpoint.csv")
    p.add_argument("--plot", default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    modes = _parse_modes(args.rule_modes)
    capacities = _parse_csv_ints(args.atlas_capacities, default=(8, 16, 32, 64))
    df, ddf, mdf, cdf, bycap, summary = run_fiber_atlas_fixedpoint_audit(
        q=int(args.q), vertices=int(args.vertices), rule_modes=modes,
        instances=int(args.instances), atlas_capacities=capacities,
        fixedpoint_iterations=int(args.fixedpoint_iterations), proliferation_iterations=int(args.proliferation_iterations),
        horizon=int(args.horizon), max_state_samples=int(args.max_state_samples), max_total_states=int(args.max_total_states),
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
        seed=int(args.seed), verbose=not bool(args.quiet),
    )
    write_outputs(df, ddf, mdf, cdf, bycap, summary, out=str(args.out), plot=(str(args.plot) or None))


if __name__ == "__main__":  # pragma: no cover
    main()
