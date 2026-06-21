"""
generatedcandidatephysicsreplayaudit.py

Replay atlas-level downstream physics only on *generated* gauge candidates.

This audit consumes the output of iteratedfiberatlasdynamicsaudit.py and
restricts all follow-up analysis to rows where a gauge/C2 sector was generated
by the atlas iteration rather than present at the initial atlas snapshot.

It does not optimize for C2, matter, charge, or conservation.  It is a gated
replay/diagnostic pass:

    iterated fiber-atlas dynamics output
      -> select generated candidates
      -> reconstruct the exact candidate atlas sequence
      -> replay atlas-level dynamic-sector, temporal, force/worldline, and
         endpoint/charge diagnostics on those candidates only.

The module is intentionally atlas-level.  It does not collapse the generated
candidate back to a boundary-only map before replaying; the replay keeps the
finite atlas profile/capacity that generated the candidate.

Example
-------
python -m relgauge.generatedcandidatephysicsreplayaudit 2 ^
  --iterated-csv example_results/iterated_fiber_atlas_q2.csv ^
  --attractors-csv example_results/iterated_fiber_atlas_q2_attractors.csv ^
  --vertices 7 ^
  --require-generated ^
  --max-candidates 20 ^
  --atlas-iterations 8 ^
  --out example_results/generated_candidate_replay_q2.csv ^
  --plot example_results/fig_generated_candidate_replay_q2.png
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
    from . import fiberatlasfixedpointaudit as FAF
    from . import iteratedfiberatlasdynamicsaudit as IFAD
except Exception:  # pragma: no cover
    import dynamicsconsistencyfixedpointaudit as DCFP  # type: ignore
    import fiberatlasfixedpointaudit as FAF  # type: ignore
    import iteratedfiberatlasdynamicsaudit as IFAD  # type: ignore



def _stable_text_hash(text, digits: int = 8) -> int:
    """Stable small integer hash; avoids Python process hash randomization."""
    return int(hashlib.sha1(str(text).encode("utf-8")).hexdigest()[: int(digits)], 16)

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


def _safe_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n", "nan", ""}:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return bool(x)


def _parse_csv_ints(text: str, default: Sequence[int]) -> List[int]:
    vals: List[int] = []
    for p in str(text or "").replace(";", ",").split(","):
        p = p.strip()
        if p:
            vals.append(int(float(p)))
    return vals or [int(x) for x in default]


def _parse_csv_text(text: str, default: Sequence[str]) -> List[str]:
    vals = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    return vals or [str(x) for x in default]


def _infer_attractors_path(iterated_csv: str, attractors_csv: str = "") -> str:
    if attractors_csv:
        return attractors_csv
    base, ext = os.path.splitext(iterated_csv)
    cand = base + "_attractors.csv"
    return cand if os.path.exists(cand) else ""


def _family_from_counts(c2: int, nontriv: int) -> str:
    if int(c2) > 0:
        return "C2"
    if int(nontriv) > 0:
        return "nontrivial"
    return "flat"


def _load_candidates(
    iterated_csv: str,
    attractors_csv: str = "",
    require_generated: bool = True,
    require_persistent: bool = False,
    include_initial_gauge: bool = False,
    profiles: Sequence[str] = (),
    capacities: Sequence[int] = (),
    rule_modes: Sequence[str] = (),
    max_candidates: int = 0,
) -> Tuple[object, object]:
    if pd is None:
        raise RuntimeError("pandas is required")
    df = pd.read_csv(iterated_csv)
    attr_path = _infer_attractors_path(iterated_csv, attractors_csv)
    if attr_path and os.path.exists(attr_path):
        adf = pd.read_csv(attr_path)
    else:
        # fall back to aggregating iteration rows
        keys = ["rule_mode", "instance", "profile", "atlas_capacity", "initial_seed"]
        recs: List[Dict[str, object]] = []
        for key, g in df.groupby(keys):
            g = g.sort_values("atlas_iteration")
            first = g.iloc[0]
            last = g.iloc[-1]
            recs.append({
                "rule_mode": key[0], "instance": int(key[1]), "profile": key[2],
                "atlas_capacity": int(key[3]), "initial_seed": int(key[4]),
                "initial_c2_count": _safe_int(first.get("initial_c2_count", first.get("n_chart_c2", 0))),
                "final_c2_count": _safe_int(last.get("n_chart_c2", 0)),
                "initial_gauge_count": _safe_int(first.get("initial_gauge_count", first.get("n_chart_nontrivial", 0))),
                "final_gauge_count": _safe_int(last.get("n_chart_nontrivial", 0)),
                "c2_generated_after_start_any": bool((g.get("c2_generated_after_start", False) == True).any()),
                "gauge_generated_after_start_any": bool((g.get("gauge_generated_after_start", False) == True).any()),
                "c2_persistent_to_final": bool(_safe_int(last.get("n_chart_c2", 0)) > 0),
                "gauge_persistent_to_final": bool(_safe_int(last.get("n_chart_nontrivial", 0)) > 0),
                "first_c2_iteration": _safe_int(g.loc[g.get("n_chart_c2", 0) > 0, "atlas_iteration"].min() if (g.get("n_chart_c2", 0) > 0).any() else -1, -1),
            })
        adf = pd.DataFrame(recs)

    if profiles:
        prof_set = {str(x) for x in profiles}
        adf = adf[adf["profile"].astype(str).isin(prof_set)]
    if capacities:
        cap_set = {int(x) for x in capacities}
        adf = adf[adf["atlas_capacity"].astype(int).isin(cap_set)]
    if rule_modes:
        mode_set = {str(x) for x in rule_modes}
        adf = adf[adf["rule_mode"].astype(str).isin(mode_set)]

    mask = np.ones(len(adf), dtype=bool)
    if require_generated:
        mask &= adf.get("c2_generated_after_start_any", False).map(_safe_bool).to_numpy()
        if not include_initial_gauge and "initial_c2_count" in adf.columns:
            mask &= (adf["initial_c2_count"].map(_safe_int).to_numpy() == 0)
    if require_persistent:
        mask &= adf.get("c2_persistent_to_final", False).map(_safe_bool).to_numpy()
    cand = adf[mask].copy()
    # Sort persistent, earlier generation, larger final count first.
    for col in ["c2_persistent_to_final", "final_c2_count", "first_c2_iteration"]:
        if col not in cand.columns:
            cand[col] = 0
    cand["_sort_persist"] = cand["c2_persistent_to_final"].map(lambda x: 1 if _safe_bool(x) else 0)
    cand["_sort_final_c2"] = cand["final_c2_count"].map(_safe_int)
    cand["_sort_first_c2"] = cand["first_c2_iteration"].map(lambda x: 999999 if _safe_int(x, -1) < 0 else _safe_int(x))
    cand = cand.sort_values(["_sort_persist", "_sort_final_c2", "_sort_first_c2"], ascending=[False, False, True])
    cand = cand.drop(columns=[c for c in ["_sort_persist", "_sort_final_c2", "_sort_first_c2"] if c in cand.columns])
    if int(max_candidates) > 0:
        cand = cand.head(int(max_candidates)).copy()
    return df, cand


def _make_args(**kwargs):
    class Args:
        pass
    a = Args()
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def _advance_effective(states, current_next, q: int, profile: str, capacity: int, rng, args, atlas_lift_mode: str):
    atlas = IFAD._one_atlas_pass(states, current_next, int(q), rng, args)
    bounded, pstats = IFAD._labels_for_profile(atlas, current_next, str(profile), int(capacity), int(args.max_signature_domains))
    rel_stats, rel_rows = FAF.temporal_relation_stats(bounded, current_next)
    eff, lift_stats, eff_rows = DCFP.extract_effective_dynamics(bounded, current_next, lift_mode=str(atlas_lift_mode))
    return atlas, bounded, pstats, rel_stats, rel_rows, eff, lift_stats, eff_rows


def _variation_distance(a: Sequence[int], b: Sequence[int]) -> float:
    ca = Counter(int(x) for x in a)
    cb = Counter(int(x) for x in b)
    keys = set(ca) | set(cb)
    na = float(max(1, sum(ca.values())))
    nb = float(max(1, sum(cb.values())))
    return float(0.5 * sum(abs(ca.get(k, 0) / na - cb.get(k, 0) / nb) for k in keys))


def _candidate_replay(
    cand_row: Dict[str, object],
    q: int,
    vertices: int,
    atlas_iterations: int,
    max_state_samples: int,
    max_total_states: int,
    max_pred: int,
    atlas_lift_mode: str,
    args,
    min_temporal_determinism: float,
    min_fiber_entropy_bits: float,
    min_class_count: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    mode = str(cand_row.get("rule_mode"))
    inst = _safe_int(cand_row.get("instance"), 0)
    profile = str(cand_row.get("profile", "fiber_preserving"))
    capacity = _safe_int(cand_row.get("atlas_capacity"), 0)
    seed = _safe_int(cand_row.get("initial_seed"), 0)
    init_rng = np.random.default_rng(seed)
    states, current_next, init_meta = DCFP.initialize_sampled_transition(
        q=int(q), vertices=int(vertices), mode=mode, rng=init_rng,
        max_state_samples=int(max_state_samples), max_total_states=int(max_total_states), max_pred=int(max_pred),
        proliferation_iterations=int(args.proliferation_iterations), horizon=int(args.horizon),
    )
    current_next = np.asarray(current_next, dtype=np.int64)
    rows: List[Dict[str, object]] = []
    transitions: List[Dict[str, object]] = []
    worldlines: List[Dict[str, object]] = []
    charges: List[Dict[str, object]] = []
    map_rows: List[Dict[str, object]] = []
    prev_row: Optional[Dict[str, object]] = None
    prev_labels: Optional[np.ndarray] = None
    family_seq: List[str] = []
    c2_seq: List[int] = []
    nontriv_seq: List[int] = []
    class_seq: List[int] = []
    det_seq: List[float] = []
    entropy_seq: List[float] = []
    effective_hashes: Dict[str, int] = {}
    fixed_iter = -1
    cycle_iter = -1

    candidate_id = f"{mode}|inst={inst}|{profile}|cap={capacity}|seed={seed}"
    for it in range(int(atlas_iterations) + 1):
        profile_hash = _stable_text_hash(profile) % 1000
        rng = np.random.default_rng(seed + 1299709 * int(capacity) + 15485863 * int(it) + 104729 * int(profile_hash))
        atlas, bounded, pstats, rel_stats, rel_rows, eff, lift_stats, eff_rows = _advance_effective(
            states, current_next, int(q), profile, int(capacity), rng, args, atlas_lift_mode
        )
        fam = _family_from_counts(int(atlas.n_chart_c2), int(atlas.n_chart_nontrivial))
        temporal_det = _safe_float(rel_stats.get("temporal_relation_determinism"), 0.0)
        moving_frac = _safe_float(rel_stats.get("temporal_relation_moving_edge_fraction", rel_stats.get("moving_edge_fraction", 0.0)), 0.0)
        class_count = _safe_int(pstats.get("bounded_atlas_classes"), 0)
        fiber_entropy = _safe_float(pstats.get("bounded_atlas_fiber_entropy_bits"), 0.0)
        defect_proxy = max(0.0, 1.0 - temporal_det) * (0.5 + 0.5 * max(0.0, moving_frac))
        eff_hash = IFAD._hash_arr(eff)
        if eff_hash in effective_hashes and cycle_iter < 0:
            cycle_iter = int(it)
        if np.array_equal(eff, current_next) and fixed_iter < 0:
            fixed_iter = int(it)
        effective_hashes.setdefault(eff_hash, int(it))
        generated_c2_now = bool(it > 0 and _safe_int(cand_row.get("initial_c2_count"), 0) == 0 and int(atlas.n_chart_c2) > 0)
        replay_dyn = bool(int(atlas.n_chart_c2) > 0 and temporal_det >= float(min_temporal_determinism) and fiber_entropy >= float(min_fiber_entropy_bits) and class_count >= int(min_class_count))
        row = {
            "candidate_id": candidate_id,
            "rule_mode": mode,
            "instance": int(inst),
            "profile": profile,
            "atlas_capacity": int(capacity),
            "initial_seed": int(seed),
            "replay_iteration": int(it),
            "q": int(q),
            "vertices": int(vertices),
            "n_states": int(len(states)),
            "initial_dynamics_kind": init_meta.get("initial_dynamics_kind", ""),
            "initial_dynamics_family": init_meta.get("initial_dynamics_family", ""),
            "random_start_unstructured": bool(init_meta.get("random_start_unstructured", False)),
            "local_reversible_start": bool(init_meta.get("local_reversible_start", False)),
            "affine_reversible_start": bool(init_meta.get("affine_reversible_start", False)),
            "atlas_family": fam,
            "chart_nontrivial_count": int(atlas.n_chart_nontrivial),
            "chart_c2_count": int(atlas.n_chart_c2),
            "chart_c3_count": int(atlas.n_chart_c3),
            "max_chart_order": int(atlas.max_chart_order),
            "chart_cycle_count": int(atlas.n_chart_cycles),
            "chart_valid_cycle_count": int(atlas.n_chart_valid_cycles),
            "generated_c2_now": bool(generated_c2_now),
            "replay_dynamic_sector_candidate": bool(replay_dyn),
            "temporal_relation_determinism": float(temporal_det),
            "temporal_relation_moving_edge_fraction": float(moving_frac),
            "temporal_defect_proxy": float(defect_proxy),
            "effective_fixed_point_now": bool(np.array_equal(eff, current_next)),
            "effective_limit_cycle_seen": bool(cycle_iter >= 0),
            "effective_hash": str(eff_hash),
            **pstats,
            **{f"rel_{k}": v for k, v in rel_stats.items()},
            **{f"lift_{k}": v for k, v in lift_stats.items()},
        }
        rows.append(row)
        # Store compact relation rows only; this is enough to audit temporal face without gigantic files.
        for rr in rel_rows:
            rrx = dict(rr)
            rrx.update({"candidate_id": candidate_id, "rule_mode": mode, "instance": int(inst), "profile": profile, "atlas_capacity": int(capacity), "replay_iteration": int(it), "row_kind": "temporal_relation"})
            map_rows.append(rrx)
        if prev_row is not None:
            labels_tv = _variation_distance(prev_labels if prev_labels is not None else [], bounded)
            changed = bool(prev_row["atlas_family"] != fam)
            became_c2 = bool(prev_row["atlas_family"] != "C2" and fam == "C2")
            left_c2 = bool(prev_row["atlas_family"] == "C2" and fam != "C2")
            trans = {
                "candidate_id": candidate_id,
                "rule_mode": mode,
                "instance": int(inst),
                "profile": profile,
                "atlas_capacity": int(capacity),
                "initial_seed": int(seed),
                "time_from": int(it - 1),
                "time_to": int(it),
                "family_from": str(prev_row["atlas_family"]),
                "family_to": fam,
                "family_changed": bool(changed),
                "flat_to_c2": bool(prev_row["atlas_family"] == "flat" and fam == "C2"),
                "nontrivial_to_flat": bool(prev_row["atlas_family"] != "flat" and fam == "flat"),
                "became_c2": bool(became_c2),
                "left_c2": bool(left_c2),
                "atlas_label_tv_from_prev": float(labels_tv),
                "delta_temporal_determinism": float(temporal_det - _safe_float(prev_row.get("temporal_relation_determinism"), 0.0)),
                "delta_class_count": int(class_count - _safe_int(prev_row.get("bounded_atlas_classes"), 0)),
                "delta_fiber_entropy_bits": float(fiber_entropy - _safe_float(prev_row.get("bounded_atlas_fiber_entropy_bits"), 0.0)),
                "transition_localized_proxy": bool(labels_tv <= 0.35 and (changed or became_c2 or left_c2)),
                "transition_force_proxy": float(abs(_safe_float(prev_row.get("temporal_defect_proxy"), 0.0) - defect_proxy) + labels_tv),
            }
            transitions.append(trans)
        prev_row = row
        prev_labels = np.asarray(bounded, dtype=np.int64)
        family_seq.append(fam)
        c2_seq.append(int(atlas.n_chart_c2))
        nontriv_seq.append(int(atlas.n_chart_nontrivial))
        class_seq.append(int(class_count))
        det_seq.append(float(temporal_det))
        entropy_seq.append(float(fiber_entropy))
        current_next = np.asarray(eff, dtype=np.int64)

    # Build simple atlas-worldline components: maximal runs of nonflat and maximal runs of C2.
    for family_target in ["nonflat", "C2"]:
        active = [f != "flat" if family_target == "nonflat" else f == "C2" for f in family_seq]
        start = None
        for i, a in enumerate(active + [False]):
            if a and start is None:
                start = i
            if (not a) and start is not None:
                end = i - 1
                span = end - start + 1
                idxs = list(range(start, end + 1))
                max_c2 = max(c2_seq[j] for j in idxs) if idxs else 0
                max_nontriv = max(nontriv_seq[j] for j in idxs) if idxs else 0
                class_range = int(max(class_seq[j] for j in idxs) - min(class_seq[j] for j in idxs)) if idxs else 0
                class_max = int(max(class_seq[j] for j in idxs)) if idxs else 1
                localized_proxy = bool(span >= 2 and class_range <= max(4, int(0.25 * max(1, class_max))))
                wg = {
                    "candidate_id": candidate_id,
                    "rule_mode": mode,
                    "instance": int(inst),
                    "profile": profile,
                    "atlas_capacity": int(capacity),
                    "initial_seed": int(seed),
                    "worldline_family": family_target,
                    "start_iteration": int(start),
                    "end_iteration": int(end),
                    "time_span": int(span),
                    "persistent_to_final": bool(end == len(family_seq) - 1),
                    "max_c2_count": int(max_c2),
                    "max_nontrivial_count": int(max_nontriv),
                    "mean_temporal_determinism": float(np.mean([det_seq[j] for j in idxs])) if idxs else 0.0,
                    "mean_fiber_entropy_bits": float(np.mean([entropy_seq[j] for j in idxs])) if idxs else 0.0,
                    "class_count_range": int(class_range),
                    "localized_worldline_proxy": bool(localized_proxy),
                    "strict_generated_c2_worldline": bool(family_target == "C2" and start > 0 and _safe_int(cand_row.get("initial_c2_count"), 0) == 0 and span >= 2),
                }
                worldlines.append(wg)
                start = None
    # Charge/endpoints = starts/ends of active C2/nonflat worldlines; q=2 parity recorded.
    for wl in worldlines:
        if str(wl.get("worldline_family")) not in {"C2", "nonflat"}:
            continue
        for endpoint_type, t in [("source", wl.get("start_iteration", 0)), ("sink", wl.get("end_iteration", 0))]:
            charges.append({
                "candidate_id": candidate_id,
                "rule_mode": mode,
                "instance": int(inst),
                "profile": profile,
                "atlas_capacity": int(capacity),
                "initial_seed": int(seed),
                "worldline_family": str(wl.get("worldline_family")),
                "endpoint_type": endpoint_type,
                "endpoint_iteration": int(t),
                "z2_endpoint": 1,
                "strict_generated_c2_worldline": bool(wl.get("strict_generated_c2_worldline", False)),
            })
    # Add per-candidate endpoint parity aggregate as rows with endpoint_type summary.
    c2_eps = [c for c in charges if c["worldline_family"] == "C2"]
    if c2_eps:
        charges.append({
            "candidate_id": candidate_id,
            "rule_mode": mode,
            "instance": int(inst),
            "profile": profile,
            "atlas_capacity": int(capacity),
            "initial_seed": int(seed),
            "worldline_family": "C2",
            "endpoint_type": "summary",
            "endpoint_iteration": -1,
            "endpoint_count": int(len(c2_eps)),
            "endpoint_parity": int(len(c2_eps) % 2),
            "z2_endpoint": int(len(c2_eps) % 2),
            "strict_generated_c2_worldline": bool(any(c.get("strict_generated_c2_worldline", False) for c in c2_eps)),
        })
    return rows, transitions, worldlines, charges, map_rows


def run_generated_candidate_physics_replay_audit(
    q: int,
    iterated_csv: str,
    attractors_csv: str = "",
    vertices: int = 7,
    require_generated: bool = True,
    require_persistent: bool = False,
    include_initial_gauge: bool = False,
    rule_modes: Sequence[str] = (),
    profiles: Sequence[str] = (),
    atlas_capacities: Sequence[int] = (),
    max_candidates: int = 30,
    atlas_iterations: int = 8,
    proliferation_iterations: int = 4,
    horizon: int = 3,
    max_state_samples: int = 512,
    max_total_states: int = 4096,
    max_pred: int = 3,
    initial_boundary: str = "sum_mod_q",
    initial_boundary_q: Optional[int] = None,
    atlas_lift_mode: str = "bijective",
    max_domains_per_depth: int = 32,
    min_live_classes: int = 2,
    min_fiber_size: int = 2,
    min_entropy_bits: float = 0.05,
    synergy_threshold: float = 0.01,
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
    min_temporal_determinism: float = 0.80,
    min_replay_fiber_entropy_bits: float = 0.25,
    min_replay_class_count: int = 2,
    verbose: bool = True,
):
    if pd is None:
        raise RuntimeError("pandas is required")
    _base_df, cand = _load_candidates(
        iterated_csv, attractors_csv=attractors_csv, require_generated=require_generated,
        require_persistent=require_persistent, include_initial_gauge=include_initial_gauge,
        profiles=profiles, capacities=atlas_capacities, rule_modes=rule_modes, max_candidates=max_candidates,
    )
    args = _make_args(
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
    )
    replay_rows: List[Dict[str, object]] = []
    transition_rows: List[Dict[str, object]] = []
    worldline_rows: List[Dict[str, object]] = []
    charge_rows: List[Dict[str, object]] = []
    map_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    for i, (_idx, cr) in enumerate(cand.iterrows()):
        cd = {k: cr[k] for k in cand.columns}
        manifest_rows.append({
            "candidate_rank": int(i),
            **{k: (bool(v) if isinstance(v, (np.bool_, bool)) else (int(v) if isinstance(v, (np.integer,)) else (float(v) if isinstance(v, (np.floating,)) else v))) for k, v in cd.items()},
        })
        if verbose:
            print(f"replay candidate {i+1}/{len(cand)} mode={cd.get('rule_mode')} inst={cd.get('instance')} profile={cd.get('profile')} cap={cd.get('atlas_capacity')} seed={cd.get('initial_seed')}")
        rr, tr, wr, ch, mr = _candidate_replay(
            cd, q=int(q), vertices=int(vertices), atlas_iterations=int(atlas_iterations),
            max_state_samples=int(max_state_samples), max_total_states=int(max_total_states),
            max_pred=int(max_pred), atlas_lift_mode=str(atlas_lift_mode), args=args,
            min_temporal_determinism=float(min_temporal_determinism),
            min_fiber_entropy_bits=float(min_replay_fiber_entropy_bits),
            min_class_count=int(min_replay_class_count),
        )
        replay_rows.extend(rr); transition_rows.extend(tr); worldline_rows.extend(wr); charge_rows.extend(ch); map_rows.extend(mr)

    rdf = pd.DataFrame(replay_rows)
    tdf = pd.DataFrame(transition_rows)
    wdf = pd.DataFrame(worldline_rows)
    cdf = pd.DataFrame(charge_rows)
    mdf = pd.DataFrame(manifest_rows)
    relmap_df = pd.DataFrame(map_rows)

    summary: Dict[str, object] = {
        "audit_version": "generated_candidate_physics_replay_v1_atlas_gated_downstream",
        "source_iterated_csv": str(iterated_csv),
        "source_attractors_csv": str(_infer_attractors_path(iterated_csv, attractors_csv)),
        "q": int(q),
        "vertices": int(vertices),
        "candidate_filter_require_generated": bool(require_generated),
        "candidate_filter_require_persistent": bool(require_persistent),
        "n_selected_candidates": int(len(mdf)),
        "n_replay_rows": int(len(rdf)),
        "n_transition_rows": int(len(tdf)),
        "n_worldline_rows": int(len(wdf)),
        "n_charge_rows": int(len(cdf)),
        "any_generated_candidate_replayed": bool(len(mdf) > 0),
        "any_replay_dynamic_sector_candidate": bool(len(rdf) and rdf.get("replay_dynamic_sector_candidate", pd.Series(dtype=bool)).map(_safe_bool).any()),
        "any_replay_flat_to_c2": bool(len(tdf) and tdf.get("flat_to_c2", pd.Series(dtype=bool)).map(_safe_bool).any()),
        "any_replay_c2_worldline": bool(len(wdf) and (wdf.get("worldline_family", pd.Series(dtype=str)).astype(str) == "C2").any()),
        "any_replay_strict_generated_c2_worldline": bool(len(wdf) and wdf.get("strict_generated_c2_worldline", pd.Series(dtype=bool)).map(_safe_bool).any()),
        "any_replay_charge_endpoint_pair": bool(len(cdf) and (cdf.get("endpoint_type", pd.Series(dtype=str)).astype(str) == "summary").any()),
        "max_replay_c2_count": int(rdf["chart_c2_count"].max()) if len(rdf) and "chart_c2_count" in rdf else 0,
        "max_replay_nontrivial_count": int(rdf["chart_nontrivial_count"].max()) if len(rdf) and "chart_nontrivial_count" in rdf else 0,
        "max_replay_worldline_span": int(wdf["time_span"].max()) if len(wdf) and "time_span" in wdf else 0,
        "max_replay_temporal_defect_proxy": float(rdf["temporal_defect_proxy"].max()) if len(rdf) and "temporal_defect_proxy" in rdf else 0.0,
    }
    if len(mdf):
        summary["selected_by_mode"] = {str(k): int(v) for k, v in mdf.groupby("rule_mode").size().to_dict().items()}
        summary["selected_by_profile"] = {str(k): int(v) for k, v in mdf.groupby("profile").size().to_dict().items()}
    if len(rdf):
        by_mode = []
        for mode, g in rdf.groupby("rule_mode"):
            wmode = wdf[wdf["rule_mode"] == mode] if len(wdf) and "rule_mode" in wdf else pd.DataFrame()
            by_mode.append({
                "rule_mode": str(mode),
                "n": int(len(g)),
                "max_c2_count": int(g["chart_c2_count"].max()),
                "mean_temporal_determinism": float(g["temporal_relation_determinism"].mean()),
                "dynamic_sector_candidate_fraction": float(g["replay_dynamic_sector_candidate"].map(_safe_bool).mean()),
                "c2_worldline_count": int(((wmode.get("worldline_family", pd.Series(dtype=str)).astype(str) == "C2")).sum()) if len(wmode) else 0,
                "strict_generated_c2_worldline_count": int(wmode.get("strict_generated_c2_worldline", pd.Series(dtype=bool)).map(_safe_bool).sum()) if len(wmode) else 0,
            })
        summary["by_mode"] = by_mode

    if not summary["any_generated_candidate_replayed"]:
        verdict = "GENERATED-CANDIDATE REPLAY EMPTY: no candidates matched the requested gate"
    elif summary["any_replay_strict_generated_c2_worldline"]:
        verdict = "GENERATED-CANDIDATE REPLAY SIGNAL: generated C2 candidates reproduce atlas-level downstream worldlines"
    elif summary["any_replay_dynamic_sector_candidate"]:
        verdict = "GENERATED-CANDIDATE REPLAY PARTIAL SIGNAL: generated C2 candidates reproduce dynamic atlas sectors"
    else:
        verdict = "GENERATED-CANDIDATE REPLAY WEAK SIGNAL: candidates selected but downstream atlas replay is weak"
    summary["verdict"] = verdict
    return rdf, tdf, wdf, cdf, mdf, relmap_df, summary


def write_outputs(base_out: str, rdf, tdf, wdf, cdf, mdf, relmap_df, summary: Dict[str, object], plot: str = ""):
    if pd is None:
        raise RuntimeError("pandas is required")
    out = str(base_out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    stem, ext = os.path.splitext(out)
    if not ext:
        out = out + ".csv"; stem = out[:-4]
    rdf.to_csv(out, index=False)
    tdf.to_csv(stem + "_transitions.csv", index=False)
    wdf.to_csv(stem + "_worldlines.csv", index=False)
    cdf.to_csv(stem + "_charges.csv", index=False)
    mdf.to_csv(stem + "_manifest.csv", index=False)
    # Map rows can be large; still useful for reproducibility.
    relmap_df.to_csv(stem + "_atlas_maps.csv", index=False)
    with open(stem + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    if plot:
        try:
            import matplotlib.pyplot as plt
            if len(rdf):
                g = rdf.groupby(["rule_mode", "replay_iteration"], as_index=False).agg(
                    mean_c2=("chart_c2_count", "mean"),
                    mean_nontriv=("chart_nontrivial_count", "mean"),
                    mean_det=("temporal_relation_determinism", "mean"),
                    mean_defect=("temporal_defect_proxy", "mean"),
                )
                fig, ax1 = plt.subplots(figsize=(12, 6))
                for mode, gg in g.groupby("rule_mode"):
                    ax1.plot(gg["replay_iteration"], gg["mean_det"], marker="o", label=f"{mode}: det")
                ax1.set_xlabel("Replay iteration")
                ax1.set_ylabel("temporal determinism")
                ax2 = ax1.twinx()
                for mode, gg in g.groupby("rule_mode"):
                    ax2.plot(gg["replay_iteration"], gg["mean_c2"], marker="^", linestyle="--", label=f"{mode}: C2")
                ax2.set_ylabel("mean C2 chart count")
                h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, fontsize=8, loc="best")
                ax1.set_title(str(summary.get("verdict", "Generated-candidate replay")))
                fig.tight_layout()
                os.makedirs(os.path.dirname(plot) or ".", exist_ok=True)
                fig.savefig(plot, dpi=150)
                plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"plot failed: {e}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Replay downstream atlas physics only on generated C2/gauge candidates")
    p.add_argument("q", type=int)
    p.add_argument("--iterated-csv", required=True)
    p.add_argument("--attractors-csv", default="")
    p.add_argument("--vertices", type=int, default=7)
    p.add_argument("--require-generated", action="store_true", help="Require C2 generated after initially non-C2 atlas")
    p.add_argument("--require-persistent", action="store_true", help="Require C2 persists to final iteration")
    p.add_argument("--include-initial-gauge", action="store_true")
    p.add_argument("--rule-modes", default="")
    p.add_argument("--profiles", default="")
    p.add_argument("--atlas-capacities", default="")
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--atlas-iterations", type=int, default=8)
    p.add_argument("--proliferation-iterations", type=int, default=4)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--max-state-samples", type=int, default=512)
    p.add_argument("--max-total-states", type=int, default=4096)
    p.add_argument("--max-pred", type=int, default=3)
    p.add_argument("--initial-boundary", default="sum_mod_q")
    p.add_argument("--initial-boundary-q", type=int, default=0)
    p.add_argument("--atlas-lift-mode", default="bijective", choices=["representative", "bijective"])
    p.add_argument("--max-domains-per-depth", type=int, default=32)
    p.add_argument("--min-live-classes", type=int, default=2)
    p.add_argument("--min-fiber-size", type=int, default=2)
    p.add_argument("--min-entropy-bits", type=float, default=0.05)
    p.add_argument("--synergy-threshold", type=float, default=0.01)
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
    p.add_argument("--min-temporal-determinism", type=float, default=0.80)
    p.add_argument("--min-replay-fiber-entropy-bits", type=float, default=0.25)
    p.add_argument("--min-replay-class-count", type=int, default=2)
    p.add_argument("--out", default="example_results/generated_candidate_replay_q2.csv")
    p.add_argument("--plot", default="")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    caps = _parse_csv_ints(args.atlas_capacities, []) if args.atlas_capacities else []
    profiles = _parse_csv_text(args.profiles, []) if args.profiles else []
    modes = _parse_csv_text(args.rule_modes, []) if args.rule_modes else []
    rdf, tdf, wdf, cdf, mdf, relmap_df, summary = run_generated_candidate_physics_replay_audit(
        q=int(args.q), iterated_csv=args.iterated_csv, attractors_csv=args.attractors_csv,
        vertices=int(args.vertices), require_generated=bool(args.require_generated), require_persistent=bool(args.require_persistent),
        include_initial_gauge=bool(args.include_initial_gauge), rule_modes=modes, profiles=profiles,
        atlas_capacities=caps, max_candidates=int(args.max_candidates), atlas_iterations=int(args.atlas_iterations),
        proliferation_iterations=int(args.proliferation_iterations), horizon=int(args.horizon),
        max_state_samples=int(args.max_state_samples), max_total_states=int(args.max_total_states), max_pred=int(args.max_pred),
        initial_boundary=str(args.initial_boundary), initial_boundary_q=(int(args.initial_boundary_q) if int(args.initial_boundary_q) > 0 else None),
        atlas_lift_mode=str(args.atlas_lift_mode), max_domains_per_depth=int(args.max_domains_per_depth),
        min_live_classes=int(args.min_live_classes), min_fiber_size=int(args.min_fiber_size), min_entropy_bits=float(args.min_entropy_bits),
        synergy_threshold=float(args.synergy_threshold), max_signature_domains=int(args.max_signature_domains),
        max_parent_domains=int(args.max_parent_domains), max_fibers_per_parent=int(args.max_fibers_per_parent),
        max_charts_per_fiber=int(args.max_charts_per_fiber), max_signature_charts=int(args.max_signature_charts),
        min_fiber_states=int(args.min_fiber_states), min_support_states=int(args.min_support_states),
        min_overlap_states=int(args.min_overlap_states), min_chart_classes=int(args.min_chart_classes),
        min_chart_entropy=float(args.min_chart_entropy), max_chart_coords=int(args.max_chart_coords), max_support_coords=int(args.max_support_coords),
        max_cycle_len=int(args.max_cycle_len), max_cycles_per_fiber=int(args.max_cycles_per_fiber),
        min_temporal_determinism=float(args.min_temporal_determinism), min_replay_fiber_entropy_bits=float(args.min_replay_fiber_entropy_bits),
        min_replay_class_count=int(args.min_replay_class_count), verbose=not bool(args.quiet),
    )
    write_outputs(args.out, rdf, tdf, wdf, cdf, mdf, relmap_df, summary, plot=args.plot)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
