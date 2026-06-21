"""
basepointawareholonomyaudit.py

Basepoint-aware chart-holonomy/isotropy classifier.

Why this module exists
----------------------
Earlier same-namespace transposition-diversity audits grouped chart-cycle loop
maps by (candidate, iteration, parent_domain_id, fiber_label), then treated the
integer labels appearing in different loop maps as if they lived in one common
label set.  That is only legitimate when the loops are based at the same chart.
In a chart atlas, a loop based at chart i is an automorphism of chart i's local
label set; a loop based at chart j is an automorphism of chart j's local label
set.  To compare them one must either conjugate along a transport path or, more
safely, classify isotropy at each base chart separately.

This audit implements the safe version.  For each reconstructed domain/fiber
atlas, and for each chart in that fiber, it enumerates all simple loops based at
that chart up to a specified length.  It collects the resulting loop maps without
conjugation (because they are already based at the same chart), groups maps by
their common label support, closes the finite permutation group, and reports the
basepoint isotropy.

A nonabelian result here is a genuine local chart isotropy signal, not a raw
integer-label artifact across different base charts.

Examples
--------
# Strict check of the known q=2,v=9 frozen transition.
python -m relgauge.basepointawareholonomyaudit 2 \
  --vertices 9 \
  --iterated-csv results/repro_witness_search.csv \
  --frozen-transition-npy data/transition_iter8_inst28_seed3221741.npy \
  --target-rule-mode random_full_permutation \
  --target-instance 28 \
  --target-profile full_atlas \
  --target-atlas-capacity 32 \
  --target-seed 3221741 \
  --target-iteration 8 \
  --target-parent-domain 75 \
  --target-fiber-label 7 \
  --out results/basepoint_isotropy_q2_v9_target.csv

# Replay search over candidates/iterations from an iterated CSV.
python -m relgauge.basepointawareholonomyaudit 2 \
  --vertices 10 \
  --iterated-csv example_results/iterated_fiber_atlas_q2_v10.csv \
  --rule-modes random_full_permutation \
  --profiles full_atlas \
  --atlas-capacities 32,64,128 \
  --atlas-iterations 12 \
  --max-candidates 40 \
  --stop-at-first-nonabelian \
  --out example_results/basepoint_isotropy_q2_v10_search.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("basepointawareholonomyaudit requires numpy and pandas") from e

try:
    from . import binarycompositegaugespectrumaudit as BCG
    from . import dynamicsconsistencyfixedpointaudit as DCFP
    from . import fiberchartconnectionaudit as FCA
    from . import generatedcandidatephysicsreplayaudit as GCPR
except Exception:  # pragma: no cover
    import binarycompositegaugespectrumaudit as BCG  # type: ignore
    import dynamicsconsistencyfixedpointaudit as DCFP  # type: ignore
    import fiberchartconnectionaudit as FCA  # type: ignore
    import generatedcandidatephysicsreplayaudit as GCPR  # type: ignore

LabelMap = Dict[int, int]
Perm = Tuple[int, ...]
EdgeKey = Tuple[int, int]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or (pd is not None and pd.isna(x)):
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (pd is not None and pd.isna(x)):
            return float(default)
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n", "nan", "", "none"}:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return bool(x)


def _stable_hash(text: Any) -> int:
    return int(hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8], 16)


def _rng_for_candidate_iteration(seed: int, capacity: int, profile: str, iteration: int) -> np.random.Generator:
    ph = _stable_hash(profile) % 1000
    return np.random.default_rng(int(seed) + 1299709 * int(capacity) + 15485863 * int(iteration) + 104729 * int(ph))


def _parse_csv_text(text: Any, default: Sequence[str] = ()) -> List[str]:
    vals = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    return vals or [str(x) for x in default]


def _parse_csv_ints(text: Any, default: Sequence[int] = ()) -> List[int]:
    vals: List[int] = []
    for p in str(text or "").replace(";", ",").split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(float(p)))
    return vals or [int(x) for x in default]


def _compact_map(m: LabelMap, limit: int = 50) -> str:
    items = sorted((int(k), int(v)) for k, v in (m or {}).items())
    if len(items) > int(limit):
        return " ".join(f"{a}->{b}" for a, b in items[: int(limit)]) + " ..."
    return " ".join(f"{a}->{b}" for a, b in items)


def _candidate_id(row: Dict[str, Any]) -> str:
    rm = str(row.get("rule_mode", "mode"))
    inst = str(_safe_int(row.get("instance", 0)))
    prof = str(row.get("profile", "profile"))
    cap = str(_safe_int(row.get("atlas_capacity", row.get("capacity", 0))))
    seed = str(_safe_int(row.get("initial_seed", row.get("seed", 0))))
    return f"{rm}|inst={inst}|{prof}|cap={cap}|seed={seed}"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _ensure_upstream_defaults(args: Any) -> None:
    defaults = {
        "proliferation_iterations": 4,
        "horizon": 3,
        "initial_boundary": "sum_mod_q",
        "initial_boundary_q": None,
        "max_domains_per_depth": 32,
        "min_live_classes": 2,
        "min_fiber_size": 2,
        "min_entropy_bits": 0.05,
        "synergy_threshold": 0.01,
        "max_signature_domains": 16,
        "max_parent_domains": 8,
        "max_fibers_per_parent": 6,
        "max_charts_per_fiber": 16,
        "max_signature_charts": 48,
        "min_fiber_states": 8,
        "min_support_states": 4,
        "min_overlap_states": 4,
        "min_chart_classes": 2,
        "min_chart_entropy": 0.05,
        "max_chart_coords": 5,
        "max_support_coords": 4,
        "max_cycle_len": 4,
        "max_cycles_per_fiber": 500,
        "max_state_samples": 512,
        "max_total_states": 200000,
        "max_pred": 0,
        "atlas_lift_mode": "bijective",
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

# ---------------------------------------------------------------------------
# Group closure / classification
# ---------------------------------------------------------------------------
def _perm_from_label_map(m: LabelMap, labels: Sequence[int]) -> Optional[Perm]:
    labs = [int(x) for x in labels]
    idx = {x: i for i, x in enumerate(labs)}
    out: List[int] = []
    for x in labs:
        if int(x) not in m:
            return None
        y = int(m[int(x)])
        if y not in idx:
            return None
        out.append(idx[y])
    if len(set(out)) != len(out):
        return None
    return tuple(out)


def _is_abelian(group: Sequence[Perm]) -> bool:
    g = list(group)
    for a in g:
        for b in g:
            if BCG._compose(a, b) != BCG._compose(b, a):
                return False
    return True


def _group_orbits(group: Sequence[Perm], n: int) -> List[List[int]]:
    seen = [False] * int(n)
    out: List[List[int]] = []
    for i in range(int(n)):
        if seen[i]:
            continue
        orb = {i}
        changed = True
        while changed:
            changed = False
            for g in group:
                for x in list(orb):
                    y = int(g[x])
                    if y not in orb:
                        orb.add(y); changed = True
        for x in orb:
            seen[x] = True
        out.append(sorted(orb))
    return sorted(out, key=lambda z: (-len(z), z))


def _classify_group(perms: Sequence[Perm], support_labels: Sequence[int], max_group_order: int = 4096) -> Dict[str, Any]:
    distinct: List[Perm] = []
    for p in perms:
        if p not in distinct:
            distinct.append(p)
    if not distinct:
        return {
            "isotropy_family": "trivial_or_absent",
            "generated_group_order": 1,
            "nonabelian": False,
            "exact_s3": False,
            "element_order_counts": {"1": 1},
            "max_orbit_size": 1 if support_labels else 0,
            "truncated": False,
        }
    group, truncated = BCG._close_group(distinct, max_group_order=int(max_group_order))
    orders = Counter(BCG._perm_order(g) for g in group)
    abelian = _is_abelian(list(group)) if not truncated and len(group) <= int(max_group_order) else False
    n = len(support_labels)
    orbits = _group_orbits(list(group), n) if group else []
    max_orbit = max([len(o) for o in orbits] or [0])
    exact_s3 = bool((not truncated) and len(group) == 6 and orders.get(1, 0) == 1 and orders.get(2, 0) == 3 and orders.get(3, 0) == 2 and not abelian)
    if exact_s3:
        fam = "S3_basepoint_isotropy"
    elif truncated:
        fam = "large_or_truncated_nonabelian_candidate" if not abelian else "large_or_truncated"
    elif not abelian:
        fam = f"nonabelian_order_{len(group)}"
    elif len(group) == 1:
        fam = "trivial"
    elif len(group) == 2 and orders.get(2, 0) == 1:
        fam = "C2_basepoint_isotropy"
    elif len(group) == 3 and orders.get(3, 0) == 2:
        fam = "C3_basepoint_isotropy"
    elif len(group) == 4:
        fam = "abelian_order4_basepoint_isotropy"
    else:
        fam = f"abelian_order_{len(group)}"
    return {
        "isotropy_family": fam,
        "generated_group_order": int(len(group)),
        "nonabelian": bool((not truncated) and not abelian),
        "exact_s3": bool(exact_s3),
        "abelian": bool(abelian) if not truncated else False,
        "truncated": bool(truncated),
        "element_order_counts": {str(k): int(v) for k, v in sorted(orders.items())},
        "max_orbit_size": int(max_orbit),
        "orbit_count": int(len(orbits)),
    }

# ---------------------------------------------------------------------------
# Basepoint loop enumeration
# ---------------------------------------------------------------------------
def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (int(min(a, b)), int(max(a, b)))


def _adjacency(n_charts: int, edge_maps: Dict[EdgeKey, Any]) -> Dict[int, List[int]]:
    adj: Dict[int, Set[int]] = {i: set() for i in range(int(n_charts))}
    for a, b in edge_maps.keys():
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    return {k: sorted(v) for k, v in adj.items()}


def enumerate_basepoint_loops(base: int, n_charts: int, edge_maps: Dict[EdgeKey, Any], max_cycle_len: int = 4, max_loops: int = 1000) -> List[List[int]]:
    """Enumerate simple loops based at ``base``.

    The returned cycle is a list of chart ids without repeating the base at the
    end; ``FCA.cycle_map`` appends the start internally.  Reverse orientations are
    retained because they represent inverse holonomies at the same base.
    """
    base = int(base)
    adj = _adjacency(int(n_charts), edge_maps)
    loops: List[List[int]] = []
    seen: Set[Tuple[int, ...]] = set()

    def dfs(path: List[int], target_len: int) -> None:
        if len(loops) >= int(max_loops):
            return
        cur = path[-1]
        if len(path) == target_len:
            if _edge_key(cur, base) in edge_maps:
                tup = tuple(path)
                if tup not in seen:
                    seen.add(tup); loops.append(list(path))
            return
        for nb in adj.get(cur, []):
            if nb == base:
                continue
            if nb in path:
                continue
            if _edge_key(cur, nb) not in edge_maps:
                continue
            path.append(nb)
            dfs(path, target_len)
            path.pop()
            if len(loops) >= int(max_loops):
                return

    for L in range(3, int(max_cycle_len) + 1):
        dfs([base], L)
        if len(loops) >= int(max_loops):
            break
    return loops


def _domain_list(atlas: Any) -> List[Any]:
    out: List[Any] = []
    seen: Set[int] = set()
    for seq_name in ["domains_current", "domains_all"]:
        for d in list(getattr(atlas, seq_name, []) or []):
            did = int(getattr(d, "domain_id", -1))
            if did in seen:
                continue
            seen.add(did); out.append(d)
    return out


def _analyze_charts_for_domain_fiber(
    charts: Sequence[Any],
    edge_maps: Dict[EdgeKey, Any],
    meta: Dict[str, Any],
    max_cycle_len: int,
    max_loops_per_base: int,
    max_group_order: int,
    include_trivial: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows: List[Dict[str, Any]] = []
    loop_rows: List[Dict[str, Any]] = []
    n_charts = int(len(charts))
    for base in range(n_charts):
        loops = enumerate_basepoint_loops(base, n_charts, edge_maps, max_cycle_len=max_cycle_len, max_loops=max_loops_per_base)
        by_support: Dict[Tuple[int, ...], List[Tuple[List[int], LabelMap, int]]] = defaultdict(list)
        valid_count = 0; nontriv_count = 0; c2_count = 0
        for cyc in loops:
            valid, mp, reason, order = FCA.cycle_map(cyc, edge_maps)
            base_support = tuple(sorted(int(x) for x in mp.keys())) if mp else tuple()
            lr = dict(meta)
            lr.update({
                "base_chart": int(base),
                "cycle_charts": " ".join(str(x) for x in cyc),
                "cycle_length": int(len(cyc)),
                "loop_valid": bool(valid),
                "loop_order": int(order),
                "loop_nontrivial": bool(valid and order > 1),
                "loop_c2": bool(valid and order == 2 and not all(int(k) == int(v) for k, v in mp.items())),
                "loop_map": _compact_map(mp),
                "loop_label_support": " ".join(str(x) for x in base_support),
                "failure_reason": str(reason),
            })
            loop_rows.append(lr)
            if valid:
                valid_count += 1
                if order > 1 and not all(int(k) == int(v) for k, v in mp.items()):
                    nontriv_count += 1
                    if order == 2:
                        c2_count += 1
                    by_support[base_support].append((cyc, {int(k): int(v) for k, v in mp.items()}, int(order)))
        if not by_support and include_trivial:
            gr = dict(meta)
            gr.update({
                "base_chart": int(base),
                "base_chart_type": str(getattr(charts[base], "chart_type", "")),
                "base_chart_label_desc": str(getattr(charts[base], "label_desc", "")),
                "base_chart_support_desc": str(getattr(charts[base], "support_desc", "")),
                "base_chart_n_support": int(getattr(charts[base], "n_support", 0)),
                "base_chart_n_labels": int(getattr(charts[base], "n_labels", 0)),
                "loops_enumerated": int(len(loops)),
                "valid_loop_count": int(valid_count),
                "nontrivial_loop_count": int(nontriv_count),
                "c2_loop_count": int(c2_count),
                "loop_label_support": "",
                "loop_label_support_size": 0,
                "distinct_loop_maps": 0,
                "generated_group_order": 1,
                "isotropy_family": "trivial_or_absent",
                "nonabelian": False,
                "exact_s3": False,
            })
            group_rows.append(gr)
        for support_key, entries in by_support.items():
            labels = list(support_key)
            perms: List[Perm] = []
            distinct_maps: Dict[Tuple[Tuple[int, int], ...], Tuple[LabelMap, int, List[str]]] = {}
            for cyc, mp, order in entries:
                p = _perm_from_label_map(mp, labels)
                if p is None:
                    continue
                perms.append(p)
                key = tuple(sorted((int(k), int(v)) for k, v in mp.items()))
                distinct_maps.setdefault(key, (mp, order, []))[2].append(" ".join(str(x) for x in cyc))
            cls = _classify_group(perms, labels, max_group_order=max_group_order)
            if (not include_trivial) and cls["generated_group_order"] <= 1:
                continue
            # For small supports, list the distinct maps so reviewer can see whether
            # multiple loops at the same base actually disagree.
            map_summ = []
            for _key, (mp, order, paths) in sorted(distinct_maps.items(), key=lambda kv: (kv[1][1], kv[0])):
                map_summ.append(f"order{order}:{_compact_map(mp)} via {len(paths)} loop(s)")
            gr = dict(meta)
            gr.update({
                "base_chart": int(base),
                "base_chart_type": str(getattr(charts[base], "chart_type", "")),
                "base_chart_label_desc": str(getattr(charts[base], "label_desc", "")),
                "base_chart_support_desc": str(getattr(charts[base], "support_desc", "")),
                "base_chart_n_support": int(getattr(charts[base], "n_support", 0)),
                "base_chart_n_labels": int(getattr(charts[base], "n_labels", 0)),
                "loops_enumerated": int(len(loops)),
                "valid_loop_count": int(valid_count),
                "nontrivial_loop_count": int(nontriv_count),
                "c2_loop_count": int(c2_count),
                "loop_label_support": " ".join(str(x) for x in labels),
                "loop_label_support_size": int(len(labels)),
                "distinct_loop_maps": int(len(distinct_maps)),
                "distinct_loop_map_summary": " || ".join(map_summ[:20]),
            })
            gr.update(cls)
            gr["element_order_counts_json"] = json.dumps(cls.get("element_order_counts", {}), sort_keys=True)
            group_rows.append(gr)
    return group_rows, loop_rows

# ---------------------------------------------------------------------------
# Reconstruct candidates/atlases
# ---------------------------------------------------------------------------
def _select_target_row(df: pd.DataFrame, args: Any) -> Dict[str, Any]:
    work = df.copy()
    mask = np.ones(len(work), dtype=bool)
    if str(args.target_rule_mode):
        mask &= work.get("rule_mode", pd.Series([""] * len(work))).astype(str).eq(str(args.target_rule_mode))
    if int(args.target_instance) >= 0:
        mask &= pd.to_numeric(work.get("instance", pd.Series([-1] * len(work))), errors="coerce").fillna(-1).astype(int).eq(int(args.target_instance))
    if str(args.target_profile):
        mask &= work.get("profile", pd.Series([""] * len(work))).astype(str).eq(str(args.target_profile))
    if int(args.target_atlas_capacity) >= 0:
        cap_col = "atlas_capacity" if "atlas_capacity" in work.columns else "capacity"
        if cap_col in work.columns:
            mask &= pd.to_numeric(work[cap_col], errors="coerce").fillna(-1).astype(int).eq(int(args.target_atlas_capacity))
    if int(args.target_seed) >= 0:
        seed_col = "initial_seed" if "initial_seed" in work.columns else "seed"
        if seed_col in work.columns:
            mask &= pd.to_numeric(work[seed_col], errors="coerce").fillna(-1).astype(int).eq(int(args.target_seed))
    cand = work[mask].copy()
    if cand.empty:
        raise ValueError("No candidate row matched target filters")
    if "atlas_iteration" in cand.columns and int(args.target_iteration) >= 0:
        cand2 = cand[pd.to_numeric(cand["atlas_iteration"], errors="coerce").fillna(-999).astype(int).eq(int(args.target_iteration))]
        if not cand2.empty:
            cand = cand2
    return cand.iloc[0].to_dict()


def _select_search_candidates(df: pd.DataFrame, args: Any) -> pd.DataFrame:
    work = df.copy()
    # Collapse to candidate-level rows.
    keys = [c for c in ["rule_mode", "instance", "profile", "atlas_capacity", "initial_seed"] if c in work.columns]
    if keys:
        agg_cols: Dict[str, Tuple[str, str]] = {}
        for c in ["c2_generated_after_start", "c2_generated_after_start_any", "initial_c2_count", "first_c2_iteration", "n_chart_c2"]:
            if c in work.columns:
                agg_cols[c] = (c, "max" if c not in {"first_c2_iteration"} else "min")
        if agg_cols:
            work = work.groupby(keys, as_index=False).agg(**agg_cols)
        else:
            work = work.drop_duplicates(subset=keys)
    mask = np.ones(len(work), dtype=bool)
    allowed_modes = _parse_csv_text(args.rule_modes, [])
    allowed_profiles = _parse_csv_text(args.profiles, [])
    allowed_caps = _parse_csv_ints(args.atlas_capacities, [])
    if allowed_modes and "rule_mode" in work.columns:
        mask &= work["rule_mode"].astype(str).isin(allowed_modes)
    if allowed_profiles and "profile" in work.columns:
        mask &= work["profile"].astype(str).isin(allowed_profiles)
    if allowed_caps:
        cap_col = "atlas_capacity" if "atlas_capacity" in work.columns else "capacity"
        if cap_col in work.columns:
            mask &= pd.to_numeric(work[cap_col], errors="coerce").fillna(-1).astype(int).isin(allowed_caps)
    if args.require_generated:
        gen_col = "c2_generated_after_start_any" if "c2_generated_after_start_any" in work.columns else "c2_generated_after_start"
        if gen_col in work.columns:
            mask &= work[gen_col].map(_safe_bool).to_numpy()
    out = work[mask].copy()
    if "first_c2_iteration" in out.columns:
        out = out.sort_values(["first_c2_iteration"] + [c for c in ["rule_mode", "instance", "profile", "atlas_capacity"] if c in out.columns])
    if int(args.max_candidates) > 0:
        out = out.head(int(args.max_candidates)).copy()
    return out.reset_index(drop=True)


def _initialize_candidate(row: Dict[str, Any], q: int, vertices: int, args: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    mode = str(row.get("rule_mode", args.target_rule_mode or "random_full_permutation"))
    seed = _safe_int(row.get("initial_seed", row.get("seed", args.target_seed)), 0)
    rng0 = np.random.default_rng(seed)
    states, current_next, init_meta = DCFP.initialize_sampled_transition(
        q=int(q), vertices=int(vertices), mode=mode, rng=rng0,
        max_state_samples=int(args.max_state_samples), max_total_states=int(args.max_total_states), max_pred=int(args.max_pred),
        proliferation_iterations=int(args.proliferation_iterations), horizon=int(args.horizon),
    )
    return np.asarray(states, dtype=np.int64), np.asarray(current_next, dtype=np.int64), init_meta


def _build_atlas_from_transition(states: np.ndarray, current_next: np.ndarray, q: int, row: Dict[str, Any], iteration: int, args: Any) -> Tuple[Any, np.ndarray]:
    profile = str(row.get("profile", args.target_profile or "full_atlas"))
    cap = _safe_int(row.get("atlas_capacity", row.get("capacity", args.target_atlas_capacity)), 32)
    seed = _safe_int(row.get("initial_seed", row.get("seed", args.target_seed)), 0)
    rng = _rng_for_candidate_iteration(seed, cap, profile, int(iteration))
    atlas, _bounded, _pstats, _rel_stats, _rel_rows, eff, _lift_stats, _eff_rows = GCPR._advance_effective(
        states, np.asarray(current_next, dtype=np.int64), int(q), profile, int(cap), rng, args, str(args.atlas_lift_mode)
    )
    return atlas, np.asarray(eff, dtype=np.int64)


def _analyze_atlas_basepoints(atlas: Any, states: np.ndarray, current_next: np.ndarray, q: int, row: Dict[str, Any], iteration: int, args: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows: List[Dict[str, Any]] = []
    loop_rows: List[Dict[str, Any]] = []
    candidate = _candidate_id(row)
    domains = _domain_list(atlas)
    target_parent = int(args.target_parent_domain)
    target_fiber = int(args.target_fiber_label)
    n_domains_seen = 0
    for domain in domains:
        did = int(getattr(domain, "domain_id", -1))
        if target_parent >= 0 and did != target_parent:
            continue
        labels = np.asarray(getattr(domain, "labels", []), dtype=np.int64)
        if len(labels) == 0:
            continue
        vals, counts = np.unique(labels, return_counts=True)
        # Analyze larger/eligible fibers first.  If target fiber is set, only that one.
        pairs = [(int(v), int(c)) for v, c in zip(vals, counts) if int(c) >= int(args.min_fiber_states)]
        pairs.sort(key=lambda x: (-x[1], x[0]))
        if target_fiber >= 0:
            pairs = [(v, c) for v, c in pairs if int(v) == target_fiber]
        if int(args.max_fibers_per_domain_scan) > 0 and target_fiber < 0:
            pairs = pairs[: int(args.max_fibers_per_domain_scan)]
        for fiber_label, fiber_count in pairs:
            charts = FCA.build_charts_for_domain_fiber(
                domain, int(fiber_label), states, current_next, int(q), int(args.horizon),
                max_chart_coords=int(args.max_chart_coords), max_support_coords=int(args.max_support_coords),
                max_charts_per_fiber=int(args.max_charts_per_fiber),
                min_chart_classes=int(args.min_chart_classes), min_chart_entropy=float(args.min_chart_entropy),
                min_support_states=int(args.min_support_states),
            )
            if len(charts) < 3:
                continue
            _tr_rows, edge_maps = FCA.build_chart_transports(charts, min_overlap_states=int(args.min_overlap_states))
            if not edge_maps:
                continue
            meta = {
                "candidate_id": candidate,
                "rule_mode": str(row.get("rule_mode", "")),
                "instance": _safe_int(row.get("instance", -1), -1),
                "profile": str(row.get("profile", "")),
                "atlas_capacity": _safe_int(row.get("atlas_capacity", row.get("capacity", -1)), -1),
                "initial_seed": _safe_int(row.get("initial_seed", row.get("seed", -1)), -1),
                "atlas_iteration": int(iteration),
                "parent_domain_id": int(did),
                "fiber_label": int(fiber_label),
                "fiber_state_count": int(fiber_count),
                "n_charts": int(len(charts)),
                "n_transport_edges": int(len(edge_maps)),
            }
            gr, lr = _analyze_charts_for_domain_fiber(
                charts, edge_maps, meta,
                max_cycle_len=int(args.max_cycle_len),
                max_loops_per_base=int(args.max_loops_per_base),
                max_group_order=int(args.max_group_order),
                include_trivial=bool(args.include_trivial),
            )
            group_rows.extend(gr); loop_rows.extend(lr)
        n_domains_seen += 1
        if int(args.max_domains_scan) > 0 and target_parent < 0 and n_domains_seen >= int(args.max_domains_scan):
            break
    return group_rows, loop_rows

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_basepoint_aware_holonomy_audit(
    q: int,
    vertices: int = 9,
    iterated_csv: str = "",
    frozen_transition_npy: str = "",
    target_rule_mode: str = "",
    target_instance: int = -1,
    target_profile: str = "",
    target_atlas_capacity: int = -1,
    target_seed: int = -1,
    target_iteration: int = 8,
    target_parent_domain: int = -1,
    target_fiber_label: int = -1,
    rule_modes: str = "",
    profiles: str = "",
    atlas_capacities: str = "",
    max_candidates: int = 20,
    atlas_iterations: int = 12,
    require_generated: bool = False,
    stop_at_first_nonabelian: bool = False,
    max_state_samples: int = 512,
    max_total_states: int = 200000,
    max_pred: int = 0,
    proliferation_iterations: int = 4,
    horizon: int = 3,
    initial_boundary: str = "sum_mod_q",
    initial_boundary_q: Optional[int] = None,
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
    max_loops_per_base: int = 1000,
    max_group_order: int = 4096,
    max_domains_scan: int = 0,
    max_fibers_per_domain_scan: int = 0,
    atlas_lift_mode: str = "bijective",
    include_trivial: bool = False,
    out: str = "",
    plot: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    args = argparse.Namespace(**locals())
    _ensure_upstream_defaults(args)

    if not iterated_csv or not os.path.exists(str(iterated_csv)):
        raise ValueError("--iterated-csv is required and must exist")
    idf = pd.read_csv(str(iterated_csv), low_memory=False)
    group_rows: List[Dict[str, Any]] = []
    loop_rows: List[Dict[str, Any]] = []

    # Mode A: exact frozen transition / target candidate.
    if str(frozen_transition_npy).strip():
        target_row = _select_target_row(idf, args)
        states, current_next, _init_meta = _initialize_candidate(target_row, int(q), int(vertices), args)
        loaded_next = np.asarray(np.load(str(frozen_transition_npy)), dtype=np.int64)
        if len(loaded_next) != len(states):
            raise ValueError(f"frozen transition length {len(loaded_next)} != state reservoir length {len(states)}")
        current_next = loaded_next
        atlas, _eff = _build_atlas_from_transition(states, current_next, int(q), target_row, int(target_iteration), args)
        gr, lr = _analyze_atlas_basepoints(atlas, states, current_next, int(q), target_row, int(target_iteration), args)
        group_rows.extend(gr); loop_rows.extend(lr)
    # Mode B: replay search across selected candidates and iterations.
    else:
        cdf = _select_search_candidates(idf, args)
        for ci, row in enumerate(cdf.to_dict(orient="records"), start=1):
            states, current_next, _init_meta = _initialize_candidate(row, int(q), int(vertices), args)
            for it in range(int(atlas_iterations) + 1):
                atlas, eff = _build_atlas_from_transition(states, current_next, int(q), row, int(it), args)
                gr, lr = _analyze_atlas_basepoints(atlas, states, current_next, int(q), row, int(it), args)
                group_rows.extend(gr); loop_rows.extend(lr)
                if stop_at_first_nonabelian and any(bool(r.get("nonabelian")) for r in gr):
                    break
                current_next = eff
            if stop_at_first_nonabelian and any(bool(r.get("nonabelian")) for r in group_rows):
                break

    rdf = pd.DataFrame(group_rows)
    ldf = pd.DataFrame(loop_rows)
    if rdf.empty:
        summary = {
            "audit_version": "basepoint_aware_holonomy_v1_chart_isotropy",
            "verdict": "BASEPOINT-AWARE HOLONOMY EMPTY/FLAT: no nontrivial basepoint loop isotropy found",
            "n_basepoint_rows": 0,
            "n_loop_rows": int(len(ldf)),
            "any_nonabelian_basepoint_isotropy": False,
            "any_exact_s3_basepoint_isotropy": False,
            "max_group_order": 0,
            "q": int(q),
            "vertices": int(vertices),
        }
    else:
        fam_counts = {str(k): int(v) for k, v in Counter(rdf["isotropy_family"].astype(str)).items()}
        any_nonab = bool(rdf.get("nonabelian", pd.Series(dtype=bool)).map(_safe_bool).any())
        any_s3 = bool(rdf.get("exact_s3", pd.Series(dtype=bool)).map(_safe_bool).any())
        max_order = int(pd.to_numeric(rdf.get("generated_group_order", pd.Series([0])), errors="coerce").fillna(0).max())
        max_comp = int(pd.to_numeric(rdf.get("max_orbit_size", pd.Series([0])), errors="coerce").fillna(0).max())
        if any_s3:
            verdict = "BASEPOINT-AWARE HOLONOMY S3 SIGNAL: exact nonabelian isotropy found at a single chart base"
        elif any_nonab:
            verdict = "BASEPOINT-AWARE HOLONOMY NONABELIAN SIGNAL: noncommuting chart-base isotropy found"
        elif max_order > 1:
            verdict = "BASEPOINT-AWARE HOLONOMY C2/ABELIAN BASELINE: loops are nontrivial but each chart-base isotropy is abelian"
        else:
            verdict = "BASEPOINT-AWARE HOLONOMY FLAT BASELINE: no nontrivial chart-base isotropy found"
        top = rdf.sort_values(["nonabelian", "exact_s3", "generated_group_order", "loop_label_support_size", "distinct_loop_maps"], ascending=[False, False, False, False, False]).head(20).to_dict(orient="records")
        summary = {
            "audit_version": "basepoint_aware_holonomy_v1_chart_isotropy",
            "verdict": verdict,
            "q": int(q),
            "vertices": int(vertices),
            "n_basepoint_rows": int(len(rdf)),
            "n_loop_rows": int(len(ldf)),
            "algebra_family_counts": fam_counts,
            "any_nonabelian_basepoint_isotropy": any_nonab,
            "any_exact_s3_basepoint_isotropy": any_s3,
            "max_group_order": max_order,
            "max_orbit_size": max_comp,
            "top_basepoints": _json_safe(top),
            "caveat": "Maps are grouped only when they are based at the same chart and act on the same label support. This fixes the raw-label artifact of mixing loops based at different charts.",
        }

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        rdf.to_csv(out, index=False)
        loops_path = os.path.splitext(out)[0] + "_loops.csv"
        ldf.to_csv(loops_path, index=False)
        with open(os.path.splitext(out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
    if plot:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(plot) or ".", exist_ok=True)
            fig, ax1 = plt.subplots(figsize=(12, 5))
            if not rdf.empty:
                labels = ["nonabelian", "exact S3", "C2/abelian", "max order"]
                vals = [
                    float(rdf["nonabelian"].map(_safe_bool).mean()),
                    float(rdf["exact_s3"].map(_safe_bool).mean()),
                    float((pd.to_numeric(rdf["generated_group_order"], errors="coerce").fillna(0) > 1).mean()),
                    0.0,
                ]
                ax1.bar(labels[:3], vals[:3], label="fractions")
                ax1.set_ylabel("fraction of basepoint rows")
                ax2 = ax1.twinx()
                ax2.plot([labels[3]], [summary.get("max_group_order", 0)], marker="o", label="max order")
                ax2.set_ylabel("max group order")
            ax1.set_title(str(summary.get("verdict", "Basepoint-aware holonomy")))
            fig.tight_layout()
            fig.savefig(plot, dpi=150)
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            summary["plot_error"] = str(e)

    return rdf, ldf, summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Basepoint-aware chart-holonomy/isotropy classifier")
    ap.add_argument("q", type=int, nargs="?", default=2)
    ap.add_argument("--vertices", type=int, default=9)
    ap.add_argument("--iterated-csv", default="")
    ap.add_argument("--frozen-transition-npy", default="")
    ap.add_argument("--target-rule-mode", default="")
    ap.add_argument("--target-instance", type=int, default=-1)
    ap.add_argument("--target-profile", default="")
    ap.add_argument("--target-atlas-capacity", type=int, default=-1)
    ap.add_argument("--target-seed", type=int, default=-1)
    ap.add_argument("--target-iteration", type=int, default=8)
    ap.add_argument("--target-parent-domain", type=int, default=-1)
    ap.add_argument("--target-fiber-label", type=int, default=-1)
    ap.add_argument("--rule-modes", default="")
    ap.add_argument("--profiles", default="")
    ap.add_argument("--atlas-capacities", default="")
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--atlas-iterations", type=int, default=12)
    ap.add_argument("--require-generated", action="store_true")
    ap.add_argument("--stop-at-first-nonabelian", action="store_true")

    # Upstream atlas controls
    ap.add_argument("--max-state-samples", type=int, default=512)
    ap.add_argument("--max-total-states", type=int, default=200000)
    ap.add_argument("--max-pred", type=int, default=0)
    ap.add_argument("--proliferation-iterations", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--initial-boundary", default="sum_mod_q")
    ap.add_argument("--initial-boundary-q", type=int, default=None)
    ap.add_argument("--max-domains-per-depth", type=int, default=32)
    ap.add_argument("--min-live-classes", type=int, default=2)
    ap.add_argument("--min-fiber-size", type=int, default=2)
    ap.add_argument("--min-entropy-bits", type=float, default=0.05)
    ap.add_argument("--synergy-threshold", type=float, default=0.01)
    ap.add_argument("--max-signature-domains", type=int, default=16)
    ap.add_argument("--max-parent-domains", type=int, default=8)
    ap.add_argument("--max-fibers-per-parent", type=int, default=6)
    ap.add_argument("--max-charts-per-fiber", type=int, default=16)
    ap.add_argument("--max-signature-charts", type=int, default=48)
    ap.add_argument("--min-fiber-states", type=int, default=8)
    ap.add_argument("--min-support-states", type=int, default=4)
    ap.add_argument("--min-overlap-states", type=int, default=4)
    ap.add_argument("--min-chart-classes", type=int, default=2)
    ap.add_argument("--min-chart-entropy", type=float, default=0.05)
    ap.add_argument("--max-chart-coords", type=int, default=5)
    ap.add_argument("--max-support-coords", type=int, default=4)
    ap.add_argument("--max-cycle-len", type=int, default=4)
    ap.add_argument("--max-cycles-per-fiber", type=int, default=500)
    ap.add_argument("--max-loops-per-base", type=int, default=1000)
    ap.add_argument("--max-group-order", type=int, default=4096)
    ap.add_argument("--max-domains-scan", type=int, default=0)
    ap.add_argument("--max-fibers-per-domain-scan", type=int, default=0)
    ap.add_argument("--atlas-lift-mode", default="bijective")
    ap.add_argument("--include-trivial", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", default="")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    rdf, ldf, summary = run_basepoint_aware_holonomy_audit(**vars(args))
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    if args.out:
        print(f"wrote {args.out}")
        print(f"wrote {os.path.splitext(args.out)[0]}_loops.csv")
        print(f"wrote {os.path.splitext(args.out)[0]}_summary.json")
    if args.plot:
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
