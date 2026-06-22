"""
iterationattractoraudit.py

Iteration-attractor audit for predictive-quotient basepoint isotropy.

This module asks whether nonabelian basepoint gauge/isotropy is an attractor,
transient, or null-sensitive feature of information-preserving finite
self-observation.

Experiments implemented
-----------------------
1. S3 rate by iteration number for q=2,3,4,5.
2. Attractor classification of the self-observation map Phi: T -> T_eff.
3. Multi-q coexistence scan for q=6 fibers with mixed effective alphabets.
4. S3 rate versus chart-compression ratio.
5. Lift-mode comparison: bijective versus representative for q=3,v=6.

The holonomy classifier is basepoint-aware. It uses the same chart-base loop
isotropy routine as basepointawareholonomyaudit, and by default builds charts
with predictivequotientchartaudit.build_predictive_quotient_charts_for_domain_fiber
when that module is present. A compatible fallback builder is included so the
module remains runnable in older RelGauge checkouts.

Example
-------
python -m relgauge.iterationattractoraudit \
  --experiments s3_by_iteration,attractors,compression,lift_mode \
  --instances 10 \
  --out-dir results/iteration_attractor_audit

Run one quick smoke:
python -m relgauge.iterationattractoraudit --experiments s3_by_iteration \
  --q-values 3 --instances 1 --atlas-iterations 2 --max-domains-scan 2 \
  --max-fibers-per-domain-scan 1 --out-dir results/iteration_attractor_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("iterationattractoraudit requires pandas") from e

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:
    from . import dynamicsconsistencyfixedpointaudit as DCFP
    from . import fiberatlasfixedpointaudit as FAF
    from . import fiberchartconnectionaudit as FCA
    from . import iteratedfiberatlasdynamicsaudit as IFAD
    from . import binarycompositegaugespectrumaudit as BCG
    try:
        from . import basepointawareholonomyaudit as BPAH
    except Exception:  # pragma: no cover
        BPAH = None  # type: ignore
    try:
        from . import predictivequotientchartaudit as PQC  # type: ignore
    except Exception:  # pragma: no cover
        PQC = None  # type: ignore
except Exception:  # pragma: no cover
    import dynamicsconsistencyfixedpointaudit as DCFP  # type: ignore
    import fiberatlasfixedpointaudit as FAF  # type: ignore
    import fiberchartconnectionaudit as FCA  # type: ignore
    import iteratedfiberatlasdynamicsaudit as IFAD  # type: ignore
    import binarycompositegaugespectrumaudit as BCG  # type: ignore
    try:
        import basepointawareholonomyaudit as BPAH  # type: ignore
    except Exception:  # pragma: no cover
        BPAH = None  # type: ignore
    try:
        import predictivequotientchartaudit as PQC  # type: ignore
    except Exception:  # pragma: no cover
        PQC = None  # type: ignore


# ---------------------------------------------------------------------------
# Embedded basepoint-aware holonomy classifier fallback
# ---------------------------------------------------------------------------
LabelMap = Dict[int, int]
Perm = Tuple[int, ...]
EdgeKey = Tuple[int, int]


def _compact_map(m: LabelMap, limit: int = 50) -> str:
    items = sorted((int(k), int(v)) for k, v in (m or {}).items())
    if len(items) > int(limit):
        return " ".join(f"{a}->{b}" for a, b in items[: int(limit)]) + " ..."
    return " ".join(f"{a}->{b}" for a, b in items)


def _perm_from_label_map_local(m: LabelMap, labels: Sequence[int]) -> Optional[Perm]:
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


def _is_abelian_local(group: Sequence[Perm]) -> bool:
    g = list(group)
    for a in g:
        for b in g:
            if BCG._compose(a, b) != BCG._compose(b, a):
                return False
    return True


def _group_orbits_local(group: Sequence[Perm], n: int) -> List[List[int]]:
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


def _classify_group_local(perms: Sequence[Perm], support_labels: Sequence[int], max_group_order: int = 4096) -> Dict[str, Any]:
    distinct: List[Perm] = []
    for p in perms:
        if p not in distinct:
            distinct.append(p)
    if not distinct:
        return {"isotropy_family": "trivial_or_absent", "generated_group_order": 1, "nonabelian": False, "exact_s3": False, "element_order_counts": {"1": 1}, "max_orbit_size": 1 if support_labels else 0, "truncated": False}
    group, truncated = BCG._close_group(distinct, max_group_order=int(max_group_order))
    orders = Counter(BCG._perm_order(g) for g in group)
    abelian = _is_abelian_local(list(group)) if not truncated and len(group) <= int(max_group_order) else False
    orbits = _group_orbits_local(list(group), len(support_labels)) if group else []
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
    return {"isotropy_family": fam, "generated_group_order": int(len(group)), "nonabelian": bool((not truncated) and not abelian), "exact_s3": bool(exact_s3), "abelian": bool(abelian) if not truncated else False, "truncated": bool(truncated), "element_order_counts": {str(k): int(v) for k, v in sorted(orders.items())}, "max_orbit_size": int(max_orbit), "orbit_count": int(len(orbits))}


def _edge_key_local(a: int, b: int) -> Tuple[int, int]:
    return (int(min(a, b)), int(max(a, b)))


def _adjacency_local(n_charts: int, edge_maps: Dict[EdgeKey, Any]) -> Dict[int, List[int]]:
    adj: Dict[int, set] = {i: set() for i in range(int(n_charts))}
    for a, b in edge_maps.keys():
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    return {k: sorted(v) for k, v in adj.items()}


def _enumerate_basepoint_loops_local(base: int, n_charts: int, edge_maps: Dict[EdgeKey, Any], max_cycle_len: int = 4, max_loops: int = 1000) -> List[List[int]]:
    base = int(base)
    adj = _adjacency_local(int(n_charts), edge_maps)
    loops: List[List[int]] = []
    seen: set = set()
    def dfs(path: List[int], target_len: int) -> None:
        if len(loops) >= int(max_loops):
            return
        cur = path[-1]
        if len(path) == target_len:
            if _edge_key_local(cur, base) in edge_maps:
                tup = tuple(path)
                if tup not in seen:
                    seen.add(tup); loops.append(list(path))
            return
        for nb in adj.get(cur, []):
            if nb == base or nb in path:
                continue
            if _edge_key_local(cur, nb) not in edge_maps:
                continue
            path.append(nb); dfs(path, target_len); path.pop()
            if len(loops) >= int(max_loops):
                return
    for L in range(3, int(max_cycle_len) + 1):
        dfs([base], L)
        if len(loops) >= int(max_loops):
            break
    return loops


def _analyze_charts_for_domain_fiber_embedded(charts: Sequence[Any], edge_maps: Dict[EdgeKey, Any], meta: Dict[str, Any], max_cycle_len: int, max_loops_per_base: int, max_group_order: int, include_trivial: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows: List[Dict[str, Any]] = []
    loop_rows: List[Dict[str, Any]] = []
    n_charts = int(len(charts))
    for base in range(n_charts):
        loops = _enumerate_basepoint_loops_local(base, n_charts, edge_maps, max_cycle_len=max_cycle_len, max_loops=max_loops_per_base)
        by_support: Dict[Tuple[int, ...], List[Tuple[List[int], LabelMap, int]]] = defaultdict(list)
        valid_count = 0; nontriv_count = 0; c2_count = 0
        for cyc in loops:
            valid, mp, reason, order = FCA.cycle_map(cyc, edge_maps)
            base_support = tuple(sorted(int(x) for x in mp.keys())) if mp else tuple()
            lr = dict(meta); lr.update({"base_chart": int(base), "cycle_charts": " ".join(str(x) for x in cyc), "cycle_length": int(len(cyc)), "loop_valid": bool(valid), "loop_order": int(order), "loop_nontrivial": bool(valid and order > 1), "loop_c2": bool(valid and order == 2 and not all(int(k) == int(v) for k, v in mp.items())), "loop_map": _compact_map(mp), "loop_label_support": " ".join(str(x) for x in base_support), "failure_reason": str(reason)})
            loop_rows.append(lr)
            if valid:
                valid_count += 1
                if order > 1 and not all(int(k) == int(v) for k, v in mp.items()):
                    nontriv_count += 1
                    if order == 2:
                        c2_count += 1
                    by_support[base_support].append((cyc, {int(k): int(v) for k, v in mp.items()}, int(order)))
        if not by_support and include_trivial:
            gr = dict(meta); ch = charts[base]
            gr.update({"base_chart": int(base), "base_chart_type": str(getattr(ch, "chart_type", "")), "base_chart_label_desc": str(getattr(ch, "label_desc", "")), "base_chart_support_desc": str(getattr(ch, "support_desc", "")), "base_chart_n_support": int(getattr(ch, "n_support", 0)), "base_chart_n_labels": int(getattr(ch, "n_labels", 0)), "loops_enumerated": int(len(loops)), "valid_loop_count": int(valid_count), "nontrivial_loop_count": int(nontriv_count), "c2_loop_count": int(c2_count), "loop_label_support": "", "loop_label_support_size": 0, "distinct_loop_maps": 0, "generated_group_order": 1, "isotropy_family": "trivial_or_absent", "nonabelian": False, "exact_s3": False})
            group_rows.append(gr)
        for support_key, entries in by_support.items():
            labels = list(support_key)
            perms: List[Perm] = []
            distinct_maps: Dict[Tuple[Tuple[int, int], ...], Tuple[LabelMap, int, List[str]]] = {}
            for cyc, mp, order in entries:
                p = _perm_from_label_map_local(mp, labels)
                if p is None:
                    continue
                perms.append(p)
                key = tuple(sorted((int(k), int(v)) for k, v in mp.items()))
                distinct_maps.setdefault(key, (mp, order, []))[2].append(" ".join(str(x) for x in cyc))
            cls = _classify_group_local(perms, labels, max_group_order=max_group_order)
            if (not include_trivial) and cls["generated_group_order"] <= 1:
                continue
            map_summ = []
            for _key, (mp, order, paths) in sorted(distinct_maps.items(), key=lambda kv: (kv[1][1], kv[0])):
                map_summ.append(f"order{order}:{_compact_map(mp)} via {len(paths)} loop(s)")
            ch = charts[base]
            gr = dict(meta); gr.update({"base_chart": int(base), "base_chart_type": str(getattr(ch, "chart_type", "")), "base_chart_label_desc": str(getattr(ch, "label_desc", "")), "base_chart_support_desc": str(getattr(ch, "support_desc", "")), "base_chart_n_support": int(getattr(ch, "n_support", 0)), "base_chart_n_labels": int(getattr(ch, "n_labels", 0)), "loops_enumerated": int(len(loops)), "valid_loop_count": int(valid_count), "nontrivial_loop_count": int(nontriv_count), "c2_loop_count": int(c2_count), "loop_label_support": " ".join(str(x) for x in labels), "loop_label_support_size": int(len(labels)), "distinct_loop_maps": int(len(distinct_maps)), "distinct_loop_map_summary": " || ".join(map_summ[:20])})
            gr.update(cls)
            gr["element_order_counts_json"] = json.dumps(cls.get("element_order_counts", {}), sort_keys=True)
            group_rows.append(gr)
    return group_rows, loop_rows


def _analyze_charts_basepoint(charts: Sequence[Any], edge_maps: Dict[EdgeKey, Any], meta: Dict[str, Any], max_cycle_len: int, max_loops_per_base: int, max_group_order: int, include_trivial: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if BPAH is not None and hasattr(BPAH, "_analyze_charts_for_domain_fiber"):
        return BPAH._analyze_charts_for_domain_fiber(charts, edge_maps, meta, max_cycle_len=max_cycle_len, max_loops_per_base=max_loops_per_base, max_group_order=max_group_order, include_trivial=include_trivial)  # type: ignore[attr-defined]
    return _analyze_charts_for_domain_fiber_embedded(charts, edge_maps, meta, max_cycle_len=max_cycle_len, max_loops_per_base=max_loops_per_base, max_group_order=max_group_order, include_trivial=include_trivial)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
Q_VERTEX_DEFAULTS = {2: 9, 3: 6, 4: 5, 5: 4, 6: 4}


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or pd.isna(x):
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return float(default)
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _parse_ints(text: Any, default: Sequence[int]) -> List[int]:
    vals: List[int] = []
    for p in str(text or "").replace(";", ",").split(","):
        p = p.strip()
        if p:
            vals.append(int(float(p)))
    return vals or [int(x) for x in default]


def _parse_texts(text: Any, default: Sequence[str]) -> List[str]:
    vals = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    return vals or [str(x) for x in default]


def _stable_hash_text(text: Any, digits: int = 8) -> int:
    return int(hashlib.sha1(str(text).encode("utf-8")).hexdigest()[: int(digits)], 16)


def _hash_arr(arr: Sequence[int]) -> str:
    return hashlib.sha1(np.asarray(arr, dtype=np.int64).tobytes()).hexdigest()[:20]


def _make_args(**kwargs: Any) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _rng_for_candidate_iter(init_seed: int, capacity: int, profile: str, iteration: int) -> np.random.Generator:
    ph = _stable_hash_text(str(profile)) % 1000
    seed = int(init_seed) + 1299709 * int(capacity) + 15485863 * int(iteration) + 104729 * int(ph)
    return np.random.default_rng(seed)


def _candidate_seed(seed: int, mode_index: int, instance: int) -> int:
    return int(seed) + 1000003 * (int(mode_index) + 1) + 7919 * int(instance)


def _q_vertices(q: int, vertices_map: Dict[int, int]) -> int:
    return int(vertices_map.get(int(q), Q_VERTEX_DEFAULTS.get(int(q), max(3, int(round(9 / max(1, math.log2(max(2, q)))))))))


def _domain_list(atlas: Any) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for seq_name in ("domains_current", "domains_all"):
        for d in list(getattr(atlas, seq_name, []) or []):
            did = int(getattr(d, "domain_id", -1))
            if did in seen:
                continue
            seen.add(did); out.append(d)
    return out


def _fiber_labels_for_domain(domain: Any, min_fiber_states: int, limit: int = 0) -> List[Tuple[int, int]]:
    labs = np.asarray(getattr(domain, "labels"), dtype=np.int64)
    cnt = Counter(int(x) for x in labs)
    pairs = [(lab, n) for lab, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if int(n) >= int(min_fiber_states)]
    if int(limit) > 0:
        pairs = pairs[: int(limit)]
    return pairs

# ---------------------------------------------------------------------------
# Predictive quotient chart construction
# ---------------------------------------------------------------------------
def _mk_chart_fallback(
    chart_id: int,
    domain: Any,
    fiber_label: int,
    kind: str,
    support_desc: str,
    label_desc: str,
    support_idx: np.ndarray,
    raw_labels: Sequence[Any],
    n_states: int,
    min_chart_classes: int,
    min_chart_entropy: float,
    min_support_states: int,
) -> Optional[Any]:
    # Prefer the exact chart constructor from fiberchartconnectionaudit.
    if hasattr(FCA, "_mk_chart"):
        return FCA._mk_chart(  # type: ignore[attr-defined]
            int(chart_id), int(getattr(domain, "domain_id", -1)), int(fiber_label),
            kind, support_desc, label_desc, np.asarray(support_idx, dtype=np.int64), raw_labels,
            int(n_states), int(min_chart_classes), float(min_chart_entropy), int(min_support_states),
        )
    # Local fallback if the private helper is unavailable.
    support_idx = np.asarray(support_idx, dtype=np.int64)
    if len(support_idx) < int(min_support_states):
        return None
    labs = FCA.relabel_on_support(raw_labels)
    n_labels = int(len(set(int(x) for x in labs)))
    ent = float(FCA.entropy(labs))
    if n_labels < int(min_chart_classes) or ent < float(min_chart_entropy):
        return None
    mask = np.zeros(int(n_states), dtype=bool)
    mask[support_idx] = True
    full = np.full(int(n_states), -1, dtype=np.int32)
    full[support_idx] = labs
    return FCA.FiberChart(
        chart_id=int(chart_id), parent_domain_id=int(getattr(domain, "domain_id", -1)), fiber_label=int(fiber_label),
        chart_type=str(kind), support_desc=str(support_desc), label_desc=str(label_desc), support_mask=mask,
        labels_full=full, n_support=int(len(support_idx)), n_labels=int(n_labels), entropy_bits=float(ent),
    )


def _fallback_build_predictive_quotient_charts_for_domain_fiber(
    domain: Any,
    fiber_label: int,
    states: Sequence[Tuple[int, ...]],
    next_idx: Sequence[int],
    q: int,
    horizon: int = 3,
    max_chart_coords: int = 5,
    max_support_coords: int = 4,
    max_charts_per_fiber: int = 32,
    max_predictive_coords: int = 6,
    max_pair_targets: int = 24,
    max_local_support_coords: int = 4,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.0,
    min_support_states: int = 3,
    include_current_basis: bool = False,
    include_parent_label: bool = True,
    include_boundary_sum: bool = True,
    include_horizon1_coords: bool = True,
    include_horizon2_coords: bool = True,
    include_pair_horizon1: bool = True,
    include_local_horizon1: bool = True,
    include_trivial: bool = False,
    chart_selection: str = "balanced",
    **_ignored: Any,
) -> List[Any]:
    """Compatible predictive-quotient chart builder.

    This fallback intentionally uses prediction-task labels, not raw singleton
    coordinate labels alone.  It mirrors the chart families used in the q=3/v=6
    predictive-quotient runs: predict_x_h1, predict_x_h2, predict_pair_h1,
    predict_parent_label_h1, predict_boundary_sum_h1, and local_predict_x_h1.
    """
    arr = np.asarray(states, dtype=np.int64)
    n_states = int(len(arr))
    if n_states <= 0 or arr.ndim != 2:
        return []
    k = int(arr.shape[1])
    q = int(q)
    idxs = np.where(np.asarray(getattr(domain, "labels"), dtype=np.int64) == int(fiber_label))[0]
    if len(idxs) < int(min_support_states):
        return []
    nxt = np.asarray(next_idx, dtype=np.int64)

    def step_indices(h: int) -> np.ndarray:
        cur = np.asarray(np.arange(n_states), dtype=np.int64)
        for _ in range(int(max(0, h))):
            cur = nxt[cur]
        return cur

    idx_h1 = step_indices(1)
    idx_h2 = step_indices(2)
    labels_parent = np.asarray(getattr(domain, "labels"), dtype=np.int64)
    charts: List[Any] = []

    def add(kind: str, sdesc: str, ldesc: str, support: np.ndarray, raw: Sequence[Any]) -> None:
        nonlocal charts
        ch = _mk_chart_fallback(
            len(charts), domain, int(fiber_label), kind, sdesc, ldesc,
            np.asarray(support, dtype=np.int64), list(raw), n_states,
            int(min_chart_classes), float(min_chart_entropy), int(min_support_states),
        )
        if ch is not None:
            charts.append(ch)

    coord_lim = min(int(max_chart_coords), k, int(max_predictive_coords))
    base_sdesc = f"parent={getattr(domain, 'domain_id', -1)};fiber={fiber_label};all"

    if include_parent_label:
        add("predict_parent_label_h1", base_sdesc, "parent_label(Tx)", idxs, [int(labels_parent[int(idx_h1[int(i)])]) for i in idxs])
    if include_boundary_sum:
        add("predict_boundary_sum_h1", base_sdesc, "sum(Tx) mod q", idxs, [int(arr[int(idx_h1[int(i)]), :].sum() % q) for i in idxs])
    if include_horizon1_coords:
        for c in range(coord_lim):
            add("predict_x_h1", base_sdesc, f"T(x)[{c}]", idxs, [int(arr[int(idx_h1[int(i)]), c]) % q for i in idxs])
    if include_horizon2_coords and int(horizon) >= 2:
        for c in range(coord_lim):
            add("predict_x_h2", base_sdesc, f"T2(x)[{c}]", idxs, [int(arr[int(idx_h2[int(i)]), c]) % q for i in idxs])
    if include_pair_horizon1:
        n_pair = 0
        for a in range(coord_lim):
            for b in range(a + 1, coord_lim):
                if n_pair >= int(max_pair_targets):
                    break
                add("predict_pair_h1", base_sdesc, f"T(x)[{a}],T(x)[{b}]", idxs,
                    [(int(arr[int(idx_h1[int(i)]), a]) % q, int(arr[int(idx_h1[int(i)]), b]) % q) for i in idxs])
                n_pair += 1
            if n_pair >= int(max_pair_targets):
                break
    if include_current_basis:
        for c in range(coord_lim):
            add("current_x", base_sdesc, f"x[{c}]", idxs, [int(arr[int(i), c]) % q for i in idxs])
    if include_local_horizon1:
        support_lim = min(int(max_support_coords), int(max_local_support_coords), k)
        target_lim = coord_lim
        for s in range(support_lim):
            vals = sorted(set(int(arr[int(i), s]) % q for i in idxs))
            for val in vals:
                sub = np.asarray([int(i) for i in idxs if int(arr[int(i), s]) % q == int(val)], dtype=np.int64)
                if len(sub) < int(min_support_states):
                    continue
                for c in range(target_lim):
                    add("local_predict_x_h1", f"x[{s}]={val}", f"T(x)[{c}]", sub, [int(arr[int(idx_h1[int(i)]), c]) % q for i in sub])

    # Deduplicate and select.  Prioritize charts with more labels, high entropy,
    # and broad support; this mirrors the predictive quotient purpose.
    dedup: Dict[Tuple[str, bytes, bytes], Any] = {}
    for ch in charts:
        key = (
            str(ch.chart_type) + ":" + str(ch.label_desc) + ":" + str(ch.support_desc),
            np.packbits(ch.support_mask).tobytes(),
            np.asarray(ch.labels_full[ch.support_mask], dtype=np.int32).tobytes(),
        )
        if key not in dedup or float(ch.entropy_bits) > float(dedup[key].entropy_bits):
            dedup[key] = ch
    charts = list(dedup.values())
    mode = str(chart_selection or "balanced").lower()
    if mode == "labels_first":
        charts.sort(key=lambda c: (int(c.n_labels), float(c.entropy_bits), int(c.n_support)), reverse=True)
    elif mode == "support_first":
        charts.sort(key=lambda c: (int(c.n_support), float(c.entropy_bits), int(c.n_labels)), reverse=True)
    else:
        charts.sort(key=lambda c: (float(c.entropy_bits), int(c.n_labels), int(c.n_support)), reverse=True)
    charts = charts[: int(max_charts_per_fiber)]
    for i, ch in enumerate(charts):
        ch.chart_id = int(i)
    return charts


def _build_predictive_quotient_charts(domain: Any, fiber_label: int, states: Sequence[Tuple[int, ...]], current_next: Sequence[int], q: int, args: Any) -> List[Any]:
    kwargs = dict(
        horizon=int(args.horizon),
        max_chart_coords=int(args.max_chart_coords),
        max_support_coords=int(args.max_support_coords),
        max_charts_per_fiber=int(args.max_charts_per_fiber),
        max_predictive_coords=int(getattr(args, "max_predictive_coords", args.max_chart_coords)),
        max_pair_targets=int(getattr(args, "max_pair_targets", 24)),
        max_local_support_coords=int(getattr(args, "max_local_support_coords", args.max_support_coords)),
        min_chart_classes=int(args.min_chart_classes),
        min_chart_entropy=float(args.min_chart_entropy),
        min_support_states=int(args.min_support_states),
        include_current_basis=bool(getattr(args, "include_current_basis", False)),
        include_parent_label=bool(getattr(args, "include_parent_label", True)),
        include_boundary_sum=bool(getattr(args, "include_boundary_sum", True)),
        include_horizon1_coords=bool(getattr(args, "include_horizon1_coords", True)),
        include_horizon2_coords=bool(getattr(args, "include_horizon2_coords", True)),
        include_pair_horizon1=bool(getattr(args, "include_pair_horizon1", True)),
        include_local_horizon1=bool(getattr(args, "include_local_horizon1", True)),
        include_trivial=bool(getattr(args, "include_trivial", False)),
        chart_selection=str(getattr(args, "chart_selection", "balanced")),
    )
    if PQC is not None and hasattr(PQC, "build_predictive_quotient_charts_for_domain_fiber"):
        try:
            return list(PQC.build_predictive_quotient_charts_for_domain_fiber(  # type: ignore[attr-defined]
                domain, int(fiber_label), states, current_next, int(q), **kwargs
            ))
        except TypeError:
            # Older signature may want q/horizon as positional arguments.
            try:
                return list(PQC.build_predictive_quotient_charts_for_domain_fiber(  # type: ignore[attr-defined]
                    domain, int(fiber_label), states, current_next, int(q), int(args.horizon), **kwargs
                ))
            except Exception:
                pass
        except Exception:
            pass
    return _fallback_build_predictive_quotient_charts_for_domain_fiber(domain, fiber_label, states, current_next, int(q), **kwargs)

# ---------------------------------------------------------------------------
# One candidate iteration pipeline
# ---------------------------------------------------------------------------
def _upstream_args(args: Any) -> argparse.Namespace:
    keys = [
        "proliferation_iterations", "horizon", "initial_boundary", "initial_boundary_q",
        "max_domains_per_depth", "min_live_classes", "min_fiber_size", "min_entropy_bits",
        "synergy_threshold", "max_signature_domains", "max_parent_domains", "max_fibers_per_parent",
        "max_charts_per_fiber", "max_signature_charts", "min_fiber_states", "min_support_states",
        "min_overlap_states", "min_chart_classes", "min_chart_entropy", "max_chart_coords",
        "max_support_coords", "max_cycle_len", "max_cycles_per_fiber", "atlas_lift_mode",
    ]
    return argparse.Namespace(**{k: getattr(args, k) for k in keys if hasattr(args, k)})


def _initial_transition(q: int, vertices: int, mode: str, init_seed: int, args: Any) -> Tuple[List[Tuple[int, ...]], np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(int(init_seed))
    states, next_idx, meta = DCFP.initialize_sampled_transition(
        q=int(q), vertices=int(vertices), mode=str(mode), rng=rng,
        max_state_samples=int(args.max_state_samples), max_total_states=int(args.max_total_states),
        max_pred=int(args.max_pred), proliferation_iterations=int(args.proliferation_iterations), horizon=int(args.horizon),
    )
    return list(states), np.asarray(next_idx, dtype=np.int64), dict(meta)


def _atlas_and_eff(states: Sequence[Tuple[int, ...]], current_next: np.ndarray, q: int, profile: str, capacity: int, iteration: int, init_seed: int, args: Any) -> Tuple[Any, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    rng = _rng_for_candidate_iter(int(init_seed), int(capacity), str(profile), int(iteration))
    atlas = IFAD._one_atlas_pass(states, current_next, int(q), rng, _upstream_args(args))
    bounded, pstats = IFAD._labels_for_profile(atlas, current_next, str(profile), int(capacity), int(args.max_signature_domains))
    eff, lift_stats, _eff_rows = DCFP.extract_effective_dynamics(bounded, current_next, lift_mode=str(args.atlas_lift_mode))
    return atlas, np.asarray(eff, dtype=np.int64), dict(pstats), dict(lift_stats)


def _analyze_predictive_isotropy(
    atlas: Any,
    states: Sequence[Tuple[int, ...]],
    current_next: Sequence[int],
    q: int,
    args: Any,
    meta: Optional[Dict[str, Any]] = None,
    return_fiber_rows: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float]]:
    meta = dict(meta or {})
    group_rows: List[Dict[str, Any]] = []
    fiber_rows: List[Dict[str, Any]] = []
    compression_ratios: List[float] = []
    domains = _domain_list(atlas)
    # Prefer rich/nontrivial domains by entropy/labels.
    domains.sort(key=lambda d: (float(getattr(d, "entropy_bits", 0.0)), int(getattr(d, "n_labels", 0))), reverse=True)
    if int(args.max_domains_scan) > 0:
        domains = domains[: int(args.max_domains_scan)]
    for d in domains:
        did = int(getattr(d, "domain_id", -1))
        fibers = _fiber_labels_for_domain(d, int(args.min_fiber_states), int(args.max_fibers_per_domain_scan))
        for flab, fsize in fibers:
            charts = _build_predictive_quotient_charts(d, int(flab), states, current_next, int(q), args)
            if not charts:
                continue
            # Per-chart compression proxy.
            for ch in charts:
                if int(getattr(ch, "n_labels", 0)) > 0:
                    compression_ratios.append(float(getattr(ch, "n_support", 0)) / max(1, int(getattr(ch, "n_labels", 1))))
            _transport_rows, edge_maps = FCA.build_chart_transports(charts, min_overlap_states=int(args.min_overlap_states))
            fmeta = dict(meta)
            fmeta.update({
                "parent_domain_id": int(did),
                "fiber_label": int(flab),
                "fiber_size": int(fsize),
                "n_charts": int(len(charts)),
                "max_chart_labels": int(max([int(getattr(ch, "n_labels", 0)) for ch in charts] or [0])),
                "median_chart_compression_ratio": float(np.median([float(getattr(ch, "n_support", 0)) / max(1, int(getattr(ch, "n_labels", 1))) for ch in charts])) if charts else 0.0,
            })
            gr, _lr = _analyze_charts_basepoint(
                charts, edge_maps, fmeta,
                max_cycle_len=int(args.max_cycle_len),
                max_loops_per_base=int(args.max_loops_per_base),
                max_group_order=int(args.max_group_order),
                include_trivial=False,
            )
            group_rows.extend(gr)
            if return_fiber_rows:
                fam_counts = Counter(str(x.get("isotropy_family", "")) for x in gr)
                fiber_rows.append({
                    **fmeta,
                    "effective_alphabet": int(fmeta["max_chart_labels"]),
                    "n_group_rows": int(len(gr)),
                    "n_c2": int(sum(1 for x in gr if str(x.get("isotropy_family", "")).startswith("C2"))),
                    "n_c3": int(sum(1 for x in gr if str(x.get("isotropy_family", "")).startswith("C3"))),
                    "n_s3": int(sum(1 for x in gr if str(x.get("isotropy_family", "")) == "S3_basepoint_isotropy")),
                    "max_group_order": int(max([_safe_int(x.get("generated_group_order"), 0) for x in gr] or [0])),
                    "holonomy_group": "S3" if fam_counts.get("S3_basepoint_isotropy", 0) else ("C3" if any(k.startswith("C3") for k in fam_counts) else ("C2" if any(k.startswith("C2") for k in fam_counts) else "flat")),
                    "s3_present": bool(fam_counts.get("S3_basepoint_isotropy", 0) > 0),
                    "c2_present": bool(any(k.startswith("C2") for k in fam_counts)),
                    "family_counts_json": json.dumps({str(k): int(v) for k, v in fam_counts.items()}, sort_keys=True),
                })
    fams = Counter(str(r.get("isotropy_family", "")) for r in group_rows)
    n_c2 = sum(int(v) for k, v in fams.items() if str(k).startswith("C2"))
    n_c3 = sum(int(v) for k, v in fams.items() if str(k).startswith("C3"))
    n_s3 = int(fams.get("S3_basepoint_isotropy", 0))
    n_total = int(len(group_rows))
    stats = {
        "n_c2": int(n_c2),
        "n_c3": int(n_c3),
        "n_s3": int(n_s3),
        "n_total": int(n_total),
        "s3_rate": float(n_s3 / max(1, n_total)),
        "max_group_order": int(max([_safe_int(r.get("generated_group_order"), 0) for r in group_rows] or [0])),
        "max_distinct_loop_maps": int(max([_safe_int(r.get("distinct_loop_maps"), 0) for r in group_rows] or [0])),
        "max_loop_label_support_size": int(max([_safe_int(r.get("loop_label_support_size"), 0) for r in group_rows] or [0])),
        "median_compression_ratio": float(np.median(compression_ratios)) if compression_ratios else 0.0,
        "mean_compression_ratio": float(np.mean(compression_ratios)) if compression_ratios else 0.0,
        "family_counts_json": json.dumps({str(k): int(v) for k, v in fams.items()}, sort_keys=True),
    }
    return stats, fiber_rows, compression_ratios

# ---------------------------------------------------------------------------
# Experiments 1,2,4 shared run
# ---------------------------------------------------------------------------
def run_s3_iteration_and_attractors(
    q_values: Sequence[int],
    vertices_map: Dict[int, int],
    instances: int,
    atlas_iterations: int,
    rule_mode: str,
    profile: str,
    atlas_capacity: int,
    seed: int,
    args: Any,
    lift_mode: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    attractor_rows: List[Dict[str, Any]] = []
    old_lift = getattr(args, "atlas_lift_mode", "bijective")
    if lift_mode is not None:
        args.atlas_lift_mode = str(lift_mode)
    try:
        for q in q_values:
            q = int(q)
            vertices = _q_vertices(q, vertices_map)
            for inst in range(int(instances)):
                init_seed = _candidate_seed(int(seed), 0, int(inst))
                states, current_next, init_meta = _initial_transition(q, vertices, str(rule_mode), int(init_seed), args)
                seen: Dict[str, int] = {}
                fixed_iter: Optional[int] = None
                cycle_iter: Optional[int] = None
                cycle_length: int = 0
                fixed_gauge = ""
                cycle_gauge = ""
                for it in range(int(atlas_iterations) + 1):
                    trans_hash = _hash_arr(current_next)
                    repeated_before = trans_hash in seen
                    if repeated_before and cycle_iter is None:
                        cycle_iter = int(it)
                        cycle_length = int(it - seen[trans_hash])
                    seen.setdefault(trans_hash, int(it))
                    atlas, eff, pstats, lift_stats = _atlas_and_eff(states, current_next, q, str(profile), int(atlas_capacity), int(it), int(init_seed), args)
                    meta = {
                        "q": int(q), "vertices": int(vertices), "instance": int(inst), "seed": int(init_seed),
                        "iteration": int(it), "rule_mode": str(rule_mode), "profile": str(profile),
                        "atlas_capacity": int(atlas_capacity), "atlas_lift_mode": str(args.atlas_lift_mode),
                    }
                    iso_stats, _fibrows, _cr = _analyze_predictive_isotropy(atlas, states, current_next, q, args, meta=meta, return_fiber_rows=False)
                    effective_fixed = bool(np.array_equal(eff, current_next))
                    fam = "S3" if int(iso_stats.get("n_s3", 0)) > 0 else ("C3" if int(iso_stats.get("n_c3", 0)) > 0 else ("C2" if int(iso_stats.get("n_c2", 0)) > 0 else "flat"))
                    if effective_fixed and fixed_iter is None:
                        fixed_iter = int(it); fixed_gauge = fam
                    if repeated_before and cycle_gauge == "":
                        cycle_gauge = fam
                    rows.append({
                        **meta,
                        "transition_hash": str(trans_hash),
                        "effective_next_hash": _hash_arr(eff),
                        "effective_fixed_point": bool(effective_fixed),
                        "effective_limit_cycle_so_far": bool(repeated_before),
                        "n_states": int(len(states)),
                        "bounded_atlas_classes": _safe_int(pstats.get("bounded_atlas_classes"), 0),
                        "bounded_atlas_entropy_bits": _safe_float(pstats.get("bounded_atlas_entropy_bits"), 0.0),
                        "lift_preferred_fraction": _safe_float(lift_stats.get("lift_preferred_fraction"), 0.0),
                        **iso_stats,
                    })
                    current_next = np.asarray(eff, dtype=np.int64)
                attractor_rows.append({
                    "q": int(q), "vertices": int(vertices), "instance": int(inst), "seed": int(init_seed),
                    "rule_mode": str(rule_mode), "profile": str(profile), "atlas_capacity": int(atlas_capacity),
                    "atlas_lift_mode": str(args.atlas_lift_mode),
                    "first_fixed_point_iteration": int(fixed_iter) if fixed_iter is not None else -1,
                    "limit_cycle_detected": bool(cycle_iter is not None),
                    "first_cycle_repeat_iteration": int(cycle_iter) if cycle_iter is not None else -1,
                    "cycle_length": int(cycle_length),
                    "gauge_at_fixed_point": str(fixed_gauge),
                    "gauge_at_cycle_repeat": str(cycle_gauge),
                    "s3_ever": bool(any(r.get("q") == q and r.get("instance") == inst and _safe_int(r.get("n_s3"), 0) > 0 for r in rows)),
                    "max_s3_rate": float(max([_safe_float(r.get("s3_rate"), 0.0) for r in rows if r.get("q") == q and r.get("instance") == inst] or [0.0])),
                })
    finally:
        args.atlas_lift_mode = old_lift
    return pd.DataFrame(rows), pd.DataFrame(attractor_rows)

# ---------------------------------------------------------------------------
# Experiment 3: q=6 coexistence
# ---------------------------------------------------------------------------
def run_multi_q_coexistence(instances: int, q: int, vertices: int, iteration: int, rule_mode: str, profile: str, capacity: int, seed: int, args: Any) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for inst in range(int(instances)):
        init_seed = _candidate_seed(int(seed), 0, int(inst))
        states, current_next, _meta = _initial_transition(int(q), int(vertices), str(rule_mode), int(init_seed), args)
        atlas = None
        for it in range(int(iteration) + 1):
            atlas, eff, _pstats, _lst = _atlas_and_eff(states, current_next, int(q), str(profile), int(capacity), int(it), int(init_seed), args)
            if it < int(iteration):
                current_next = np.asarray(eff, dtype=np.int64)
        if atlas is None:
            continue
        meta = {"q": int(q), "vertices": int(vertices), "instance": int(inst), "seed": int(init_seed), "iteration": int(iteration), "rule_mode": rule_mode, "profile": profile, "atlas_capacity": int(capacity)}
        _stats, fibrows, _cr = _analyze_predictive_isotropy(atlas, states, current_next, int(q), args, meta=meta, return_fiber_rows=True)
        rows.extend(fibrows)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Experiment 5: lift-mode comparison
# ---------------------------------------------------------------------------
def run_lift_mode_comparison(q: int, vertices: int, instances: int, atlas_iterations: int, rule_mode: str, profile: str, capacity: int, seed: int, lift_modes: Sequence[str], args: Any) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for lm in lift_modes:
        df, _adf = run_s3_iteration_and_attractors([int(q)], {int(q): int(vertices)}, int(instances), int(atlas_iterations), str(rule_mode), str(profile), int(capacity), int(seed), args, lift_mode=str(lm))
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# ---------------------------------------------------------------------------
# Plots and summaries
# ---------------------------------------------------------------------------
def _plot_s3_by_iteration(df: pd.DataFrame, path: str) -> None:
    if plt is None or df.empty:
        return
    qvals = sorted(df["q"].unique())
    fig, axes = plt.subplots(len(qvals), 1, figsize=(9, max(3, 2.8 * len(qvals))), sharex=True)
    if len(qvals) == 1:
        axes = [axes]
    for ax, q in zip(axes, qvals):
        sub = df[df["q"] == q]
        for (_inst, _seed), g in sub.groupby(["instance", "seed"]):
            g = g.sort_values("iteration")
            ax.plot(g["iteration"], g["s3_rate"], alpha=0.25, linewidth=1)
        mean = sub.groupby("iteration", as_index=False)["s3_rate"].mean().sort_values("iteration")
        ax.plot(mean["iteration"], mean["s3_rate"], linewidth=3, label=f"q={q} mean")
        ax.set_ylabel("S3 rate")
        ax.set_title(f"q={q}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("atlas iteration")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_s3_vs_compression(df: pd.DataFrame, path: str) -> None:
    if plt is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for q, g in df.groupby("q"):
        ax.scatter(g["median_compression_ratio"], g["s3_rate"], s=18, alpha=0.45, label=f"q={q}")
    ax.set_xlabel("median chart compression ratio: support size / chart labels")
    ax.set_ylabel("S3 rate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def _summary(out_dir: str, files: Dict[str, str]) -> Dict[str, Any]:
    summ: Dict[str, Any] = {"audit_version": "iteration_attractor_audit_v1_predictive_quotient_basepoint_isotropy", "files": files}
    for key, path in files.items():
        if path.endswith(".csv") and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                summ[key + "_rows"] = int(len(df))
                if "n_s3" in df.columns:
                    summ[key + "_any_s3"] = bool((pd.to_numeric(df["n_s3"], errors="coerce").fillna(0) > 0).any())
                if "s3_rate" in df.columns:
                    summ[key + "_max_s3_rate"] = float(pd.to_numeric(df["s3_rate"], errors="coerce").fillna(0).max())
            except Exception as e:
                summ[key + "_error"] = str(e)
    # Explicit top-line summaries.
    p = os.path.join(out_dir, "s3_rate_by_iteration.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        if len(df):
            byq = []
            for q, g in df.groupby("q"):
                byq.append({
                    "q": int(q),
                    "max_mean_s3_rate": float(g.groupby("iteration")["s3_rate"].mean().max()),
                    "max_any_s3_rate": float(g["s3_rate"].max()),
                    "iterations_with_any_s3": int((g.groupby("iteration")["n_s3"].sum() > 0).sum()),
                })
            summ["by_q_s3_iteration"] = byq
    return summ

# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_iteration_attractor_audit(
    experiments: Sequence[str] = ("all",),
    out_dir: str = "results/iteration_attractor_audit",
    q_values: Sequence[int] = (2, 3, 4, 5),
    vertices_map: Optional[Dict[int, int]] = None,
    instances: int = 10,
    atlas_iterations: int = 12,
    rule_mode: str = "random_full_permutation",
    profile: str = "full_atlas",
    atlas_capacity: int = 32,
    seed: int = 2000006,
    # Upstream / chart args
    proliferation_iterations: int = 4,
    horizon: int = 3,
    max_state_samples: int = 4096,
    max_total_states: int = 200000,
    max_pred: int = 0,
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
    max_charts_per_fiber: int = 32,
    max_signature_charts: int = 48,
    min_fiber_states: int = 6,
    min_support_states: int = 3,
    min_overlap_states: int = 3,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.0,
    max_chart_coords: int = 5,
    max_support_coords: int = 4,
    max_cycle_len: int = 5,
    max_cycles_per_fiber: int = 500,
    max_loops_per_base: int = 500,
    max_group_order: int = 4096,
    max_domains_scan: int = 20,
    max_fibers_per_domain_scan: int = 3,
    atlas_lift_mode: str = "bijective",
    max_predictive_coords: int = 6,
    max_pair_targets: int = 24,
    max_local_support_coords: int = 4,
    chart_selection: str = "balanced",
    include_current_basis: bool = False,
    include_parent_label: bool = True,
    include_boundary_sum: bool = True,
    include_horizon1_coords: bool = True,
    include_horizon2_coords: bool = True,
    include_pair_horizon1: bool = True,
    include_local_horizon1: bool = True,
    # Experiment-specific
    multi_q: int = 6,
    multi_q_vertices: int = 4,
    multi_q_iteration: int = 4,
    multi_q_instances: int = 5,
    lift_modes: Sequence[str] = ("bijective", "representative"),
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    vertices_map = vertices_map or dict(Q_VERTEX_DEFAULTS)
    exps = set(str(x).strip() for x in experiments)
    if "all" in exps:
        exps = {"s3_by_iteration", "attractors", "multi_q", "compression", "lift_mode"}
    args = _make_args(**locals())
    # Clean fields that should not confuse upstream helpers.
    args.experiments = list(exps)
    files: Dict[str, str] = {}

    main_df: Optional[pd.DataFrame] = None
    attract_df: Optional[pd.DataFrame] = None
    if exps & {"s3_by_iteration", "attractors", "compression"}:
        main_df, attract_df = run_s3_iteration_and_attractors(
            list(q_values), vertices_map, int(instances), int(atlas_iterations), str(rule_mode), str(profile), int(atlas_capacity), int(seed), args
        )
        if "s3_by_iteration" in exps:
            path = os.path.join(out_dir, "s3_rate_by_iteration.csv")
            _write(main_df, path); files["s3_rate_by_iteration"] = path
            plot = os.path.join(out_dir, "s3_rate_by_iteration.png")
            _plot_s3_by_iteration(main_df, plot); files["s3_rate_by_iteration_plot"] = plot
        if "attractors" in exps:
            path = os.path.join(out_dir, "attractor_classification.csv")
            _write(attract_df, path); files["attractor_classification"] = path
        if "compression" in exps:
            cols = [c for c in ["q", "vertices", "instance", "seed", "iteration", "atlas_lift_mode", "median_compression_ratio", "mean_compression_ratio", "s3_rate", "n_s3", "n_total", "max_group_order"] if c in main_df.columns]
            cdf = main_df[cols].copy()
            path = os.path.join(out_dir, "s3_vs_compression.csv")
            _write(cdf, path); files["s3_vs_compression"] = path
            plot = os.path.join(out_dir, "s3_vs_compression.png")
            _plot_s3_vs_compression(cdf, plot); files["s3_vs_compression_plot"] = plot
    if "multi_q" in exps:
        mdf = run_multi_q_coexistence(int(multi_q_instances), int(multi_q), int(multi_q_vertices), int(multi_q_iteration), str(rule_mode), str(profile), int(atlas_capacity), int(seed), args)
        path = os.path.join(out_dir, "multi_q_coexistence.csv")
        _write(mdf, path); files["multi_q_coexistence"] = path
    if "lift_mode" in exps:
        # Follow the requested q=3,v=6 comparison.
        ldf = run_lift_mode_comparison(3, 6, int(instances), int(atlas_iterations), str(rule_mode), str(profile), int(atlas_capacity), int(seed), list(lift_modes), args)
        path = os.path.join(out_dir, "lift_mode_comparison.csv")
        _write(ldf, path); files["lift_mode_comparison"] = path
    summ = _summary(out_dir, files)
    summ.update({
        "experiments": sorted(exps),
        "q_values": [int(x) for x in q_values],
        "instances": int(instances),
        "atlas_iterations": int(atlas_iterations),
        "rule_mode": str(rule_mode),
        "profile": str(profile),
        "atlas_capacity": int(atlas_capacity),
        "chart_builder": "predictivequotientchartaudit" if PQC is not None and hasattr(PQC, "build_predictive_quotient_charts_for_domain_fiber") else "fallback_predictive_quotient",
        "caveat": "Nonabelian claims are basepoint isotropy claims only; raw same-fiber label mixing is not used.",
    })
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, sort_keys=True)
    files["summary"] = os.path.join(out_dir, "summary.json")
    return summ

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Iteration-attractor audit for predictive-quotient basepoint isotropy")
    ap.add_argument("--experiments", default="all", help="Comma list: all,s3_by_iteration,attractors,multi_q,compression,lift_mode")
    ap.add_argument("--out-dir", default="results/iteration_attractor_audit")
    ap.add_argument("--q-values", default="2,3,4,5")
    ap.add_argument("--vertices-map", default="2:9,3:6,4:5,5:4,6:4", help="Comma map q:vertices")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--atlas-iterations", type=int, default=12)
    ap.add_argument("--rule-mode", default="random_full_permutation")
    ap.add_argument("--profile", default="full_atlas")
    ap.add_argument("--atlas-capacity", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2000006)
    ap.add_argument("--proliferation-iterations", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--max-state-samples", type=int, default=4096)
    ap.add_argument("--max-total-states", type=int, default=200000)
    ap.add_argument("--max-pred", type=int, default=0)
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
    ap.add_argument("--max-charts-per-fiber", type=int, default=32)
    ap.add_argument("--max-signature-charts", type=int, default=48)
    ap.add_argument("--min-fiber-states", type=int, default=6)
    ap.add_argument("--min-support-states", type=int, default=3)
    ap.add_argument("--min-overlap-states", type=int, default=3)
    ap.add_argument("--min-chart-classes", type=int, default=2)
    ap.add_argument("--min-chart-entropy", type=float, default=0.0)
    ap.add_argument("--max-chart-coords", type=int, default=5)
    ap.add_argument("--max-support-coords", type=int, default=4)
    ap.add_argument("--max-cycle-len", type=int, default=5)
    ap.add_argument("--max-cycles-per-fiber", type=int, default=500)
    ap.add_argument("--max-loops-per-base", type=int, default=500)
    ap.add_argument("--max-group-order", type=int, default=4096)
    ap.add_argument("--max-domains-scan", type=int, default=20)
    ap.add_argument("--max-fibers-per-domain-scan", type=int, default=3)
    ap.add_argument("--atlas-lift-mode", default="bijective")
    ap.add_argument("--max-predictive-coords", type=int, default=6)
    ap.add_argument("--max-pair-targets", type=int, default=24)
    ap.add_argument("--max-local-support-coords", type=int, default=4)
    ap.add_argument("--chart-selection", default="balanced")
    ap.add_argument("--include-current-basis", action="store_true")
    ap.add_argument("--no-parent-label", action="store_true")
    ap.add_argument("--no-boundary-sum", action="store_true")
    ap.add_argument("--no-horizon1-coords", action="store_true")
    ap.add_argument("--no-horizon2-coords", action="store_true")
    ap.add_argument("--no-pair-horizon1", action="store_true")
    ap.add_argument("--no-local-horizon1", action="store_true")
    ap.add_argument("--multi-q", type=int, default=6)
    ap.add_argument("--multi-q-vertices", type=int, default=4)
    ap.add_argument("--multi-q-iteration", type=int, default=4)
    ap.add_argument("--multi-q-instances", type=int, default=5)
    ap.add_argument("--lift-modes", default="bijective,representative")
    args = ap.parse_args(argv)

    vmap: Dict[int, int] = {}
    for part in str(args.vertices_map or "").split(","):
        if not part.strip():
            continue
        if ":" not in part:
            continue
        a, b = part.split(":", 1)
        vmap[int(float(a))] = int(float(b))
    summary = run_iteration_attractor_audit(
        experiments=_parse_texts(args.experiments, ["all"]),
        out_dir=str(args.out_dir),
        q_values=_parse_ints(args.q_values, [2, 3, 4, 5]),
        vertices_map=vmap,
        instances=int(args.instances),
        atlas_iterations=int(args.atlas_iterations),
        rule_mode=str(args.rule_mode),
        profile=str(args.profile),
        atlas_capacity=int(args.atlas_capacity),
        seed=int(args.seed),
        proliferation_iterations=int(args.proliferation_iterations),
        horizon=int(args.horizon),
        max_state_samples=int(args.max_state_samples),
        max_total_states=int(args.max_total_states),
        max_pred=int(args.max_pred),
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
        max_loops_per_base=int(args.max_loops_per_base),
        max_group_order=int(args.max_group_order),
        max_domains_scan=int(args.max_domains_scan),
        max_fibers_per_domain_scan=int(args.max_fibers_per_domain_scan),
        atlas_lift_mode=str(args.atlas_lift_mode),
        max_predictive_coords=int(args.max_predictive_coords),
        max_pair_targets=int(args.max_pair_targets),
        max_local_support_coords=int(args.max_local_support_coords),
        chart_selection=str(args.chart_selection),
        include_current_basis=bool(args.include_current_basis),
        include_parent_label=not bool(args.no_parent_label),
        include_boundary_sum=not bool(args.no_boundary_sum),
        include_horizon1_coords=not bool(args.no_horizon1_coords),
        include_horizon2_coords=not bool(args.no_horizon2_coords),
        include_pair_horizon1=not bool(args.no_pair_horizon1),
        include_local_horizon1=not bool(args.no_local_horizon1),
        multi_q=int(args.multi_q),
        multi_q_vertices=int(args.multi_q_vertices),
        multi_q_iteration=int(args.multi_q_iteration),
        multi_q_instances=int(args.multi_q_instances),
        lift_modes=_parse_texts(args.lift_modes, ["bijective", "representative"]),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
