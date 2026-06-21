"""
predictivequotientchartaudit.py

Predictive-quotient chart-basis audit.

Motivation
----------
The basepoint-aware holonomy audit showed that raw same-fiber integer-label
composition can overstate nonabelian structure when loop maps are based at
different charts. The temporal-capacity chart-basis audit then showed that
simply adding pair-coordinate charts can create 3+ label charts but those charts
can remain too globally regular: many loops close trivially, and nontrivial
holonomy remains confined to 2-label cores.

This module tests a different, more observer-native chart basis.  For each
hidden fiber, a chart is a *predictive quotient*: two microstates receive the
same chart label iff they are indistinguishable for a specified prediction task
(target observable, horizon, support condition).  Different prediction tasks can
partition the same fiber incompatibly, which is precisely the kind of structure
that can create nontrivial overlap transports.

Implemented target observables
------------------------------
For a state x and transition T, each (target, horizon, support) triple gives one
chart.  The built-in target classes include:

  * x[p] at horizon 1 and horizon 2,
  * (x[p], x[r]) at horizon 1,
  * boundary sum at horizon 1,
  * parent-domain label at horizon 1,
  * x[p] at horizon 1 restricted to local support x[s]=val.

All nonabelian claims are basepoint-aware.  The module builds the overlap graph,
enumerates loops based at each chart, and closes only loop maps that are already
based at the same chart and act on the same label support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("predictivequotientchartaudit requires pandas") from e

try:
    from . import basepointawareholonomyaudit as BPH
    from . import boundaryproliferationaudit as BPA
    from . import dynamicsconsistencyfixedpointaudit as DCFP
    from . import fiberchartconnectionaudit as FCA
    from . import generatedcandidatephysicsreplayaudit as GCPR
except Exception:  # pragma: no cover
    import basepointawareholonomyaudit as BPH  # type: ignore
    import boundaryproliferationaudit as BPA  # type: ignore
    import dynamicsconsistencyfixedpointaudit as DCFP  # type: ignore
    import fiberchartconnectionaudit as FCA  # type: ignore
    import generatedcandidatephysicsreplayaudit as GCPR  # type: ignore


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
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
        return v if math.isfinite(v) else float(default)
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
    if s in {"false", "f", "0", "no", "n", "none", "nan", ""}:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return bool(x)


def _parse_csv_text(text: Any, default: Sequence[str] = ()) -> List[str]:
    vals = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    return vals or [str(x) for x in default]


def _parse_csv_ints(text: Any, default: Sequence[int] = ()) -> List[int]:
    vals: List[int] = []
    for p in str(text or "").replace(";", ",").split(","):
        p = p.strip()
        if p:
            vals.append(int(float(p)))
    return vals or [int(x) for x in default]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _candidate_id(row: Dict[str, Any]) -> str:
    return BPH._candidate_id(row)


def _domain_list(atlas: Any) -> List[Any]:
    return BPH._domain_list(atlas)


def _fiber_indices(domain: Any, fiber_label: int) -> np.ndarray:
    labels = np.asarray(getattr(domain, "labels", []), dtype=np.int64)
    return np.where(labels == int(fiber_label))[0]


def _mk_chart(chart_id: int, parent_id: int, fiber_label: int, kind: str, sdesc: str, ldesc: str,
              support_idx: Sequence[int], raw_labels: Sequence[Any], n_states: int,
              min_chart_classes: int, min_chart_entropy: float, min_support_states: int) -> Optional[Any]:
    support_idx = np.asarray(support_idx, dtype=np.int64)
    if len(support_idx) < int(min_support_states):
        return None
    if len(raw_labels) != len(support_idx):
        return None
    labs = FCA.relabel_on_support(list(raw_labels))
    n_labels = int(len(set(int(x) for x in labs)))
    ent = float(FCA.entropy(labs))
    if n_labels < int(min_chart_classes) or ent < float(min_chart_entropy):
        return None
    mask = np.zeros(int(n_states), dtype=bool)
    mask[support_idx] = True
    full = np.full(int(n_states), -1, dtype=np.int32)
    full[support_idx] = labs
    return FCA.FiberChart(
        chart_id=int(chart_id), parent_domain_id=int(parent_id), fiber_label=int(fiber_label),
        chart_type=str(kind), support_desc=str(sdesc), label_desc=str(ldesc),
        support_mask=mask, labels_full=full, n_support=int(len(support_idx)),
        n_labels=int(n_labels), entropy_bits=float(ent),
    )


def _transition_powers(next_idx: Sequence[int], max_h: int) -> Dict[int, np.ndarray]:
    nxt = np.asarray(next_idx, dtype=np.int64)
    powers: Dict[int, np.ndarray] = {0: np.arange(len(nxt), dtype=np.int64), 1: nxt.copy()}
    for h in range(2, int(max_h) + 1):
        prev = powers[h - 1]
        powers[h] = nxt[prev]
    return powers


def _balanced_select_charts(charts: List[Any], max_charts: int, mode: str = "balanced") -> List[Any]:
    if int(max_charts) <= 0 or len(charts) <= int(max_charts):
        for i, ch in enumerate(charts):
            ch.chart_id = int(i)
        return charts
    mode = str(mode or "balanced").lower()
    # Deduplicate first.
    dedup: Dict[Tuple[str, bytes, bytes], Any] = {}
    for ch in charts:
        key = (str(ch.chart_type) + ":" + str(ch.support_desc) + ":" + str(ch.label_desc),
               np.packbits(ch.support_mask).tobytes(), ch.labels_full[ch.support_mask].tobytes())
        if key not in dedup or float(ch.entropy_bits) > float(dedup[key].entropy_bits):
            dedup[key] = ch
    charts = list(dedup.values())
    if len(charts) <= int(max_charts):
        for i, ch in enumerate(charts):
            ch.chart_id = int(i)
        return charts
    if mode in {"entropy", "priority"}:
        charts.sort(key=lambda c: (float(c.entropy_bits), int(c.n_labels), int(c.n_support)), reverse=True)
        out = charts[: int(max_charts)]
    else:
        by_type: Dict[str, List[Any]] = defaultdict(list)
        for ch in charts:
            by_type[str(ch.chart_type)].append(ch)
        for typ in by_type:
            by_type[typ].sort(key=lambda c: (float(c.entropy_bits), int(c.n_labels), int(c.n_support)), reverse=True)
        # Round-robin by chart type so singleton/local/pair/predictive charts all survive.
        out: List[Any] = []
        types = sorted(by_type.keys())
        while len(out) < int(max_charts) and any(by_type[t] for t in types):
            for t in types:
                if by_type[t] and len(out) < int(max_charts):
                    out.append(by_type[t].pop(0))
    for i, ch in enumerate(out):
        ch.chart_id = int(i)
    return out


# ---------------------------------------------------------------------------
# Predictive quotient charts
# ---------------------------------------------------------------------------
def build_predictive_quotient_charts_for_domain_fiber(
    domain: Any,
    fiber_label: int,
    states: Sequence[Tuple[int, ...]],
    next_idx: Sequence[int],
    q: int,
    horizon: int,
    *,
    max_predictive_coords: int = 6,
    max_pair_targets: int = 24,
    max_local_support_coords: int = 4,
    max_charts_per_fiber: int = 64,
    chart_selection: str = "balanced",
    include_current_basis: bool = False,
    include_horizon1_coords: bool = True,
    include_horizon2_coords: bool = True,
    include_pair_horizon1: bool = True,
    include_boundary_sum: bool = True,
    include_parent_label: bool = True,
    include_local_horizon1: bool = True,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.0,
    min_support_states: int = 4,
) -> List[Any]:
    """Construct predictive quotient charts over one hidden fiber.

    For each chart, labels are canonical equivalence classes of target observable
    values over the chosen support.  The construction intentionally uses many
    different target observables so that charts can be incompatible sufficient
    statistics for different finite prediction tasks.
    """
    arr = np.asarray(states, dtype=np.int64)
    n_states = int(len(arr))
    if n_states == 0 or arr.ndim != 2:
        return []
    k = int(arr.shape[1])
    idxs = _fiber_indices(domain, int(fiber_label))
    if len(idxs) < int(min_support_states):
        return []
    max_h = max(2, int(horizon), 1)
    powers = _transition_powers(next_idx, max_h)
    # Parent-domain labels are a global array over state indices.
    parent_labels = np.asarray(getattr(domain, "labels", []), dtype=np.int64)
    charts: List[Any] = []
    parent_id = int(getattr(domain, "domain_id", -1))
    coord_lim = min(int(max_predictive_coords), k)

    def add(kind: str, sdesc: str, ldesc: str, support: Sequence[int], raw: Sequence[Any]) -> None:
        ch = _mk_chart(len(charts), parent_id, int(fiber_label), kind, sdesc, ldesc,
                       np.asarray(support, dtype=np.int64), list(raw), n_states,
                       min_chart_classes, min_chart_entropy, min_support_states)
        if ch is not None:
            charts.append(ch)

    if include_current_basis:
        try:
            charts.extend(FCA.build_charts_for_domain_fiber(
                domain, fiber_label, states, next_idx, q, horizon,
                max_chart_coords=coord_lim, max_support_coords=max_local_support_coords,
                max_charts_per_fiber=max_charts_per_fiber,
                min_chart_classes=min_chart_classes, min_chart_entropy=min_chart_entropy,
                min_support_states=min_support_states,
            ))
        except Exception:
            pass

    # x[p] after one step and two steps.
    if include_horizon1_coords:
        h = 1; image = powers[h]
        for p in range(coord_lim):
            add("predict_x_h1", "all", f"x[{p}]@h1", idxs,
                [int(arr[int(image[int(i)]), p]) % int(q) for i in idxs])

    if include_horizon2_coords:
        h = 2; image = powers[h]
        for p in range(coord_lim):
            add("predict_x_h2", "all", f"x[{p}]@h2", idxs,
                [int(arr[int(image[int(i)]), p]) % int(q) for i in idxs])

    # Pair target observable after one step.
    if include_pair_horizon1:
        image = powers[1]
        count = 0
        for p in range(coord_lim):
            for r in range(p + 1, coord_lim):
                add("predict_pair_h1", "all", f"(x[{p}],x[{r}])@h1", idxs,
                    [(int(arr[int(image[int(i)]), p]) % int(q), int(arr[int(image[int(i)]), r]) % int(q)) for i in idxs])
                count += 1
                if int(max_pair_targets) > 0 and count >= int(max_pair_targets):
                    break
            if int(max_pair_targets) > 0 and count >= int(max_pair_targets):
                break

    # Boundary sum after one step: use all vertices in the sampled state vector.
    if include_boundary_sum:
        image = powers[1]
        add("predict_boundary_sum_h1", "all", "sum(x)@h1", idxs,
            [int(np.sum(arr[int(image[int(i)])]) % int(q)) for i in idxs])

    # Parent-domain label after one step.
    if include_parent_label:
        image = powers[1]
        add("predict_parent_label_h1", "all", "parent_label@h1", idxs,
            [int(parent_labels[int(image[int(i)])]) if 0 <= int(image[int(i)]) < len(parent_labels) else -1 for i in idxs])

    # Local support x[s]=val, target x[p] after one step.
    if include_local_horizon1:
        image = powers[1]
        support_lim = min(int(max_local_support_coords), k)
        for s in range(support_lim):
            vals = sorted(set(int(arr[int(i), s]) % int(q) for i in idxs))
            for val in vals:
                sub = np.asarray([int(i) for i in idxs if int(arr[int(i), s]) % int(q) == int(val)], dtype=np.int64)
                if len(sub) < int(min_support_states):
                    continue
                for p in range(coord_lim):
                    add("local_predict_x_h1", f"x[{s}]={val}", f"x[{p}]@h1", sub,
                        [int(arr[int(image[int(i)]), p]) % int(q) for i in sub])

    return _balanced_select_charts(charts, int(max_charts_per_fiber), mode=str(chart_selection))


# ---------------------------------------------------------------------------
# Candidate reconstruction and analysis
# ---------------------------------------------------------------------------
def _make_args(**kwargs: Any) -> Any:
    return argparse.Namespace(**kwargs)


def _ensure_args(args: Any) -> None:
    BPH._ensure_upstream_defaults(args)
    defaults = {
        "max_predictive_coords": 6,
        "max_pair_targets": 24,
        "max_local_support_coords": 4,
        "chart_selection": "balanced",
        "include_current_basis": False,
        "include_horizon1_coords": True,
        "include_horizon2_coords": True,
        "include_pair_horizon1": True,
        "include_boundary_sum": True,
        "include_parent_label": True,
        "include_local_horizon1": True,
        "max_loops_per_base": 3000,
        "max_group_order": 4096,
        "include_trivial": False,
        "target_parent_domain": -1,
        "target_fiber_label": -1,
        "max_domains_scan": 0,
        "max_fibers_per_domain_scan": 0,
        "stop_at_first_nonabelian": False,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)


def _build_atlas(states: np.ndarray, current_next: np.ndarray, q: int, row: Dict[str, Any], iteration: int, args: Any):
    return BPH._build_atlas_from_transition(states, current_next, int(q), row, int(iteration), args)


def _analyze_predictive_quotient(atlas: Any, states: np.ndarray, current_next: np.ndarray, q: int, row: Dict[str, Any], iteration: int, args: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows: List[Dict[str, Any]] = []
    loop_rows: List[Dict[str, Any]] = []
    chart_rows: List[Dict[str, Any]] = []
    candidate = _candidate_id(row)
    target_parent = int(args.target_parent_domain)
    target_fiber = int(args.target_fiber_label)
    n_domains_seen = 0
    for domain in _domain_list(atlas):
        did = int(getattr(domain, "domain_id", -1))
        if target_parent >= 0 and did != target_parent:
            continue
        labels = np.asarray(getattr(domain, "labels", []), dtype=np.int64)
        if len(labels) == 0:
            continue
        vals, counts = np.unique(labels, return_counts=True)
        pairs = [(int(v), int(c)) for v, c in zip(vals, counts) if int(c) >= int(args.min_fiber_states)]
        pairs.sort(key=lambda x: (-x[1], x[0]))
        if target_fiber >= 0:
            pairs = [(v, c) for v, c in pairs if int(v) == target_fiber]
        if int(args.max_fibers_per_domain_scan) > 0 and target_fiber < 0:
            pairs = pairs[: int(args.max_fibers_per_domain_scan)]
        for fiber_label, fiber_count in pairs:
            charts = build_predictive_quotient_charts_for_domain_fiber(
                domain, int(fiber_label), states, current_next, int(q), int(args.horizon),
                max_predictive_coords=int(args.max_predictive_coords),
                max_pair_targets=int(args.max_pair_targets),
                max_local_support_coords=int(args.max_local_support_coords),
                max_charts_per_fiber=int(args.max_charts_per_fiber),
                chart_selection=str(args.chart_selection),
                include_current_basis=bool(args.include_current_basis),
                include_horizon1_coords=bool(args.include_horizon1_coords),
                include_horizon2_coords=bool(args.include_horizon2_coords),
                include_pair_horizon1=bool(args.include_pair_horizon1),
                include_boundary_sum=bool(args.include_boundary_sum),
                include_parent_label=bool(args.include_parent_label),
                include_local_horizon1=bool(args.include_local_horizon1),
                min_chart_classes=int(args.min_chart_classes),
                min_chart_entropy=float(args.min_chart_entropy),
                min_support_states=int(args.min_support_states),
            )
            if len(charts) < 3:
                continue
            _tr, edge_maps = FCA.build_chart_transports(charts, min_overlap_states=int(args.min_overlap_states))
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
                "chart_basis": "predictive_quotient",
                "n_charts": int(len(charts)),
                "n_transport_edges": int(len(edge_maps)),
            }
            for ch in charts:
                chart_rows.append({**meta,
                    "chart_id": int(ch.chart_id),
                    "chart_type": str(ch.chart_type),
                    "support_desc": str(ch.support_desc),
                    "label_desc": str(ch.label_desc),
                    "n_support": int(ch.n_support),
                    "n_labels": int(ch.n_labels),
                    "entropy_bits": float(ch.entropy_bits),
                })
            if not edge_maps:
                continue
            gr, lr = BPH._analyze_charts_for_domain_fiber(
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
    return group_rows, loop_rows, chart_rows


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_predictive_quotient_chart_audit(
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
    max_charts_per_fiber: int = 64,
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
    max_loops_per_base: int = 3000,
    max_group_order: int = 4096,
    max_domains_scan: int = 0,
    max_fibers_per_domain_scan: int = 0,
    atlas_lift_mode: str = "bijective",
    include_trivial: bool = False,
    stop_at_first_nonabelian: bool = False,
    max_predictive_coords: int = 6,
    max_pair_targets: int = 24,
    max_local_support_coords: int = 4,
    chart_selection: str = "balanced",
    include_current_basis: bool = False,
    include_horizon1_coords: bool = True,
    include_horizon2_coords: bool = True,
    include_pair_horizon1: bool = True,
    include_boundary_sum: bool = True,
    include_parent_label: bool = True,
    include_local_horizon1: bool = True,
    out: str = "",
    plot: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    args = argparse.Namespace(**locals())
    _ensure_args(args)
    if not iterated_csv or not os.path.exists(str(iterated_csv)):
        raise ValueError("--iterated-csv is required and must exist")
    idf = pd.read_csv(str(iterated_csv), low_memory=False)
    group_rows: List[Dict[str, Any]] = []
    loop_rows: List[Dict[str, Any]] = []
    chart_rows: List[Dict[str, Any]] = []

    if str(frozen_transition_npy).strip():
        target_row = BPH._select_target_row(idf, args)
        states, current_next, _init_meta = BPH._initialize_candidate(target_row, int(q), int(vertices), args)
        loaded = np.asarray(np.load(str(frozen_transition_npy)), dtype=np.int64)
        if len(loaded) != len(states):
            raise ValueError(f"frozen transition length {len(loaded)} != state reservoir length {len(states)}")
        current_next = loaded
        atlas, _eff = _build_atlas(states, current_next, int(q), target_row, int(target_iteration), args)
        gr, lr, cr = _analyze_predictive_quotient(atlas, states, current_next, int(q), target_row, int(target_iteration), args)
        group_rows.extend(gr); loop_rows.extend(lr); chart_rows.extend(cr)
    else:
        cdf = BPH._select_search_candidates(idf, args)
        for row in cdf.to_dict(orient="records"):
            states, current_next, _init_meta = BPH._initialize_candidate(row, int(q), int(vertices), args)
            for it in range(int(atlas_iterations) + 1):
                atlas, eff = _build_atlas(states, current_next, int(q), row, int(it), args)
                gr, lr, cr = _analyze_predictive_quotient(atlas, states, current_next, int(q), row, int(it), args)
                group_rows.extend(gr); loop_rows.extend(lr); chart_rows.extend(cr)
                if stop_at_first_nonabelian and any(bool(r.get("nonabelian")) for r in gr):
                    break
                current_next = eff
            if stop_at_first_nonabelian and any(bool(r.get("nonabelian")) for r in group_rows):
                break

    rdf = pd.DataFrame(group_rows)
    ldf = pd.DataFrame(loop_rows)
    cdf_out = pd.DataFrame(chart_rows)
    chart_type_counts = {}
    if not cdf_out.empty and "chart_type" in cdf_out.columns:
        chart_type_counts = {str(k): int(v) for k, v in Counter(cdf_out["chart_type"].astype(str)).items()}
    n_charts3 = int((pd.to_numeric(cdf_out.get("n_labels", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 3).sum()) if not cdf_out.empty else 0
    max_chart_labels = int(pd.to_numeric(cdf_out.get("n_labels", pd.Series([0])), errors="coerce").fillna(0).max()) if not cdf_out.empty else 0
    if rdf.empty:
        summary = {
            "audit_version": "predictive_quotient_chart_audit_v1_prediction_task_charts",
            "verdict": "PREDICTIVE-QUOTIENT CHART BASIS EMPTY/FLAT: no nontrivial basepoint loop isotropy found",
            "q": int(q), "vertices": int(vertices),
            "n_group_rows": 0, "n_loop_rows": int(len(ldf)), "n_chart_rows": int(len(cdf_out)),
            "n_charts_with_3plus_labels": int(n_charts3), "max_chart_label_count": int(max_chart_labels),
            "chart_type_counts": chart_type_counts,
            "any_nonabelian_basepoint_isotropy": False,
            "any_exact_s3_basepoint_isotropy": False,
            "max_generated_group_order": 0,
            "max_loop_label_support_size": 0,
            "max_orbit_size": 0,
        }
    else:
        fam_counts = {str(k): int(v) for k, v in Counter(rdf["isotropy_family"].astype(str)).items()}
        any_nonab = bool(rdf.get("nonabelian", pd.Series(dtype=bool)).map(_safe_bool).any())
        any_s3 = bool(rdf.get("exact_s3", pd.Series(dtype=bool)).map(_safe_bool).any())
        max_order = int(pd.to_numeric(rdf.get("generated_group_order", pd.Series([0])), errors="coerce").fillna(0).max())
        max_support = int(pd.to_numeric(rdf.get("loop_label_support_size", pd.Series([0])), errors="coerce").fillna(0).max())
        max_orbit = int(pd.to_numeric(rdf.get("max_orbit_size", pd.Series([0])), errors="coerce").fillna(0).max())
        max_distinct = int(pd.to_numeric(rdf.get("distinct_loop_maps", pd.Series([0])), errors="coerce").fillna(0).max())
        if any_s3:
            verdict = "PREDICTIVE-QUOTIENT CHART BASIS S3 SIGNAL: exact S3 basepoint isotropy found"
        elif any_nonab:
            verdict = "PREDICTIVE-QUOTIENT CHART BASIS NONABELIAN SIGNAL: noncommuting basepoint isotropy found"
        elif max_order > 1:
            verdict = "PREDICTIVE-QUOTIENT CHART BASIS C2/ABELIAN BASELINE: nontrivial loops remain abelian"
        else:
            verdict = "PREDICTIVE-QUOTIENT CHART BASIS FLAT BASELINE: no nontrivial basepoint isotropy found"
        by_type = []
        if "base_chart_type" in rdf.columns:
            for typ, g in rdf.groupby("base_chart_type", dropna=False):
                by_type.append({
                    "base_chart_type": str(typ),
                    "n": int(len(g)),
                    "exact_s3_fraction": float(g.get("exact_s3", pd.Series(dtype=bool)).map(_safe_bool).mean()),
                    "nonabelian_fraction": float(g.get("nonabelian", pd.Series(dtype=bool)).map(_safe_bool).mean()),
                    "max_group_order": int(pd.to_numeric(g.get("generated_group_order", pd.Series([0])), errors="coerce").fillna(0).max()),
                    "max_support_size": int(pd.to_numeric(g.get("loop_label_support_size", pd.Series([0])), errors="coerce").fillna(0).max()),
                    "max_distinct_loop_maps": int(pd.to_numeric(g.get("distinct_loop_maps", pd.Series([0])), errors="coerce").fillna(0).max()),
                })
        summary = {
            "audit_version": "predictive_quotient_chart_audit_v1_prediction_task_charts",
            "verdict": verdict,
            "q": int(q), "vertices": int(vertices),
            "n_group_rows": int(len(rdf)), "n_loop_rows": int(len(ldf)), "n_chart_rows": int(len(cdf_out)),
            "n_charts_with_3plus_labels": int(n_charts3), "max_chart_label_count": int(max_chart_labels),
            "chart_type_counts": chart_type_counts,
            "algebra_family_counts": fam_counts,
            "any_nonabelian_basepoint_isotropy": bool(any_nonab),
            "any_exact_s3_basepoint_isotropy": bool(any_s3),
            "max_generated_group_order": int(max_order),
            "max_loop_label_support_size": int(max_support),
            "max_distinct_loop_maps": int(max_distinct),
            "max_orbit_size": int(max_orbit),
            "by_base_chart_type": by_type,
        }
    summary["args"] = {k: _json_safe(v) for k, v in vars(args).items() if k not in {"idf"}}

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        rdf.to_csv(out, index=False)
        ldf.to_csv(os.path.splitext(out)[0] + "_loops.csv", index=False)
        cdf_out.to_csv(os.path.splitext(out)[0] + "_charts.csv", index=False)
        with open(os.path.splitext(out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
    if plot:
        try:
            import matplotlib.pyplot as plt
            fam = summary.get("algebra_family_counts", {}) or {"none": 0}
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(list(fam.keys()), list(fam.values()))
            ax.set_title("Predictive-quotient basepoint isotropy families")
            ax.set_ylabel("count")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            os.makedirs(os.path.dirname(plot) or ".", exist_ok=True)
            fig.savefig(plot, dpi=160)
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"plot skipped: {e}")
    return rdf, ldf, cdf_out, summary


# ---------------------------------------------------------------------------
# Synthetic smoke
# ---------------------------------------------------------------------------
def run_synthetic_smoke(out: str = "", plot: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # Create five charts and edge maps whose loops at base 0 generate S3.
    n = 6
    mask = np.ones(n, dtype=bool)
    def fake_chart(cid: int, labels: List[int]) -> Any:
        full = np.asarray(labels, dtype=np.int32)
        return FCA.FiberChart(cid, 0, 0, "synthetic_predictive", "all", f"synthetic{cid}", mask.copy(), full, n, len(set(labels)), FCA.entropy(labels))
    charts = [
        fake_chart(0, [0, 1, 2, 0, 1, 2]),
        fake_chart(1, [0, 1, 2, 0, 1, 2]),
        fake_chart(2, [1, 0, 2, 1, 0, 2]),
        fake_chart(3, [2, 1, 0, 2, 1, 0]),
        fake_chart(4, [0, 1, 2, 0, 1, 2]),
    ]
    # Build a hand-coded exact transport graph.  Loops [0,1,2] and [0,3,4]
    # are both based at chart 0 and induce (0 1) and (0 2), respectively.
    def tr(a, b, mp):
        return FCA.ChartTransport(
            source=int(a), target=int(b), overlap_size=n, source_label_count=3, target_label_count=3,
            relation_pair_count=3, forward_deterministic=True, backward_deterministic=True,
            exact_bijective_transport=True, forward_accuracy=1.0, backward_accuracy=1.0,
            forward_map={int(k): int(v) for k, v in mp.items()},
            backward_map={int(v): int(k) for k, v in mp.items()}, failure_reason="",
        )
    ident = {0: 0, 1: 1, 2: 2}
    swap01 = {0: 1, 1: 0, 2: 2}
    swap02 = {0: 2, 2: 0, 1: 1}
    edge_maps = {
        (0, 1): tr(0, 1, ident),
        (1, 2): tr(1, 2, ident),
        (0, 2): tr(0, 2, swap01),
        (0, 3): tr(0, 3, ident),
        (3, 4): tr(3, 4, ident),
        (0, 4): tr(0, 4, swap02),
    }
    meta = {"candidate_id": "synthetic", "rule_mode": "synthetic", "instance": 0, "profile": "synthetic", "atlas_capacity": 0, "initial_seed": 0, "atlas_iteration": 0, "parent_domain_id": 0, "fiber_label": 0, "fiber_state_count": n, "chart_basis": "predictive_quotient", "n_charts": len(charts), "n_transport_edges": len(edge_maps)}
    gr, lr = BPH._analyze_charts_for_domain_fiber(charts, edge_maps, meta, max_cycle_len=4, max_loops_per_base=100, max_group_order=4096, include_trivial=False)
    rdf = pd.DataFrame(gr); ldf = pd.DataFrame(lr)
    cdf = pd.DataFrame([{**meta, "chart_id": c.chart_id, "chart_type": c.chart_type, "label_desc": c.label_desc, "support_desc": c.support_desc, "n_support": c.n_support, "n_labels": c.n_labels, "entropy_bits": c.entropy_bits} for c in charts])
    any_s3 = bool((not rdf.empty) and rdf.get("exact_s3", pd.Series(dtype=bool)).map(_safe_bool).any())
    fam_counts = {str(k): int(v) for k, v in Counter(rdf.get("isotropy_family", pd.Series(dtype=str)).astype(str)).items()} if not rdf.empty else {}
    summary = {
        "audit_version": "predictive_quotient_chart_audit_v1_prediction_task_charts",
        "q": 2, "vertices": 3,
        "synthetic_smoke": True,
        "verdict": "PREDICTIVE-QUOTIENT SYNTHETIC S3 SIGNAL" if any_s3 else "PREDICTIVE-QUOTIENT SYNTHETIC FAILED",
        "algebra_family_counts": fam_counts,
        "any_exact_s3_basepoint_isotropy": any_s3,
        "any_nonabelian_basepoint_isotropy": bool((not rdf.empty) and rdf.get("nonabelian", pd.Series(dtype=bool)).map(_safe_bool).any()),
        "max_generated_group_order": int(pd.to_numeric(rdf.get("generated_group_order", pd.Series([0])), errors="coerce").fillna(0).max()) if not rdf.empty else 0,
        "max_loop_label_support_size": int(pd.to_numeric(rdf.get("loop_label_support_size", pd.Series([0])), errors="coerce").fillna(0).max()) if not rdf.empty else 0,
        "n_group_rows": int(len(rdf)), "n_loop_rows": int(len(ldf)), "n_chart_rows": int(len(cdf)),
    }
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        rdf.to_csv(out, index=False)
        ldf.to_csv(os.path.splitext(out)[0] + "_loops.csv", index=False)
        cdf.to_csv(os.path.splitext(out)[0] + "_charts.csv", index=False)
        with open(os.path.splitext(out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
    if plot:
        try:
            import matplotlib.pyplot as plt
            fam = summary.get("algebra_family_counts", {}) or {"none": 0}
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(list(fam.keys()), list(fam.values()))
            ax.set_title("Predictive-quotient synthetic smoke")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout(); os.makedirs(os.path.dirname(plot) or ".", exist_ok=True); fig.savefig(plot, dpi=160); plt.close(fig)
        except Exception:
            pass
    return rdf, ldf, cdf, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Predictive-quotient chart-basis basepoint isotropy audit")
    ap.add_argument("q", nargs="?", type=int, default=2)
    ap.add_argument("--synthetic-smoke", action="store_true")
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
    ap.add_argument("--max-signature-charts", type=int, default=48)
    ap.add_argument("--atlas-lift-mode", default="bijective")
    ap.add_argument("--max-charts-per-fiber", type=int, default=64)
    ap.add_argument("--min-fiber-states", type=int, default=6)
    ap.add_argument("--min-support-states", type=int, default=3)
    ap.add_argument("--min-overlap-states", type=int, default=3)
    ap.add_argument("--min-chart-classes", type=int, default=2)
    ap.add_argument("--min-chart-entropy", type=float, default=0.0)
    ap.add_argument("--max-chart-coords", type=int, default=5)
    ap.add_argument("--max-support-coords", type=int, default=4)
    ap.add_argument("--max-cycle-len", type=int, default=5)
    ap.add_argument("--max-cycles-per-fiber", type=int, default=500)
    ap.add_argument("--max-loops-per-base", type=int, default=3000)
    ap.add_argument("--max-group-order", type=int, default=4096)
    ap.add_argument("--max-domains-scan", type=int, default=0)
    ap.add_argument("--max-fibers-per-domain-scan", type=int, default=0)
    ap.add_argument("--include-trivial", action="store_true")
    ap.add_argument("--stop-at-first-nonabelian", action="store_true")
    # Predictive quotient options.
    ap.add_argument("--max-predictive-coords", type=int, default=6)
    ap.add_argument("--max-pair-targets", type=int, default=24)
    ap.add_argument("--max-local-support-coords", type=int, default=4)
    ap.add_argument("--chart-selection", default="balanced", choices=["balanced", "entropy", "priority"])
    ap.add_argument("--include-current-basis", action="store_true")
    ap.add_argument("--no-horizon1-coords", action="store_true")
    ap.add_argument("--no-horizon2-coords", action="store_true")
    ap.add_argument("--no-pair-horizon1", action="store_true")
    ap.add_argument("--no-boundary-sum", action="store_true")
    ap.add_argument("--no-parent-label", action="store_true")
    ap.add_argument("--no-local-horizon1", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", default="")
    ns = ap.parse_args(argv)
    if ns.synthetic_smoke:
        _rdf, _ldf, _cdf, summary = run_synthetic_smoke(ns.out, ns.plot)
    else:
        _rdf, _ldf, _cdf, summary = run_predictive_quotient_chart_audit(
            q=int(ns.q), vertices=int(ns.vertices), iterated_csv=str(ns.iterated_csv), frozen_transition_npy=str(ns.frozen_transition_npy),
            target_rule_mode=str(ns.target_rule_mode), target_instance=int(ns.target_instance), target_profile=str(ns.target_profile),
            target_atlas_capacity=int(ns.target_atlas_capacity), target_seed=int(ns.target_seed), target_iteration=int(ns.target_iteration),
            target_parent_domain=int(ns.target_parent_domain), target_fiber_label=int(ns.target_fiber_label),
            rule_modes=str(ns.rule_modes), profiles=str(ns.profiles), atlas_capacities=str(ns.atlas_capacities), max_candidates=int(ns.max_candidates),
            atlas_iterations=int(ns.atlas_iterations), require_generated=bool(ns.require_generated), max_state_samples=int(ns.max_state_samples),
            max_total_states=int(ns.max_total_states), max_pred=int(ns.max_pred), proliferation_iterations=int(ns.proliferation_iterations), horizon=int(ns.horizon),
            initial_boundary=str(ns.initial_boundary), initial_boundary_q=ns.initial_boundary_q, max_domains_per_depth=int(ns.max_domains_per_depth),
            min_live_classes=int(ns.min_live_classes), min_fiber_size=int(ns.min_fiber_size), min_entropy_bits=float(ns.min_entropy_bits),
            synergy_threshold=float(ns.synergy_threshold), max_signature_domains=int(ns.max_signature_domains), max_parent_domains=int(ns.max_parent_domains),
            max_fibers_per_parent=int(ns.max_fibers_per_parent), max_charts_per_fiber=int(ns.max_charts_per_fiber),
            max_signature_charts=int(ns.max_signature_charts), min_fiber_states=int(ns.min_fiber_states), min_support_states=int(ns.min_support_states),
            min_overlap_states=int(ns.min_overlap_states), min_chart_classes=int(ns.min_chart_classes), min_chart_entropy=float(ns.min_chart_entropy),
            max_chart_coords=int(ns.max_chart_coords), max_support_coords=int(ns.max_support_coords), max_cycle_len=int(ns.max_cycle_len),
            max_cycles_per_fiber=int(ns.max_cycles_per_fiber), max_loops_per_base=int(ns.max_loops_per_base), max_group_order=int(ns.max_group_order),
            max_domains_scan=int(ns.max_domains_scan), max_fibers_per_domain_scan=int(ns.max_fibers_per_domain_scan), atlas_lift_mode=str(ns.atlas_lift_mode),
            include_trivial=bool(ns.include_trivial), stop_at_first_nonabelian=bool(ns.stop_at_first_nonabelian),
            max_predictive_coords=int(ns.max_predictive_coords), max_pair_targets=int(ns.max_pair_targets), max_local_support_coords=int(ns.max_local_support_coords),
            chart_selection=str(ns.chart_selection), include_current_basis=bool(ns.include_current_basis), include_horizon1_coords=not bool(ns.no_horizon1_coords),
            include_horizon2_coords=not bool(ns.no_horizon2_coords), include_pair_horizon1=not bool(ns.no_pair_horizon1), include_boundary_sum=not bool(ns.no_boundary_sum),
            include_parent_label=not bool(ns.no_parent_label), include_local_horizon1=not bool(ns.no_local_horizon1), out=str(ns.out), plot=str(ns.plot)
        )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    if ns.out:
        print(f"wrote {ns.out}")
        print(f"wrote {os.path.splitext(ns.out)[0]}_loops.csv")
        print(f"wrote {os.path.splitext(ns.out)[0]}_charts.csv")
        print(f"wrote {os.path.splitext(ns.out)[0]}_summary.json")
    if ns.plot:
        print(f"wrote {ns.plot}")


if __name__ == "__main__":
    main()
