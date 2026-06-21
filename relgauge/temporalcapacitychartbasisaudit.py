"""
temporalcapacitychartbasisaudit.py

Endogenous / temporal-capacity chart-basis audit.

Why this module exists
----------------------
The stricter basepoint-aware holonomy audit showed that the earlier raw-label
S3 witness collapses to C2 when loop maps are classified at a single chart
basepoint.  A likely reason is structural: the current chart basis contains
mostly singleton coordinate charts (binary for q=2) plus predictive-history
charts.  Singleton coordinate charts naturally generate C2 flips, while the
multi-label predictive charts tend to glue flatly or restrict to two-label
subcores.

This module tests the proposed fix: chart richness should be induced by the
observer's own temporal/boundary capacity rather than chosen as an external
knob.  If an observer cannot resolve several coordinate samples within one
effective time window, the effective chart is a joint coordinate block such as
(x[i], x[j]) or (x[i], T(x)[j]).  Such charts can have 3+ labels at q=2 and may
support nonabelian basepoint isotropy if overlap loops act noncommutatively on a
common base chart.

Evidence tiers
--------------
1. current_basis_baseline
   Existing chart construction.  Expected to produce mostly C2 basepoint
   isotropy.

2. manual_multi_coord
   Adds explicit multi-coordinate block charts.  Diagnostic: can richer charts
   express nonabelian basepoint isotropy at all?

3. capacity_induced
   Builds joint charts from temporal-unresolved coordinate blocks.  This is the
   endogenous chart-basis test: chart arity is induced by an observer temporal
   window rather than manually selected as a physical target.

4. rich
   Adds current + capacity-induced + predictive-coordinate mixed charts.  This
   is a high-recall search mode.

All nonabelian claims are basepoint-aware: loop maps are only composed when they
are based at the same chart and share the same label support.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("temporalcapacitychartbasisaudit requires pandas") from e

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
# Helpers
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


def _hash_arr(vals: Sequence[int]) -> str:
    return hashlib.sha1(np.asarray(list(vals), dtype=np.int64).tobytes()).hexdigest()[:20]


def _candidate_id(row: Dict[str, Any]) -> str:
    return BPH._candidate_id(row)


def _domain_list(atlas: Any) -> List[Any]:
    return BPH._domain_list(atlas)


def _mk_chart(chart_id: int, parent_id: int, fiber_label: int, kind: str, sdesc: str, ldesc: str,
              support_idx: np.ndarray, raw_labels: Sequence[Any], n_states: int, min_chart_classes: int,
              min_chart_entropy: float, min_support_states: int) -> Optional[Any]:
    """Local wrapper matching fiberchartconnectionaudit._mk_chart behavior."""
    support_idx = np.asarray(support_idx, dtype=np.int64)
    if len(support_idx) < int(min_support_states):
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


def _fiber_indices(domain: Any, fiber_label: int) -> np.ndarray:
    labels = np.asarray(getattr(domain, "labels", []), dtype=np.int64)
    return np.where(labels == int(fiber_label))[0]


# ---------------------------------------------------------------------------
# Temporal-capacity chart basis
# ---------------------------------------------------------------------------
def _coord_blocks(arr: np.ndarray, next_arr: np.ndarray, idxs: np.ndarray, q: int, *,
                  arity: int, max_coord: int, max_blocks: int, mode: str) -> List[Tuple[int, ...]]:
    """Derive unresolved temporal coordinate blocks.

    Modes:
      sliding    : deterministic sliding windows over the first max_coord coordinates.
      contiguous : disjoint contiguous blocks.
      cyclic     : cyclic sliding windows.
      change_corr: high-recall pairs/triples ranked by correlated change patterns.
    """
    k = int(arr.shape[1])
    coords = list(range(min(int(max_coord), k)))
    b = max(1, min(int(arity), len(coords)))
    mode = str(mode or "sliding").lower()
    out: List[Tuple[int, ...]] = []
    if b <= 0 or not coords:
        return []
    if b == 1:
        return [(c,) for c in coords[: int(max_blocks) if int(max_blocks) > 0 else len(coords)]]
    if mode in {"contiguous", "disjoint"}:
        for i in range(0, len(coords), b):
            block = tuple(coords[i:i + b])
            if len(block) == b:
                out.append(block)
    elif mode in {"cyclic"}:
        for i in range(len(coords)):
            block = tuple(coords[(i + j) % len(coords)] for j in range(b))
            if len(set(block)) == b:
                out.append(block)
    elif mode in {"change_corr", "correlated", "change_correlation"}:
        # Rank coordinate tuples by covariance of change indicators over the fiber.
        changes = (np.asarray(next_arr[idxs], dtype=np.int64) - np.asarray(arr[idxs], dtype=np.int64)) % int(q)
        # binary-ish nonzero change indicator works for all q.
        ch = (changes[:, coords] != 0).astype(float)
        tuples: List[Tuple[float, Tuple[int, ...]]] = []
        import itertools
        for block in itertools.combinations(coords, b):
            if ch.shape[0] <= 1:
                score = 0.0
            else:
                sub = ch[:, [coords.index(c) for c in block]]
                # average pairwise absolute covariance + entropy of joint observed current block.
                cov = np.cov(sub, rowvar=False)
                if np.ndim(cov) == 0:
                    corr_score = 0.0
                else:
                    corr_score = float(np.mean(np.abs(cov[np.triu_indices_from(cov, k=1)]))) if b > 1 else 0.0
                vals = [tuple(int(arr[int(i), c]) % int(q) for c in block) for i in idxs]
                ent = float(FCA.entropy(FCA.relabel_on_support(vals))) if vals else 0.0
                score = corr_score + 0.05 * ent
            tuples.append((float(score), tuple(int(c) for c in block)))
        tuples.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out = [b for _s, b in tuples]
    else:  # sliding
        for i in range(0, max(0, len(coords) - b + 1)):
            out.append(tuple(coords[i:i + b]))
    # Deduplicate and cap.
    seen = set(); ans: List[Tuple[int, ...]] = []
    for block in out:
        block = tuple(int(x) for x in block)
        if len(set(block)) != len(block):
            continue
        if block in seen:
            continue
        seen.add(block); ans.append(block)
        if int(max_blocks) > 0 and len(ans) >= int(max_blocks):
            break
    return ans


def build_temporal_capacity_charts_for_domain_fiber(
    domain: Any,
    fiber_label: int,
    states: Sequence[Tuple[int, ...]],
    next_idx: Sequence[int],
    q: int,
    horizon: int,
    *,
    chart_basis: str = "capacity_induced",
    temporal_window_size: int = 2,
    temporal_block_mode: str = "sliding",
    max_temporal_blocks: int = 32,
    include_current_charts: bool = True,
    include_predictive_mix: bool = True,
    max_chart_coords: int = 5,
    max_support_coords: int = 4,
    max_charts_per_fiber: int = 32,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.05,
    min_support_states: int = 8,
) -> List[Any]:
    """Construct current/manual/capacity-induced charts over one hidden fiber."""
    basis = str(chart_basis or "capacity_induced").lower().strip()
    if basis in {"current", "baseline", "existing"}:
        return FCA.build_charts_for_domain_fiber(
            domain, fiber_label, states, next_idx, q, horizon,
            max_chart_coords=max_chart_coords, max_support_coords=max_support_coords,
            max_charts_per_fiber=max_charts_per_fiber,
            min_chart_classes=min_chart_classes, min_chart_entropy=min_chart_entropy,
            min_support_states=min_support_states,
        )

    arr = np.asarray(states, dtype=np.int64)
    n_states = int(len(arr))
    if n_states == 0 or arr.ndim != 2:
        return []
    k = int(arr.shape[1])
    idxs = _fiber_indices(domain, int(fiber_label))
    if len(idxs) < int(min_support_states):
        return []
    next_arr = arr[np.asarray(next_idx, dtype=np.int64)]
    histories = BPA.future_label_history(domain.labels, next_idx, horizon=int(horizon), include_current=True)
    charts: List[Any] = []

    def add(kind: str, sdesc: str, ldesc: str, support: Sequence[int], raw: Sequence[Any]) -> None:
        nonlocal charts
        ch = _mk_chart(len(charts), int(getattr(domain, "domain_id", -1)), int(fiber_label), kind, sdesc, ldesc,
                       np.asarray(support, dtype=np.int64), list(raw), n_states,
                       min_chart_classes, min_chart_entropy, min_support_states)
        if ch is not None:
            charts.append(ch)

    if include_current_charts:
        # Add existing charts as a baseline substrate, but they do not dominate once richer
        # capacity charts have higher entropy/support.
        try:
            charts.extend(FCA.build_charts_for_domain_fiber(
                domain, fiber_label, states, next_idx, q, horizon,
                max_chart_coords=max_chart_coords, max_support_coords=max_support_coords,
                max_charts_per_fiber=max_charts_per_fiber,
                min_chart_classes=min_chart_classes, min_chart_entropy=min_chart_entropy,
                min_support_states=min_support_states,
            ))
        except Exception:
            pass

    # Always include the global predictive chart as the observer's parent-boundary face.
    add("predictive_history", f"parent={getattr(domain,'domain_id',-1)};fiber={fiber_label};all", "future(parent_label)", idxs, [histories[int(i)] for i in idxs])

    # Temporal-capacity blocks.  The unresolved temporal window determines arity.
    block_sizes = [int(temporal_window_size)]
    if basis in {"rich", "manual_multi_coord", "multi_coord"} and int(temporal_window_size) > 1:
        # Include all arities up to the requested one as a diagnostic high-recall basis.
        block_sizes = list(range(2, int(temporal_window_size) + 1))
    elif basis in {"manual_multi_coord", "multi_coord"}:
        block_sizes = [2]

    all_blocks: List[Tuple[int, ...]] = []
    for b in block_sizes:
        all_blocks.extend(_coord_blocks(arr, next_arr, idxs, int(q), arity=b, max_coord=max_chart_coords,
                                        max_blocks=max_temporal_blocks, mode=temporal_block_mode))
    # Dedup.
    seen_blocks = set(); blocks: List[Tuple[int, ...]] = []
    for block in all_blocks:
        if block not in seen_blocks:
            seen_blocks.add(block); blocks.append(block)

    def block_name(block: Tuple[int, ...]) -> str:
        return ",".join(str(x) for x in block)

    for block in blocks:
        bname = block_name(block)
        add("capacity_coord_block", f"temporal_window={temporal_window_size};all", f"x[{bname}]", idxs,
            [tuple(int(arr[int(i), c]) % int(q) for c in block) for i in idxs])
        add("capacity_next_block", f"temporal_window={temporal_window_size};all", f"T(x)[{bname}]", idxs,
            [tuple(int(next_arr[int(i), c]) % int(q) for c in block) for i in idxs])
        add("capacity_delta_block", f"temporal_window={temporal_window_size};all", f"delta[{bname}]", idxs,
            [tuple((int(next_arr[int(i), c]) - int(arr[int(i), c])) % int(q) for c in block) for i in idxs])
        if include_predictive_mix or basis in {"rich", "predictive_coord", "capacity_induced"}:
            add("capacity_predictive_coord", f"temporal_window={temporal_window_size};all", f"future+x[{bname}]", idxs,
                [(histories[int(i)], tuple(int(arr[int(i), c]) % int(q) for c in block)) for i in idxs])
            add("capacity_predictive_next", f"temporal_window={temporal_window_size};all", f"future+T(x)[{bname}]", idxs,
                [(histories[int(i)], tuple(int(next_arr[int(i), c]) % int(q) for c in block)) for i in idxs])

    # Local support charts.  These are what create overlaps rather than just a single global factor.
    support_lim = min(int(max_support_coords), k)
    for s in range(support_lim):
        vals = sorted(set(int(arr[int(i), s]) % int(q) for i in idxs))
        for val in vals:
            sub = np.asarray([int(i) for i in idxs if int(arr[int(i), s]) % int(q) == int(val)], dtype=np.int64)
            if len(sub) < int(min_support_states):
                continue
            for block in blocks:
                bname = block_name(block)
                add("local_capacity_coord_block", f"x[{s}]={val}", f"x[{bname}]", sub,
                    [tuple(int(arr[int(i), c]) % int(q) for c in block) for i in sub])
                add("local_capacity_next_block", f"x[{s}]={val}", f"T(x)[{bname}]", sub,
                    [tuple(int(next_arr[int(i), c]) % int(q) for c in block) for i in sub])
                add("local_capacity_delta_block", f"x[{s}]={val}", f"delta[{bname}]", sub,
                    [tuple((int(next_arr[int(i), c]) - int(arr[int(i), c])) % int(q) for c in block) for i in sub])
                if include_predictive_mix or basis in {"rich", "predictive_coord", "capacity_induced"}:
                    add("local_capacity_predictive_coord", f"x[{s}]={val}", f"future+x[{bname}]", sub,
                        [(histories[int(i)], tuple(int(arr[int(i), c]) % int(q) for c in block)) for i in sub])

    # Deduplicate and sort.  Avoid duplicate support/label arrays, not merely duplicate text.
    dedup: Dict[Tuple[str, bytes, bytes], Any] = {}
    for ch in charts:
        key = (str(ch.chart_type) + ":" + str(ch.label_desc) + ":" + str(ch.support_desc),
               np.packbits(ch.support_mask).tobytes(), ch.labels_full[ch.support_mask].tobytes())
        if key not in dedup or float(ch.entropy_bits) > float(dedup[key].entropy_bits):
            dedup[key] = ch
    charts = list(dedup.values())
    # For enriched bases, reserve priority for the newly induced block charts.
    # Otherwise high-entropy predictive charts can crowd them out, making the
    # audit accidentally identical to the current singleton basis.  Then prefer
    # charts with 3+ labels, entropy, and broad support.
    def _chart_priority(c):
        ctype = str(getattr(c, "chart_type", ""))
        is_block = 1 if "block" in ctype else 0
        return (is_block, int(getattr(c, "n_labels", 0)) >= 3, float(getattr(c, "entropy_bits", 0.0)), int(getattr(c, "n_support", 0)), int(getattr(c, "n_labels", 0)))
    charts.sort(key=_chart_priority, reverse=True)
    charts = charts[: int(max_charts_per_fiber)]
    for i, ch in enumerate(charts):
        ch.chart_id = int(i)
    return charts


@contextlib.contextmanager
def _patched_chart_builder(args: Any):
    orig = FCA.build_charts_for_domain_fiber
    if not bool(getattr(args, "apply_basis_to_atlas_signature", False)):
        yield
        return

    def wrapper(domain, fiber_label, states, next_idx, q, horizon,
                max_chart_coords=5, max_support_coords=4, max_charts_per_fiber=32,
                min_chart_classes=2, min_chart_entropy=0.05, min_support_states=8):
        return build_temporal_capacity_charts_for_domain_fiber(
            domain, fiber_label, states, next_idx, q, horizon,
            chart_basis=str(getattr(args, "chart_basis", "capacity_induced")),
            temporal_window_size=int(getattr(args, "temporal_window_size", 2)),
            temporal_block_mode=str(getattr(args, "temporal_block_mode", "sliding")),
            max_temporal_blocks=int(getattr(args, "max_temporal_blocks", 32)),
            include_current_charts=bool(getattr(args, "include_current_charts", True)),
            include_predictive_mix=bool(getattr(args, "include_predictive_mix", True)),
            max_chart_coords=int(max_chart_coords), max_support_coords=int(max_support_coords),
            max_charts_per_fiber=int(max_charts_per_fiber),
            min_chart_classes=int(min_chart_classes), min_chart_entropy=float(min_chart_entropy),
            min_support_states=int(min_support_states),
        )

    FCA.build_charts_for_domain_fiber = wrapper
    try:
        yield
    finally:
        FCA.build_charts_for_domain_fiber = orig


# ---------------------------------------------------------------------------
# Candidate reconstruction and analysis
# ---------------------------------------------------------------------------
def _make_args(**kwargs: Any) -> Any:
    return argparse.Namespace(**kwargs)


def _ensure_args(args: Any) -> None:
    BPH._ensure_upstream_defaults(args)
    defaults = {
        "chart_basis": "capacity_induced",
        "temporal_window_size": 2,
        "temporal_block_mode": "sliding",
        "max_temporal_blocks": 32,
        "include_current_charts": True,
        "include_predictive_mix": True,
        "apply_basis_to_atlas_signature": False,
        "max_loops_per_base": 5000,
        "max_group_order": 4096,
        "include_trivial": False,
        "target_parent_domain": -1,
        "target_fiber_label": -1,
        "max_domains_scan": 0,
        "max_fibers_per_domain_scan": 0,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)


def _build_atlas(states: np.ndarray, current_next: np.ndarray, q: int, row: Dict[str, Any], iteration: int, args: Any):
    with _patched_chart_builder(args):
        return BPH._build_atlas_from_transition(states, current_next, int(q), row, int(iteration), args)


def _analyze_with_capacity_basis(atlas: Any, states: np.ndarray, current_next: np.ndarray, q: int, row: Dict[str, Any], iteration: int, args: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows: List[Dict[str, Any]] = []
    loop_rows: List[Dict[str, Any]] = []
    chart_rows: List[Dict[str, Any]] = []
    domains = _domain_list(atlas)
    target_parent = int(getattr(args, "target_parent_domain", -1))
    target_fiber = int(getattr(args, "target_fiber_label", -1))
    n_domains_seen = 0
    candidate = _candidate_id(row)
    for domain in domains:
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
            charts = build_temporal_capacity_charts_for_domain_fiber(
                domain, int(fiber_label), states, current_next, int(q), int(args.horizon),
                chart_basis=str(args.chart_basis), temporal_window_size=int(args.temporal_window_size),
                temporal_block_mode=str(args.temporal_block_mode), max_temporal_blocks=int(args.max_temporal_blocks),
                include_current_charts=bool(args.include_current_charts), include_predictive_mix=bool(args.include_predictive_mix),
                max_chart_coords=int(args.max_chart_coords), max_support_coords=int(args.max_support_coords),
                max_charts_per_fiber=int(args.max_charts_per_fiber),
                min_chart_classes=int(args.min_chart_classes), min_chart_entropy=float(args.min_chart_entropy),
                min_support_states=int(args.min_support_states),
            )
            if len(charts) < 3:
                continue
            tr_rows, edge_maps = FCA.build_chart_transports(charts, min_overlap_states=int(args.min_overlap_states))
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
                "chart_basis": str(args.chart_basis),
                "temporal_window_size": int(args.temporal_window_size),
                "temporal_block_mode": str(args.temporal_block_mode),
                "apply_basis_to_atlas_signature": bool(args.apply_basis_to_atlas_signature),
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


def _select_rows_from_iterated(iterated_csv: str, args: Any) -> pd.DataFrame:
    if not iterated_csv or not os.path.exists(iterated_csv):
        raise ValueError("--iterated-csv is required unless using --synthetic-smoke")
    df = pd.read_csv(iterated_csv)
    if int(args.target_instance) >= 0 or str(args.target_rule_mode) or str(args.target_profile) or int(args.target_atlas_capacity) >= 0 or int(args.target_seed) >= 0:
        return pd.DataFrame([BPH._select_target_row(df, args)])
    return BPH._select_search_candidates(df, args)


def run_temporal_capacity_chart_basis_audit(
    q: int,
    vertices: int = 9,
    iterated_csv: str = "",
    frozen_transition_npy: str = "",
    chart_basis: str = "capacity_induced",
    temporal_window_sizes: Sequence[int] = (2,),
    temporal_block_modes: Sequence[str] = ("sliding",),
    apply_basis_to_atlas_signature: bool = False,
    include_current_charts: bool = True,
    include_predictive_mix: bool = True,
    target_rule_mode: str = "",
    target_instance: int = -1,
    target_profile: str = "",
    target_atlas_capacity: int = -1,
    target_seed: int = -1,
    target_iteration: int = -1,
    target_parent_domain: int = -1,
    target_fiber_label: int = -1,
    rule_modes: Sequence[str] = (),
    profiles: Sequence[str] = (),
    atlas_capacities: Sequence[int] = (),
    atlas_iterations: int = 12,
    require_generated: bool = False,
    max_candidates: int = 20,
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
    max_charts_per_fiber: int = 32,
    max_signature_charts: int = 64,
    min_fiber_states: int = 6,
    min_support_states: int = 3,
    min_overlap_states: int = 3,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.0,
    max_chart_coords: int = 6,
    max_support_coords: int = 5,
    max_temporal_blocks: int = 32,
    max_cycle_len: int = 5,
    max_loops_per_base: int = 5000,
    max_cycles_per_fiber: int = 1000,
    max_group_order: int = 4096,
    max_domains_scan: int = 0,
    max_fibers_per_domain_scan: int = 0,
    atlas_lift_mode: str = "bijective",
    include_trivial: bool = False,
    stop_at_first_nonabelian: bool = False,
    out: str = "",
    plot: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    args = _make_args(**locals())
    # Normalize target fields for basepoint helper compatibility.
    args.rule_modes = ",".join(rule_modes) if not isinstance(rule_modes, str) else rule_modes
    args.profiles = ",".join(profiles) if not isinstance(profiles, str) else profiles
    args.atlas_capacities = ",".join(str(x) for x in atlas_capacities) if not isinstance(atlas_capacities, str) else atlas_capacities
    _ensure_args(args)
    rows_df = _select_rows_from_iterated(iterated_csv, args)
    if int(max_candidates) > 0 and not (int(target_instance) >= 0 or frozen_transition_npy):
        rows_df = rows_df.head(int(max_candidates)).copy()
    group_rows: List[Dict[str, Any]] = []
    loop_rows: List[Dict[str, Any]] = []
    chart_rows: List[Dict[str, Any]] = []
    n_processed = 0
    for _ri, rr in rows_df.iterrows():
        row = rr.to_dict()
        states, current_next, _meta = BPH._initialize_candidate(row, int(q), int(vertices), args)
        if frozen_transition_npy:
            current_next = np.load(str(frozen_transition_npy)).astype(np.int64)
        # iterations to inspect
        if frozen_transition_npy:
            its = [int(target_iteration) if int(target_iteration) >= 0 else 0]
        elif int(target_iteration) >= 0:
            its = [int(target_iteration)]
        else:
            its = list(range(int(atlas_iterations)))
        for it in range(0, max(its) + 1):
            # Build atlas using either current or patched basis. Analyze only requested iterations.
            for tw in temporal_window_sizes:
                for tbm in temporal_block_modes:
                    args.temporal_window_size = int(tw)
                    args.temporal_block_mode = str(tbm)
                    atlas, eff = _build_atlas(states, current_next, int(q), row, int(it), args)
                    if int(it) in its:
                        gr, lr, cr = _analyze_with_capacity_basis(atlas, states, current_next, int(q), row, int(it), args)
                        group_rows.extend(gr); loop_rows.extend(lr); chart_rows.extend(cr)
                        n_processed += 1
                        if stop_at_first_nonabelian and any(bool(x.get("nonabelian")) for x in gr):
                            gdf = pd.DataFrame(group_rows); ldf = pd.DataFrame(loop_rows); cdf = pd.DataFrame(chart_rows)
                            summary = _summarize(gdf, ldf, cdf, args, n_processed)
                            _write_outputs(gdf, ldf, cdf, summary, out, plot)
                            return gdf, ldf, cdf, summary
                    # Avoid evolving multiple times for tw/tbm variants; atlas construction can differ under patched basis.
                    # Effective dynamics for iteration progression should use the first configured basis only.
                    if tw == list(temporal_window_sizes)[0] and tbm == list(temporal_block_modes)[0]:
                        next_eff = eff
            if not frozen_transition_npy:
                current_next = np.asarray(next_eff, dtype=np.int64)
    gdf = pd.DataFrame(group_rows)
    ldf = pd.DataFrame(loop_rows)
    cdf = pd.DataFrame(chart_rows)
    summary = _summarize(gdf, ldf, cdf, args, n_processed)
    _write_outputs(gdf, ldf, cdf, summary, out, plot)
    return gdf, ldf, cdf, summary


def _summarize(gdf: pd.DataFrame, ldf: pd.DataFrame, cdf: pd.DataFrame, args: Any, n_processed: int) -> Dict[str, Any]:
    if gdf is None or gdf.empty:
        fam_counts: Dict[str, int] = {}
        any_nonabelian = any_s3 = False
        max_order = max_orbit = max_labels = 0
    else:
        fam_counts = {str(k): int(v) for k, v in gdf.get("isotropy_family", pd.Series(dtype=str)).value_counts().to_dict().items()}
        any_nonabelian = bool(gdf.get("nonabelian", pd.Series(dtype=bool)).map(_safe_bool).any())
        any_s3 = bool(gdf.get("exact_s3", pd.Series(dtype=bool)).map(_safe_bool).any())
        max_order = int(pd.to_numeric(gdf.get("generated_group_order", pd.Series([0])), errors="coerce").fillna(0).max())
        max_orbit = int(pd.to_numeric(gdf.get("max_orbit_size", pd.Series([0])), errors="coerce").fillna(0).max())
        max_labels = int(pd.to_numeric(gdf.get("loop_label_support_size", pd.Series([0])), errors="coerce").fillna(0).max())
    if cdf is not None and not cdf.empty:
        chart_type_counts = {str(k): int(v) for k, v in cdf.get("chart_type", pd.Series(dtype=str)).value_counts().head(30).to_dict().items()}
        n_charts_ge3 = int((pd.to_numeric(cdf.get("n_labels", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 3).sum())
        max_chart_labels = int(pd.to_numeric(cdf.get("n_labels", pd.Series([0])), errors="coerce").fillna(0).max())
    else:
        chart_type_counts = {}; n_charts_ge3 = 0; max_chart_labels = 0
    if not any_nonabelian:
        verdict = "TEMPORAL-CAPACITY CHART BASIS C2/ABELIAN BASELINE: no nonabelian basepoint isotropy found"
    elif any_s3:
        verdict = "TEMPORAL-CAPACITY CHART BASIS S3 SIGNAL: exact S3 basepoint isotropy found"
    else:
        verdict = "TEMPORAL-CAPACITY CHART BASIS NONABELIAN SIGNAL: nonabelian basepoint isotropy found"
    return {
        "verdict": verdict,
        "audit_version": "temporal_capacity_chart_basis_audit_v1_endogenous_chart_richness",
        "q": int(args.q),
        "vertices": int(args.vertices),
        "chart_basis": str(args.chart_basis),
        "temporal_window_sizes": [int(x) for x in args.temporal_window_sizes],
        "temporal_block_modes": [str(x) for x in args.temporal_block_modes],
        "apply_basis_to_atlas_signature": bool(args.apply_basis_to_atlas_signature),
        "n_processed_atlases": int(n_processed),
        "n_group_rows": int(0 if gdf is None else len(gdf)),
        "n_loop_rows": int(0 if ldf is None else len(ldf)),
        "n_chart_rows": int(0 if cdf is None else len(cdf)),
        "chart_type_counts": chart_type_counts,
        "n_charts_with_3plus_labels": int(n_charts_ge3),
        "max_chart_label_count": int(max_chart_labels),
        "algebra_family_counts": fam_counts,
        "any_nonabelian_basepoint_isotropy": bool(any_nonabelian),
        "any_exact_s3_basepoint_isotropy": bool(any_s3),
        "max_generated_group_order": int(max_order),
        "max_orbit_size": int(max_orbit),
        "max_loop_label_support_size": int(max_labels),
    }


def _write_outputs(gdf: pd.DataFrame, ldf: pd.DataFrame, cdf: pd.DataFrame, summary: Dict[str, Any], out: str, plot: str) -> None:
    if out:
        base = os.path.splitext(str(out))[0]
        os.makedirs(os.path.dirname(str(out)) or ".", exist_ok=True)
        gdf.to_csv(out, index=False)
        ldf.to_csv(base + "_loops.csv", index=False)
        cdf.to_csv(base + "_charts.csv", index=False)
        with open(base + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(BPH._json_safe(summary), f, indent=2, sort_keys=True)
    if plot:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(str(plot)) or ".", exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            counts = summary.get("algebra_family_counts", {}) or {}
            if counts:
                labels = list(counts.keys()); vals = [counts[k] for k in labels]
                ax.bar(range(len(labels)), vals)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_ylabel("basepoint rows")
            else:
                ax.text(0.5, 0.5, "No group rows", ha="center", va="center")
                ax.set_axis_off()
            ax.set_title("Temporal-capacity chart-basis isotropy")
            fig.tight_layout()
            fig.savefig(plot, dpi=160)
            plt.close(fig)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Synthetic smoke
# ---------------------------------------------------------------------------
def _synthetic_smoke(out: str = "", plot: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # Directly exercise the basepoint closure code on a synthetic three-label chart set.
    # Two distinct loops based at chart 0 realize (0 1) and (0 2), so basepoint
    # isotropy closes to exact S3.
    charts: List[Any] = []
    n = 6
    def fake_chart(cid: int, labels: List[int]) -> Any:
        mask = np.ones(n, dtype=bool)
        full = np.asarray(labels, dtype=np.int32)
        return FCA.FiberChart(cid, 0, 0, "synthetic", "all", f"ch{cid}", mask, full, n, len(set(labels)), FCA.entropy(labels))
    charts = [fake_chart(i, [0,0,1,1,2,2]) for i in range(5)]
    def ct(a,b,mp):
        return FCA.ChartTransport(a,b,6,3,3,3,True,True,True,1.0,1.0,mp,{v:k for k,v in mp.items()},"")
    ident = {0:0,1:1,2:2}
    flip01 = {0:1,1:0,2:2}
    flip02 = {0:2,2:0,1:1}
    edge_maps = {
        (0,1): ct(0,1,ident), (1,2): ct(1,2,ident), (0,2): ct(0,2,flip01),
        (0,3): ct(0,3,ident), (3,4): ct(3,4,ident), (0,4): ct(0,4,flip02),
    }
    meta = {"candidate_id":"synthetic", "atlas_iteration":0, "parent_domain_id":0, "fiber_label":0, "chart_basis":"synthetic", "temporal_window_size":2, "temporal_block_mode":"manual"}
    gr, lr = BPH._analyze_charts_for_domain_fiber(charts, edge_maps, meta, max_cycle_len=3, max_loops_per_base=50, max_group_order=4096, include_trivial=True)
    gdf = pd.DataFrame(gr); ldf = pd.DataFrame(lr)
    cdf = pd.DataFrame([{"chart_id": c.chart_id, "chart_type": c.chart_type, "n_labels": c.n_labels, "n_support": c.n_support} for c in charts])
    args = _make_args(q=2, vertices=3, chart_basis="synthetic", temporal_window_sizes=[2], temporal_block_modes=["manual"], apply_basis_to_atlas_signature=False)
    summary = _summarize(gdf, ldf, cdf, args, 1)
    _write_outputs(gdf, ldf, cdf, summary, out, plot)
    return gdf, ldf, cdf, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal-capacity/endogenous chart-basis audit")
    ap.add_argument("q", nargs="?", type=int, default=2)
    ap.add_argument("--vertices", type=int, default=9)
    ap.add_argument("--iterated-csv", default="")
    ap.add_argument("--frozen-transition-npy", default="")
    ap.add_argument("--synthetic-smoke", action="store_true")
    ap.add_argument("--chart-basis", default="capacity_induced", choices=["current", "manual_multi_coord", "multi_coord", "capacity_induced", "predictive_coord", "rich"])
    ap.add_argument("--temporal-window-sizes", default="2")
    ap.add_argument("--temporal-block-modes", default="sliding")
    ap.add_argument("--max-temporal-blocks", type=int, default=32)
    ap.add_argument("--apply-basis-to-atlas-signature", action="store_true")
    ap.add_argument("--no-current-charts", action="store_true")
    ap.add_argument("--no-predictive-mix", action="store_true")
    ap.add_argument("--target-rule-mode", default="")
    ap.add_argument("--target-instance", type=int, default=-1)
    ap.add_argument("--target-profile", default="")
    ap.add_argument("--target-atlas-capacity", type=int, default=-1)
    ap.add_argument("--target-seed", type=int, default=-1)
    ap.add_argument("--target-iteration", type=int, default=-1)
    ap.add_argument("--target-parent-domain", type=int, default=-1)
    ap.add_argument("--target-fiber-label", type=int, default=-1)
    ap.add_argument("--rule-modes", default="")
    ap.add_argument("--profiles", default="")
    ap.add_argument("--atlas-capacities", default="")
    ap.add_argument("--atlas-iterations", type=int, default=12)
    ap.add_argument("--require-generated", action="store_true")
    ap.add_argument("--max-candidates", type=int, default=20)
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
    ap.add_argument("--max-charts-per-fiber", type=int, default=32)
    ap.add_argument("--max-signature-charts", type=int, default=64)
    ap.add_argument("--min-fiber-states", type=int, default=6)
    ap.add_argument("--min-support-states", type=int, default=3)
    ap.add_argument("--min-overlap-states", type=int, default=3)
    ap.add_argument("--min-chart-classes", type=int, default=2)
    ap.add_argument("--min-chart-entropy", type=float, default=0.0)
    ap.add_argument("--max-chart-coords", type=int, default=6)
    ap.add_argument("--max-support-coords", type=int, default=5)
    ap.add_argument("--max-cycle-len", type=int, default=5)
    ap.add_argument("--max-loops-per-base", type=int, default=5000)
    ap.add_argument("--max-cycles-per-fiber", type=int, default=1000)
    ap.add_argument("--max-group-order", type=int, default=4096)
    ap.add_argument("--max-domains-scan", type=int, default=0)
    ap.add_argument("--max-fibers-per-domain-scan", type=int, default=0)
    ap.add_argument("--atlas-lift-mode", default="bijective")
    ap.add_argument("--include-trivial", action="store_true")
    ap.add_argument("--stop-at-first-nonabelian", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", default="")
    ns = ap.parse_args()
    if ns.synthetic_smoke:
        gdf, ldf, cdf, summary = _synthetic_smoke(ns.out, ns.plot)
    else:
        gdf, ldf, cdf, summary = run_temporal_capacity_chart_basis_audit(
            q=int(ns.q), vertices=int(ns.vertices), iterated_csv=str(ns.iterated_csv), frozen_transition_npy=str(ns.frozen_transition_npy),
            chart_basis=str(ns.chart_basis), temporal_window_sizes=_parse_csv_ints(ns.temporal_window_sizes, [2]),
            temporal_block_modes=_parse_csv_text(ns.temporal_block_modes, ["sliding"]),
            apply_basis_to_atlas_signature=bool(ns.apply_basis_to_atlas_signature),
            include_current_charts=not bool(ns.no_current_charts), include_predictive_mix=not bool(ns.no_predictive_mix),
            target_rule_mode=str(ns.target_rule_mode), target_instance=int(ns.target_instance), target_profile=str(ns.target_profile),
            target_atlas_capacity=int(ns.target_atlas_capacity), target_seed=int(ns.target_seed), target_iteration=int(ns.target_iteration),
            target_parent_domain=int(ns.target_parent_domain), target_fiber_label=int(ns.target_fiber_label),
            rule_modes=_parse_csv_text(ns.rule_modes, []), profiles=_parse_csv_text(ns.profiles, []), atlas_capacities=_parse_csv_ints(ns.atlas_capacities, []),
            atlas_iterations=int(ns.atlas_iterations), require_generated=bool(ns.require_generated), max_candidates=int(ns.max_candidates),
            max_state_samples=int(ns.max_state_samples), max_total_states=int(ns.max_total_states), max_pred=int(ns.max_pred),
            proliferation_iterations=int(ns.proliferation_iterations), horizon=int(ns.horizon), initial_boundary=str(ns.initial_boundary), initial_boundary_q=ns.initial_boundary_q,
            max_domains_per_depth=int(ns.max_domains_per_depth), min_live_classes=int(ns.min_live_classes), min_fiber_size=int(ns.min_fiber_size),
            min_entropy_bits=float(ns.min_entropy_bits), synergy_threshold=float(ns.synergy_threshold), max_signature_domains=int(ns.max_signature_domains),
            max_parent_domains=int(ns.max_parent_domains), max_fibers_per_parent=int(ns.max_fibers_per_parent), max_charts_per_fiber=int(ns.max_charts_per_fiber),
            max_signature_charts=int(ns.max_signature_charts), min_fiber_states=int(ns.min_fiber_states), min_support_states=int(ns.min_support_states),
            min_overlap_states=int(ns.min_overlap_states), min_chart_classes=int(ns.min_chart_classes), min_chart_entropy=float(ns.min_chart_entropy),
            max_chart_coords=int(ns.max_chart_coords), max_support_coords=int(ns.max_support_coords), max_temporal_blocks=int(ns.max_temporal_blocks),
            max_cycle_len=int(ns.max_cycle_len), max_loops_per_base=int(ns.max_loops_per_base), max_cycles_per_fiber=int(ns.max_cycles_per_fiber),
            max_group_order=int(ns.max_group_order), max_domains_scan=int(ns.max_domains_scan), max_fibers_per_domain_scan=int(ns.max_fibers_per_domain_scan),
            atlas_lift_mode=str(ns.atlas_lift_mode), include_trivial=bool(ns.include_trivial), stop_at_first_nonabelian=bool(ns.stop_at_first_nonabelian),
            out=str(ns.out), plot=str(ns.plot),
        )
    print(json.dumps(BPH._json_safe(summary), indent=2, sort_keys=True))
    if ns.out:
        print(f"wrote {ns.out}")
        base = os.path.splitext(ns.out)[0]
        print(f"wrote {base}_loops.csv")
        print(f"wrote {base}_charts.csv")
        print(f"wrote {base}_summary.json")
    if ns.plot:
        print(f"wrote {ns.plot}")


if __name__ == "__main__":
    main()
