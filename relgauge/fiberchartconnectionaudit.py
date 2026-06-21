"""
fiberchartconnectionaudit.py

Inside-out boundary-chart connection audit.

The common-factor closure audit C(P(A)) creates lateral loop candidates by
matching domains with similar/repeated predictive profiles.  The exact closure
holonomy audit then showed that those common-factor loops can carry exact
transports, but the exact loop maps are flat/order-1: common-factor labels are
already globalized, so round trips tend to be identity.

This module tests the next, more observer-relative construction:

    same hidden fiber + different boundary charts -> local chart transports

For each live fiber of a generated boundary domain, the audit builds several
local charts on the same unresolved interior.  A chart is a finite boundary
projection defined only on a support subset of the fiber, such as a coordinate
view, a one-step coordinate view, or a predictive-history view.  Transports are
learned only on overlaps between charts.  Cycles of overlapping charts are then
checked for exact automorphism/holonomy.

This is intentionally different from common-factor closure:

* common-factor closure connects domains that look the same;
* fiber-chart connection connects different boundary charts over shared hidden
  content and asks whether their overlap maps glue flatly or with a cocycle.

The audit is a strict filter, not a theorem prover.  It reconstructs the finite
arena from boundaryproliferationaudit metadata, samples bounded state closures,
and compares observed chart holonomy against shuffled-label nulls.

Example
-------
python -m relgauge.fiberchartconnectionaudit ^
  --proliferation-csv example_results/boundary_proliferation_q2.csv ^
  --proliferation-summary example_results/boundary_proliferation_q2_summary.json ^
  --null-shuffles 4 ^
  --out example_results/fiber_chart_connection_q2.csv ^
  --plot example_results/fig_fiber_chart_connection_q2.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import boundaryproliferationaudit as BPA
    from . import exactclosureholonomyaudit as ECH
except Exception:  # pragma: no cover
    import boundaryproliferationaudit as BPA  # type: ignore
    import exactclosureholonomyaudit as ECH  # type: ignore


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, float) and not np.isfinite(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        y = float(x)
        if not np.isfinite(y):
            return float(default)
        return y
    except Exception:
        return float(default)


def _read_csv(path: str):
    if pd is None:
        raise RuntimeError("pandas is required")
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _maybe(path: str, base_dir: str = "") -> str:
    if not path:
        return ""
    if os.path.exists(path):
        return path
    if base_dir:
        p2 = os.path.join(base_dir, os.path.basename(path))
        if os.path.exists(p2):
            return p2
    p3 = os.path.join(os.getcwd(), path)
    if os.path.exists(p3):
        return p3
    return path


def _stem(path: str) -> str:
    return os.path.splitext(str(path))[0]


def _load_json(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def companion_paths(proliferation_csv: str) -> Dict[str, str]:
    stem = _stem(proliferation_csv)
    return {
        "domains": stem + "_domains.csv",
        "dependencies": stem + "_dependencies.csv",
        "summary": stem + "_summary.json",
    }


def compact_map_string(mp: Optional[Dict[int, int]], max_items: int = 80) -> str:
    if not mp:
        return ""
    items = sorted((int(k), int(v)) for k, v in mp.items())
    if len(items) > int(max_items):
        items = items[: int(max_items)]
    return " ".join(f"{a}->{b}" for a, b in items)


def _is_identity_map(mp: Dict[int, int]) -> bool:
    return all(int(k) == int(v) for k, v in mp.items())


def _compose_maps(left: Dict[int, int], right: Dict[int, int]) -> Optional[Dict[int, int]]:
    """Return right ∘ left: x -> right[left[x]], or None if not closed."""
    out: Dict[int, int] = {}
    for k, v in left.items():
        if int(v) not in right:
            return None
        out[int(k)] = int(right[int(v)])
    return out


def permutation_order(mp: Dict[int, int], max_order: int = 256) -> int:
    keys = set(int(k) for k in mp.keys())
    vals = set(int(v) for v in mp.values())
    if keys != vals or len(keys) != len(mp):
        return 0
    seen = set()
    lcm = 1
    for x in sorted(keys):
        if x in seen:
            continue
        cur = int(x)
        cyc = 0
        while cur not in seen:
            seen.add(cur)
            cyc += 1
            cur = int(mp[cur])
            if cur not in keys or cyc > max_order * max(1, len(keys)):
                return 0
        if cyc > 0:
            lcm = math.lcm(lcm, cyc)
        if lcm > max_order:
            return int(lcm)
    return int(lcm)


def entropy(labels: Sequence[int]) -> float:
    return BPA.entropy_of_labels([int(x) for x in labels])


def relabel_on_support(raw: Sequence[object]) -> np.ndarray:
    return BPA.canonical_relabel(list(raw)).astype(np.int32)


# ---------------------------------------------------------------------------
# Chart and transport objects
# ---------------------------------------------------------------------------
@dataclass
class FiberChart:
    chart_id: int
    parent_domain_id: int
    fiber_label: int
    chart_type: str
    support_desc: str
    label_desc: str
    support_mask: np.ndarray  # bool mask over global state index order
    labels_full: np.ndarray   # -1 outside support, canonical labels inside
    n_support: int
    n_labels: int
    entropy_bits: float


@dataclass
class ChartTransport:
    source: int
    target: int
    overlap_size: int
    source_label_count: int
    target_label_count: int
    relation_pair_count: int
    forward_deterministic: bool
    backward_deterministic: bool
    exact_bijective_transport: bool
    forward_accuracy: float
    backward_accuracy: float
    forward_map: Optional[Dict[int, int]]
    backward_map: Optional[Dict[int, int]]
    failure_reason: str


# ---------------------------------------------------------------------------
# Chart construction
# ---------------------------------------------------------------------------
def _fiber_indices(labels: np.ndarray, lab: int) -> np.ndarray:
    return np.where(np.asarray(labels, dtype=np.int64) == int(lab))[0]


def _mk_chart(chart_id: int, parent_id: int, fiber_label: int, chart_type: str,
              support_desc: str, label_desc: str, support_idx: np.ndarray,
              raw_labels: Sequence[object], n_states: int,
              min_chart_classes: int, min_chart_entropy: float,
              min_support_states: int) -> Optional[FiberChart]:
    support_idx = np.asarray(support_idx, dtype=np.int64)
    if len(support_idx) < int(min_support_states):
        return None
    labs = relabel_on_support(raw_labels)
    n_labels = int(len(set(int(x) for x in labs)))
    ent = float(entropy(labs))
    if n_labels < int(min_chart_classes) or ent < float(min_chart_entropy):
        return None
    mask = np.zeros(int(n_states), dtype=bool)
    mask[support_idx] = True
    full = np.full(int(n_states), -1, dtype=np.int32)
    full[support_idx] = labs
    return FiberChart(
        chart_id=int(chart_id), parent_domain_id=int(parent_id), fiber_label=int(fiber_label),
        chart_type=str(chart_type), support_desc=str(support_desc), label_desc=str(label_desc),
        support_mask=mask, labels_full=full, n_support=int(len(support_idx)), n_labels=int(n_labels),
        entropy_bits=float(ent),
    )


def build_charts_for_domain_fiber(
    domain: BPA.BoundaryDomain,
    fiber_label: int,
    states: Sequence[Tuple[int, ...]],
    next_idx: Sequence[int],
    q: int,
    horizon: int,
    max_chart_coords: int = 5,
    max_support_coords: int = 4,
    max_charts_per_fiber: int = 32,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.05,
    min_support_states: int = 8,
) -> List[FiberChart]:
    """Construct an atlas of boundary charts over one hidden fiber.

    Full-fiber charts are global views of the same fiber.  Local charts add
    support conditions, creating atlas overlaps.  Nontrivial holonomy, if any,
    must come from these local overlap maps rather than from a single global
    common factor.
    """
    arr = np.asarray(states, dtype=np.int64)
    n_states = int(len(arr))
    if n_states == 0:
        return []
    k = int(arr.shape[1]) if arr.ndim == 2 else 0
    if k <= 0:
        return []
    idxs = _fiber_indices(domain.labels, int(fiber_label))
    if len(idxs) < int(min_support_states):
        return []
    next_arr = arr[np.asarray(next_idx, dtype=np.int64)]
    histories = BPA.future_label_history(domain.labels, next_idx, horizon=int(horizon), include_current=True)
    charts: List[FiberChart] = []

    def add(kind: str, sdesc: str, ldesc: str, support: np.ndarray, raw: Sequence[object]) -> None:
        nonlocal charts
        ch = _mk_chart(len(charts), int(domain.domain_id), int(fiber_label), kind, sdesc, ldesc, support, raw, n_states, min_chart_classes, min_chart_entropy, min_support_states)
        if ch is not None:
            charts.append(ch)

    # Global predictive chart: unresolved distinctions that change future parent-boundary histories.
    add("predictive_history", f"parent={domain.domain_id};fiber={fiber_label};all", "future(parent_label)", idxs, [histories[int(i)] for i in idxs])

    # Global coordinate and one-step coordinate charts.
    coord_lim = min(int(max_chart_coords), k)
    for c in range(coord_lim):
        add("coord", f"parent={domain.domain_id};fiber={fiber_label};all", f"x[{c}]", idxs, [int(arr[int(i), c]) % int(q) for i in idxs])
        add("next_coord", f"parent={domain.domain_id};fiber={fiber_label};all", f"T(x)[{c}]", idxs, [int(next_arr[int(i), c]) % int(q) for i in idxs])

    # Local support charts: finite boundary charts on overlapping sub-patches of the same fiber.
    # These create an atlas-like cover instead of a single global factor.
    support_lim = min(int(max_support_coords), k)
    for s in range(support_lim):
        vals = sorted(set(int(arr[int(i), s]) % int(q) for i in idxs))
        for val in vals:
            sub = np.asarray([int(i) for i in idxs if int(arr[int(i), s]) % int(q) == int(val)], dtype=np.int64)
            if len(sub) < int(min_support_states):
                continue
            # Local predictive view on this patch.
            add("local_predictive", f"x[{s}]={val}", "future(parent_label)", sub, [histories[int(i)] for i in sub])
            # Neighbor coordinate view; this is intentionally local and chart-like.
            p = (s + 1) % k
            add("local_coord", f"x[{s}]={val}", f"x[{p}]", sub, [int(arr[int(i), p]) % int(q) for i in sub])
            add("local_next_coord", f"x[{s}]={val}", f"T(x)[{p}]", sub, [int(next_arr[int(i), p]) % int(q) for i in sub])

    # Keep highest-entropy, broadest charts.  Avoid duplicate label/support pairs.
    dedup: Dict[Tuple[str, bytes, bytes], FiberChart] = {}
    for ch in charts:
        key = (ch.chart_type + ":" + ch.label_desc + ":" + ch.support_desc, np.packbits(ch.support_mask).tobytes(), ch.labels_full[ch.support_mask].tobytes())
        if key not in dedup or ch.entropy_bits > dedup[key].entropy_bits:
            dedup[key] = ch
    charts = list(dedup.values())
    charts.sort(key=lambda c: (c.entropy_bits, c.n_support, c.n_labels), reverse=True)
    charts = charts[: int(max_charts_per_fiber)]
    for i, ch in enumerate(charts):
        ch.chart_id = int(i)
    return charts


# ---------------------------------------------------------------------------
# Exact chart transports and chart-cycle holonomy
# ---------------------------------------------------------------------------
def chart_transport_between(a: FiberChart, b: FiberChart, min_overlap_states: int = 8) -> ChartTransport:
    overlap = np.where(a.support_mask & b.support_mask)[0]
    if len(overlap) < int(min_overlap_states):
        return ChartTransport(a.chart_id, b.chart_id, int(len(overlap)), 0, 0, 0, False, False, False, 0.0, 0.0, None, None, "insufficient_overlap")
    la = np.asarray(a.labels_full[overlap], dtype=np.int64)
    lb = np.asarray(b.labels_full[overlap], dtype=np.int64)
    src_vals = sorted(set(int(x) for x in la if int(x) >= 0))
    tgt_vals = sorted(set(int(x) for x in lb if int(x) >= 0))
    rel = Counter((int(x), int(y)) for x, y in zip(la, lb) if int(x) >= 0 and int(y) >= 0)
    by_src: Dict[int, Counter] = defaultdict(Counter)
    by_tgt: Dict[int, Counter] = defaultdict(Counter)
    for (x, y), c in rel.items():
        by_src[int(x)][int(y)] += int(c)
        by_tgt[int(y)][int(x)] += int(c)
    f_map: Dict[int, int] = {}
    b_map: Dict[int, int] = {}
    f_good = 0; b_good = 0
    f_det = True; b_det = True
    for x in src_vals:
        cnt = by_src.get(int(x), Counter())
        if len(cnt) != 1:
            f_det = False
        if cnt:
            y, c = cnt.most_common(1)[0]
            f_map[int(x)] = int(y)
            f_good += int(c)
    for y in tgt_vals:
        cnt = by_tgt.get(int(y), Counter())
        if len(cnt) != 1:
            b_det = False
        if cnt:
            x, c = cnt.most_common(1)[0]
            b_map[int(y)] = int(x)
            b_good += int(c)
    n = int(len(overlap))
    f_acc = float(f_good / max(1, n))
    b_acc = float(b_good / max(1, n))
    exact = bool(
        f_det and b_det and len(src_vals) >= 2 and len(tgt_vals) >= 2 and
        len(f_map) == len(src_vals) and len(b_map) == len(tgt_vals) and
        len(set(f_map.values())) == len(f_map)
    )
    if exact:
        reason = ""
    elif len(src_vals) < 2 or len(tgt_vals) < 2:
        reason = "trivial_overlap_labels"
    elif not f_det and not b_det:
        reason = "nondeterministic_both_directions"
    elif not f_det:
        reason = "forward_nondeterministic"
    elif not b_det:
        reason = "backward_nondeterministic"
    else:
        reason = "not_bijective"
    return ChartTransport(
        source=int(a.chart_id), target=int(b.chart_id), overlap_size=n,
        source_label_count=int(len(src_vals)), target_label_count=int(len(tgt_vals)), relation_pair_count=int(len(rel)),
        forward_deterministic=bool(f_det), backward_deterministic=bool(b_det), exact_bijective_transport=bool(exact),
        forward_accuracy=float(f_acc), backward_accuracy=float(b_acc),
        forward_map=f_map if f_det else None, backward_map=b_map if b_det else None, failure_reason=reason,
    )


def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (int(min(a, b)), int(max(a, b)))


def build_chart_transports(charts: Sequence[FiberChart], min_overlap_states: int = 8) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, int], ChartTransport]]:
    rows: List[Dict[str, object]] = []
    edge_maps: Dict[Tuple[int, int], ChartTransport] = {}
    for i, j in itertools.combinations(range(len(charts)), 2):
        et = chart_transport_between(charts[i], charts[j], min_overlap_states=min_overlap_states)
        row = {
            "source_chart": int(i), "target_chart": int(j),
            "source_type": charts[i].chart_type, "target_type": charts[j].chart_type,
            "source_support": charts[i].support_desc, "target_support": charts[j].support_desc,
            "source_label_desc": charts[i].label_desc, "target_label_desc": charts[j].label_desc,
            "overlap_size": int(et.overlap_size),
            "source_label_count": int(et.source_label_count), "target_label_count": int(et.target_label_count),
            "relation_pair_count": int(et.relation_pair_count),
            "forward_deterministic": bool(et.forward_deterministic),
            "backward_deterministic": bool(et.backward_deterministic),
            "exact_bijective_transport": bool(et.exact_bijective_transport),
            "forward_accuracy": float(et.forward_accuracy), "backward_accuracy": float(et.backward_accuracy),
            "failure_reason": et.failure_reason,
            "forward_map": compact_map_string(et.forward_map),
            "backward_map": compact_map_string(et.backward_map),
        }
        rows.append(row)
        if et.exact_bijective_transport:
            edge_maps[_edge_key(i, j)] = et
    return rows, edge_maps


def map_for_orientation(et: ChartTransport, a: int, b: int) -> Optional[Dict[int, int]]:
    if int(et.source) == int(a) and int(et.target) == int(b):
        return et.forward_map
    if int(et.source) == int(b) and int(et.target) == int(a):
        return et.backward_map
    return None


def cycle_map(cycle: Sequence[int], edge_maps: Dict[Tuple[int, int], ChartTransport]) -> Tuple[bool, Dict[int, int], str, int]:
    if len(cycle) < 3:
        return False, {}, "cycle_too_short", 0
    cyc = [int(x) for x in cycle] + [int(cycle[0])]
    current: Optional[Dict[int, int]] = None
    for a, b in zip(cyc[:-1], cyc[1:]):
        et = edge_maps.get(_edge_key(a, b))
        if et is None or not et.exact_bijective_transport:
            return False, {}, f"missing_or_invalid_edge_{a}_{b}", 0
        mp = map_for_orientation(et, a, b)
        if not mp:
            return False, {}, f"missing_map_{a}_{b}", 0
        if current is None:
            current = {int(k): int(v) for k, v in mp.items()}
        else:
            nxt = _compose_maps(current, mp)
            if nxt is None:
                return False, {}, f"composition_domain_mismatch_{a}_{b}", 0
            current = nxt
    if current is None:
        return False, {}, "empty_map", 0
    keys = set(current.keys()); vals = set(current.values())
    if keys != vals:
        return False, current, "loop_not_closed_on_start_labels", 0
    order = permutation_order(current, max_order=256)
    if order <= 0:
        return False, current, "not_a_permutation", 0
    return True, current, "", int(order)


def enumerate_chart_cycles(n_charts: int, edge_maps: Dict[Tuple[int, int], ChartTransport], max_cycle_len: int = 4, max_cycles: int = 2000) -> List[List[int]]:
    adj: Dict[int, set] = {i: set() for i in range(n_charts)}
    for a, b in edge_maps.keys():
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    cycles: List[List[int]] = []
    seen = set()
    # triangles and 4-cycles by combinations; enough for small atlas graphs and deterministic.
    for L in range(3, int(max_cycle_len) + 1):
        for nodes in itertools.combinations(range(n_charts), L):
            # Check all circular permutations with first fixed at min node.
            start = min(nodes)
            rest = [x for x in nodes if x != start]
            for perm in itertools.permutations(rest):
                cyc = [start] + list(perm)
                ok = all(_edge_key(cyc[i], cyc[(i + 1) % L]) in edge_maps for i in range(L))
                if not ok:
                    continue
                # canonicalize undirected cycle to avoid rotations/reversals.
                rev = [cyc[0]] + list(reversed(cyc[1:]))
                key = min(tuple(cyc), tuple(rev))
                if key in seen:
                    continue
                seen.add(key)
                cycles.append(list(key))
                if len(cycles) >= int(max_cycles):
                    return cycles
    return cycles


def analyze_chart_cycles(charts: Sequence[FiberChart], edge_maps: Dict[Tuple[int, int], ChartTransport], max_cycle_len: int = 4, max_cycles: int = 2000) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    cycles = enumerate_chart_cycles(len(charts), edge_maps, max_cycle_len=max_cycle_len, max_cycles=max_cycles)
    for cid, cyc in enumerate(cycles):
        valid, mp, reason, order = cycle_map(cyc, edge_maps)
        identity = bool(valid and _is_identity_map(mp))
        nontriv = bool(valid and order > 1 and not identity)
        rows.append({
            "cycle_id": int(cid),
            "cycle_charts": " ".join(str(x) for x in cyc),
            "cycle_length": int(len(cyc)),
            "chart_loop_valid": bool(valid),
            "chart_loop_order": int(order),
            "chart_loop_identity": bool(identity),
            "chart_nontrivial_holonomy": bool(nontriv),
            "chart_c2_holonomy": bool(nontriv and order == 2),
            "chart_c3_holonomy": bool(nontriv and order == 3),
            "chart_holonomy_family": ("C2" if nontriv and order == 2 else "C3" if nontriv and order == 3 else f"C{order}" if nontriv else "trivial" if valid else "invalid"),
            "failure_reason": str(reason),
            "loop_map": compact_map_string(mp),
        })
    return rows


def shuffle_charts(charts: Sequence[FiberChart], rng: np.random.Generator) -> List[FiberChart]:
    out: List[FiberChart] = []
    for ch in charts:
        full = ch.labels_full.copy()
        idx = np.where(ch.support_mask)[0]
        vals = full[idx].copy()
        if len(vals) > 1:
            vals = vals[rng.permutation(len(vals))]
        full[idx] = vals
        out.append(FiberChart(
            chart_id=ch.chart_id, parent_domain_id=ch.parent_domain_id, fiber_label=ch.fiber_label,
            chart_type=ch.chart_type, support_desc=ch.support_desc, label_desc=ch.label_desc,
            support_mask=ch.support_mask.copy(), labels_full=full, n_support=ch.n_support,
            n_labels=ch.n_labels, entropy_bits=ch.entropy_bits,
        ))
    return out


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def _infer_summary_path(proliferation_csv: str, provided: str = "") -> str:
    if provided:
        return provided
    if proliferation_csv:
        return _stem(proliferation_csv) + "_summary.json"
    return ""


def _group_rows_from_proliferation(proliferation_csv: str, params: ECH.ReplayParams, rule_modes: Sequence[str], instances: int, iterations: Sequence[int], seed: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if proliferation_csv and os.path.exists(proliferation_csv) and pd is not None:
        df = pd.read_csv(proliferation_csv)
        if "graph_id" not in df.columns and "instance" in df.columns:
            df["graph_id"] = df["instance"]
        if "seed" not in df.columns:
            df["seed"] = -1
        keep_cols = [c for c in ["q", "rule_mode", "instance", "graph_id", "seed", "iteration"] if c in df.columns]
        for _, r in df[keep_cols].drop_duplicates().iterrows():
            rows.append({c: r[c] for c in keep_cols})
        return rows
    # Fallback: synthesize the same seed schedule as boundaryproliferationaudit.
    modes = list(rule_modes) if rule_modes else ["affine_mix", "random_table"]
    its = list(iterations) if iterations else list(range(int(params.iterations) + 1))
    for mi, mode in enumerate(modes):
        for inst in range(int(instances)):
            sub_seed = int(seed) + 1000003 * (mi + 1) + 7919 * inst
            for it in its:
                rows.append({"q": params.q, "rule_mode": mode, "instance": inst, "graph_id": inst, "seed": sub_seed, "iteration": it})
    return rows


def run_fiber_chart_connection_audit(
    proliferation_csv: str = "",
    proliferation_summary: str = "",
    q: int = 0,
    vertices: int = 0,
    iterations: Optional[int] = None,
    horizon: Optional[int] = None,
    max_state_samples: Optional[int] = None,
    max_total_states: Optional[int] = None,
    initial_boundary: str = "",
    initial_boundary_q: Optional[int] = None,
    max_domains_per_depth: int = 32,
    min_live_classes: int = 2,
    min_fiber_size: int = 2,
    min_entropy_bits: float = 0.05,
    max_pred: int = 3,
    rule_modes: Sequence[str] = (),
    instances: int = 0,
    seed: int = 0,
    max_groups: int = 0,
    min_fiber_states: int = 16,
    min_support_states: int = 8,
    min_overlap_states: int = 8,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.05,
    max_parent_domains: int = 12,
    max_fibers_per_parent: int = 12,
    max_charts_per_fiber: int = 32,
    max_chart_coords: int = 5,
    max_support_coords: int = 4,
    max_cycle_len: int = 4,
    max_cycles_per_fiber: int = 2000,
    null_shuffles: int = 4,
    only_live_iterations: bool = True,
) -> Tuple[object, object, object, object, Dict[str, object]]:
    if pd is None:
        raise RuntimeError("pandas is required")
    base_dir = os.path.dirname(os.path.abspath(proliferation_csv)) if proliferation_csv else "."
    proliferation_summary = _maybe(_infer_summary_path(proliferation_csv, proliferation_summary), base_dir)
    prolif_summary = _load_json(proliferation_summary)

    class _Args:
        pass
    a = _Args()
    a.q = q
    a.vertices = vertices
    a.iterations = iterations
    a.horizon = horizon
    a.max_state_samples = max_state_samples
    a.max_total_states = max_total_states
    a.initial_boundary = initial_boundary
    a.initial_boundary_q = initial_boundary_q
    a.max_domains_per_depth = max_domains_per_depth
    a.min_live_classes = min_live_classes
    a.min_fiber_size = min_fiber_size
    a.min_entropy_bits = min_entropy_bits
    a.max_pred = max_pred
    params = ECH.params_from_summary(prolif_summary, a)

    # Iterations selected from CSV rows or generated fallback.
    groups = _group_rows_from_proliferation(proliferation_csv, params, rule_modes, instances or _safe_int(prolif_summary.get("instances", 1), 1), list(range(params.iterations + 1)), seed)
    if only_live_iterations and proliferation_csv and os.path.exists(proliferation_csv):
        try:
            dfp = pd.read_csv(proliferation_csv)
            if "live_boundary_fraction" in dfp.columns:
                live_keys = set()
                for _, r in dfp[dfp["live_boundary_fraction"].astype(float) > 0].iterrows():
                    live_keys.add((str(r.get("rule_mode", "")), str(r.get("instance", "")), str(r.get("iteration", ""))))
                groups = [g for g in groups if (str(g.get("rule_mode", "")), str(g.get("instance", "")), str(g.get("iteration", ""))) in live_keys]
        except Exception:
            pass

    rows: List[Dict[str, object]] = []
    chart_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    cycle_rows: List[Dict[str, object]] = []
    cache: Dict[Tuple, Tuple[Dict[int, BPA.BoundaryDomain], Sequence[Tuple[int, ...]], Sequence[int]]] = {}
    rng_null = np.random.default_rng(int(seed) + 424242)
    processed = 0

    for g in groups:
        if int(max_groups) > 0 and processed >= int(max_groups):
            break
        rule_mode = str(g.get("rule_mode", "unknown"))
        inst = _safe_int(g.get("instance", -1), -1)
        graph_id = _safe_int(g.get("graph_id", inst), inst)
        it = _safe_int(g.get("iteration", 0), 0)
        seed_val = _safe_int(g.get("seed", -1), -1)
        if seed_val < 0:
            # Same fallback as proliferation audit when CSV lacks seed.
            mode_list = list(rule_modes) if rule_modes else [rule_mode]
            mi = mode_list.index(rule_mode) if rule_mode in mode_list else 0
            seed_val = int(seed) + 1000003 * (mi + 1) + 7919 * max(0, inst)
        cache_key = (rule_mode, int(seed_val), int(it), int(params.q), int(params.vertices))
        if cache_key not in cache:
            cache[cache_key] = ECH.replay_domains_for_iteration(rule_mode, int(seed_val), int(it), params)
        domains, states, next_idx = cache[cache_key]
        if not domains:
            continue
        processed += 1
        # High-entropy parent domains first to keep runtime bounded.
        parent_list = list(domains.values())
        parent_list.sort(key=lambda d: (d.entropy_bits, d.live_fiber_count, d.n_labels), reverse=True)
        parent_list = parent_list[: int(max_parent_domains)]

        for domain in parent_list:
            labs = np.asarray(domain.labels, dtype=np.int64)
            counts = Counter(int(x) for x in labs)
            fiber_values = [lab for lab, cnt in counts.most_common() if cnt >= int(min_fiber_states)]
            fiber_values = fiber_values[: int(max_fibers_per_parent)]
            for flab in fiber_values:
                charts = build_charts_for_domain_fiber(
                    domain=domain, fiber_label=int(flab), states=states, next_idx=next_idx, q=params.q, horizon=params.horizon,
                    max_chart_coords=max_chart_coords, max_support_coords=max_support_coords,
                    max_charts_per_fiber=max_charts_per_fiber, min_chart_classes=min_chart_classes,
                    min_chart_entropy=min_chart_entropy, min_support_states=min_support_states,
                )
                if len(charts) < 2:
                    continue
                erows, edge_maps = build_chart_transports(charts, min_overlap_states=min_overlap_states)
                crows = analyze_chart_cycles(charts, edge_maps, max_cycle_len=max_cycle_len, max_cycles=max_cycles_per_fiber)
                n_edges = len(erows)
                n_exact_edges = int(sum(1 for r in erows if bool(r.get("exact_bijective_transport", False))))
                n_cycles = len(crows)
                n_valid_cycles = int(sum(1 for r in crows if bool(r.get("chart_loop_valid", False))))
                n_nontriv = int(sum(1 for r in crows if bool(r.get("chart_nontrivial_holonomy", False))))
                n_c2 = int(sum(1 for r in crows if bool(r.get("chart_c2_holonomy", False))))
                n_c3 = int(sum(1 for r in crows if bool(r.get("chart_c3_holonomy", False))))
                max_order = int(max([_safe_int(r.get("chart_loop_order", 0), 0) for r in crows] or [0]))

                # Null: shuffle labels on each chart support, preserving supports and marginal label counts.
                null_nontriv: List[int] = []
                null_valid: List[int] = []
                for _ in range(int(null_shuffles)):
                    sh = shuffle_charts(charts, rng_null)
                    _erows_n, emaps_n = build_chart_transports(sh, min_overlap_states=min_overlap_states)
                    crows_n = analyze_chart_cycles(sh, emaps_n, max_cycle_len=max_cycle_len, max_cycles=max_cycles_per_fiber)
                    null_nontriv.append(int(sum(1 for r in crows_n if bool(r.get("chart_nontrivial_holonomy", False)))))
                    null_valid.append(int(sum(1 for r in crows_n if bool(r.get("chart_loop_valid", False)))))
                null_nontriv_mean = float(np.mean(null_nontriv) if null_nontriv else 0.0)
                null_valid_mean = float(np.mean(null_valid) if null_valid else 0.0)
                nontriv_above_null = float(n_nontriv - null_nontriv_mean)
                chart_signal = bool(n_nontriv > 0 and n_nontriv > null_nontriv_mean)
                global_flat = bool(n_valid_cycles > 0 and n_nontriv == 0)
                row_base = {
                    "q": int(params.q), "vertices": int(params.vertices), "rule_mode": rule_mode,
                    "instance": int(inst), "graph_id": int(graph_id), "seed": int(seed_val), "iteration": int(it),
                    "parent_domain_id": int(domain.domain_id), "parent_depth": int(domain.depth),
                    "parent_name": str(domain.name), "fiber_label": int(flab), "fiber_size": int(counts[int(flab)]),
                }
                rows.append({
                    **row_base,
                    "n_charts": int(len(charts)),
                    "n_chart_edges_tested": int(n_edges),
                    "exact_chart_edge_count": int(n_exact_edges),
                    "exact_chart_edge_fraction": float(n_exact_edges / max(1, n_edges)),
                    "n_chart_cycles_tested": int(n_cycles),
                    "exact_chart_loop_valid_count": int(n_valid_cycles),
                    "exact_chart_loop_valid_fraction": float(n_valid_cycles / max(1, n_cycles)),
                    "exact_chart_nontrivial_holonomy_count": int(n_nontriv),
                    "exact_chart_c2_holonomy_count": int(n_c2),
                    "exact_chart_c3_holonomy_count": int(n_c3),
                    "max_exact_chart_holonomy_order": int(max_order),
                    "null_chart_nontrivial_holonomy_mean": float(null_nontriv_mean),
                    "exact_chart_nontrivial_holonomy_above_null": float(nontriv_above_null),
                    "null_chart_loop_valid_mean": float(null_valid_mean),
                    "chart_connection_proof_available": bool(n_cycles > 0 and n_valid_cycles > 0),
                    "chart_nontrivial_holonomy_proven": bool(chart_signal),
                    "global_flat_gaugeable": bool(global_flat),
                })
                for ch in charts:
                    chart_rows.append({
                        **row_base,
                        "chart_id": int(ch.chart_id), "chart_type": ch.chart_type,
                        "support_desc": ch.support_desc, "label_desc": ch.label_desc,
                        "n_support": int(ch.n_support), "n_labels": int(ch.n_labels),
                        "entropy_bits": float(ch.entropy_bits),
                    })
                for er in erows:
                    edge_rows.append({**row_base, **er})
                for cr in crows:
                    cycle_rows.append({**row_base, **cr})

    df = pd.DataFrame(rows)
    chart_df = pd.DataFrame(chart_rows)
    edge_df = pd.DataFrame(edge_rows)
    cycle_df = pd.DataFrame(cycle_rows)

    any_proof = bool(not df.empty and (df["chart_connection_proof_available"] == True).any())
    any_nontriv = bool(not df.empty and (df["chart_nontrivial_holonomy_proven"] == True).any())
    any_c2 = bool(not df.empty and (df["exact_chart_c2_holonomy_count"] > 0).any())
    any_c3 = bool(not df.empty and (df["exact_chart_c3_holonomy_count"] > 0).any())
    if any_nontriv:
        verdict = "FIBER-CHART HOLONOMY SIGNAL: same-fiber boundary charts carry nontrivial exact cocycles"
    elif any_proof:
        verdict = "FIBER-CHART CONNECTION FLAT/WEAK SIGNAL: chart transports exist but nontrivial holonomy not established"
    elif not df.empty:
        verdict = "FIBER-CHART CONNECTION WEAK SIGNAL: chart atlases generated but exact loop transport is scarce"
    else:
        verdict = "FIBER-CHART CONNECTION NEGATIVE SIGNAL: no usable same-fiber chart atlas found"
    summary: Dict[str, object] = {
        "verdict": verdict,
        "audit_version": "fiber_chart_connection_v1_same_fiber_local_boundary_charts",
        "source_proliferation_csv": str(proliferation_csv),
        "source_proliferation_summary": str(proliferation_summary),
        "q": int(params.q),
        "vertices": int(params.vertices),
        "horizon": int(params.horizon),
        "max_state_samples": int(params.max_state_samples),
        "n_rows": int(len(df)),
        "n_chart_rows": int(len(chart_df)),
        "n_chart_edge_rows": int(len(edge_df)),
        "n_chart_cycle_rows": int(len(cycle_df)),
        "any_chart_connection_proof": bool(any_proof),
        "any_exact_chart_nontrivial_holonomy": bool(any_nontriv),
        "any_exact_chart_c2_holonomy": bool(any_c2),
        "any_exact_chart_c3_holonomy": bool(any_c3),
        "max_exact_chart_edge_fraction": float(df["exact_chart_edge_fraction"].max()) if not df.empty else 0.0,
        "max_exact_chart_loop_valid_fraction": float(df["exact_chart_loop_valid_fraction"].max()) if not df.empty else 0.0,
        "max_exact_chart_nontrivial_holonomy_count": int(df["exact_chart_nontrivial_holonomy_count"].max()) if not df.empty else 0,
        "max_exact_chart_holonomy_order": int(df["max_exact_chart_holonomy_order"].max()) if not df.empty else 0,
        "mean_exact_chart_nontrivial_holonomy_above_null": float(df["exact_chart_nontrivial_holonomy_above_null"].mean()) if not df.empty else 0.0,
        "global_flat_gaugeable_fraction": float((df["global_flat_gaugeable"] == True).mean()) if not df.empty else 0.0,
    }
    if not df.empty:
        by_mode = []
        for mode, gg in df.groupby("rule_mode"):
            by_mode.append({
                "rule_mode": str(mode),
                "n": int(len(gg)),
                "mean_exact_chart_edge_fraction": float(gg["exact_chart_edge_fraction"].mean()),
                "mean_exact_chart_loop_valid_fraction": float(gg["exact_chart_loop_valid_fraction"].mean()),
                "max_exact_chart_nontrivial_holonomy_count": int(gg["exact_chart_nontrivial_holonomy_count"].max()),
                "max_exact_chart_c2_holonomy_count": int(gg["exact_chart_c2_holonomy_count"].max()),
                "max_exact_chart_c3_holonomy_count": int(gg["exact_chart_c3_holonomy_count"].max()),
                "proof_fraction": float((gg["chart_connection_proof_available"] == True).mean()),
                "nontrivial_proven_fraction": float((gg["chart_nontrivial_holonomy_proven"] == True).mean()),
                "global_flat_gaugeable_fraction": float((gg["global_flat_gaugeable"] == True).mean()),
            })
        summary["by_mode"] = by_mode
    return df, chart_df, edge_df, cycle_df, summary


# ---------------------------------------------------------------------------
# Output / CLI
# ---------------------------------------------------------------------------
def maybe_write_csv(df, path: str) -> None:
    if pd is None:
        raise RuntimeError("pandas is required")
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def plot_summary(df, path: str) -> None:
    if not path or df is None or len(df) == 0:
        return
    import matplotlib.pyplot as plt
    g = df.groupby("rule_mode").agg({
        "exact_chart_edge_fraction": "mean",
        "exact_chart_loop_valid_fraction": "mean",
        "exact_chart_nontrivial_holonomy_count": "max",
        "exact_chart_c2_holonomy_count": "max",
        "exact_chart_c3_holonomy_count": "max",
        "global_flat_gaugeable": "mean",
    }).reset_index()
    x = np.arange(len(g))
    w = 0.17
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x - 1.5*w, g["exact_chart_edge_fraction"], width=w, label="chart edge bijective frac")
    ax1.bar(x - 0.5*w, g["exact_chart_loop_valid_fraction"], width=w, label="chart loop valid frac")
    ax1.bar(x + 0.5*w, g["global_flat_gaugeable"], width=w, label="flat gaugeable frac")
    ax1.set_ylabel("fraction")
    ax1.set_xticks(x)
    ax1.set_xticklabels(g["rule_mode"], rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, g["exact_chart_nontrivial_holonomy_count"], marker="o", label="nontrivial count")
    ax2.plot(x, g["exact_chart_c2_holonomy_count"], marker="s", label="C2 count")
    ax2.plot(x, g["exact_chart_c3_holonomy_count"], marker="^", label="C3 count")
    ax2.set_ylabel("cycle count")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("Fiber-chart connection: same hidden fiber -> local boundary-chart transports")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inside-out fiber-chart connection/holonomy audit")
    p.add_argument("--proliferation-csv", default="", help="Output CSV from boundaryproliferationaudit.py")
    p.add_argument("--proliferation-summary", default="", help="Boundary proliferation summary JSON used to reconstruct the arena")
    p.add_argument("--q", type=int, default=0)
    p.add_argument("--vertices", type=int, default=0)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--max-state-samples", type=int, default=None)
    p.add_argument("--max-total-states", type=int, default=None)
    p.add_argument("--initial-boundary", default="")
    p.add_argument("--initial-boundary-q", type=int, default=None)
    p.add_argument("--max-domains-per-depth", type=int, default=32)
    p.add_argument("--min-live-classes", type=int, default=2)
    p.add_argument("--min-fiber-size", type=int, default=2)
    p.add_argument("--min-entropy-bits", type=float, default=0.05)
    p.add_argument("--max-pred", type=int, default=3)
    p.add_argument("--rule-modes", default="", help="Fallback comma-separated rule modes when no proliferation CSV is supplied")
    p.add_argument("--instances", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-groups", type=int, default=0)
    p.add_argument("--min-fiber-states", type=int, default=16)
    p.add_argument("--min-support-states", type=int, default=8)
    p.add_argument("--min-overlap-states", type=int, default=8)
    p.add_argument("--min-chart-classes", type=int, default=2)
    p.add_argument("--min-chart-entropy", type=float, default=0.05)
    p.add_argument("--max-parent-domains", type=int, default=12)
    p.add_argument("--max-fibers-per-parent", type=int, default=12)
    p.add_argument("--max-charts-per-fiber", type=int, default=32)
    p.add_argument("--max-chart-coords", type=int, default=5)
    p.add_argument("--max-support-coords", type=int, default=4)
    p.add_argument("--max-cycle-len", type=int, default=4)
    p.add_argument("--max-cycles-per-fiber", type=int, default=2000)
    p.add_argument("--null-shuffles", type=int, default=4)
    p.add_argument("--include-inert-iterations", action="store_true", help="Do not filter to iterations with live_boundary_fraction > 0")
    p.add_argument("--out", default="example_results/fiber_chart_connection.csv")
    p.add_argument("--plot", default="")
    return p


def _parse_modes(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    df, chart_df, edge_df, cycle_df, summary = run_fiber_chart_connection_audit(
        proliferation_csv=args.proliferation_csv,
        proliferation_summary=args.proliferation_summary,
        q=args.q,
        vertices=args.vertices,
        iterations=args.iterations,
        horizon=args.horizon,
        max_state_samples=args.max_state_samples,
        max_total_states=args.max_total_states,
        initial_boundary=args.initial_boundary,
        initial_boundary_q=args.initial_boundary_q,
        max_domains_per_depth=args.max_domains_per_depth,
        min_live_classes=args.min_live_classes,
        min_fiber_size=args.min_fiber_size,
        min_entropy_bits=args.min_entropy_bits,
        max_pred=args.max_pred,
        rule_modes=_parse_modes(args.rule_modes),
        instances=args.instances,
        seed=args.seed,
        max_groups=args.max_groups,
        min_fiber_states=args.min_fiber_states,
        min_support_states=args.min_support_states,
        min_overlap_states=args.min_overlap_states,
        min_chart_classes=args.min_chart_classes,
        min_chart_entropy=args.min_chart_entropy,
        max_parent_domains=args.max_parent_domains,
        max_fibers_per_parent=args.max_fibers_per_parent,
        max_charts_per_fiber=args.max_charts_per_fiber,
        max_chart_coords=args.max_chart_coords,
        max_support_coords=args.max_support_coords,
        max_cycle_len=args.max_cycle_len,
        max_cycles_per_fiber=args.max_cycles_per_fiber,
        null_shuffles=args.null_shuffles,
        only_live_iterations=not bool(args.include_inert_iterations),
    )
    out = args.out
    maybe_write_csv(df, out)
    stem = _stem(out)
    maybe_write_csv(chart_df, stem + "_charts.csv")
    maybe_write_csv(edge_df, stem + "_chart_edges.csv")
    maybe_write_csv(cycle_df, stem + "_chart_cycles.csv")
    with open(stem + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.plot:
        plot_summary(df, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(f"wrote {stem}_charts.csv")
    print(f"wrote {stem}_chart_edges.csv")
    print(f"wrote {stem}_chart_cycles.csv")
    print(f"wrote {stem}_summary.json")


if __name__ == "__main__":  # pragma: no cover
    main()
