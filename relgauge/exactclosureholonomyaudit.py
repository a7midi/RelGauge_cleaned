"""
exactclosureholonomyaudit.py

Exact transport/holonomy audit for proliferation-closure loop candidates.

The proliferation audit P creates live-fiber observer domains.  The closure audit
C adds lateral common-factor edges between generated domains with repeated
predictive-factor metadata, producing loop candidates.  This module is the next
screening stage:

    candidate closure loop -> reconstructed exact domain labels -> exact loop map

It replays the boundaryproliferationaudit arena from the saved seeds/parameters,
reconstructs the domain label arrays at each proliferation iteration, and then
checks whether the closure graph cycles carry exact state-supported transports.

A closure edge A--B has an exact raw transport only when the relation

    label_A(state) -> label_B(state)

is single-valued and bijective on the observed active labels.  A closure cycle
has exact holonomy only when all edge transports along the cycle compose into a
bijective automorphism of the starting domain's labels.  Non-identity cycle maps
are reported as exact C2/C3/etc. holonomy.

This module intentionally distinguishes three tiers:

* closure candidates: from proliferationclosureaudit metadata,
* exact edge transports: state-supported bijective label maps,
* exact loop holonomy: nontrivial automorphism after composing a full cycle.

Example
-------
python -m relgauge.exactclosureholonomyaudit ^
  --closure-csv example_results/proliferation_closure_q2.csv ^
  --closure-edges-csv example_results/proliferation_closure_q2_closure_edges.csv ^
  --closure-cycles-csv example_results/proliferation_closure_q2_closure_cycles.csv ^
  --proliferation-summary example_results/boundary_proliferation_q2_summary.json ^
  --null-shuffles 4 ^
  --out example_results/exact_closure_holonomy_q2.csv ^
  --plot example_results/fig_exact_closure_holonomy_q2.png
"""
from __future__ import annotations

import argparse
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
except Exception:  # pragma: no cover
    import boundaryproliferationaudit as BPA  # type: ignore


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


def companion_paths(closure_csv: str) -> Dict[str, str]:
    stem = _stem(closure_csv)
    return {
        "edges": stem + "_closure_edges.csv",
        "cycles": stem + "_closure_cycles.csv",
        "summary": stem + "_summary.json",
    }


def _load_json(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _key_cols(df) -> List[str]:
    cols = []
    for c in ["q", "rule_mode", "instance", "graph_id", "seed", "iteration"]:
        if c in df.columns:
            cols.append(c)
    return cols


def _meta_key(row_or_dict, cols: Sequence[str]) -> Tuple:
    return tuple(row_or_dict[c] for c in cols)


def parse_cycle_nodes(text: object) -> List[int]:
    if text is None:
        return []
    if isinstance(text, float) and not np.isfinite(text):
        return []
    s = str(text).replace(",", " ").replace(";", " ")
    out = []
    for tok in s.split():
        try:
            out.append(int(tok))
        except Exception:
            pass
    return out


def ordered_edge_key(a: int, b: int) -> Tuple[int, int]:
    return (int(min(a, b)), int(max(a, b)))


def _is_identity_map(mp: Dict[int, int]) -> bool:
    return all(int(k) == int(v) for k, v in mp.items())


def permutation_order(mp: Dict[int, int], max_order: int = 64) -> int:
    """Return order of a bijective self-map, or 0 when invalid/not closed."""
    keys = set(int(k) for k in mp.keys())
    vals = set(int(v) for v in mp.values())
    if keys != vals or len(keys) != len(mp):
        return 0
    seen = set()
    lcm = 1
    for x in sorted(keys):
        if x in seen:
            continue
        cur = x
        cyc = 0
        while cur not in seen:
            seen.add(cur)
            cyc += 1
            cur = int(mp[cur])
            if cyc > max_order * max(1, len(keys)):
                return 0
            if cur not in keys:
                return 0
        if cyc > 0:
            lcm = math.lcm(lcm, cyc)
        if lcm > max_order:
            # Still exact, but cap to keep classification finite.
            return int(lcm)
    return int(lcm)


def compact_map_string(mp: Optional[Dict[int, int]], max_items: int = 64) -> str:
    if not mp:
        return ""
    items = sorted((int(k), int(v)) for k, v in mp.items())
    if len(items) > max_items:
        items = items[:max_items]
    return " ".join(f"{a}->{b}" for a, b in items)


# ---------------------------------------------------------------------------
# Reconstruction of exact domain labels
# ---------------------------------------------------------------------------
@dataclass
class ReplayParams:
    q: int
    vertices: int
    iterations: int
    horizon: int
    max_state_samples: int
    max_total_states: int
    initial_boundary: str
    initial_boundary_q: Optional[int]
    max_domains_per_depth: int
    min_live_classes: int
    min_fiber_size: int
    min_entropy_bits: float
    max_pred: int


def params_from_summary(summary: Dict[str, object], args: argparse.Namespace) -> ReplayParams:
    q = int(args.q or _safe_int(summary.get("q", 2), 2))
    vertices = int(args.vertices or _safe_int(summary.get("vertices", 7), 7))
    iterations = int(args.iterations if args.iterations is not None else _safe_int(summary.get("iterations", 5), 5))
    horizon = int(args.horizon if args.horizon is not None else _safe_int(summary.get("horizon", 3), 3))
    max_state_samples = int(args.max_state_samples if args.max_state_samples is not None else _safe_int(summary.get("max_state_samples", 512), 512))
    max_total_states = int(args.max_total_states if args.max_total_states is not None else _safe_int(summary.get("max_total_states", 4096), 4096))
    initial_boundary = str(args.initial_boundary or summary.get("initial_boundary", "sum_mod_q"))
    ibq_raw = args.initial_boundary_q if args.initial_boundary_q is not None else summary.get("initial_boundary_q", None)
    initial_boundary_q = None if ibq_raw in [None, "", 0, "0"] else int(ibq_raw)
    return ReplayParams(
        q=q,
        vertices=vertices,
        iterations=iterations,
        horizon=horizon,
        max_state_samples=max_state_samples,
        max_total_states=max_total_states,
        initial_boundary=initial_boundary,
        initial_boundary_q=initial_boundary_q,
        max_domains_per_depth=int(args.max_domains_per_depth),
        min_live_classes=int(args.min_live_classes),
        min_fiber_size=int(args.min_fiber_size),
        min_entropy_bits=float(args.min_entropy_bits),
        max_pred=int(args.max_pred),
    )


def replay_domains_for_iteration(rule_mode: str, seed: int, iteration: int, params: ReplayParams):
    """Replay boundary proliferation and return exact current-domain labels."""
    rng = np.random.default_rng(int(seed))
    dyn = BPA.make_dynamics(params.q, params.vertices, str(rule_mode), rng, max_pred=int(params.max_pred))
    closure_steps = max(1, int(params.iterations) * (int(params.horizon) + 1) + 2)
    initial_states = BPA.sample_initial_states(params.q, params.vertices, int(params.max_state_samples), rng)
    states, _state_to_idx, next_idx = BPA.build_state_closure(dyn, initial_states, closure_steps, int(params.max_total_states))
    labels0 = BPA.initial_boundary_labels(states, params.q, params.initial_boundary, rng, boundary_q=params.initial_boundary_q)
    root = BPA.BoundaryDomain(0, 0, -1, "root", labels0, int(len(set(labels0))), BPA.entropy_of_labels(labels0), active=True)
    domains: List[BPA.BoundaryDomain] = [root]
    current: List[BPA.BoundaryDomain] = [root]
    if int(iteration) <= 0:
        return {int(d.domain_id): d for d in current}, states, next_idx
    for _it in range(int(iteration)):
        children: List[BPA.BoundaryDomain] = []
        for parent in current:
            for child in BPA.proliferate_children(parent, next_idx, horizon=int(params.horizon), min_live_classes=int(params.min_live_classes), min_fiber_size=int(params.min_fiber_size)):
                if child.active and float(child.entropy_bits) >= float(params.min_entropy_bits):
                    child.domain_id = len(domains) + len(children)
                    children.append(child)
        children.sort(key=lambda d: (d.entropy_bits, d.live_fiber_count, d.n_labels), reverse=True)
        children = children[: int(params.max_domains_per_depth)]
        for c in children:
            c.domain_id = len(domains)
            domains.append(c)
        current = children
        if not current:
            break
    return {int(d.domain_id): d for d in current}, states, next_idx


# ---------------------------------------------------------------------------
# Exact edge transport and cycle holonomy
# ---------------------------------------------------------------------------
@dataclass
class EdgeTransport:
    source: int
    target: int
    n_states: int
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


def exact_transport_between(labels_a: Sequence[int], labels_b: Sequence[int], source: int, target: int) -> EdgeTransport:
    aa = np.asarray(labels_a, dtype=np.int64)
    bb = np.asarray(labels_b, dtype=np.int64)
    n = int(min(len(aa), len(bb)))
    aa = aa[:n]; bb = bb[:n]
    src_vals = sorted(set(int(x) for x in aa))
    tgt_vals = sorted(set(int(x) for x in bb))
    rel = Counter((int(a), int(b)) for a, b in zip(aa, bb))
    by_src: Dict[int, Counter] = defaultdict(Counter)
    by_tgt: Dict[int, Counter] = defaultdict(Counter)
    for (a, b), c in rel.items():
        by_src[int(a)][int(b)] += int(c)
        by_tgt[int(b)][int(a)] += int(c)
    forward_map: Dict[int, int] = {}
    backward_map: Dict[int, int] = {}
    f_good = 0; b_good = 0
    f_det = True; b_det = True
    for a in src_vals:
        cnt = by_src.get(int(a), Counter())
        if len(cnt) != 1:
            f_det = False
        if cnt:
            b, c = cnt.most_common(1)[0]
            forward_map[int(a)] = int(b)
            f_good += int(c)
    for b in tgt_vals:
        cnt = by_tgt.get(int(b), Counter())
        if len(cnt) != 1:
            b_det = False
        if cnt:
            a, c = cnt.most_common(1)[0]
            backward_map[int(b)] = int(a)
            b_good += int(c)
    f_acc = float(f_good / max(1, n))
    b_acc = float(b_good / max(1, n))
    exact_bij = bool(f_det and b_det and len(forward_map) == len(src_vals) and len(backward_map) == len(tgt_vals) and len(set(forward_map.values())) == len(forward_map))
    if exact_bij:
        reason = ""
    elif not f_det and not b_det:
        reason = "nondeterministic_both_directions"
    elif not f_det:
        reason = "forward_nondeterministic"
    elif not b_det:
        reason = "backward_nondeterministic"
    else:
        reason = "not_bijective"
    return EdgeTransport(
        source=int(source), target=int(target), n_states=n,
        source_label_count=int(len(src_vals)), target_label_count=int(len(tgt_vals)),
        relation_pair_count=int(len(rel)), forward_deterministic=bool(f_det), backward_deterministic=bool(b_det),
        exact_bijective_transport=bool(exact_bij), forward_accuracy=float(f_acc), backward_accuracy=float(b_acc),
        forward_map=forward_map if f_det else None, backward_map=backward_map if b_det else None,
        failure_reason=reason,
    )


def _compose_maps(left: Dict[int, int], right: Dict[int, int]) -> Optional[Dict[int, int]]:
    """Return right ∘ left: x -> right[left[x]]."""
    out: Dict[int, int] = {}
    for k, v in left.items():
        if int(v) not in right:
            return None
        out[int(k)] = int(right[int(v)])
    return out


def cycle_loop_map(nodes: Sequence[int], edge_map_by_pair: Dict[Tuple[int, int], EdgeTransport]) -> Tuple[bool, Dict[int, int], str, int]:
    if len(nodes) < 2:
        return False, {}, "cycle_too_short", 0
    cyc = [int(x) for x in nodes] + [int(nodes[0])]
    # Start labels are the domain labels of cyc[0].  Initialize lazily from the first edge map.
    current: Optional[Dict[int, int]] = None
    for a, b in zip(cyc[:-1], cyc[1:]):
        key = ordered_edge_key(a, b)
        et = edge_map_by_pair.get(key)
        if et is None:
            return False, {}, f"missing_edge_{a}_{b}", 0
        if not et.exact_bijective_transport:
            return False, {}, f"edge_not_bijective_{a}_{b}:{et.failure_reason}", 0
        if int(et.source) == int(a) and int(et.target) == int(b):
            step_map = et.forward_map or {}
        elif int(et.source) == int(b) and int(et.target) == int(a):
            step_map = et.backward_map or {}
        else:
            # Should not happen because key matched, but keep safe.
            return False, {}, f"edge_orientation_error_{a}_{b}", 0
        if current is None:
            current = {int(k): int(v) for k, v in step_map.items()}
        else:
            nxt = _compose_maps(current, step_map)
            if nxt is None:
                return False, {}, f"composition_domain_mismatch_{a}_{b}", 0
            current = nxt
    if current is None:
        return False, {}, "empty_map", 0
    # The current map should now be an automorphism of the starting labels.
    keys = set(current.keys())
    vals = set(current.values())
    if keys != vals:
        return False, current, "loop_not_closed_on_start_labels", 0
    order = permutation_order(current, max_order=256)
    if order <= 0:
        return False, current, "loop_map_not_permutation", 0
    return True, current, "", int(order)


def shuffle_domain_labels(domains: Dict[int, BPA.BoundaryDomain], rng: np.random.Generator) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for did, d in domains.items():
        arr = np.asarray(d.labels, dtype=np.int64).copy()
        if len(arr) > 1:
            arr = arr[rng.permutation(len(arr))]
        out[int(did)] = arr
    return out


def build_edge_transports(edges_sub, domains: Dict[int, BPA.BoundaryDomain], labels_override: Optional[Dict[int, np.ndarray]] = None) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, int], EdgeTransport]]:
    rows: List[Dict[str, object]] = []
    edge_map_by_pair: Dict[Tuple[int, int], EdgeTransport] = {}
    for _, e in edges_sub.iterrows():
        s = _safe_int(e.get("source_domain", e.get("source", -1)), -1)
        t = _safe_int(e.get("target_domain", e.get("target", -1)), -1)
        if s not in domains or t not in domains:
            continue
        la = labels_override.get(s) if labels_override is not None else np.asarray(domains[s].labels, dtype=np.int64)
        lb = labels_override.get(t) if labels_override is not None else np.asarray(domains[t].labels, dtype=np.int64)
        et = exact_transport_between(la, lb, source=s, target=t)
        edge_map_by_pair[ordered_edge_key(s, t)] = et
        row = {
            "source_domain": int(s),
            "target_domain": int(t),
            "source_label_count": int(et.source_label_count),
            "target_label_count": int(et.target_label_count),
            "relation_pair_count": int(et.relation_pair_count),
            "forward_deterministic": bool(et.forward_deterministic),
            "backward_deterministic": bool(et.backward_deterministic),
            "exact_bijective_transport": bool(et.exact_bijective_transport),
            "forward_accuracy": float(et.forward_accuracy),
            "backward_accuracy": float(et.backward_accuracy),
            "failure_reason": str(et.failure_reason),
            "forward_map": compact_map_string(et.forward_map),
            "backward_map": compact_map_string(et.backward_map),
        }
        # Carry closure metadata when present.
        for col in ["edge_type", "factor_key", "factor_order", "transport_shift", "source_name", "target_name", "closure_edge_id"]:
            if col in e.index:
                row[col] = e.get(col)
        rows.append(row)
    return rows, edge_map_by_pair


def analyze_cycles_exact(cycles_sub, edge_map_by_pair: Dict[Tuple[int, int], EdgeTransport], max_cycles: int = 5000) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if cycles_sub is None or cycles_sub.empty:
        return rows
    # Deterministic downsample if needed.
    if len(cycles_sub) > int(max_cycles):
        idxs = np.linspace(0, len(cycles_sub) - 1, int(max_cycles)).round().astype(int)
        cycles_iter = cycles_sub.iloc[idxs]
    else:
        cycles_iter = cycles_sub
    for _, c in cycles_iter.iterrows():
        nodes = parse_cycle_nodes(c.get("cycle_nodes", ""))
        valid, loop_map, reason, order = cycle_loop_map(nodes, edge_map_by_pair)
        identity = bool(valid and _is_identity_map(loop_map))
        nontriv = bool(valid and order > 1 and not identity)
        rows.append({
            "cycle_id": _safe_int(c.get("cycle_id", len(rows)), len(rows)),
            "cycle_nodes": " ".join(str(x) for x in nodes),
            "cycle_length": int(len(nodes)),
            "exact_loop_transport_valid": bool(valid),
            "exact_loop_bijective": bool(valid and order > 0),
            "exact_loop_order": int(order),
            "exact_loop_identity": bool(identity),
            "exact_nontrivial_holonomy": bool(nontriv),
            "exact_c2_holonomy": bool(nontriv and order == 2),
            "exact_c3_holonomy": bool(nontriv and order == 3),
            "exact_holonomy_family": ("C2" if nontriv and order == 2 else "C3" if nontriv and order == 3 else f"C{order}" if nontriv else "trivial" if valid else "invalid"),
            "failure_reason": str(reason),
            "loop_map": compact_map_string(loop_map),
        })
    return rows


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_exact_closure_holonomy_audit(
    closure_csv: str,
    closure_edges_csv: str = "",
    closure_cycles_csv: str = "",
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
    null_shuffles: int = 4,
    max_cycles_per_group: int = 5000,
    seed: int = 0,
    max_groups: int = 0,
    only_positive_closure: bool = False,
) -> Tuple[object, object, object, Dict[str, object]]:
    if pd is None:
        raise RuntimeError("pandas is required")
    base_dir = os.path.dirname(os.path.abspath(closure_csv)) or "."
    companions = companion_paths(closure_csv)
    closure_edges_csv = _maybe(closure_edges_csv or companions["edges"], base_dir)
    closure_cycles_csv = _maybe(closure_cycles_csv or companions["cycles"], base_dir)
    closure_summary_path = _maybe(companions["summary"], base_dir)
    closure_summary = _load_json(closure_summary_path)

    if not proliferation_summary:
        src = str(closure_summary.get("source_proliferation_csv", ""))
        if src:
            proliferation_summary = os.path.splitext(src)[0] + "_summary.json"
    proliferation_summary = _maybe(proliferation_summary, base_dir)
    prolif_summary = _load_json(proliferation_summary)

    # argparse-like object for shared parser.
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
    params = params_from_summary(prolif_summary, a)

    closure_df = _read_csv(closure_csv)
    edges_df = _read_csv(closure_edges_csv)
    cycles_df = _read_csv(closure_cycles_csv)
    if closure_df.empty:
        raise ValueError("closure CSV is empty or missing")
    if edges_df.empty:
        raise ValueError("closure edges CSV is empty or missing")

    # Normalize group columns from edge/cycle/main data.
    for df in [closure_df, edges_df, cycles_df]:
        if df is None or df.empty:
            continue
        if "graph_id" not in df.columns and "instance" in df.columns:
            df["graph_id"] = df["instance"]
        if "instance" not in df.columns and "graph_id" in df.columns:
            df["instance"] = df["graph_id"]
        if "q" not in df.columns:
            df["q"] = params.q
    group_cols = [c for c in ["q", "rule_mode", "instance", "graph_id", "seed", "iteration"] if c in edges_df.columns]
    rows: List[Dict[str, object]] = []
    exact_edge_rows: List[Dict[str, object]] = []
    exact_cycle_rows: List[Dict[str, object]] = []
    rng_null = np.random.default_rng(int(seed))
    cache: Dict[Tuple, Tuple[Dict[int, BPA.BoundaryDomain], Sequence[Tuple[int, ...]], Sequence[int]]] = {}

    processed_groups = 0
    positive_lookup = set()
    if only_positive_closure and closure_df is not None and not closure_df.empty:
        ccols = [c for c in group_cols if c in closure_df.columns]
        if ccols and "closure_edges_above_null" in closure_df.columns:
            for _, rr in closure_df[closure_df["closure_edges_above_null"].astype(float) > 0].iterrows():
                positive_lookup.add(tuple(str(rr[c]) for c in ccols))

    for key, egroup in edges_df.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(group_cols, key_tuple))
        if only_positive_closure and positive_lookup:
            ckey = tuple(str(meta[c]) for c in group_cols if c in closure_df.columns)
            if ckey not in positive_lookup:
                continue
        if int(max_groups) > 0 and processed_groups >= int(max_groups):
            break
        processed_groups += 1
        q_val = _safe_int(meta.get("q", params.q), params.q)
        rule_mode = str(meta.get("rule_mode", "unknown"))
        seed_val = _safe_int(meta.get("seed", -1), -1)
        it_val = _safe_int(meta.get("iteration", 0), 0)
        if seed_val < 0:
            # Deterministic fallback matching the proliferation audit's usual seed pattern.
            inst = _safe_int(meta.get("instance", 0), 0)
            seed_val = 1000003 + inst
        # q is usually fixed from summary; per-group q only overrides label arithmetic, not replay q unless explicit.
        params_group = ReplayParams(
            q=int(params.q if params.q > 0 else q_val),
            vertices=params.vertices,
            iterations=params.iterations,
            horizon=params.horizon,
            max_state_samples=params.max_state_samples,
            max_total_states=params.max_total_states,
            initial_boundary=params.initial_boundary,
            initial_boundary_q=params.initial_boundary_q,
            max_domains_per_depth=params.max_domains_per_depth,
            min_live_classes=params.min_live_classes,
            min_fiber_size=params.min_fiber_size,
            min_entropy_bits=params.min_entropy_bits,
            max_pred=params.max_pred,
        )
        cache_key = (rule_mode, int(seed_val), int(it_val), int(params_group.q), int(params_group.vertices))
        if cache_key not in cache:
            cache[cache_key] = replay_domains_for_iteration(rule_mode, int(seed_val), int(it_val), params_group)
        domains, _states, _next_idx = cache[cache_key]
        edge_rows, edge_maps = build_edge_transports(egroup, domains)
        for er in edge_rows:
            er.update(meta)
            exact_edge_rows.append(er)

        if cycles_df is not None and not cycles_df.empty:
            csub = cycles_df.copy()
            mask = np.ones(len(csub), dtype=bool)
            for col, val in meta.items():
                if col in csub.columns:
                    mask &= (csub[col].astype(str).values == str(val))
            csub = csub[mask]
        else:
            csub = pd.DataFrame()
        cyc_rows = analyze_cycles_exact(csub, edge_maps, max_cycles=int(max_cycles_per_group))
        for cr in cyc_rows:
            cr.update(meta)
            exact_cycle_rows.append(cr)

        # Null: shuffle labels by state within each reconstructed domain and recompute exact cycles.
        null_nontriv: List[int] = []
        null_valid: List[int] = []
        for _ in range(int(null_shuffles)):
            labels_null = shuffle_domain_labels(domains, rng_null)
            _erows_n, edge_maps_n = build_edge_transports(egroup, domains, labels_override=labels_null)
            cr_n = analyze_cycles_exact(csub, edge_maps_n, max_cycles=int(max_cycles_per_group))
            null_nontriv.append(int(sum(1 for r in cr_n if bool(r.get("exact_nontrivial_holonomy", False)))))
            null_valid.append(int(sum(1 for r in cr_n if bool(r.get("exact_loop_transport_valid", False)))))

        n_edges = len(edge_rows)
        n_bij_edges = int(sum(1 for r in edge_rows if bool(r.get("exact_bijective_transport", False))))
        n_cycles = len(cyc_rows)
        n_valid_cycles = int(sum(1 for r in cyc_rows if bool(r.get("exact_loop_transport_valid", False))))
        n_nontriv = int(sum(1 for r in cyc_rows if bool(r.get("exact_nontrivial_holonomy", False))))
        n_c2 = int(sum(1 for r in cyc_rows if bool(r.get("exact_c2_holonomy", False))))
        n_c3 = int(sum(1 for r in cyc_rows if bool(r.get("exact_c3_holonomy", False))))
        max_order = int(max([_safe_int(r.get("exact_loop_order", 0), 0) for r in cyc_rows] or [0]))
        null_nontriv_mean = float(np.mean(null_nontriv) if null_nontriv else 0.0)
        row = {
            "q": int(q_val),
            "rule_mode": rule_mode,
            "instance": _safe_int(meta.get("instance", -1), -1),
            "graph_id": _safe_int(meta.get("graph_id", -1), -1),
            "seed": int(seed_val),
            "iteration": int(it_val),
            "n_domains_reconstructed": int(len(domains)),
            "n_closure_edges": int(n_edges),
            "exact_edge_bijective_count": int(n_bij_edges),
            "exact_edge_bijective_fraction": float(n_bij_edges / max(1, n_edges)),
            "n_cycles_tested": int(n_cycles),
            "exact_loop_valid_count": int(n_valid_cycles),
            "exact_loop_valid_fraction": float(n_valid_cycles / max(1, n_cycles)),
            "exact_nontrivial_holonomy_count": int(n_nontriv),
            "exact_c2_holonomy_count": int(n_c2),
            "exact_c3_holonomy_count": int(n_c3),
            "max_exact_holonomy_order": int(max_order),
            "null_exact_nontrivial_holonomy_mean": float(null_nontriv_mean),
            "exact_nontrivial_holonomy_above_null": float(n_nontriv - null_nontriv_mean),
            "null_exact_loop_valid_mean": float(np.mean(null_valid) if null_valid else 0.0),
            "exact_holonomy_proof_available": bool(n_cycles > 0 and n_valid_cycles > 0),
            "exact_nontrivial_holonomy_proven": bool(n_nontriv > null_nontriv_mean and n_nontriv > 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    edge_df = pd.DataFrame(exact_edge_rows)
    cycle_df = pd.DataFrame(exact_cycle_rows)

    any_proof = bool(not df.empty and (df["exact_holonomy_proof_available"] == True).any())
    any_nontriv = bool(not df.empty and (df["exact_nontrivial_holonomy_proven"] == True).any())
    any_c2 = bool(not df.empty and (df["exact_c2_holonomy_count"] > 0).any())
    any_c3 = bool(not df.empty and (df["exact_c3_holonomy_count"] > 0).any())
    if any_nontriv:
        verdict = "EXACT CLOSURE HOLONOMY SIGNAL: closure-generated loops carry nontrivial exact transport"
    elif any_proof:
        verdict = "EXACT CLOSURE TRANSPORT FLAT/WEAK SIGNAL: exact loop maps exist but nontrivial holonomy not established"
    else:
        verdict = "EXACT CLOSURE HOLONOMY NEGATIVE/WEAK SIGNAL: closure candidates do not yet carry exact loop transports"
    summary: Dict[str, object] = {
        "verdict": verdict,
        "audit_version": "exact_closure_holonomy_v1_replayed_domain_label_transport",
        "source_closure_csv": str(closure_csv),
        "source_closure_edges_csv": str(closure_edges_csv),
        "source_closure_cycles_csv": str(closure_cycles_csv),
        "source_proliferation_summary": str(proliferation_summary),
        "q": int(params.q),
        "vertices": int(params.vertices),
        "horizon": int(params.horizon),
        "max_state_samples": int(params.max_state_samples),
        "n_rows": int(len(df)),
        "n_exact_edge_rows": int(len(edge_df)),
        "n_exact_cycle_rows": int(len(cycle_df)),
        "any_exact_loop_transport": bool(any_proof),
        "any_exact_nontrivial_holonomy": bool(any_nontriv),
        "any_exact_c2_holonomy": bool(any_c2),
        "any_exact_c3_holonomy": bool(any_c3),
        "max_exact_edge_bijective_fraction": float(df["exact_edge_bijective_fraction"].max()) if not df.empty else 0.0,
        "max_exact_loop_valid_fraction": float(df["exact_loop_valid_fraction"].max()) if not df.empty else 0.0,
        "max_exact_nontrivial_holonomy_count": int(df["exact_nontrivial_holonomy_count"].max()) if not df.empty else 0,
        "max_exact_holonomy_order": int(df["max_exact_holonomy_order"].max()) if not df.empty else 0,
        "mean_exact_nontrivial_holonomy_above_null": float(df["exact_nontrivial_holonomy_above_null"].mean()) if not df.empty else 0.0,
    }
    if not df.empty:
        by_mode = []
        for mode, g in df.groupby("rule_mode"):
            by_mode.append({
                "rule_mode": str(mode),
                "n": int(len(g)),
                "mean_exact_edge_bijective_fraction": float(g["exact_edge_bijective_fraction"].mean()),
                "mean_exact_loop_valid_fraction": float(g["exact_loop_valid_fraction"].mean()),
                "max_exact_nontrivial_holonomy_count": int(g["exact_nontrivial_holonomy_count"].max()),
                "max_exact_c2_holonomy_count": int(g["exact_c2_holonomy_count"].max()),
                "max_exact_c3_holonomy_count": int(g["exact_c3_holonomy_count"].max()),
                "proof_fraction": float((g["exact_holonomy_proof_available"] == True).mean()),
                "nontrivial_proven_fraction": float((g["exact_nontrivial_holonomy_proven"] == True).mean()),
            })
        summary["by_mode"] = by_mode
    return df, edge_df, cycle_df, summary


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
        "exact_edge_bijective_fraction": "mean",
        "exact_loop_valid_fraction": "mean",
        "exact_nontrivial_holonomy_count": "max",
        "exact_c2_holonomy_count": "max",
        "exact_c3_holonomy_count": "max",
    }).reset_index()
    x = np.arange(len(g))
    w = 0.18
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x - 1.5*w, g["exact_edge_bijective_fraction"], width=w, label="edge bijective frac")
    ax1.bar(x - 0.5*w, g["exact_loop_valid_fraction"], width=w, label="loop valid frac")
    ax1.set_ylabel("fraction")
    ax1.set_xticks(x)
    ax1.set_xticklabels(g["rule_mode"], rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, g["exact_nontrivial_holonomy_count"], marker="o", label="nontrivial holonomy count")
    ax2.plot(x, g["exact_c2_holonomy_count"], marker="s", label="C2 count")
    ax2.plot(x, g["exact_c3_holonomy_count"], marker="^", label="C3 count")
    ax2.set_ylabel("cycle count")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("Exact closure holonomy: candidate loops -> exact transport maps")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exact transport/holonomy audit for proliferation-closure loop candidates")
    p.add_argument("--closure-csv", required=True, help="Output CSV from proliferationclosureaudit.py")
    p.add_argument("--closure-edges-csv", default="", help="Closure edges CSV; defaults to <closure stem>_closure_edges.csv")
    p.add_argument("--closure-cycles-csv", default="", help="Closure cycles CSV; defaults to <closure stem>_closure_cycles.csv")
    p.add_argument("--proliferation-summary", default="", help="Boundary proliferation summary JSON used to reconstruct the arena")
    p.add_argument("--q", type=int, default=0, help="Override q from proliferation summary")
    p.add_argument("--vertices", type=int, default=0, help="Override vertex count from proliferation summary")
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
    p.add_argument("--null-shuffles", type=int, default=4)
    p.add_argument("--max-cycles-per-group", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-groups", type=int, default=0, help="Process at most this many closure groups; 0 means all")
    p.add_argument("--only-positive-closure", action="store_true", help="Only process groups whose closure_edges_above_null is positive in the closure CSV")
    p.add_argument("--out", default="example_results/exact_closure_holonomy.csv")
    p.add_argument("--plot", default="")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    df, edge_df, cycle_df, summary = run_exact_closure_holonomy_audit(
        closure_csv=args.closure_csv,
        closure_edges_csv=args.closure_edges_csv,
        closure_cycles_csv=args.closure_cycles_csv,
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
        null_shuffles=args.null_shuffles,
        max_cycles_per_group=args.max_cycles_per_group,
        seed=args.seed,
        max_groups=args.max_groups,
        only_positive_closure=args.only_positive_closure,
    )
    out = args.out
    maybe_write_csv(df, out)
    stem = _stem(out)
    maybe_write_csv(edge_df, stem + "_exact_edges.csv")
    maybe_write_csv(cycle_df, stem + "_exact_cycles.csv")
    with open(stem + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.plot:
        plot_summary(df, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(f"wrote {stem + '_exact_edges.csv'}")
    print(f"wrote {stem + '_exact_cycles.csv'}")
    print(f"wrote {stem + '_summary.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()
