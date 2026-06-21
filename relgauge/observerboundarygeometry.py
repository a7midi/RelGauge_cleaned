"""
observerboundarygeometry.py

Observer-relative boundary geometry audit for finite relational systems.

This module is deliberately diagnostic rather than a new physics postulate.  It
implements the closed-system idea that the whole system has no exterior, but
proper subsystems can have observer-relative boundaries/cuts.  It then asks what
geometry is visible through those cuts.

Core workflow
-------------
1. Generate finite directed relational systems, or analyze a supplied finite
   graph/rule instance in code.
2. Identify SCCs as candidate observer pockets.
3. For every proper SCC, compute its boundary relative to the complement.
4. Estimate a passive one-step boundary channel from complement inputs into the
   observer's boundary panel.
5. Build the observer graph whose nodes are SCC observers and whose edges are
   live inter-SCC channels.
6. Find minimal cycles in the *observer-relative* graph.  These cycles are
   candidate plaquettes discovered from cuts, not imposed from the outside.

The module does NOT insert a square, triangle, lattice, gauge group, or matter
field.  It only reports what SCC observers can resolve through their own cuts.

CLI example
-----------
python -m relgauge.observerboundarygeometry 4 \
  --graphs 100 --vertices 18 --graph-mode componented \
  --components 5 --inter-prob 0.35 --extra-intra-prob 0.25 \
  --capacity-threshold 0.05 \
  --out example_results/observer_boundary_geometry.csv \
  --plot example_results/fig_observer_boundary_geometry.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # optional at import time; required for CLI CSV output
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


Edge = Tuple[int, int]


# ---------------------------------------------------------------------------
# Small finite relational system
# ---------------------------------------------------------------------------
@dataclass
class FiniteRelationalSystem:
    q: int
    preds: List[Tuple[int, ...]]
    tables: List[np.ndarray]

    @property
    def k(self) -> int:
        return len(self.preds)

    def edges(self) -> List[Edge]:
        return [(u, v) for v, ps in enumerate(self.preds) for u in ps]

    def eval_vertex_from_assignment(self, v: int, assignment: Dict[int, int]) -> int:
        code = 0
        for p in self.preds[v]:
            code = code * self.q + int(assignment.get(p, 0))
        return int(self.tables[v][code])

    def eval_vertex(self, v: int, state: Sequence[int]) -> int:
        code = 0
        for p in self.preds[v]:
            code = code * self.q + int(state[p])
        return int(self.tables[v][code])

    def step_parallel(self, state: Sequence[int]) -> Tuple[int, ...]:
        return tuple(self.eval_vertex(v, state) for v in range(self.k))


def _encode_tuple(vals: Sequence[int], q: int) -> int:
    code = 0
    for x in vals:
        code = code * q + int(x)
    return int(code)


def _random_table(q: int, arity: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, q, size=q ** int(arity), dtype=np.int16)


def make_random_system(
    q: int,
    n: int,
    edge_prob: float,
    rng: np.random.Generator,
    max_pred: Optional[int] = None,
    allow_self: bool = False,
) -> FiniteRelationalSystem:
    """Erdos-Renyi directed finite system with random local tables."""
    preds: List[Tuple[int, ...]] = []
    for v in range(n):
        ps = [u for u in range(n) if (allow_self or u != v) and rng.random() < edge_prob]
        if max_pred is not None and len(ps) > max_pred:
            ps = list(rng.choice(ps, size=max_pred, replace=False))
        ps = sorted(set(int(x) for x in ps))
        preds.append(tuple(ps))
    tables = [_random_table(q, len(ps), rng) for ps in preds]
    return FiniteRelationalSystem(q=q, preds=preds, tables=tables)


def _partition_sizes(n: int, components: int) -> List[int]:
    base = n // components
    rem = n % components
    return [base + (1 if i < rem else 0) for i in range(components)]


def make_componented_system(
    q: int,
    n: int,
    components: int,
    inter_prob: float,
    extra_intra_prob: float,
    rng: np.random.Generator,
    max_pred: Optional[int] = None,
) -> FiniteRelationalSystem:
    """Random closed system with several internally recurrent SCC-like blocks.

    Each block receives a directed cycle to guarantee feedback.  Inter-block
    edges are drawn in a random acyclic order, so the whole closed universe can
    contain multiple proper SCC observers with observer-relative boundaries.
    """
    sizes = _partition_sizes(n, max(1, int(components)))
    blocks: List[List[int]] = []
    idx = 0
    for s in sizes:
        blocks.append(list(range(idx, idx + s)))
        idx += s

    pred_sets: List[Set[int]] = [set() for _ in range(n)]

    # Internal directed cycle per block, plus optional extra edges.
    for block in blocks:
        if len(block) == 1:
            # singletons are not genuine observers; optionally give self-loop? no.
            continue
        for a, b in zip(block, block[1:] + block[:1]):
            pred_sets[b].add(a)
        for u in block:
            for v in block:
                if u != v and rng.random() < extra_intra_prob:
                    pred_sets[v].add(u)

    # Inter-block edges in a random condensation order.  This may create
    # undirected cycles/diamonds while keeping SCCs distinct.
    order = list(range(len(blocks)))
    rng.shuffle(order)
    rank = {c: i for i, c in enumerate(order)}
    for ci, bi in enumerate(blocks):
        for cj, bj in enumerate(blocks):
            if ci == cj:
                continue
            if rank[ci] < rank[cj] and rng.random() < inter_prob:
                u = int(rng.choice(bi)); v = int(rng.choice(bj))
                pred_sets[v].add(u)
                # sometimes add a second independent edge to thicken the cut
                if len(bi) > 1 and len(bj) > 1 and rng.random() < 0.35:
                    u2 = int(rng.choice(bi)); v2 = int(rng.choice(bj))
                    pred_sets[v2].add(u2)

    preds: List[Tuple[int, ...]] = []
    for v, ps in enumerate(pred_sets):
        psl = sorted(ps)
        if max_pred is not None and len(psl) > max_pred:
            # Preserve at least one internal predecessor if possible.
            psl = list(rng.choice(psl, size=max_pred, replace=False))
            psl = sorted(set(int(x) for x in psl))
        preds.append(tuple(psl))
    tables = [_random_table(q, len(ps), rng) for ps in preds]
    return FiniteRelationalSystem(q=q, preds=preds, tables=tables)


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------
def tarjan_scc(n: int, edges: Iterable[Edge]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v))

    index = 0
    stack: List[int] = []
    on_stack: Set[int] = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    comps: List[List[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj[v]:
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: List[int] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(sorted(comp))

    for v in range(n):
        if v not in indices:
            strongconnect(v)
    return comps


def boundary_edges(subset: Set[int], edges: Sequence[Edge]) -> Tuple[List[Edge], List[Edge]]:
    incoming, outgoing = [], []
    for u, v in edges:
        if u not in subset and v in subset:
            incoming.append((u, v))
        elif u in subset and v not in subset:
            outgoing.append((u, v))
    return incoming, outgoing


def weak_components(nodes: Sequence[int], undirected_edges: Sequence[Tuple[int, int]]) -> int:
    node_set = set(nodes)
    adj = {v: set() for v in node_set}
    for a, b in undirected_edges:
        if a in node_set and b in node_set and a != b:
            adj[a].add(b); adj[b].add(a)
    seen: Set[int] = set(); comps = 0
    for s in nodes:
        if s in seen:
            continue
        comps += 1
        st = [s]
        while st:
            x = st.pop()
            if x in seen:
                continue
            seen.add(x)
            st.extend(adj[x] - seen)
    return comps


def canonical_cycle(cyc: Sequence[int]) -> Tuple[int, ...]:
    """Canonicalize an undirected simple cycle."""
    c = list(cyc)
    if len(c) == 0:
        return tuple()
    variants = []
    for seq in (c, list(reversed(c))):
        for i in range(len(seq)):
            rot = tuple(seq[i:] + seq[:i])
            variants.append(rot)
    return min(variants)


def find_undirected_cycles(nodes: Sequence[int], edges: Sequence[Tuple[int, int]], max_len: int = 6, limit: int = 500) -> List[Tuple[int, ...]]:
    """Find simple undirected cycles up to max_len.  Small-graph diagnostic."""
    adj = {v: set() for v in nodes}
    for a, b in edges:
        if a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    found: Set[Tuple[int, ...]] = set()
    for start in nodes:
        stack = [(start, [start])]
        while stack and len(found) < limit:
            cur, path = stack.pop()
            for nb in adj.get(cur, ()):
                if nb == start and len(path) >= 3:
                    found.add(canonical_cycle(path))
                elif nb not in path and len(path) < max_len and nb >= start:
                    stack.append((nb, path + [nb]))
    return sorted(found, key=lambda x: (len(x), x))


# ---------------------------------------------------------------------------
# Information diagnostics
# ---------------------------------------------------------------------------
def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def mutual_information_from_joint(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=float)
    if joint.sum() <= 0:
        return 0.0
    return entropy_from_counts(joint.sum(axis=1)) + entropy_from_counts(joint.sum(axis=0)) - entropy_from_counts(joint)


def _all_assignments(q: int, nodes: Sequence[int], max_configs: int, rng: np.random.Generator) -> List[Tuple[int, ...]]:
    nodes = list(nodes)
    total = q ** len(nodes)
    if len(nodes) == 0:
        return [tuple()]
    if total <= max_configs:
        return list(itertools.product(range(q), repeat=len(nodes)))
    out = []
    for _ in range(max_configs):
        out.append(tuple(int(x) for x in rng.integers(0, q, size=len(nodes))))
    return out


def estimate_channel_mi(
    sys: FiniteRelationalSystem,
    input_nodes: Sequence[int],
    target_nodes: Sequence[int],
    max_input_configs: int,
    max_background_configs: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Estimate I(input_nodes ; post-update target_nodes) with uniform inputs.

    Background consists of all predecessors of target nodes not in input_nodes.
    This is a passive local channel proxy, not a full causal intervention.
    """
    q = sys.q
    input_nodes = tuple(sorted(set(int(x) for x in input_nodes)))
    target_nodes = tuple(sorted(set(int(x) for x in target_nodes)))
    if not input_nodes or not target_nodes:
        return dict(mi_bits=0.0, h_input_bits=0.0, h_output_bits=0.0, mi_norm=0.0,
                    n_input_nodes=len(input_nodes), n_target_nodes=len(target_nodes))

    needed: Set[int] = set()
    for v in target_nodes:
        needed.update(sys.preds[v])
    bg_nodes = tuple(sorted(needed - set(input_nodes)))

    input_assignments = _all_assignments(q, input_nodes, max_input_configs, rng)
    bg_assignments = _all_assignments(q, bg_nodes, max_background_configs, rng)
    n_i = q ** len(input_nodes)
    n_y = q ** len(target_nodes)
    # If sampled input assignments do not cover all possible inputs, compress to sampled index space.
    input_codes = sorted(set(_encode_tuple(a, q) for a in input_assignments))
    code_to_row = {c: i for i, c in enumerate(input_codes)}
    joint = np.zeros((len(input_codes), n_y), dtype=np.int64)

    for ia in input_assignments:
        assignment: Dict[int, int] = {node: val for node, val in zip(input_nodes, ia)}
        i_code = _encode_tuple(ia, q)
        row = code_to_row[i_code]
        for ba in bg_assignments:
            a2 = dict(assignment)
            a2.update({node: val for node, val in zip(bg_nodes, ba)})
            y = [sys.eval_vertex_from_assignment(v, a2) for v in target_nodes]
            joint[row, _encode_tuple(y, q)] += 1

    mi = mutual_information_from_joint(joint)
    h_i = entropy_from_counts(joint.sum(axis=1))
    h_y = entropy_from_counts(joint.sum(axis=0))
    mi_norm = float(mi / h_i) if h_i > 1e-12 else 0.0
    return dict(mi_bits=float(mi), h_input_bits=float(h_i), h_output_bits=float(h_y), mi_norm=mi_norm,
                n_input_nodes=len(input_nodes), n_target_nodes=len(target_nodes),
                n_background_nodes=len(bg_nodes), sampled_input_fraction=float(len(input_codes) / max(1, n_i)))


# ---------------------------------------------------------------------------
# Observer-relative audit
# ---------------------------------------------------------------------------
def analyze_system(
    sys: FiniteRelationalSystem,
    graph_id: int = 0,
    capacity_threshold: float = 0.05,
    max_channel_inputs: int = 256,
    max_channel_backgrounds: int = 256,
    cycle_max_len: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    if rng is None:
        rng = np.random.default_rng(0)
    edges = sys.edges()
    comps = tarjan_scc(sys.k, edges)
    comp_of: Dict[int, int] = {}
    for ci, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = ci

    full_set = set(range(sys.k))
    full_in, full_out = boundary_edges(full_set, edges)
    full_boundary_violation = int(len(full_in) + len(full_out) != 0)

    observer_rows: List[Dict] = []
    for ci, comp in enumerate(comps):
        S = set(comp)
        incoming, outgoing = boundary_edges(S, edges)
        is_whole = (len(S) == sys.k)
        genuine_feedback = len(S) >= 2
        in_nodes = sorted({u for u, _ in incoming})
        boundary_targets = sorted({v for _, v in incoming})
        channel = estimate_channel_mi(sys, in_nodes, boundary_targets, max_channel_inputs, max_channel_backgrounds, rng)
        observer_rows.append(dict(
            graph_id=graph_id,
            observer_id=ci,
            size=len(S),
            is_whole=int(is_whole),
            genuine_feedback=int(genuine_feedback),
            boundary_in_edges=len(incoming),
            boundary_out_edges=len(outgoing),
            boundary_total_edges=len(incoming) + len(outgoing),
            boundary_in_nodes=len(in_nodes),
            boundary_targets=len(boundary_targets),
            boundary_mi_bits=channel.get("mi_bits", 0.0),
            boundary_mi_norm=channel.get("mi_norm", 0.0),
            boundary_h_input_bits=channel.get("h_input_bits", 0.0),
            live_boundary=int(channel.get("mi_bits", 0.0) >= capacity_threshold),
            vertices=" ".join(map(str, comp)),
        ))

    # Component-level directed edges and capacities.
    pair_edges: Dict[Tuple[int, int], List[Edge]] = {}
    for u, v in edges:
        cu, cv = comp_of[u], comp_of[v]
        if cu != cv:
            pair_edges.setdefault((cu, cv), []).append((u, v))

    edge_rows: List[Dict] = []
    undirected_edges_set: Set[Tuple[int, int]] = set()
    cap_by_undir: Dict[Tuple[int, int], float] = {}
    for (cu, cv), evs in sorted(pair_edges.items()):
        source_nodes = sorted({u for u, _ in evs})
        target_nodes = sorted({v for _, v in evs})
        channel = estimate_channel_mi(sys, source_nodes, target_nodes, max_channel_inputs, max_channel_backgrounds, rng)
        live = int(channel.get("mi_bits", 0.0) >= capacity_threshold)
        edge_rows.append(dict(
            graph_id=graph_id,
            source_observer=cu,
            target_observer=cv,
            n_edges=len(evs),
            source_boundary_nodes=len(source_nodes),
            target_boundary_nodes=len(target_nodes),
            edge_mi_bits=channel.get("mi_bits", 0.0),
            edge_mi_norm=channel.get("mi_norm", 0.0),
            edge_h_input_bits=channel.get("h_input_bits", 0.0),
            live_edge=live,
            edges=" ".join(f"{u}->{v}" for u, v in evs),
        ))
        a, b = sorted((cu, cv))
        undirected_edges_set.add((a, b))
        cap_by_undir[(a, b)] = max(cap_by_undir.get((a, b), 0.0), float(channel.get("mi_bits", 0.0)))

    nodes = list(range(len(comps)))
    undirected_edges = sorted(undirected_edges_set)
    cycles = find_undirected_cycles(nodes, undirected_edges, max_len=cycle_max_len)
    cycle_rows: List[Dict] = []
    for cidx, cyc in enumerate(cycles):
        edge_caps = []
        edge_pairs = []
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            key = tuple(sorted((a, b)))
            edge_pairs.append(key)
            edge_caps.append(cap_by_undir.get(key, 0.0))
        min_cap = min(edge_caps) if edge_caps else 0.0
        prod_score = float(np.prod([min(1.0, x) for x in edge_caps])) if edge_caps else 0.0
        cycle_rows.append(dict(
            graph_id=graph_id,
            cycle_id=cidx,
            length=len(cyc),
            observers=" ".join(map(str, cyc)),
            min_edge_mi_bits=float(min_cap),
            mean_edge_mi_bits=float(np.mean(edge_caps)) if edge_caps else 0.0,
            product_edge_score=prod_score,
            live_cycle=int(min_cap >= capacity_threshold),
            edge_pairs=" ".join(f"{a}-{b}" for a, b in edge_pairs),
        ))

    n_obs = len(comps)
    n_uedges = len(undirected_edges)
    n_wcc = weak_components(nodes, undirected_edges) if nodes else 0
    beta1 = n_uedges - n_obs + n_wcc
    summary = dict(
        graph_id=graph_id,
        q=sys.q,
        n_vertices=sys.k,
        n_edges=len(edges),
        n_scc=len(comps),
        n_proper_scc=sum(1 for c in comps if len(c) < sys.k),
        n_feedback_scc=sum(1 for c in comps if len(c) >= 2),
        n_observers_with_boundary=sum(1 for r in observer_rows if r["boundary_total_edges"] > 0 and not r["is_whole"]),
        n_live_observers=sum(int(r["live_boundary"]) for r in observer_rows if not r["is_whole"]),
        n_observer_edges=len(edge_rows),
        n_live_observer_edges=sum(int(r["live_edge"]) for r in edge_rows),
        observer_graph_beta1=int(beta1),
        n_observer_cycles=len(cycle_rows),
        n_live_observer_cycles=sum(int(r["live_cycle"]) for r in cycle_rows),
        full_boundary_violation=full_boundary_violation,
        mean_boundary_mi_bits=float(np.mean([r["boundary_mi_bits"] for r in observer_rows if not r["is_whole"]])) if any(not r["is_whole"] for r in observer_rows) else 0.0,
        mean_edge_mi_bits=float(np.mean([r["edge_mi_bits"] for r in edge_rows])) if edge_rows else 0.0,
    )
    return observer_rows, edge_rows, cycle_rows, summary


def run_audit(
    q: int,
    graphs: int,
    vertices: int,
    graph_mode: str,
    components: int,
    edge_prob: float,
    inter_prob: float,
    extra_intra_prob: float,
    max_pred: Optional[int],
    capacity_threshold: float,
    max_channel_inputs: int,
    max_channel_backgrounds: int,
    cycle_max_len: int,
    seed: int,
    verbose: bool = True,
) -> Tuple[object, object, object, Dict]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for run_audit/CLI output")
    all_obs: List[Dict] = []
    all_edges: List[Dict] = []
    all_cycles: List[Dict] = []
    summaries: List[Dict] = []
    rng0 = np.random.default_rng(seed)
    for gi in range(graphs):
        grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
        if graph_mode == "er":
            sys = make_random_system(q, vertices, edge_prob, grng, max_pred=max_pred)
        elif graph_mode == "componented":
            sys = make_componented_system(q, vertices, components, inter_prob, extra_intra_prob, grng, max_pred=max_pred)
        else:
            raise ValueError(f"unknown graph mode: {graph_mode}")
        obs, edg, cyc, summ = analyze_system(sys, gi, capacity_threshold, max_channel_inputs, max_channel_backgrounds, cycle_max_len, grng)
        all_obs.extend(obs); all_edges.extend(edg); all_cycles.extend(cyc); summaries.append(summ)
        if verbose:
            print(f"observer-boundary graph={gi} scc={summ['n_scc']} cycles={summ['n_observer_cycles']} live_cycles={summ['n_live_observer_cycles']}", flush=True)

    obs_df = pd.DataFrame(all_obs)
    edge_df = pd.DataFrame(all_edges)
    cycle_df = pd.DataFrame(all_cycles)
    summ_df = pd.DataFrame(summaries)

    total_cycles = int(summ_df["n_observer_cycles"].sum()) if len(summ_df) else 0
    live_cycles = int(summ_df["n_live_observer_cycles"].sum()) if len(summ_df) else 0
    live_obs = int(summ_df["n_live_observers"].sum()) if len(summ_df) else 0
    boundary_viol = int(summ_df["full_boundary_violation"].sum()) if len(summ_df) else 0

    if live_cycles > 0:
        verdict = "OBSERVER-RELATIVE CYCLE GEOMETRY SIGNAL: live cycles appear in observer-resolvable cut graph"
    elif total_cycles > 0:
        verdict = "OBSERVER-RELATIVE GEOMETRY SIGNAL: cycles appear, but live boundary capacity is weak in this regime"
    elif live_obs > 0:
        verdict = "OBSERVER-BOUNDARY SIGNAL: live observer cuts appear, but no observer-relative cycles were found"
    else:
        verdict = "NO ROBUST OBSERVER-BOUNDARY GEOMETRY SIGNAL in this regime"

    summary = dict(
        verdict=verdict,
        q=q,
        graph_mode=graph_mode,
        n_graphs=int(graphs),
        n_vertices=int(vertices),
        mean_scc=float(summ_df["n_scc"].mean()) if len(summ_df) else 0.0,
        mean_feedback_scc=float(summ_df["n_feedback_scc"].mean()) if len(summ_df) else 0.0,
        mean_observers_with_boundary=float(summ_df["n_observers_with_boundary"].mean()) if len(summ_df) else 0.0,
        mean_live_observers=float(summ_df["n_live_observers"].mean()) if len(summ_df) else 0.0,
        mean_observer_edges=float(summ_df["n_observer_edges"].mean()) if len(summ_df) else 0.0,
        mean_live_observer_edges=float(summ_df["n_live_observer_edges"].mean()) if len(summ_df) else 0.0,
        mean_observer_beta1=float(summ_df["observer_graph_beta1"].mean()) if len(summ_df) else 0.0,
        total_observer_cycles=total_cycles,
        total_live_observer_cycles=live_cycles,
        graph_fraction_with_cycles=float((summ_df["n_observer_cycles"] > 0).mean()) if len(summ_df) else 0.0,
        graph_fraction_with_live_cycles=float((summ_df["n_live_observer_cycles"] > 0).mean()) if len(summ_df) else 0.0,
        mean_boundary_mi_bits=float(summ_df["mean_boundary_mi_bits"].mean()) if len(summ_df) else 0.0,
        mean_edge_mi_bits=float(summ_df["mean_edge_mi_bits"].mean()) if len(summ_df) else 0.0,
        full_boundary_violations=boundary_viol,
    )
    return obs_df, edge_df, cycle_df, summary


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------
def _derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def write_outputs(obs_df, edge_df, cycle_df, summary: Dict, out: str) -> None:
    obs_df.to_csv(out, index=False)
    edge_df.to_csv(_derived_path(out, "_edges"), index=False)
    cycle_df.to_csv(_derived_path(out, "_cycles"), index=False)
    with open(_derived_path(out, "_summary").replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def make_plot(obs_df, edge_df, cycle_df, summary: Dict, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    if len(obs_df):
        axes[0].hist(obs_df.loc[obs_df["is_whole"] == 0, "boundary_mi_bits"].astype(float), bins=20)
    axes[0].set_title("Observer boundary MI")
    axes[0].set_xlabel("bits")
    axes[0].set_ylabel("count")

    if len(edge_df):
        axes[1].hist(edge_df["edge_mi_bits"].astype(float), bins=20)
    axes[1].set_title("Inter-observer edge MI")
    axes[1].set_xlabel("bits")

    if len(cycle_df):
        vals = cycle_df["min_edge_mi_bits"].astype(float)
        axes[2].hist(vals, bins=20)
    axes[2].set_title("Discovered cycle min edge MI")
    axes[2].set_xlabel("bits")

    fig.suptitle(summary.get("verdict", "observer boundary geometry"), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Observer-relative boundary geometry audit")
    p.add_argument("q", type=int, help="alphabet size")
    p.add_argument("--graphs", type=int, default=50, help="number of random finite systems")
    p.add_argument("--vertices", type=int, default=18, help="vertices per finite system")
    p.add_argument("--graph-mode", choices=["componented", "er"], default="componented")
    p.add_argument("--components", type=int, default=5, help="blocks for componented graph mode")
    p.add_argument("--edge-prob", type=float, default=0.12, help="edge probability for ER mode")
    p.add_argument("--inter-prob", type=float, default=0.35, help="component DAG edge probability")
    p.add_argument("--extra-intra-prob", type=float, default=0.25, help="extra internal edge probability")
    p.add_argument("--max-pred", type=int, default=4, help="cap local rule arity; <=0 disables")
    p.add_argument("--capacity-threshold", type=float, default=0.05, help="MI threshold for a live cut/edge/cycle")
    p.add_argument("--max-channel-inputs", type=int, default=256)
    p.add_argument("--max-channel-backgrounds", type=int, default=256)
    p.add_argument("--cycle-max-len", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/observer_boundary_geometry.csv")
    p.add_argument("--plot", default="example_results/fig_observer_boundary_geometry.png")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    max_pred = None if args.max_pred is None or args.max_pred <= 0 else int(args.max_pred)
    obs_df, edge_df, cycle_df, summary = run_audit(
        q=args.q,
        graphs=args.graphs,
        vertices=args.vertices,
        graph_mode=args.graph_mode,
        components=args.components,
        edge_prob=args.edge_prob,
        inter_prob=args.inter_prob,
        extra_intra_prob=args.extra_intra_prob,
        max_pred=max_pred,
        capacity_threshold=args.capacity_threshold,
        max_channel_inputs=args.max_channel_inputs,
        max_channel_backgrounds=args.max_channel_backgrounds,
        cycle_max_len=args.cycle_max_len,
        seed=args.seed,
        verbose=not args.quiet,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_outputs(obs_df, edge_df, cycle_df, summary, args.out)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        make_plot(obs_df, edge_df, cycle_df, summary, args.plot)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {_derived_path(args.out, '_edges')}")
    print(f"wrote {_derived_path(args.out, '_cycles')}")
    if args.plot:
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
