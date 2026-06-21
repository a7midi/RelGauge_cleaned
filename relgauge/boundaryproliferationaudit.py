"""
boundaryproliferationaudit.py

Boundary proliferation operator audit.

This module implements the generative endpoint suggested by the observer-relative
framework: topology should not be inserted as an observer graph, but inferred
from live unresolved boundary fibers.

Given a finite deterministic arena and an initial finite boundary map b, the
operator iterates:

    boundary -> fiber -> predictive quotient -> sub-boundary -> dependency graph

A fiber proliferates only when hidden distinctions inside the fiber change
future boundary-visible histories.  The generated observer topology is the
predictive-dependency graph among the live sub-boundaries.  The module then
canonicalizes the generated architecture at each iteration and checks for fixed
points or limit cycles.

This is an audit, not a theorem prover.  The exact theoretical equivalence
"same future histories under all admissible schedules and contexts" is
approximated by finite sampled states, a bounded prediction horizon, and a
chosen deterministic update arena.

CLI example
-----------
python -m relgauge.boundaryproliferationaudit 2 ^
  --vertices 7 ^
  --rule-modes affine_mix,shift_xor,random_table ^
  --instances 10 ^
  --iterations 5 ^
  --horizon 3 ^
  --max-state-samples 512 ^
  --initial-boundary sum_mod_q ^
  --out example_results/boundary_proliferation_q2.csv ^
  --plot example_results/fig_boundary_proliferation_q2.png
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import observerboundarygeometry as OBG
except Exception:  # pragma: no cover
    import observerboundarygeometry as OBG  # type: ignore

State = Tuple[int, ...]
Edge = Tuple[int, int]


# ---------------------------------------------------------------------------
# Basic finite information helpers
# ---------------------------------------------------------------------------
def entropy_from_counts(counts: Sequence[int]) -> float:
    arr = np.asarray(list(counts), dtype=float)
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    p = arr[arr > 0] / total
    return float(-(p * np.log2(p)).sum())


def entropy_of_labels(labels: Sequence[int]) -> float:
    return entropy_from_counts(Counter(int(x) for x in labels).values())


def encode_tuple(vals: Sequence[int], q: int) -> int:
    code = 0
    for x in vals:
        code = code * int(q) + int(x)
    return int(code)


def canonical_relabel(labels: Sequence[object]) -> np.ndarray:
    """Relabel arbitrary hashable labels to 0..m-1 in first-occurrence order."""
    mp: Dict[object, int] = {}
    out: List[int] = []
    for x in labels:
        if x not in mp:
            mp[x] = len(mp)
        out.append(mp[x])
    return np.asarray(out, dtype=np.int32)


def mutual_information_discrete(x: Sequence[int], y: Sequence[int]) -> float:
    if len(x) != len(y) or not x:
        return 0.0
    joint = Counter((int(a), int(b)) for a, b in zip(x, y))
    cx = Counter(int(a) for a in x)
    cy = Counter(int(b) for b in y)
    n = float(len(x))
    mi = 0.0
    for (a, b), c in joint.items():
        p = c / n
        pa = cx[a] / n
        pb = cy[b] / n
        if p > 0 and pa > 0 and pb > 0:
            mi += p * math.log2(p / (pa * pb))
    return float(max(0.0, mi))


def normalized_l1(a: Sequence[float], b: Sequence[float]) -> float:
    m = max(len(a), len(b))
    if m == 0:
        return 0.0
    aa = list(a) + [0.0] * (m - len(a))
    bb = list(b) + [0.0] * (m - len(b))
    return float(sum(abs(float(x) - float(y)) for x, y in zip(aa, bb)) / m)


# ---------------------------------------------------------------------------
# Deterministic arenas used only to generate boundary histories.
# ---------------------------------------------------------------------------
@dataclass
class ArenaDynamics:
    q: int
    k: int
    mode: str
    A: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    sys: Optional[OBG.FiniteRelationalSystem] = None

    def step(self, state: Sequence[int]) -> State:
        q = int(self.q)
        st = np.asarray(state, dtype=np.int64) % q
        if self.sys is not None:
            return tuple(int(x) for x in self.sys.step_parallel(tuple(int(v) for v in st)))
        if self.A is None:
            # Defensive fallback: identity.
            return tuple(int(x) for x in st)
        out = (self.A.dot(st) + (self.b if self.b is not None else 0)) % q
        return tuple(int(x) for x in out)

    def summary(self) -> Dict[str, object]:
        return {
            "q": int(self.q),
            "vertices": int(self.k),
            "rule_mode": str(self.mode),
            "uses_local_tables": bool(self.sys is not None),
        }


def _invertible_like_matrix(q: int, k: int, rng: np.random.Generator, density: float = 0.25) -> np.ndarray:
    """A sparse-ish matrix with a strong diagonal, not guaranteed invertible for composite q."""
    A = np.eye(k, dtype=np.int64)
    # ring terms give persistence and feedback without an observer topology.
    for i in range(k):
        A[i, (i - 1) % k] = (A[i, (i - 1) % k] + 1) % q
        if rng.random() < 0.5:
            A[i, (i + 1) % k] = (A[i, (i + 1) % k] + 1) % q
    for i in range(k):
        for j in range(k):
            if i != j and rng.random() < density:
                A[i, j] = (A[i, j] + int(rng.integers(0, q))) % q
    return A % q


def make_dynamics(q: int, k: int, mode: str, rng: np.random.Generator, max_pred: int = 3) -> ArenaDynamics:
    q = int(q); k = int(k); mode = str(mode).strip().lower()
    if mode in {"identity", "id"}:
        return ArenaDynamics(q, k, mode, A=np.eye(k, dtype=np.int64), b=np.zeros(k, dtype=np.int64))
    if mode in {"shift", "cycle_shift"}:
        A = np.zeros((k, k), dtype=np.int64)
        for i in range(k):
            A[i, (i - 1) % k] = 1
        return ArenaDynamics(q, k, mode, A=A, b=np.zeros(k, dtype=np.int64))
    if mode in {"shift_xor", "xor_shift", "additive_shift"}:
        A = np.zeros((k, k), dtype=np.int64)
        for i in range(k):
            A[i, i] = 1
            A[i, (i - 1) % k] = (A[i, (i - 1) % k] + 1) % q
        b = rng.integers(0, q, size=k, dtype=np.int64)
        return ArenaDynamics(q, k, mode, A=A % q, b=b)
    if mode in {"affine", "affine_mix", "permutive_mix", "linear_mix"}:
        A = _invertible_like_matrix(q, k, rng, density=0.20)
        b = rng.integers(0, q, size=k, dtype=np.int64)
        return ArenaDynamics(q, k, mode, A=A, b=b)
    if mode in {"dense_affine", "random_affine"}:
        A = _invertible_like_matrix(q, k, rng, density=0.55)
        b = rng.integers(0, q, size=k, dtype=np.int64)
        return ArenaDynamics(q, k, mode, A=A, b=b)
    if mode in {"random_table", "random_local"}:
        sys = OBG.make_random_system(q=q, n=k, edge_prob=min(0.6, max(0.15, max_pred / max(1, k - 1))), rng=rng, max_pred=max_pred)
        return ArenaDynamics(q, k, mode, sys=sys)
    if mode in {"componented_random", "componented"}:
        sys = OBG.make_componented_system(q=q, n=k, components=max(2, min(5, k // 2)), inter_prob=0.35, extra_intra_prob=0.35, rng=rng, max_pred=max_pred)
        return ArenaDynamics(q, k, mode, sys=sys)
    raise ValueError(f"unknown rule mode: {mode}")


# ---------------------------------------------------------------------------
# Boundary domains and proliferation
# ---------------------------------------------------------------------------
@dataclass
class BoundaryDomain:
    domain_id: int
    depth: int
    parent_id: int
    name: str
    labels: np.ndarray
    n_labels: int
    entropy_bits: float
    live_fiber_count: int = 0
    mean_predictive_classes_per_fiber: float = 1.0
    max_predictive_classes_per_fiber: int = 1
    mean_fiber_entropy_bits: float = 0.0
    active: bool = True


@dataclass
class ProliferationStep:
    iteration: int
    architecture_hash: str
    n_domains_total: int
    n_domains_current: int
    n_new_domains: int
    mean_domain_entropy_bits: float
    max_domain_entropy_bits: float
    mean_predictive_classes_per_fiber: float
    max_predictive_classes_per_fiber: int
    live_boundary_fraction: float
    dependency_edges: int
    dependency_edges_null_mean: float
    dependency_edges_above_null: float
    dependency_beta1: int
    dependency_mean_synergy_bits: float
    approximate_residual_from_previous: float
    repeated_hash: bool
    repeat_period: int
    nontriviality_score: float
    fixed_point_candidate: bool
    limit_cycle_candidate: bool


# ---------------------------------------------------------------------------
# State sampling and closure under dynamics
# ---------------------------------------------------------------------------
def sample_initial_states(q: int, k: int, max_state_samples: int, rng: np.random.Generator) -> List[State]:
    total = int(q) ** int(k)
    if total <= int(max_state_samples):
        return [tuple(int(x) for x in vals) for vals in itertools.product(range(q), repeat=k)]
    seen: Set[State] = set()
    out: List[State] = []
    attempts = 0
    while len(out) < int(max_state_samples) and attempts < int(max_state_samples) * 20:
        attempts += 1
        st = tuple(int(x) for x in rng.integers(0, q, size=k))
        if st not in seen:
            seen.add(st)
            out.append(st)
    return out


def build_state_closure(
    dyn: ArenaDynamics,
    initial: Sequence[State],
    steps: int,
    max_total_states: int,
) -> Tuple[List[State], Dict[State, int], List[int]]:
    """Closure under the deterministic update for a bounded number of steps."""
    states: List[State] = []
    idx: Dict[State, int] = {}
    q = deque()
    for st in initial:
        if st not in idx and len(states) < max_total_states:
            idx[st] = len(states); states.append(st); q.append((st, 0))
    ptr = 0
    while ptr < len(states):
        st = states[ptr]; ptr += 1
        # We do not need breadth depth here; simply add successor until budget.
        if len(states) >= max_total_states:
            continue
        nxt = dyn.step(st)
        if nxt not in idx:
            idx[nxt] = len(states); states.append(nxt)
    # Make sure multi-step futures of all sampled states are present if possible.
    for _ in range(max(0, int(steps))):
        current = list(states)
        for st in current:
            if len(states) >= max_total_states:
                break
            nxt = dyn.step(st)
            if nxt not in idx:
                idx[nxt] = len(states); states.append(nxt)
    next_idx: List[int] = []
    for st in states:
        nxt = dyn.step(st)
        if nxt not in idx:
            if len(states) < max_total_states:
                idx[nxt] = len(states); states.append(nxt)
                next_idx.append(idx[nxt])
            else:
                next_idx.append(idx[st])  # budget fallback self-loop
        else:
            next_idx.append(idx[nxt])
    # If adding successors extended states, finish their next pointers.
    while len(next_idx) < len(states):
        st = states[len(next_idx)]
        nxt = dyn.step(st)
        next_idx.append(idx.get(nxt, len(next_idx)))
    return states, idx, next_idx


def future_label_history(labels: np.ndarray, next_idx: Sequence[int], horizon: int, include_current: bool = True) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for i in range(len(labels)):
        cur = int(i)
        vals: List[int] = []
        if include_current:
            vals.append(int(labels[cur]))
        for _ in range(int(horizon)):
            cur = int(next_idx[cur])
            vals.append(int(labels[cur]))
        out.append(tuple(vals))
    return out


# ---------------------------------------------------------------------------
# Boundary maps
# ---------------------------------------------------------------------------
def initial_boundary_labels(states: Sequence[State], q: int, mode: str, rng: np.random.Generator, boundary_q: Optional[int] = None) -> np.ndarray:
    q = int(q)
    bq = int(boundary_q or q)
    mode = str(mode).strip().lower()
    arr = np.asarray(states, dtype=np.int64) % q
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.zeros(len(states), dtype=np.int32)
    if mode in {"first", "coordinate0", "coord0"}:
        labels = arr[:, 0] % bq
    elif mode in {"last", "coordinate_last", "coordlast"}:
        labels = arr[:, -1] % bq
    elif mode in {"sum", "sum_mod_q", "sum_mod"}:
        labels = arr.sum(axis=1) % bq
    elif mode in {"parity", "sum_mod_2"}:
        labels = arr.sum(axis=1) % 2
    elif mode in {"pair", "first_pair"}:
        labels = (arr[:, 0] * q + arr[:, 1 % arr.shape[1]]) % max(1, bq * bq)
    elif mode in {"random_linear", "random_partition"}:
        w = rng.integers(0, max(2, bq), size=arr.shape[1], dtype=np.int64)
        labels = arr.dot(w) % bq
    else:
        raise ValueError(f"unknown initial boundary mode: {mode}")
    return canonical_relabel([int(x) for x in labels])


def proliferate_children(
    parent: BoundaryDomain,
    next_idx: Sequence[int],
    horizon: int,
    min_live_classes: int,
    min_fiber_size: int,
    include_combined_child: bool = True,
) -> List[BoundaryDomain]:
    """Generate sub-boundaries from every live fiber of a parent boundary.

    Each live fiber b^{-1}(label) gets its own child boundary whose labels are:

        0 outside the fiber,
        1..m for predictive classes inside the fiber.

    This is closer to the theoretical proliferation rule than a single global
    refinement: live unresolved cells of the boundary become separate
    sub-boundary domains.  We optionally also include the combined quotient
    child as a coarse summary domain.
    """
    histories = future_label_history(parent.labels, next_idx, horizon=horizon, include_current=True)
    by_fiber: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(parent.labels):
        by_fiber[int(lab)].append(i)

    combined_raw: List[Tuple[int, int]] = [(int(lab), 0) for lab in parent.labels]
    children: List[BoundaryDomain] = []
    combined_live_fibers = 0
    combined_class_counts: List[int] = []
    combined_fiber_entropies: List[float] = []

    for lab, idxs in sorted(by_fiber.items()):
        if len(idxs) < int(min_fiber_size):
            combined_class_counts.append(1)
            combined_fiber_entropies.append(0.0)
            continue
        hist_to_class: Dict[Tuple[int, ...], int] = {}
        cls_counts: Counter = Counter()
        for ii in idxs:
            h = histories[ii]
            if h not in hist_to_class:
                hist_to_class[h] = len(hist_to_class)
            c = hist_to_class[h]
            combined_raw[ii] = (int(lab), int(c))
            cls_counts[c] += 1
        ncls = len(hist_to_class)
        combined_class_counts.append(ncls)
        combined_fiber_entropies.append(entropy_from_counts(cls_counts.values()))
        if ncls < int(min_live_classes):
            continue
        combined_live_fibers += 1

        # Local sub-boundary generated by this one live fiber.
        local = np.zeros(len(parent.labels), dtype=np.int32)
        for ii in idxs:
            local[ii] = int(hist_to_class[histories[ii]]) + 1
        local = canonical_relabel([int(x) for x in local])
        n_labels = int(len(set(int(x) for x in local)))
        ent = entropy_of_labels(local)
        children.append(BoundaryDomain(
            domain_id=-1,
            depth=int(parent.depth) + 1,
            parent_id=int(parent.domain_id),
            name=f"{parent.name}.fiber{lab}",
            labels=local,
            n_labels=n_labels,
            entropy_bits=float(ent),
            live_fiber_count=1,
            mean_predictive_classes_per_fiber=float(ncls),
            max_predictive_classes_per_fiber=int(ncls),
            mean_fiber_entropy_bits=float(entropy_from_counts(cls_counts.values())),
            active=True,
        ))

    if include_combined_child and combined_live_fibers > 0:
        child_labels = canonical_relabel(combined_raw)
        n_labels = int(len(set(int(x) for x in child_labels)))
        ent = entropy_of_labels(child_labels)
        children.append(BoundaryDomain(
            domain_id=-1,
            depth=int(parent.depth) + 1,
            parent_id=int(parent.domain_id),
            name=f"{parent.name}.fiberq",
            labels=child_labels,
            n_labels=n_labels,
            entropy_bits=float(ent),
            live_fiber_count=int(combined_live_fibers),
            mean_predictive_classes_per_fiber=float(np.mean(combined_class_counts) if combined_class_counts else 1.0),
            max_predictive_classes_per_fiber=int(max(combined_class_counts) if combined_class_counts else 1),
            mean_fiber_entropy_bits=float(np.mean(combined_fiber_entropies) if combined_fiber_entropies else 0.0),
            active=bool(n_labels > parent.n_labels),
        ))
    return children


# ---------------------------------------------------------------------------
# Dependency topology from predictive dependence
# ---------------------------------------------------------------------------
def pair_synergy_bits(a: np.ndarray, b: np.ndarray, target: Sequence[int]) -> Tuple[float, float, float, float]:
    aa = [int(x) for x in a]
    bb = [int(x) for x in b]
    tt = [int(x) for x in target]
    joint_ab = canonical_relabel(list(zip(aa, bb)))
    mi_a = mutual_information_discrete(aa, tt)
    mi_b = mutual_information_discrete(bb, tt)
    mi_joint = mutual_information_discrete([int(x) for x in joint_ab], tt)
    synergy = float(mi_joint - max(mi_a, mi_b))
    return float(max(0.0, synergy)), float(mi_joint), float(mi_a), float(mi_b)


def dependency_edges_for_domains(
    domains: Sequence[BoundaryDomain],
    target: Sequence[int],
    synergy_threshold: float,
) -> Tuple[List[Dict[str, object]], float, int]:
    edges: List[Dict[str, object]] = []
    synergies: List[float] = []
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            s, mij, mia, mib = pair_synergy_bits(domains[i].labels, domains[j].labels, target)
            synergies.append(s)
            if s >= float(synergy_threshold):
                edges.append({
                    "source_domain": domains[i].domain_id,
                    "target_domain": domains[j].domain_id,
                    "synergy_bits": float(s),
                    "joint_mi_bits": float(mij),
                    "source_mi_bits": float(mia),
                    "target_mi_bits": float(mib),
                })
    beta1 = graph_beta1([d.domain_id for d in domains], [(int(e["source_domain"]), int(e["target_domain"])) for e in edges])
    return edges, float(np.mean(synergies) if synergies else 0.0), int(beta1)


def graph_beta1(nodes: Sequence[int], edges: Sequence[Edge]) -> int:
    ns = list(dict.fromkeys(int(x) for x in nodes))
    if not ns:
        return 0
    parent = {x: x for x in ns}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    used_edges: Set[Tuple[int, int]] = set()
    for a, b in edges:
        if a == b or a not in parent or b not in parent:
            continue
        aa, bb = sorted((int(a), int(b)))
        used_edges.add((aa, bb))
        union(aa, bb)
    comps = len(set(find(x) for x in ns))
    return max(0, len(used_edges) - len(ns) + comps)


def null_dependency_edge_count(
    domains: Sequence[BoundaryDomain],
    target: Sequence[int],
    synergy_threshold: float,
    null_shuffles: int,
    rng: np.random.Generator,
) -> float:
    if null_shuffles <= 0 or len(domains) < 2:
        return 0.0
    counts: List[int] = []
    for _ in range(int(null_shuffles)):
        shuf_domains: List[BoundaryDomain] = []
        for d in domains:
            labs = np.asarray(d.labels).copy()
            rng.shuffle(labs)
            shuf_domains.append(BoundaryDomain(d.domain_id, d.depth, d.parent_id, d.name, labs, d.n_labels, d.entropy_bits))
        edges, _m, _b = dependency_edges_for_domains(shuf_domains, target, synergy_threshold)
        counts.append(len(edges))
    return float(np.mean(counts) if counts else 0.0)


# ---------------------------------------------------------------------------
# Canonical architecture signatures
# ---------------------------------------------------------------------------
def architecture_signature(domains: Sequence[BoundaryDomain], current_domains: Sequence[BoundaryDomain], dep_edges: Sequence[Dict[str, object]], dep_beta1: int) -> Dict[str, object]:
    label_spectrum = sorted(int(d.n_labels) for d in current_domains)
    entropy_spectrum = sorted(round(float(d.entropy_bits), 3) for d in current_domains)
    live_spectrum = sorted(int(d.live_fiber_count) for d in current_domains)
    pred_spectrum = sorted(round(float(d.mean_predictive_classes_per_fiber), 3) for d in current_domains)
    parent_degrees = Counter(int(d.parent_id) for d in current_domains)
    parent_degree_spectrum = sorted(int(v) for v in parent_degrees.values())
    return {
        "n_total": int(len(domains)),
        "n_current": int(len(current_domains)),
        "label_spectrum": label_spectrum,
        "entropy_spectrum": entropy_spectrum,
        "live_spectrum": live_spectrum,
        "predictive_class_spectrum": pred_spectrum,
        "parent_degree_spectrum": parent_degree_spectrum,
        "dependency_edges": int(len(dep_edges)),
        "dependency_beta1": int(dep_beta1),
    }


def architecture_hash(sig: Dict[str, object]) -> str:
    text = json.dumps(sig, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def signature_vector(sig: Dict[str, object]) -> List[float]:
    out: List[float] = [
        float(sig.get("n_current", 0)),
        float(sig.get("dependency_edges", 0)),
        float(sig.get("dependency_beta1", 0)),
    ]
    for key in ["label_spectrum", "entropy_spectrum", "live_spectrum", "predictive_class_spectrum", "parent_degree_spectrum"]:
        vals = [float(x) for x in sig.get(key, [])]
        out.extend([float(len(vals)), float(np.mean(vals) if vals else 0.0), float(max(vals) if vals else 0.0)])
    return out


def compute_nontriviality(current_domains: Sequence[BoundaryDomain], dep_edges: Sequence[Dict[str, object]], dep_beta1: int) -> float:
    if not current_domains:
        return 0.0
    mean_ent = float(np.mean([d.entropy_bits for d in current_domains]))
    max_ent = float(max(1e-9, max(d.entropy_bits for d in current_domains)))
    ent_score = min(1.0, mean_ent / max(1.0, max_ent))
    live_score = float(np.mean([1.0 if d.live_fiber_count > 0 else 0.0 for d in current_domains]))
    edge_score = min(1.0, len(dep_edges) / max(1.0, len(current_domains)))
    cycle_score = 1.0 if dep_beta1 > 0 else 0.0
    return float(0.35 * live_score + 0.30 * ent_score + 0.25 * edge_score + 0.10 * cycle_score)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_proliferation_instance(
    q: int,
    vertices: int,
    rule_mode: str,
    rng: np.random.Generator,
    iterations: int = 5,
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
    null_shuffles: int = 1,
    max_pred: int = 3,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    q = int(q); vertices = int(vertices)
    dyn = make_dynamics(q, vertices, rule_mode, rng, max_pred=max_pred)
    closure_steps = max(1, int(iterations) * (int(horizon) + 1) + 2)
    initial_states = sample_initial_states(q, vertices, max_state_samples, rng)
    states, state_to_idx, next_idx = build_state_closure(dyn, initial_states, closure_steps, max_total_states)
    labels0 = initial_boundary_labels(states, q, initial_boundary, rng, boundary_q=initial_boundary_q)
    root = BoundaryDomain(0, 0, -1, "root", labels0, int(len(set(labels0))), entropy_of_labels(labels0), active=True)
    domains: List[BoundaryDomain] = [root]
    current: List[BoundaryDomain] = [root]

    rows: List[Dict[str, object]] = []
    domain_rows: List[Dict[str, object]] = []
    dependency_rows: List[Dict[str, object]] = []

    seen_hash: Dict[str, int] = {}
    prev_sig_vec: Optional[List[float]] = None
    fixed_found = False
    cycle_found = False
    best_nontriv = 0.0
    min_residual = float("inf")

    for it in range(0, int(iterations) + 1):
        # Target for dependency topology: future history of the root boundary.
        root_future = canonical_relabel(future_label_history(root.labels, next_idx, horizon=horizon, include_current=False))
        dep_edges, mean_syn, beta1 = dependency_edges_for_domains(current, root_future, synergy_threshold)
        null_edges = null_dependency_edge_count(current, root_future, synergy_threshold, null_shuffles, rng)
        sig = architecture_signature(domains, current, dep_edges, beta1)
        h = architecture_hash(sig)
        sig_vec = signature_vector(sig)
        residual = normalized_l1(sig_vec, prev_sig_vec) if prev_sig_vec is not None else 0.0
        repeated = h in seen_hash
        repeat_period = int(it - seen_hash[h]) if repeated else 0
        seen_hash.setdefault(h, it)
        nontriv = compute_nontriviality(current, dep_edges, beta1)
        best_nontriv = max(best_nontriv, nontriv)
        if it > 0:
            min_residual = min(min_residual, residual)
        fixed = bool(repeated and repeat_period == 1 and nontriv >= 0.4)
        cyc = bool(repeated and repeat_period > 1 and nontriv >= 0.4)
        fixed_found = fixed_found or fixed
        cycle_found = cycle_found or cyc
        step = ProliferationStep(
            iteration=it,
            architecture_hash=h,
            n_domains_total=len(domains),
            n_domains_current=len(current),
            n_new_domains=0 if it == 0 else len(current),
            mean_domain_entropy_bits=float(np.mean([d.entropy_bits for d in current]) if current else 0.0),
            max_domain_entropy_bits=float(max([d.entropy_bits for d in current], default=0.0)),
            mean_predictive_classes_per_fiber=float(np.mean([d.mean_predictive_classes_per_fiber for d in current]) if current else 1.0),
            max_predictive_classes_per_fiber=int(max([d.max_predictive_classes_per_fiber for d in current], default=1)),
            live_boundary_fraction=float(np.mean([1.0 if d.live_fiber_count > 0 else 0.0 for d in current]) if current else 0.0),
            dependency_edges=len(dep_edges),
            dependency_edges_null_mean=float(null_edges),
            dependency_edges_above_null=float(len(dep_edges) - null_edges),
            dependency_beta1=int(beta1),
            dependency_mean_synergy_bits=float(mean_syn),
            approximate_residual_from_previous=float(residual),
            repeated_hash=bool(repeated),
            repeat_period=int(repeat_period),
            nontriviality_score=float(nontriv),
            fixed_point_candidate=bool(fixed),
            limit_cycle_candidate=bool(cyc),
        )
        row = {k: getattr(step, k) for k in step.__dataclass_fields__}
        row.update({
            "q": q,
            "vertices": vertices,
            "rule_mode": rule_mode,
            "initial_boundary": initial_boundary,
            "n_states": len(states),
            "arena_mode": dyn.mode,
            "signature_json": json.dumps(sig, sort_keys=True),
        })
        rows.append(row)
        for d in current:
            domain_rows.append({
                "iteration": it,
                "domain_id": d.domain_id,
                "depth": d.depth,
                "parent_id": d.parent_id,
                "name": d.name,
                "n_labels": d.n_labels,
                "entropy_bits": d.entropy_bits,
                "live_fiber_count": d.live_fiber_count,
                "mean_predictive_classes_per_fiber": d.mean_predictive_classes_per_fiber,
                "max_predictive_classes_per_fiber": d.max_predictive_classes_per_fiber,
                "mean_fiber_entropy_bits": d.mean_fiber_entropy_bits,
                "active": d.active,
            })
        for e in dep_edges:
            er = dict(e)
            er.update({"iteration": it, "q": q, "rule_mode": rule_mode})
            dependency_rows.append(er)
        prev_sig_vec = sig_vec

        if it >= int(iterations):
            break

        # Proliferate every active current domain.  Keep the highest-entropy
        # children if the fanout grows too large.
        children: List[BoundaryDomain] = []
        for parent in current:
            for child in proliferate_children(parent, next_idx, horizon=horizon, min_live_classes=min_live_classes, min_fiber_size=min_fiber_size):
                if child.active and child.entropy_bits >= min_entropy_bits:
                    child.domain_id = len(domains) + len(children)
                    children.append(child)
        children.sort(key=lambda d: (d.entropy_bits, d.live_fiber_count, d.n_labels), reverse=True)
        children = children[: int(max_domains_per_depth)]
        for c in children:
            c.domain_id = len(domains)
            domains.append(c)
        current = children
        if not current:
            # Once no fiber is live, the proliferation operator has reached an
            # inert architecture; continue one inert row and stop early.
            break

    summary = {
        "q": q,
        "vertices": vertices,
        "rule_mode": rule_mode,
        "initial_boundary": initial_boundary,
        "n_states": len(states),
        "iterations_requested": int(iterations),
        "iterations_completed": int(rows[-1]["iteration"] if rows else 0),
        "fixed_point_found": bool(fixed_found),
        "limit_cycle_found": bool(cycle_found),
        "best_nontriviality_score": float(best_nontriv),
        "min_architecture_residual": float(min_residual if math.isfinite(min_residual) else 0.0),
        "final_hash": str(rows[-1]["architecture_hash"] if rows else ""),
        "final_domains": int(rows[-1]["n_domains_current"] if rows else 0),
        "final_dependency_edges": int(rows[-1]["dependency_edges"] if rows else 0),
    }
    return rows, domain_rows, dependency_rows, summary


def parse_str_list(text: str) -> List[str]:
    return [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]


def run_boundary_proliferation_audit(
    q: int,
    vertices: int = 7,
    rule_modes: Sequence[str] = ("affine_mix", "shift_xor", "random_table"),
    instances: int = 10,
    iterations: int = 5,
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
    null_shuffles: int = 1,
    seed: int = 0,
    max_pred: int = 3,
    verbose: bool = True,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame", Dict[str, object]]:
    if pd is None:
        raise RuntimeError("pandas is required for boundaryproliferationaudit CLI")
    rows: List[Dict[str, object]] = []
    drows: List[Dict[str, object]] = []
    erows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    rule_modes = list(rule_modes)
    for mode in rule_modes:
        for inst in range(int(instances)):
            sub_seed = int(seed) + 1000003 * (rule_modes.index(mode) + 1) + 7919 * inst
            rng = np.random.default_rng(sub_seed)
            r, dr, er, summ = run_proliferation_instance(
                q=q,
                vertices=vertices,
                rule_mode=mode,
                rng=rng,
                iterations=iterations,
                horizon=horizon,
                max_state_samples=max_state_samples,
                max_total_states=max_total_states,
                initial_boundary=initial_boundary,
                initial_boundary_q=initial_boundary_q,
                max_domains_per_depth=max_domains_per_depth,
                min_live_classes=min_live_classes,
                min_fiber_size=min_fiber_size,
                min_entropy_bits=min_entropy_bits,
                synergy_threshold=synergy_threshold,
                null_shuffles=null_shuffles,
                max_pred=max_pred,
            )
            for rr in r:
                rr.update({"instance": inst, "seed": sub_seed})
            for rr in dr:
                rr.update({"instance": inst, "seed": sub_seed, "rule_mode": mode, "q": q})
            for rr in er:
                rr.update({"instance": inst, "seed": sub_seed})
            summ.update({"instance": inst, "seed": sub_seed})
            rows.extend(r); drows.extend(dr); erows.extend(er); summaries.append(summ)
            if verbose:
                print(
                    f"proliferation mode={mode} inst={inst+1}/{instances} "
                    f"fixed={int(summ['fixed_point_found'])} cycle={int(summ['limit_cycle_found'])} "
                    f"domains={summ['final_domains']} edges={summ['final_dependency_edges']} "
                    f"nontriv={summ['best_nontriviality_score']:.3f}"
                )
    df = pd.DataFrame(rows)
    dom_df = pd.DataFrame(drows)
    dep_df = pd.DataFrame(erows)

    observed = df[df["iteration"] > 0] if not df.empty else df
    by_mode: List[Dict[str, object]] = []
    if not observed.empty:
        for mode, g in observed.groupby("rule_mode"):
            last = g.sort_values("iteration").groupby("instance").tail(1)
            by_mode.append({
                "rule_mode": str(mode),
                "n_instances": int(last["instance"].nunique()),
                "fixed_point_fraction": float(last["fixed_point_candidate"].mean()),
                "limit_cycle_fraction": float(last["limit_cycle_candidate"].mean()),
                "mean_final_domains": float(last["n_domains_current"].mean()),
                "mean_final_dependency_edges": float(last["dependency_edges"].mean()),
                "mean_final_dependency_edges_above_null": float(last["dependency_edges_above_null"].mean()),
                "mean_final_beta1": float(last["dependency_beta1"].mean()),
                "mean_best_nontriviality": float(pd.DataFrame([s for s in summaries if s.get("rule_mode") == mode])["best_nontriviality_score"].mean()) if summaries else 0.0,
                "mean_final_residual": float(last["approximate_residual_from_previous"].mean()),
            })
    fixed_any = bool(df.get("fixed_point_candidate", pd.Series(dtype=bool)).any()) if not df.empty else False
    cycle_any = bool(df.get("limit_cycle_candidate", pd.Series(dtype=bool)).any()) if not df.empty else False
    topology_any = bool((df.get("dependency_edges_above_null", pd.Series(dtype=float)) > 0).any()) if not df.empty else False
    live_any = bool((df.get("live_boundary_fraction", pd.Series(dtype=float)) > 0).any()) if not df.empty else False
    if fixed_any:
        verdict = "BOUNDARY PROLIFERATION FIXED-POINT SIGNAL: live fibers regenerate a canonical observer architecture"
    elif cycle_any:
        verdict = "BOUNDARY PROLIFERATION LIMIT-CYCLE SIGNAL: live fibers enter a canonical architecture cycle"
    elif topology_any and live_any:
        verdict = "BOUNDARY PROLIFERATION TOPOLOGY SIGNAL: live fibers generate dependency topology without fixed point yet"
    elif live_any:
        verdict = "BOUNDARY PROLIFERATION FIBER SIGNAL: live predictive fibers found, dependency topology weak"
    else:
        verdict = "BOUNDARY PROLIFERATION NEGATIVE/INERT: no live predictive fibers found under tested settings"

    summary: Dict[str, object] = {
        "verdict": verdict,
        "audit_version": "boundary_proliferation_v1_live_fiber_predictive_topology",
        "q": int(q),
        "vertices": int(vertices),
        "rule_modes": list(rule_modes),
        "instances": int(instances),
        "iterations": int(iterations),
        "horizon": int(horizon),
        "max_state_samples": int(max_state_samples),
        "initial_boundary": str(initial_boundary),
        "initial_boundary_q": int(initial_boundary_q) if initial_boundary_q is not None else int(q),
        "n_rows": int(len(df)),
        "n_domain_rows": int(len(dom_df)),
        "n_dependency_rows": int(len(dep_df)),
        "any_fixed_point_candidate": fixed_any,
        "any_limit_cycle_candidate": cycle_any,
        "any_live_fiber": live_any,
        "any_dependency_topology_above_null": topology_any,
        "max_nontriviality_score": float(df["nontriviality_score"].max()) if not df.empty else 0.0,
        "max_dependency_beta1": int(df["dependency_beta1"].max()) if not df.empty else 0,
        "max_dependency_edges_above_null": float(df["dependency_edges_above_null"].max()) if not df.empty else 0.0,
        "by_mode": by_mode,
    }
    return df, dom_df, dep_df, summary


# ---------------------------------------------------------------------------
# I/O and plotting
# ---------------------------------------------------------------------------
def maybe_write_csv(df, path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path}")


def maybe_plot(df, path: str) -> None:
    if not path or df.empty:
        return
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    for mode, g in df.groupby("rule_mode"):
        gg = g.groupby("iteration").agg({
            "n_domains_current": "mean",
            "dependency_edges_above_null": "mean",
            "nontriviality_score": "mean",
        }).reset_index()
        ax1.plot(gg["iteration"], gg["n_domains_current"], marker="o", label=f"{mode}: domains")
    ax1.set_xlabel("Proliferation iteration")
    ax1.set_ylabel("Mean current generated boundaries")
    ax2 = ax1.twinx()
    for mode, g in df.groupby("rule_mode"):
        gg = g.groupby("iteration").agg({"dependency_edges_above_null": "mean"}).reset_index()
        ax2.plot(gg["iteration"], gg["dependency_edges_above_null"], linestyle="--", alpha=0.6, label=f"{mode}: edges-null")
    ax2.set_ylabel("Mean dependency edges above null")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")
    ax1.set_title("Boundary proliferation: live fibers -> generated dependency topology")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Boundary proliferation operator audit")
    ap.add_argument("q", type=int, help="microscopic alphabet size for the finite arena")
    ap.add_argument("--vertices", type=int, default=7)
    ap.add_argument("--rule-modes", default="affine_mix,shift_xor,random_table")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--max-state-samples", type=int, default=512)
    ap.add_argument("--max-total-states", type=int, default=4096)
    ap.add_argument("--initial-boundary", default="sum_mod_q", choices=["first", "last", "sum_mod_q", "sum", "parity", "pair", "random_linear", "random_partition"])
    ap.add_argument("--initial-boundary-q", type=int, default=0, help="boundary alphabet for the initial map; 0 means q")
    ap.add_argument("--max-domains-per-depth", type=int, default=32)
    ap.add_argument("--min-live-classes", type=int, default=2)
    ap.add_argument("--min-fiber-size", type=int, default=2)
    ap.add_argument("--min-entropy-bits", type=float, default=0.05)
    ap.add_argument("--synergy-threshold", type=float, default=0.01)
    ap.add_argument("--null-shuffles", type=int, default=1)
    ap.add_argument("--max-pred", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df, dom_df, dep_df, summary = run_boundary_proliferation_audit(
        q=int(args.q),
        vertices=int(args.vertices),
        rule_modes=parse_str_list(args.rule_modes),
        instances=int(args.instances),
        iterations=int(args.iterations),
        horizon=int(args.horizon),
        max_state_samples=int(args.max_state_samples),
        max_total_states=int(args.max_total_states),
        initial_boundary=str(args.initial_boundary),
        initial_boundary_q=(int(args.initial_boundary_q) if int(args.initial_boundary_q) > 0 else None),
        max_domains_per_depth=int(args.max_domains_per_depth),
        min_live_classes=int(args.min_live_classes),
        min_fiber_size=int(args.min_fiber_size),
        min_entropy_bits=float(args.min_entropy_bits),
        synergy_threshold=float(args.synergy_threshold),
        null_shuffles=int(args.null_shuffles),
        seed=int(args.seed),
        max_pred=int(args.max_pred),
        verbose=not bool(args.quiet),
    )
    if args.plot:
        maybe_plot(df, args.plot)
    print(json.dumps(summary, indent=2))
    if args.out:
        stem, ext = os.path.splitext(args.out)
        maybe_write_csv(df, args.out)
        maybe_write_csv(dom_df, stem + "_domains.csv")
        maybe_write_csv(dep_df, stem + "_dependencies.csv")
        with open(stem + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {stem + '_summary.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()
