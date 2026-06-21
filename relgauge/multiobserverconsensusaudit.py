"""
multiobserverconsensusaudit.py

Multi-observer consensus audit for generated q=2 finite-atlas critical events.

Motivation
----------
A same-namespace spatial S3 witness was found in the q=2 v9 atlas branch:
two C2 chart-cycle flips, e.g. (0 1) and (0 2), overlapped in one local
parent-domain/fiber namespace and generated S3.  Subsequent pulse-train audits
showed this was a one-tick local critical flash.  That leaves a key question:

    Is the S3 structural in the frozen boundary transition table, or was it an
    artifact of one particular atlas/chart construction?

This module tests that by reconstructing a target candidate up to a target
iteration, freezing the transition table, then rebuilding the atlas many times
with independent atlas-construction RNG seeds.  It asks whether independent
observer atlases detect S3 in the same target namespace, or elsewhere.

Interpretation
--------------
- target_exact_s3_consensus_fraction high:
    independent observers agree on S3 at the same boundary namespace.  This is
    evidence for structural/matter-like consensus.

- any_exact_s3_fraction high but target fraction low:
    S3 is present as an observer-dependent decomposition elsewhere; structural
    location is not certified.

- both low:
    the original S3 was likely window/atlas-construction dependent, not a
    robust structural feature of the frozen transition table.

The module does not insert S3; it classifies the chart-cycle loop maps produced
by each independently reconstructed atlas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

try:
    import numpy as np
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("multiobserverconsensusaudit requires numpy and pandas") from e

try:  # pragma: no cover - available inside relgauge package
    from . import binarycompositegaugespectrumaudit as BCG
    from . import dynamicsconsistencyfixedpointaudit as DCFP
    from . import fiberchartconnectionaudit as FCA
    from . import generatedcandidatephysicsreplayaudit as GCPR
except Exception:  # pragma: no cover
    try:
        import binarycompositegaugespectrumaudit as BCG  # type: ignore
        import dynamicsconsistencyfixedpointaudit as DCFP  # type: ignore
        import fiberchartconnectionaudit as FCA  # type: ignore
        import generatedcandidatephysicsreplayaudit as GCPR  # type: ignore
    except Exception:
        BCG = None  # type: ignore
        DCFP = None  # type: ignore
        FCA = None  # type: ignore
        GCPR = None  # type: ignore

LabelMap = Dict[int, int]
Perm = Tuple[int, ...]

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


def _stable_hash(text: Any, digits: int = 8) -> int:
    return int(hashlib.sha1(str(text).encode("utf-8")).hexdigest()[: int(digits)], 16)


def _parse_loop_map(text: Any) -> LabelMap:
    if text is None:
        return {}
    try:
        if pd.isna(text):
            return {}
    except Exception:
        pass
    s = str(text).strip()
    if not s or s.lower() in {"nan", "none"}:
        return {}
    out: Dict[int, int] = {}
    for a, b in re.findall(r"(-?\d+)\s*->\s*(-?\d+)", s):
        out[int(a)] = int(b)
    if out:
        return out
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {int(k): int(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def _compact_label_map(m: LabelMap, limit: int = 40) -> str:
    items = sorted((int(k), int(v)) for k, v in m.items())
    out = " ".join(f"{a}->{b}" for a, b in items[: int(limit)])
    if len(items) > int(limit):
        out += " ..."
    return out


def _candidate_id_from_row(row: Dict[str, Any]) -> str:
    if row.get("candidate_id") not in (None, ""):
        try:
            if not pd.isna(row.get("candidate_id")):
                return str(row.get("candidate_id"))
        except Exception:
            return str(row.get("candidate_id"))
    rm = str(row.get("rule_mode", "mode"))
    inst = str(_safe_int(row.get("instance"), 0))
    prof = str(row.get("profile", "profile"))
    cap = str(_safe_int(row.get("atlas_capacity", row.get("capacity")), 0))
    seed = str(_safe_int(row.get("initial_seed", row.get("seed")), 0))
    return f"{rm}|inst={inst}|{prof}|cap={cap}|seed={seed}"


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


def _ensure_upstream_defaults(args: Any) -> Any:
    """Supply fields consumed by the upstream atlas replay modules."""
    defaults = {
        "atlas_lift_mode": "bijective",
        "proliferation_iterations": 4,
        "horizon": 3,
        "max_pred": 0,
        "max_total_states": 200000,
        "max_state_samples": 512,
        "min_fiber_states": 2,
        "min_support_states": 4,
        "min_overlap_states": 4,
        "max_cycle_len": 4,
        "max_cycles_per_fiber": 500,
        "max_chart_coords": 3,
        "max_support_coords": 3,
        "max_charts_per_fiber": 12,
        "min_chart_classes": 2,
        "min_chart_entropy": 0.0,
        "min_chart_transition_determinism": 0.98,
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
        "max_signature_charts": 48,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args

# ---------------------------------------------------------------------------
# Permutation/group helpers for transposition diversity
# ---------------------------------------------------------------------------
def _perm_from_label_map(m: LabelMap, labels: Sequence[int]) -> Optional[Perm]:
    labs = [int(x) for x in labels]
    idx = {x: i for i, x in enumerate(labs)}
    arr: List[int] = []
    for x in labs:
        if x not in m:
            return None
        y = int(m[x])
        if y not in idx:
            return None
        arr.append(idx[y])
    if len(set(arr)) != len(arr):
        return None
    return tuple(arr)


def _compose(p: Perm, q: Perm) -> Perm:
    return tuple(p[q[i]] for i in range(len(p)))


def _perm_inverse(p: Perm) -> Perm:
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def _perm_order(p: Perm, max_order: int = 100000) -> int:
    if BCG is not None:
        try:
            return int(BCG._perm_order(p))
        except Exception:
            pass
    n = len(p)
    seen = [False] * n
    lcm = 1
    for i in range(n):
        if seen[i]:
            continue
        cur = i
        length = 0
        while not seen[cur]:
            seen[cur] = True
            cur = p[cur]
            length += 1
            if length > max_order:
                return max_order + 1
        if length > 0:
            lcm = lcm * length // math.gcd(lcm, length)
            if lcm > max_order:
                return max_order + 1
    return int(lcm)


def _close_group(gens: Sequence[Perm], max_group_order: int = 4096) -> Tuple[Set[Perm], bool]:
    if BCG is not None:
        try:
            return BCG._close_group(list(gens), max_group_order=int(max_group_order))
        except Exception:
            pass
    if not gens:
        return set(), False
    n = len(gens[0])
    ident = tuple(range(n))
    full_gens: List[Perm] = []
    for g in gens:
        if len(g) != n:
            continue
        if g not in full_gens:
            full_gens.append(g)
        inv = _perm_inverse(g)
        if inv not in full_gens:
            full_gens.append(inv)
    group: Set[Perm] = {ident}
    dq: deque[Perm] = deque([ident])
    while dq:
        a = dq.popleft()
        for g in full_gens:
            for h in (_compose(g, a), _compose(a, g)):
                if h not in group:
                    group.add(h)
                    if len(group) > int(max_group_order):
                        return group, True
                    dq.append(h)
    return group, False


def _transposition_edges(m: LabelMap) -> List[Tuple[int, int]]:
    labs = sorted(set(map(int, m.keys())) | set(map(int, m.values())))
    edges: List[Tuple[int, int]] = []
    used: Set[int] = set()
    for a in labs:
        if a in used:
            continue
        b = int(m.get(a, a))
        if b != a and int(m.get(b, b)) == a:
            edges.append(tuple(sorted((int(a), int(b)))))
            used.add(a); used.add(b)
    return sorted(set(edges))


def _components_from_edges(edges: Sequence[Tuple[int, int]]) -> List[Set[int]]:
    adj: Dict[int, Set[int]] = defaultdict(set)
    for a, b in edges:
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    seen: Set[int] = set()
    comps: List[Set[int]] = []
    for x in sorted(adj):
        if x in seen:
            continue
        stack = [x]
        comp: Set[int] = set()
        seen.add(x)
        while stack:
            y = stack.pop(); comp.add(y)
            for z in adj[y]:
                if z not in seen:
                    seen.add(z); stack.append(z)
        comps.append(comp)
    return comps


def _classify_maps(maps: Sequence[LabelMap], max_group_order: int = 4096) -> Dict[str, Any]:
    distinct: List[LabelMap] = []
    seen = set()
    for m in maps:
        key = tuple(sorted((int(k), int(v)) for k, v in m.items()))
        if key not in seen:
            seen.add(key); distinct.append({int(k): int(v) for k, v in m.items()})
    all_edges: List[Tuple[int, int]] = []
    for m in distinct:
        all_edges.extend(_transposition_edges(m))
    all_edges = sorted(set(all_edges))
    comps = _components_from_edges(all_edges)
    max_comp = max([len(c) for c in comps], default=0)
    shared_pair = False
    disjoint_pair = False
    for i, e1 in enumerate(all_edges):
        for e2 in all_edges[i + 1:]:
            inter = set(e1) & set(e2)
            if len(inter) == 1:
                shared_pair = True
            if len(inter) == 0:
                disjoint_pair = True

    labels = sorted(set(x for m in distinct for kv in m.items() for x in kv))
    perms: List[Perm] = []
    if labels:
        for m in distinct:
            extended = {x: x for x in labels}
            extended.update(m)
            p = _perm_from_label_map(extended, labels)
            if p is not None:
                perms.append(p)
    group: Set[Perm] = set()
    trunc = False
    if perms:
        group, trunc = _close_group(perms, max_group_order=max_group_order)
    exact_s3 = bool((not trunc) and len(group) == 6 and max_comp == 3 and shared_pair)
    nonabelian = False
    if group and not trunc:
        glist = list(group)
        nonabelian = any(_compose(a, b) != _compose(b, a) for a in glist for b in glist)
    if exact_s3:
        fam = "S3_consensus"
    elif max_comp >= 4:
        fam = "Sn_ge4_consensus"
    elif len(distinct) >= 2 and disjoint_pair and not shared_pair:
        fam = "disjoint_C2_product"
    elif len(distinct) >= 1:
        fam = "C2_single_or_duplicate"
    else:
        fam = "flat_or_absent"
    return {
        "n_distinct_c2_maps": int(len(distinct)),
        "n_transposition_edges": int(len(all_edges)),
        "transposition_edges": ";".join(f"{a}-{b}" for a, b in all_edges),
        "max_transposition_component_size": int(max_comp),
        "component_size_histogram": ";".join(f"{k}:{v}" for k, v in sorted(Counter(len(c) for c in comps).items())),
        "shared_label_transposition_pair": bool(shared_pair),
        "disjoint_transposition_pair": bool(disjoint_pair),
        "generated_group_order": int(len(group)),
        "group_truncated": bool(trunc),
        "exact_s3_group_closure": bool(exact_s3),
        "nonabelian_group_closure": bool(nonabelian),
        "algebra_family": fam,
        "loop_maps": "; ".join(_compact_label_map(m) for m in distinct),
    }


# ---------------------------------------------------------------------------
# Support helpers for observer-independent matching
# ---------------------------------------------------------------------------
def _find_domain(atlas: Any, parent_domain_id: int) -> Any:
    for coll_name in ("domains_current", "domains_all"):
        for d in list(getattr(atlas, coll_name, []) or []):
            try:
                if int(getattr(d, "domain_id", -999999)) == int(parent_domain_id):
                    return d
            except Exception:
                pass
    return None


def _support_for_namespace(atlas: Any, parent_domain_id: int, fiber_label: int) -> Set[int]:
    d = _find_domain(atlas, int(parent_domain_id))
    if d is None:
        return set()
    labels = getattr(d, "labels", None)
    if labels is None:
        return set()
    try:
        arr = np.asarray(labels, dtype=np.int64)
    except Exception:
        return set()
    return {int(i) for i, x in enumerate(arr) if int(x) == int(fiber_label)}


def _domain_fiber_supports(atlas: Any, min_size: int = 1) -> Dict[Tuple[int, int], Set[int]]:
    out: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    seen_domains: Set[int] = set()
    for coll_name in ("domains_current", "domains_all"):
        for d in list(getattr(atlas, coll_name, []) or []):
            try:
                did = int(getattr(d, "domain_id"))
            except Exception:
                continue
            # domains_current is usually a subset of domains_all; do not double-count.
            if did in seen_domains:
                continue
            seen_domains.add(did)
            labels = getattr(d, "labels", None)
            if labels is None:
                continue
            try:
                arr = np.asarray(labels, dtype=np.int64)
            except Exception:
                continue
            for fib in sorted(set(int(x) for x in arr.tolist())):
                supp = {int(i) for i, x in enumerate(arr) if int(x) == int(fib)}
                if len(supp) >= int(min_size):
                    out[(did, int(fib))] |= supp
    return dict(out)


def _support_hash(support: Iterable[int]) -> str:
    vals = np.asarray(sorted(int(x) for x in support), dtype=np.int64)
    if vals.size == 0:
        return ""
    return hashlib.sha1(vals.tobytes()).hexdigest()[:20]


def _support_sample(support: Iterable[int], limit: int = 16) -> str:
    xs = sorted(int(x) for x in support)
    if len(xs) > int(limit):
        return " ".join(str(x) for x in xs[: int(limit)]) + " ..."
    return " ".join(str(x) for x in xs)


def _overlap_stats(target: Set[int], observer: Set[int]) -> Dict[str, Any]:
    target = set(int(x) for x in target)
    observer = set(int(x) for x in observer)
    inter = target & observer
    union = target | observer
    return {
        "overlap_count": int(len(inter)),
        "target_overlap_fraction": float(len(inter) / max(1, len(target))),
        "observer_overlap_fraction": float(len(inter) / max(1, len(observer))),
        "jaccard": float(len(inter) / max(1, len(union))),
    }


def _parse_support_indices(text: Any) -> Set[int]:
    if text is None:
        return set()
    s = str(text).strip()
    if not s:
        return set()
    out: Set[int] = set()
    for tok in re.split(r"[\s,;]+", s):
        if not tok:
            continue
        try:
            out.add(int(float(tok)))
        except Exception:
            pass
    return out


def _load_target_support_from_args(args: Any) -> Set[int]:
    support: Set[int] = set()
    path = str(getattr(args, "target_support_npy", "") or "").strip()
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"--target-support-npy not found: {path}")
        arr = np.asarray(np.load(path)).reshape(-1)
        # Accept bool masks or integer-index arrays.  Treat 0/1 vector of same
        # kind as a mask only when it is longer than two entries and all values
        # are binary.
        uniq = set(np.unique(arr).tolist()) if arr.size else set()
        if arr.dtype == np.bool_ or (arr.size > 2 and uniq.issubset({0, 1, False, True})):
            support |= {int(i) for i, v in enumerate(arr) if bool(v)}
        else:
            support |= {int(x) for x in arr.astype(np.int64).tolist()}
    csv_path = str(getattr(args, "target_support_csv", "") or "").strip()
    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"--target-support-csv not found: {csv_path}")
        df = pd.read_csv(csv_path)
        col = None
        for c in ["state_index", "microstate_index", "index", "i"]:
            if c in df.columns:
                col = c; break
        if col is None and len(df.columns):
            col = df.columns[0]
        if col is not None:
            support |= {int(x) for x in pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist()}
    support |= _parse_support_indices(getattr(args, "target_support_indices", ""))
    return support

# ---------------------------------------------------------------------------
# Candidate reconstruction and atlas observer resampling
# ---------------------------------------------------------------------------
def _select_target_row(df: pd.DataFrame, args: Any) -> Dict[str, Any]:
    work = df.copy()
    # Construct candidate ids if absent.
    if "candidate_id" not in work.columns:
        work["candidate_id"] = work.apply(lambda r: _candidate_id_from_row(r.to_dict()), axis=1)
    mask = pd.Series(True, index=work.index)
    if str(args.target_candidate).strip():
        mask &= work["candidate_id"].astype(str).eq(str(args.target_candidate).strip())
    if str(args.target_rule_mode).strip():
        mask &= work.get("rule_mode", "").astype(str).eq(str(args.target_rule_mode).strip())
    if int(args.target_instance) >= 0 and "instance" in work.columns:
        mask &= pd.to_numeric(work["instance"], errors="coerce").fillna(-999999).astype(int).eq(int(args.target_instance))
    if str(args.target_profile).strip() and "profile" in work.columns:
        mask &= work["profile"].astype(str).eq(str(args.target_profile).strip())
    if int(args.target_atlas_capacity) > 0 and "atlas_capacity" in work.columns:
        mask &= pd.to_numeric(work["atlas_capacity"], errors="coerce").fillna(-999999).astype(int).eq(int(args.target_atlas_capacity))
    if int(args.target_seed) >= 0:
        seed_col = "initial_seed" if "initial_seed" in work.columns else ("seed" if "seed" in work.columns else "")
        if seed_col:
            mask &= pd.to_numeric(work[seed_col], errors="coerce").fillna(-999999).astype(int).eq(int(args.target_seed))
    if "atlas_iteration" in work.columns:
        # Keep all candidate rows but prefer target iteration or candidate summary rows.
        pass
    cand = work[mask].copy()
    if cand.empty:
        raise ValueError("No candidate row matched target filters")
    # Prefer rows at or before target iteration if present; otherwise first.
    if "atlas_iteration" in cand.columns:
        cand2 = cand[pd.to_numeric(cand["atlas_iteration"], errors="coerce").fillna(-1).astype(int) == int(args.target_iteration)]
        if not cand2.empty:
            cand = cand2
    return cand.iloc[0].to_dict()


def _rng_for_candidate_iteration(seed: int, capacity: int, profile: str, iteration: int, mode: str = "stable") -> np.random.Generator:
    # Deterministic on every Python process. The legacy Python-hash path is intentionally disabled.
    ph = _stable_hash(profile) % 1000
    return np.random.default_rng(int(seed) + 1299709 * int(capacity) + 15485863 * int(iteration) + 104729 * int(ph))


def _reconstruct_frozen_transition(row: Dict[str, Any], q: int, vertices: int, target_iteration: int, args: Any) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """Return sampled states and the frozen transition table for the target.

    If --frozen-transition-npy is provided, load that exact transition table and
    bypass atlas replay.  This is the preferred path for multi-observer tests,
    because it avoids replay ambiguity and Python hash randomization in old
    runs.  States are still reconstructed from the candidate metadata; for the
    q=2, v=9/10 full-state runs this is deterministic because all states fit in
    max_state_samples.
    """
    if DCFP is None or GCPR is None:
        raise RuntimeError("relgauge replay modules are required for non-synthetic runs")
    mode = str(row.get("rule_mode"))
    profile = str(row.get("profile", "fiber_preserving"))
    capacity = _safe_int(row.get("atlas_capacity", row.get("capacity")), 0)
    seed = _safe_int(row.get("initial_seed", row.get("seed")), 0)
    frozen_path = str(getattr(args, "frozen_transition_npy", "") or "").strip()

    # If a frozen transition was saved directly by iteratedfiberatlasdynamicsaudit,
    # use its length to ensure the sampled state reservoir has matching size.
    loaded_next: Optional[np.ndarray] = None
    if frozen_path:
        if not os.path.exists(frozen_path):
            raise FileNotFoundError(f"--frozen-transition-npy not found: {frozen_path}")
        loaded_next = np.asarray(np.load(frozen_path), dtype=np.int64)
        if loaded_next.ndim != 1:
            raise ValueError(f"frozen transition must be a 1D array, got shape {loaded_next.shape}")
        if int(getattr(args, "max_state_samples", 0)) < len(loaded_next):
            setattr(args, "max_state_samples", int(len(loaded_next)))

    rng0 = np.random.default_rng(seed)
    states, current_next, init_meta = DCFP.initialize_sampled_transition(
        q=int(q), vertices=int(vertices), mode=mode, rng=rng0,
        max_state_samples=int(args.max_state_samples), max_total_states=int(args.max_total_states), max_pred=int(args.max_pred),
        proliferation_iterations=int(args.proliferation_iterations), horizon=int(args.horizon),
    )
    current_next = np.asarray(current_next, dtype=np.int64)
    if loaded_next is not None:
        if len(states) != len(loaded_next):
            raise ValueError(
                f"state reservoir length {len(states)} does not match frozen transition length {len(loaded_next)}. "
                "Use the same q/vertices/max-state-samples as the saved transition run."
            )
        current_next = loaded_next.copy()
        frozen_source = "npy"
    else:
        # Advance to the input transition table for target_iteration.
        for it in range(int(target_iteration)):
            rng = _rng_for_candidate_iteration(seed, capacity, profile, it, mode=str(args.replay_rng_hash_mode))
            _atlas, _bounded, _pstats, _rel_stats, _rel_rows, eff, _lift_stats, _eff_rows = GCPR._advance_effective(
                states, current_next, int(q), profile, int(capacity), rng, args, str(args.atlas_lift_mode)
            )
            current_next = np.asarray(eff, dtype=np.int64)
        frozen_source = "replay"
    meta = {
        "rule_mode": mode,
        "profile": profile,
        "atlas_capacity": int(capacity),
        "initial_seed": int(seed),
        "candidate_id": _candidate_id_from_row(row),
        "frozen_transition_source": frozen_source,
        "frozen_transition_npy": frozen_path,
    }
    return states, current_next, meta


def _analyze_atlas_chart_cycles(
    atlas: Any,
    target_parent: int,
    target_fiber: int,
    max_cycles: int,
    max_group_order: int,
    target_support: Optional[Set[int]] = None,
    min_support_overlap_fraction: float = 0.5,
    min_support_jaccard: float = 0.05,
    min_support_overlap_states: int = 1,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Classify exact target namespace and support-overlap consensus groups.

    Exact target matching uses observer-internal coordinates
    (parent_domain_id, fiber_label).  Support matching compares microstate sets
    to the reference target S3 orbit support, so independent observers may use
    different domain/fiber names while still describing the same boundary
    region.
    """
    supports = _domain_fiber_supports(atlas, min_size=1)
    groups: Dict[Tuple[int, int], List[LabelMap]] = defaultdict(list)
    raw_counts: Counter = Counter()
    for cr in list(getattr(atlas, "chart_cycle_rows", [])):
        lm = _parse_loop_map(cr.get("loop_map", "") if isinstance(cr, dict) else getattr(cr, "loop_map", ""))
        if not lm:
            continue
        flag = cr.get("chart_c2_holonomy", False) if isinstance(cr, dict) else False
        if not _safe_bool(flag):
            labs0 = sorted(set(lm.keys()) | set(lm.values()))
            p0 = _perm_from_label_map(lm, labs0)
            if p0 is None or _perm_order(p0) != 2:
                continue
        parent = _safe_int(cr.get("parent_domain_id", -1) if isinstance(cr, dict) else -1, -1)
        fib = _safe_int(cr.get("fiber_label", -1) if isinstance(cr, dict) else -1, -1)
        groups[(parent, fib)].append(lm)
        raw_counts[(parent, fib)] += 1

    # Include flat support-overlap namespaces in group rows too, not only C2
    # namespaces.  This lets the audit distinguish "same support recovered but
    # flat" from "support not recovered".
    all_keys: Set[Tuple[int, int]] = set(groups.keys()) | set(supports.keys())
    group_rows: List[Dict[str, Any]] = []
    max_comp = 0
    max_order = 0
    any_s3 = False
    any_shared = False
    any_two_plus = False
    target_result: Dict[str, Any] = {}

    target_support_set: Set[int] = set(target_support or set())
    support_best: Optional[Dict[str, Any]] = None
    support_overlap_rows: List[Dict[str, Any]] = []

    for (parent, fib) in sorted(all_keys):
        maps = groups.get((parent, fib), [])
        maps_use = maps[: int(max_cycles)] if int(max_cycles) > 0 else maps
        cls = _classify_maps(maps_use, max_group_order=max_group_order) if maps_use else {
            "n_distinct_c2_maps": 0,
            "n_transposition_edges": 0,
            "transposition_edges": "",
            "max_transposition_component_size": 0,
            "component_size_histogram": "",
            "shared_label_transposition_pair": False,
            "disjoint_transposition_pair": False,
            "generated_group_order": 0,
            "group_truncated": False,
            "exact_s3_group_closure": False,
            "nonabelian_group_closure": False,
            "algebra_family": "flat_or_absent",
            "loop_maps": "",
        }
        supp = supports.get((parent, fib), set())
        row = {
            "parent_domain_id": int(parent),
            "fiber_label": int(fib),
            "n_raw_c2_rows": int(raw_counts.get((parent, fib), 0)),
            "support_size": int(len(supp)),
            "support_hash": _support_hash(supp),
            "support_sample": _support_sample(supp),
            **cls,
        }
        if target_support_set:
            ov = _overlap_stats(target_support_set, set(supp))
            row.update({
                "target_support_overlap_count": ov["overlap_count"],
                "target_support_jaccard": ov["jaccard"],
                "target_support_overlap_fraction": ov["target_overlap_fraction"],
                "observer_support_overlap_fraction": ov["observer_overlap_fraction"],
            })
            passes = bool(
                ov["overlap_count"] >= int(min_support_overlap_states)
                and (
                    ov["target_overlap_fraction"] >= float(min_support_overlap_fraction)
                    or ov["jaccard"] >= float(min_support_jaccard)
                )
            )
            row["support_overlap_match"] = bool(passes)
            # Track best by target fraction, then jaccard, then count, then S3.
            score_tuple = (
                float(ov["target_overlap_fraction"]),
                float(ov["jaccard"]),
                int(ov["overlap_count"]),
                int(bool(cls.get("exact_s3_group_closure", False))),
            )
            if support_best is None or score_tuple > support_best.get("_score_tuple", (-1, -1, -1, -1)):
                support_best = dict(row)
                support_best["_score_tuple"] = score_tuple
            if passes:
                support_overlap_rows.append(row)
        else:
            row.update({
                "target_support_overlap_count": 0,
                "target_support_jaccard": 0.0,
                "target_support_overlap_fraction": 0.0,
                "observer_support_overlap_fraction": 0.0,
                "support_overlap_match": False,
            })
        group_rows.append(row)

        max_comp = max(max_comp, int(cls.get("max_transposition_component_size", 0)))
        max_order = max(max_order, int(cls.get("generated_group_order", 0)))
        any_s3 = any_s3 or bool(cls.get("exact_s3_group_closure"))
        any_shared = any_shared or bool(cls.get("shared_label_transposition_pair"))
        any_two_plus = any_two_plus or int(cls.get("n_distinct_c2_maps", 0)) >= 2
        if int(parent) == int(target_parent) and int(fib) == int(target_fiber):
            target_result = dict(row)

    if not target_result:
        target_result = {
            "parent_domain_id": int(target_parent),
            "fiber_label": int(target_fiber),
            "n_raw_c2_rows": 0,
            "support_size": int(len(supports.get((int(target_parent), int(target_fiber)), set()))),
            "support_hash": _support_hash(supports.get((int(target_parent), int(target_fiber)), set())),
            "support_sample": _support_sample(supports.get((int(target_parent), int(target_fiber)), set())),
            "n_distinct_c2_maps": 0,
            "max_transposition_component_size": 0,
            "generated_group_order": 0,
            "exact_s3_group_closure": False,
            "shared_label_transposition_pair": False,
            "algebra_family": "target_namespace_absent_or_flat",
            "loop_maps": "",
            "target_support_overlap_count": 0,
            "target_support_jaccard": 0.0,
            "target_support_overlap_fraction": 0.0,
            "observer_support_overlap_fraction": 0.0,
            "support_overlap_match": False,
        }

    best = support_best or {}
    overlap_matches = [r for r in support_overlap_rows if bool(r.get("support_overlap_match", False))]
    support_s3_rows = [r for r in overlap_matches if bool(r.get("exact_s3_group_closure", False))]
    support_shared_rows = [r for r in overlap_matches if bool(r.get("shared_label_transposition_pair", False))]
    support_two_plus_rows = [r for r in overlap_matches if int(r.get("n_distinct_c2_maps", 0)) >= 2]

    summary = {
        "n_namespace_groups": int(len(groups)),
        "n_support_namespaces": int(len(supports)),
        "any_exact_s3_group_closure": bool(any_s3),
        "any_shared_label_transposition_pair": bool(any_shared),
        "any_two_plus_distinct_maps": bool(any_two_plus),
        "max_transposition_component_size": int(max_comp),
        "max_generated_group_order": int(max_order),
        "target_namespace_observed": bool(int(target_result.get("n_raw_c2_rows", 0)) > 0),
        "target_namespace_support_observed": bool(int(target_result.get("support_size", 0)) > 0),
        "target_exact_s3_group_closure": bool(target_result.get("exact_s3_group_closure", False)),
        "target_shared_label_transposition_pair": bool(target_result.get("shared_label_transposition_pair", False)),
        "target_n_distinct_c2_maps": int(target_result.get("n_distinct_c2_maps", 0)),
        "target_max_transposition_component_size": int(target_result.get("max_transposition_component_size", 0)),
        "target_generated_group_order": int(target_result.get("generated_group_order", 0)),
        "target_algebra_family": str(target_result.get("algebra_family", "")),
        "target_loop_maps": str(target_result.get("loop_maps", "")),
        "target_support_size": int(len(target_support_set)),
        "support_best_parent_domain_id": int(best.get("parent_domain_id", -1)) if best else -1,
        "support_best_fiber_label": int(best.get("fiber_label", -1)) if best else -1,
        "support_best_overlap_count": int(best.get("target_support_overlap_count", 0)) if best else 0,
        "support_best_jaccard": float(best.get("target_support_jaccard", 0.0)) if best else 0.0,
        "support_best_target_overlap_fraction": float(best.get("target_support_overlap_fraction", 0.0)) if best else 0.0,
        "support_best_observer_overlap_fraction": float(best.get("observer_support_overlap_fraction", 0.0)) if best else 0.0,
        "support_best_algebra_family": str(best.get("algebra_family", "")) if best else "",
        "support_best_exact_s3": bool(best.get("exact_s3_group_closure", False)) if best else False,
        "support_best_shared_pair": bool(best.get("shared_label_transposition_pair", False)) if best else False,
        "support_best_n_distinct_c2_maps": int(best.get("n_distinct_c2_maps", 0)) if best else 0,
        "support_best_max_component_size": int(best.get("max_transposition_component_size", 0)) if best else 0,
        "support_overlap_namespace_count": int(len(overlap_matches)),
        "support_overlap_observed": bool(len(overlap_matches) > 0),
        "support_exact_s3_group_closure": bool(len(support_s3_rows) > 0),
        "support_shared_label_transposition_pair": bool(len(support_shared_rows) > 0),
        "support_two_plus_distinct_maps": bool(len(support_two_plus_rows) > 0),
        "support_max_transposition_component_size": int(max([int(r.get("max_transposition_component_size", 0)) for r in overlap_matches], default=0)),
        "support_max_generated_group_order": int(max([int(r.get("generated_group_order", 0)) for r in overlap_matches], default=0)),
    }
    return summary, group_rows
# ---------------------------------------------------------------------------
# Main audit logic
# ---------------------------------------------------------------------------
def run_multiobserver_consensus_audit(
    q: int = 2,
    vertices: int = 9,
    iterated_csv: str = "",
    frozen_transition_npy: str = "",
    target_candidate: str = "",
    target_rule_mode: str = "",
    target_instance: int = -1,
    target_profile: str = "",
    target_atlas_capacity: int = 0,
    target_seed: int = -1,
    target_iteration: int = 8,
    target_parent_domain: int = 75,
    target_fiber_label: int = 7,
    target_support_npy: str = "",
    target_support_csv: str = "",
    target_support_indices: str = "",
    observer_runs: int = 32,
    observer_seed_start: int = 0,
    observer_seed_stride: int = 1,
    observer_seed_mode: str = "independent",
    reference_observer_seed: int = -1,
    reference_seed_mode: str = "original_iteration",
    min_consensus_fraction: float = 0.5,
    max_spatial_cycles_per_namespace: int = 0,
    max_group_order: int = 4096,
    min_support_overlap_fraction: float = 0.5,
    min_support_jaccard: float = 0.05,
    min_support_overlap_states: int = 1,
    replay_rng_hash_mode: str = "stable",
    out: str = "example_results/multi_observer_consensus.csv",
    plot: str = "",
    synthetic_smoke: bool = False,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if synthetic_smoke:
        return _run_synthetic(out=out, plot=plot)

    if not iterated_csv or not os.path.exists(iterated_csv):
        raise ValueError("Provide --iterated-csv")
    args = argparse.Namespace(**locals())
    for k, v in kwargs.items():
        setattr(args, k, v)
    _ensure_upstream_defaults(args)

    df = pd.read_csv(iterated_csv, low_memory=False)
    row = _select_target_row(df, args)
    states, frozen_next, meta = _reconstruct_frozen_transition(row, int(q), int(vertices), int(target_iteration), args)

    rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    profile = str(meta["profile"])
    capacity = int(meta["atlas_capacity"])
    base_seed = int(meta["initial_seed"])
    candidate_id = str(meta["candidate_id"])

    # Define the observer-independent target support.  Prefer an explicit support
    # file/list; otherwise reconstruct one reference atlas over the exact frozen
    # transition table and extract the support of the target namespace there.
    target_support: Set[int] = _load_target_support_from_args(args)
    target_support_source = "explicit_support" if target_support else ""
    if int(reference_observer_seed) >= 0:
        ref_seed = int(reference_observer_seed)
        ref_seed_source = "manual_reference_seed"
    elif str(reference_seed_mode) == "observer_seed_start":
        ref_seed = int(observer_seed_start)
        ref_seed_source = "observer_seed_start"
    else:
        # Same integer seed formula used by the iterated atlas pass at target_iteration.
        ph = _stable_hash(profile) % 1000
        ref_seed = int(base_seed) + 1299709 * int(capacity) + 15485863 * int(target_iteration) + 104729 * int(ph)
        ref_seed_source = "original_iteration_seed_formula"

    ref_summary: Dict[str, Any] = {}
    if not target_support:
        ref_rng = np.random.default_rng(int(ref_seed))
        ref_atlas, _ref_bounded, _ref_pstats, _ref_rel_stats, _ref_rel_rows, _ref_eff, _ref_lift_stats, _ref_eff_rows = GCPR._advance_effective(
            states, frozen_next, int(q), profile, int(capacity), ref_rng, args, str(args.atlas_lift_mode)
        )
        target_support = _support_for_namespace(ref_atlas, int(target_parent_domain), int(target_fiber_label))
        target_support_source = "reference_exact_namespace" if target_support else "reference_exact_namespace_absent"
        ref_summary, ref_group_rows = _analyze_atlas_chart_cycles(
            ref_atlas,
            int(target_parent_domain), int(target_fiber_label),
            max_cycles=int(max_spatial_cycles_per_namespace),
            max_group_order=int(max_group_order),
            target_support=target_support,
            min_support_overlap_fraction=float(min_support_overlap_fraction),
            min_support_jaccard=float(min_support_jaccard),
            min_support_overlap_states=int(min_support_overlap_states),
        )
    else:
        # Still build a reference atlas for context, but do not overwrite explicit support.
        ref_rng = np.random.default_rng(int(ref_seed))
        ref_atlas, _ref_bounded, _ref_pstats, _ref_rel_stats, _ref_rel_rows, _ref_eff, _ref_lift_stats, _ref_eff_rows = GCPR._advance_effective(
            states, frozen_next, int(q), profile, int(capacity), ref_rng, args, str(args.atlas_lift_mode)
        )
        ref_summary, ref_group_rows = _analyze_atlas_chart_cycles(
            ref_atlas,
            int(target_parent_domain), int(target_fiber_label),
            max_cycles=int(max_spatial_cycles_per_namespace),
            max_group_order=int(max_group_order),
            target_support=target_support,
            min_support_overlap_fraction=float(min_support_overlap_fraction),
            min_support_jaccard=float(min_support_jaccard),
            min_support_overlap_states=int(min_support_overlap_states),
        )

    args.reference_observer_seed_used = int(ref_seed)
    args.reference_observer_seed_source = str(ref_seed_source)
    args.reference_target_support_source = str(target_support_source)
    args.reference_target_support_size = int(len(target_support))
    args.reference_target_support_hash = _support_hash(target_support)
    args.reference_target_support_sample = _support_sample(target_support)
    args.reference_target_exact_s3 = bool(ref_summary.get("target_exact_s3_group_closure", False))
    args.reference_target_n_distinct_c2_maps = int(ref_summary.get("target_n_distinct_c2_maps", 0))
    args.reference_target_max_component_size = int(ref_summary.get("target_max_transposition_component_size", 0))
    args.reference_target_generated_group_order = int(ref_summary.get("target_generated_group_order", 0))

    for r in range(int(observer_runs)):
        if str(observer_seed_mode) == "candidate_offset":
            obs_seed = int(base_seed + int(observer_seed_start) + r * int(observer_seed_stride))
        else:
            obs_seed = int(observer_seed_start + r * int(observer_seed_stride))
        rng = np.random.default_rng(obs_seed)
        atlas, bounded, pstats, rel_stats, rel_rows, eff, lift_stats, eff_rows = GCPR._advance_effective(
            states, frozen_next, int(q), profile, int(capacity), rng, args, str(args.atlas_lift_mode)
        )
        obs_summary, obs_group_rows = _analyze_atlas_chart_cycles(
            atlas,
            int(target_parent_domain), int(target_fiber_label),
            max_cycles=int(max_spatial_cycles_per_namespace),
            max_group_order=int(max_group_order),
            target_support=target_support,
            min_support_overlap_fraction=float(min_support_overlap_fraction),
            min_support_jaccard=float(min_support_jaccard),
            min_support_overlap_states=int(min_support_overlap_states),
        )
        obs_row = {
            "candidate_id": candidate_id,
            "rule_mode": meta["rule_mode"],
            "instance": _safe_int(row.get("instance"), -1),
            "profile": profile,
            "atlas_capacity": capacity,
            "initial_seed": base_seed,
            "frozen_transition_source": str(meta.get("frozen_transition_source", "")),
            "frozen_transition_npy": str(meta.get("frozen_transition_npy", "")),
            "target_iteration": int(target_iteration),
            "target_parent_domain": int(target_parent_domain),
            "target_fiber_label": int(target_fiber_label),
            "target_support_size": int(len(target_support)),
            "target_support_hash": _support_hash(target_support),
            "target_support_source": str(target_support_source),
            "reference_observer_seed": int(ref_seed),
            "reference_observer_seed_source": str(ref_seed_source),
            "observer_run": int(r),
            "observer_seed": int(obs_seed),
            "bounded_atlas_classes": int(pstats.get("bounded_atlas_classes", 0)) if isinstance(pstats, dict) else 0,
            "bounded_atlas_fiber_entropy_bits": float(pstats.get("bounded_atlas_fiber_entropy_bits", 0.0)) if isinstance(pstats, dict) else 0.0,
            "temporal_relation_determinism": float(rel_stats.get("temporal_relation_determinism", 0.0)) if isinstance(rel_stats, dict) else 0.0,
            "chart_c2_count": int(getattr(atlas, "n_chart_c2", 0)),
            "chart_nontrivial_count": int(getattr(atlas, "n_chart_nontrivial", 0)),
            **obs_summary,
        }
        rows.append(obs_row)
        for gr in obs_group_rows:
            gr2 = {
                "candidate_id": candidate_id,
                "observer_run": int(r),
                "observer_seed": int(obs_seed),
                "target_iteration": int(target_iteration),
                **gr,
            }
            group_rows.append(gr2)
        print(
            f"multiobs run={r+1}/{observer_runs} seed={obs_seed} "
            f"target_s3={int(obs_summary.get('target_exact_s3_group_closure', False))} "
            f"support_s3={int(obs_summary.get('support_exact_s3_group_closure', False))} "
            f"any_s3={int(obs_summary.get('any_exact_s3_group_closure', False))} "
            f"target_maps={obs_summary.get('target_n_distinct_c2_maps',0)} "
            f"support_match={int(obs_summary.get('support_overlap_observed', False))} "
            f"jacc={float(obs_summary.get('support_best_jaccard',0.0)):.3f}"
        )

    rdf = pd.DataFrame(rows)
    gdf = pd.DataFrame(group_rows)
    summary = _summarize(rdf, gdf, args, candidate_id)
    _write_outputs(rdf, gdf, summary, out, plot)
    return rdf, gdf, summary

def _summarize(rdf: pd.DataFrame, gdf: pd.DataFrame, args: Any, candidate_id: str) -> Dict[str, Any]:
    if rdf.empty:
        return {"verdict": "MULTI-OBSERVER CONSENSUS EMPTY: no observer atlas rows", "n_rows": 0}
    target_frac = float(rdf["target_exact_s3_group_closure"].map(_safe_bool).mean()) if "target_exact_s3_group_closure" in rdf else 0.0
    target_seen_frac = float(rdf["target_namespace_observed"].map(_safe_bool).mean()) if "target_namespace_observed" in rdf else 0.0
    any_s3_frac = float(rdf["any_exact_s3_group_closure"].map(_safe_bool).mean()) if "any_exact_s3_group_closure" in rdf else 0.0
    shared_frac = float(rdf["target_shared_label_transposition_pair"].map(_safe_bool).mean()) if "target_shared_label_transposition_pair" in rdf else 0.0
    support_seen_frac = float(rdf["support_overlap_observed"].map(_safe_bool).mean()) if "support_overlap_observed" in rdf else 0.0
    support_s3_frac = float(rdf["support_exact_s3_group_closure"].map(_safe_bool).mean()) if "support_exact_s3_group_closure" in rdf else 0.0
    support_shared_frac = float(rdf["support_shared_label_transposition_pair"].map(_safe_bool).mean()) if "support_shared_label_transposition_pair" in rdf else 0.0
    support_two_plus_frac = float(rdf["support_two_plus_distinct_maps"].map(_safe_bool).mean()) if "support_two_plus_distinct_maps" in rdf else 0.0
    max_support_jaccard = float(pd.to_numeric(rdf.get("support_best_jaccard", pd.Series([0.0])), errors="coerce").fillna(0.0).max())
    max_support_overlap = float(pd.to_numeric(rdf.get("support_best_target_overlap_fraction", pd.Series([0.0])), errors="coerce").fillna(0.0).max())
    max_comp = int(pd.to_numeric(rdf.get("target_max_transposition_component_size", pd.Series([0])), errors="coerce").fillna(0).max())
    any_max_comp = int(pd.to_numeric(rdf.get("max_transposition_component_size", pd.Series([0])), errors="coerce").fillna(0).max())
    max_order = int(pd.to_numeric(rdf.get("target_generated_group_order", pd.Series([0])), errors="coerce").fillna(0).max())
    support_max_comp = int(pd.to_numeric(rdf.get("support_max_transposition_component_size", pd.Series([0])), errors="coerce").fillna(0).max())
    support_max_order = int(pd.to_numeric(rdf.get("support_max_generated_group_order", pd.Series([0])), errors="coerce").fillna(0).max())

    if target_frac >= float(args.min_consensus_fraction):
        verdict = "MULTI-OBSERVER STRUCTURAL S3 CONSENSUS: independent atlases repeatedly find exact target S3"
    elif support_s3_frac >= float(args.min_consensus_fraction):
        verdict = "MULTI-OBSERVER SUPPORT-LEVEL STRUCTURAL S3 CONSENSUS: independent atlases find S3 on the target microstate support"
    elif target_frac > 0:
        verdict = "MULTI-OBSERVER WEAK EXACT-NAMESPACE S3 CONSENSUS: target S3 recurs in a minority of independent atlases"
    elif support_s3_frac > 0:
        verdict = "MULTI-OBSERVER WEAK SUPPORT-LEVEL S3 CONSENSUS: S3 recurs on overlapping target support in a minority of atlases"
    elif any_s3_frac > 0:
        verdict = "MULTI-OBSERVER RELOCATED S3 SIGNAL: independent atlases find S3, but not on target namespace/support"
    elif support_seen_frac > 0:
        verdict = "MULTI-OBSERVER SUPPORT C2/FLAT CONSENSUS: target support is recovered, but S3 is not"
    elif target_seen_frac > 0:
        verdict = "MULTI-OBSERVER EXACT-NAMESPACE C2/FLAT CONSENSUS: target namespace observed but S3 not reproduced"
    else:
        verdict = "MULTI-OBSERVER OBSERVER-DEPENDENT/ABSENT SIGNAL: target namespace/support not reproduced as S3"

    fam_counts = Counter(map(str, rdf.get("target_algebra_family", pd.Series([], dtype=str)).fillna("")))
    support_fam_counts = Counter(map(str, rdf.get("support_best_algebra_family", pd.Series([], dtype=str)).fillna("")))
    summary = {
        "verdict": verdict,
        "audit_version": "multi_observer_consensus_audit_v2_frozen_transition_support_overlap",
        "source_iterated_csv": str(args.iterated_csv),
        "candidate_id": str(candidate_id),
        "q": int(args.q),
        "vertices": int(args.vertices),
        "target_iteration": int(args.target_iteration),
        "target_parent_domain": int(args.target_parent_domain),
        "target_fiber_label": int(args.target_fiber_label),
        "observer_runs": int(len(rdf)),
        "target_namespace_observed_fraction": float(target_seen_frac),
        "target_exact_s3_consensus_fraction": float(target_frac),
        "target_shared_pair_consensus_fraction": float(shared_frac),
        "support_overlap_observed_fraction": float(support_seen_frac),
        "support_exact_s3_consensus_fraction": float(support_s3_frac),
        "support_shared_pair_consensus_fraction": float(support_shared_frac),
        "support_two_plus_maps_consensus_fraction": float(support_two_plus_frac),
        "max_support_best_jaccard": float(max_support_jaccard),
        "max_support_best_target_overlap_fraction": float(max_support_overlap),
        "any_exact_s3_fraction": float(any_s3_frac),
        "max_target_transposition_component_size": int(max_comp),
        "max_any_transposition_component_size": int(any_max_comp),
        "max_target_generated_group_order": int(max_order),
        "max_support_transposition_component_size": int(support_max_comp),
        "max_support_generated_group_order": int(support_max_order),
        "mean_target_distinct_c2_maps": float(pd.to_numeric(rdf.get("target_n_distinct_c2_maps", pd.Series([0])), errors="coerce").fillna(0).mean()),
        "target_support_size": int(pd.to_numeric(rdf.get("target_support_size", pd.Series([0])), errors="coerce").fillna(0).max()),
        "target_support_source_counts": dict(Counter(map(str, rdf.get("target_support_source", pd.Series([], dtype=str)).fillna("")))),
        "target_algebra_family_counts": dict(fam_counts),
        "support_best_algebra_family_counts": dict(support_fam_counts),
        "reference_observer_seed_used": int(getattr(args, "reference_observer_seed_used", -1)),
        "reference_target_support_size": int(getattr(args, "reference_target_support_size", 0)),
        "reference_target_support_hash": str(getattr(args, "reference_target_support_hash", "")),
        "reference_target_support_sample": str(getattr(args, "reference_target_support_sample", "")),
        "reference_target_support_source": str(getattr(args, "reference_target_support_source", "")),
        "reference_target_exact_s3": bool(getattr(args, "reference_target_exact_s3", False)),
        "reference_target_n_distinct_c2_maps": int(getattr(args, "reference_target_n_distinct_c2_maps", 0)),
        "reference_target_max_component_size": int(getattr(args, "reference_target_max_component_size", 0)),
        "reference_target_generated_group_order": int(getattr(args, "reference_target_generated_group_order", 0)),
        "structural_consensus_candidate": bool(target_frac >= float(args.min_consensus_fraction)),
        "support_structural_consensus_candidate": bool(support_s3_frac >= float(args.min_consensus_fraction)),
        "observer_dependent_candidate": bool(target_frac == 0.0 and support_s3_frac == 0.0 and any_s3_frac == 0.0),
        "n_group_rows": int(len(gdf)),
        "args": {k: _json_safe(v) for k, v in vars(args).items() if k not in {"synthetic_smoke"}},
    }
    return summary


def _write_outputs(rdf: pd.DataFrame, gdf: pd.DataFrame, summary: Dict[str, Any], out: str, plot: str) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        rdf.to_csv(out, index=False)
        gpath = out[:-4] + "_groups.csv" if out.endswith(".csv") else out + "_groups.csv"
        gdf.to_csv(gpath, index=False)
        spath = out[:-4] + "_summary.json" if out.endswith(".csv") else out + "_summary.json"
        with open(spath, "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
        print(f"wrote {out}")
        print(f"wrote {gpath}")
        print(f"wrote {spath}")
    if plot:
        try:
            import matplotlib.pyplot as plt
            labels = ["target seen", "target S3", "support seen", "support S3", "any S3", "support comp", "support order"]
            vals = [
                float(summary.get("target_namespace_observed_fraction", 0.0)),
                float(summary.get("target_exact_s3_consensus_fraction", 0.0)),
                float(summary.get("support_overlap_observed_fraction", 0.0)),
                float(summary.get("support_exact_s3_consensus_fraction", 0.0)),
                float(summary.get("any_exact_s3_fraction", 0.0)),
                float(summary.get("max_support_transposition_component_size", 0.0)),
                float(summary.get("max_support_generated_group_order", 0.0)),
            ]
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax2 = ax1.twinx()
            ax1.bar(labels[:5], vals[:5])
            ax2.plot(labels[5:], vals[5:], marker="o")
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("fraction")
            ax2.set_ylabel("size/order")
            ax1.set_title(str(summary.get("verdict", "multi-observer consensus")))
            fig.tight_layout()
            os.makedirs(os.path.dirname(plot) or ".", exist_ok=True)
            fig.savefig(plot, dpi=160)
            plt.close(fig)
            print(f"wrote {plot}")
        except Exception as e:
            print(f"plot failed: {e}")

# ---------------------------------------------------------------------------
# Synthetic test
# ---------------------------------------------------------------------------
def _run_synthetic(out: str = "", plot: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    target_parent, target_fiber = 75, 7
    observer_maps = [
        [{0: 1, 1: 0}, {0: 2, 2: 0}],  # S3
        [{0: 1, 1: 0}, {0: 2, 2: 0}],  # S3
        [{0: 1, 1: 0}],                # C2 only
        [],                            # absent
    ]
    for r, maps in enumerate(observer_maps):
        cls = _classify_maps(maps)
        row = {
            "candidate_id": "synthetic|inst=0|full_atlas|cap=32|seed=1",
            "observer_run": r,
            "observer_seed": r,
            "target_iteration": 8,
            "target_parent_domain": target_parent,
            "target_fiber_label": target_fiber,
            "n_namespace_groups": 1 if maps else 0,
            "any_exact_s3_group_closure": bool(cls.get("exact_s3_group_closure", False)),
            "target_namespace_observed": bool(maps),
            "target_exact_s3_group_closure": bool(cls.get("exact_s3_group_closure", False)),
            "target_shared_label_transposition_pair": bool(cls.get("shared_label_transposition_pair", False)),
            "target_n_distinct_c2_maps": int(cls.get("n_distinct_c2_maps", 0)),
            "target_max_transposition_component_size": int(cls.get("max_transposition_component_size", 0)),
            "target_generated_group_order": int(cls.get("generated_group_order", 0)),
            "target_algebra_family": str(cls.get("algebra_family", "flat_or_absent")),
            "target_loop_maps": str(cls.get("loop_maps", "")),
            "max_transposition_component_size": int(cls.get("max_transposition_component_size", 0)),
        }
        rows.append(row)
        if maps:
            groups.append({"observer_run": r, "parent_domain_id": target_parent, "fiber_label": target_fiber, **cls})
    rdf = pd.DataFrame(rows)
    gdf = pd.DataFrame(groups)
    args = argparse.Namespace(
        iterated_csv="synthetic", q=2, vertices=9, target_iteration=8,
        target_parent_domain=target_parent, target_fiber_label=target_fiber,
        observer_runs=len(rdf), min_consensus_fraction=0.5,
    )
    summary = _summarize(rdf, gdf, args, "synthetic|inst=0|full_atlas|cap=32|seed=1")
    _write_outputs(rdf, gdf, summary, out, plot)
    return rdf, gdf, summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-observer consensus audit for critical S3 events")
    ap.add_argument("q", type=int, nargs="?", default=2)
    ap.add_argument("--vertices", type=int, default=9)
    ap.add_argument("--iterated-csv", default="")
    ap.add_argument("--frozen-transition-npy", default="", help="Load an exact saved transition table and bypass replay to the target iteration")
    ap.add_argument("--target-candidate", default="")
    ap.add_argument("--target-rule-mode", default="")
    ap.add_argument("--target-instance", type=int, default=-1)
    ap.add_argument("--target-profile", default="")
    ap.add_argument("--target-atlas-capacity", type=int, default=0)
    ap.add_argument("--target-seed", type=int, default=-1)
    ap.add_argument("--target-iteration", type=int, default=8)
    ap.add_argument("--target-parent-domain", type=int, default=75)
    ap.add_argument("--target-fiber-label", type=int, default=7)
    ap.add_argument("--target-support-npy", default="", help="Optional bool mask or integer-index array defining target microstate support")
    ap.add_argument("--target-support-csv", default="", help="Optional CSV with state_index/microstate_index column defining target support")
    ap.add_argument("--target-support-indices", default="", help="Optional comma/space separated target support indices")
    ap.add_argument("--reference-observer-seed", type=int, default=-1, help="Seed for reference atlas used to derive target support when no support file is supplied")
    ap.add_argument("--reference-seed-mode", default="original_iteration", choices=["original_iteration", "observer_seed_start", "literal"], help="How to choose the reference atlas seed if --reference-observer-seed is unset")
    ap.add_argument("--observer-runs", type=int, default=32)
    ap.add_argument("--observer-seed-start", type=int, default=0)
    ap.add_argument("--observer-seed-stride", type=int, default=1)
    ap.add_argument("--observer-seed-mode", default="independent", choices=["independent", "candidate_offset"])
    ap.add_argument("--min-consensus-fraction", type=float, default=0.5)
    ap.add_argument("--max-spatial-cycles-per-namespace", type=int, default=0)
    ap.add_argument("--max-group-order", type=int, default=4096)
    ap.add_argument("--min-support-overlap-fraction", type=float, default=0.5)
    ap.add_argument("--min-support-jaccard", type=float, default=0.05)
    ap.add_argument("--min-support-overlap-states", type=int, default=1)
    ap.add_argument("--replay-rng-hash-mode", default="stable", choices=["stable"], help="Stable SHA1-based replay seeding (legacy Python-hash mode removed).")

    # Upstream replay knobs.
    ap.add_argument("--atlas-lift-mode", default="bijective")
    ap.add_argument("--proliferation-iterations", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--max-state-samples", type=int, default=512)
    ap.add_argument("--max-total-states", type=int, default=200000)
    ap.add_argument("--max-pred", type=int, default=0)
    ap.add_argument("--min-fiber-states", type=int, default=2)
    ap.add_argument("--min-support-states", type=int, default=4)
    ap.add_argument("--min-overlap-states", type=int, default=4)
    ap.add_argument("--max-cycle-len", type=int, default=4)
    ap.add_argument("--max-cycles-per-fiber", type=int, default=500)
    ap.add_argument("--max-chart-coords", type=int, default=3)
    ap.add_argument("--max-support-coords", type=int, default=3)
    ap.add_argument("--max-charts-per-fiber", type=int, default=12)
    ap.add_argument("--min-chart-classes", type=int, default=2)
    ap.add_argument("--min-chart-entropy", type=float, default=0.0)
    ap.add_argument("--min-chart-transition-determinism", type=float, default=0.98)
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

    ap.add_argument("--out", default="example_results/multi_observer_consensus.csv")
    ap.add_argument("--plot", default="")
    ap.add_argument("--synthetic-smoke", action="store_true")
    args = ap.parse_args()
    rdf, gdf, summary = run_multiobserver_consensus_audit(**vars(args))
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
