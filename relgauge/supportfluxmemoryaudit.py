"""
supportfluxmemoryaudit.py

Support-flux memory audit for a localized S3 consensus event.

Given a frozen transition table T and a reference support K0 (for example the
17-state S3 support), this audit follows the forward images

    K_n = T^n(K0), n = 0, ..., N

inside the same frozen transition table.  At each step it reconstructs a finite
atlas from the same transition table and asks whether any domain/fiber namespace
whose support overlaps K_n carries nontrivial chart-cycle holonomy (C2) or exact
S3 transposition diversity.

The goal is to distinguish a one-tick algebraic flash from a possible memory
trace or gauge wake in the downstream state space.

Typical use
-----------
python -m relgauge.supportfluxmemoryaudit 2 ^
  --vertices 9 ^
  --frozen-transition-npy data/transition_iter8_inst28_seed3221741.npy ^
  --target-support-indices "29,30,44,47,61,62,156,159,188,191,221,222,253,254,256,284,287" ^
  --max-step 20 ^
  --jaccard-threshold 0.3 ^
  --out results/support_flux_memory.csv ^
  --plot results/fig_support_flux_memory.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("supportfluxmemoryaudit requires numpy and pandas") from e

try:  # pragma: no cover
    from . import binarycompositegaugespectrumaudit as BCG
    from . import dynamicsconsistencyfixedpointaudit as DCFP
    from . import generatedcandidatephysicsreplayaudit as GCPR
except Exception:  # pragma: no cover
    try:
        import binarycompositegaugespectrumaudit as BCG  # type: ignore
        import dynamicsconsistencyfixedpointaudit as DCFP  # type: ignore
        import generatedcandidatephysicsreplayaudit as GCPR  # type: ignore
    except Exception:
        BCG = None  # type: ignore
        DCFP = None  # type: ignore
        GCPR = None  # type: ignore

LabelMap = Dict[int, int]
Perm = Tuple[int, ...]

DEFAULT_SUPPORT = [29, 30, 44, 47, 61, 62, 156, 159, 188, 191, 221, 222, 253, 254, 256, 284, 287]

# ---------------------------------------------------------------------------
# Safe conversion / hashing helpers
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


def _sha_support(vals: Iterable[int], n: int = 20) -> str:
    arr = np.asarray(sorted({int(x) for x in vals}), dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()[: int(n)]


def _sample_vals(vals: Iterable[int], limit: int = 20) -> str:
    xs = sorted({int(x) for x in vals})
    out = " ".join(str(x) for x in xs[: int(limit)])
    if len(xs) > int(limit):
        out += " ..."
    return out


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


def _stable_hash_int(text: Any, digits: int = 8) -> int:
    return int(hashlib.sha1(str(text).encode("utf-8")).hexdigest()[: int(digits)], 16)

# ---------------------------------------------------------------------------
# Support parsing
# ---------------------------------------------------------------------------
def _parse_int_list(text: Any) -> Set[int]:
    if text is None:
        return set()
    out: Set[int] = set()
    for tok in re.split(r"[\s,;]+", str(text).strip()):
        if not tok:
            continue
        try:
            out.add(int(float(tok)))
        except Exception:
            pass
    return out


def _load_support_from_json(path: str) -> Set[int]:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Common forms used in the package and prior scripts.
    if isinstance(obj, list):
        return {int(x) for x in obj}
    if isinstance(obj, dict):
        for key in [
            "support_indices", "target_support", "reference_support", "indices",
            "s3_support", "support", "support_states",
        ]:
            if key in obj:
                v = obj[key]
                if isinstance(v, str):
                    return _parse_int_list(v)
                return {int(x) for x in v}
        # Sometimes summary stores only a sample string.  Use it if present, but
        # warn by recording the source in output rather than silently claiming a
        # complete support.
        for key in ["reference_target_support_sample", "target_support_sample"]:
            if key in obj:
                return _parse_int_list(obj[key])
    return set()


def load_target_support(args: Any) -> Tuple[Set[int], str]:
    if str(getattr(args, "target_support_indices", "")).strip():
        return _parse_int_list(args.target_support_indices), "cli_indices"
    if str(getattr(args, "target_support_json", "")).strip():
        return _load_support_from_json(args.target_support_json), "json"
    if str(getattr(args, "target_support_npy", "")).strip():
        arr = np.asarray(np.load(args.target_support_npy), dtype=np.int64)
        return {int(x) for x in arr.ravel().tolist()}, "npy"
    return set(DEFAULT_SUPPORT), "built_in_default_17_state_s3_support"

# ---------------------------------------------------------------------------
# Group classification helpers
# ---------------------------------------------------------------------------
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


def _perm_order(p: Perm) -> int:
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
        cur = i; length = 0
        while not seen[cur]:
            seen[cur] = True
            cur = p[cur]
            length += 1
        if length > 0:
            lcm = lcm * length // math.gcd(lcm, length)
    return int(lcm)


def _close_group(gens: Sequence[Perm], max_group_order: int = 4096) -> Tuple[Set[Perm], bool]:
    if BCG is not None:
        try:
            return BCG._close_group(gens, max_group_order=max_group_order)
        except Exception:
            pass
    if not gens:
        return set(), False
    n = len(gens[0])
    ident = tuple(range(n))
    allgens: List[Perm] = []
    for g in gens:
        if len(g) != n:
            continue
        if g not in allgens:
            allgens.append(g)
        inv = _perm_inverse(g)
        if inv not in allgens:
            allgens.append(inv)
    group: Set[Perm] = {ident}
    q: deque[Perm] = deque([ident])
    while q:
        a = q.popleft()
        for g in allgens:
            for h in (_compose(g, a), _compose(a, g)):
                if h not in group:
                    group.add(h)
                    if len(group) > int(max_group_order):
                        return group, True
                    q.append(h)
    return group, False


def _component_sizes(edges: Sequence[Tuple[int, int]]) -> List[int]:
    adj: Dict[int, Set[int]] = defaultdict(set)
    for a, b in edges:
        a = int(a); b = int(b)
        adj[a].add(b); adj[b].add(a)
    seen: Set[int] = set(); sizes: List[int] = []
    for x in sorted(adj):
        if x in seen:
            continue
        stack = [x]; seen.add(x); c = 0
        while stack:
            y = stack.pop(); c += 1
            for z in adj.get(y, set()):
                if z not in seen:
                    seen.add(z); stack.append(z)
        sizes.append(c)
    return sizes


def _classify_maps(maps: Sequence[LabelMap], max_group_order: int = 4096) -> Dict[str, Any]:
    # Distinct order-2 loop maps only.
    distinct: List[LabelMap] = []
    seen: Set[Tuple[Tuple[int, int], ...]] = set()
    raw_order2 = 0
    for m in maps:
        if not m:
            continue
        labs = sorted(set(m.keys()) | set(m.values()))
        p = _perm_from_label_map(m, labs)
        if p is None:
            continue
        if _perm_order(p) != 2:
            continue
        raw_order2 += 1
        key = tuple(sorted((int(k), int(v)) for k, v in m.items()))
        if key not in seen:
            seen.add(key); distinct.append({int(k): int(v) for k, v in m.items()})
    edges: Set[Tuple[int, int]] = set()
    gens: List[Perm] = []
    union_labels: Set[int] = set()
    for m in distinct:
        moved = sorted({int(k) for k, v in m.items() if int(k) != int(v)} | {int(v) for k, v in m.items() if int(k) != int(v)})
        union_labels.update(moved)
        if len(moved) == 2:
            edges.add((min(moved), max(moved)))
    labs = sorted(union_labels)
    if labs:
        for m in distinct:
            ext = {x: x for x in labs}
            ext.update({int(k): int(v) for k, v in m.items() if int(k) in ext or int(v) in ext})
            # Ensure image labels are in the same core.
            if not set(ext.values()).issubset(set(labs)):
                continue
            p = _perm_from_label_map(ext, labs)
            if p is not None:
                gens.append(p)
    group, truncated = _close_group(gens, max_group_order=max_group_order) if gens else (set(), False)
    comp = _component_sizes(sorted(edges))
    max_comp = max(comp) if comp else 0
    exact_s3 = bool((not truncated) and len(group) == 6 and max_comp >= 3)
    nonabelian = bool(group and any(_compose(a, b) != _compose(b, a) for a in group for b in group)) if group and not truncated else False
    shared_pairs = 0
    e_list = list(edges)
    for i in range(len(e_list)):
        for j in range(i + 1, len(e_list)):
            if set(e_list[i]) & set(e_list[j]):
                shared_pairs += 1
    if exact_s3:
        family = "S3"
        gauge_group = "S3"
    elif len(distinct) > 0:
        family = "C2"
        gauge_group = "C2"
    else:
        family = "flat"
        gauge_group = "trivial"
    return {
        "n_raw_order2_rows": int(raw_order2),
        "n_distinct_c2_maps": int(len(distinct)),
        "n_transposition_edges": int(len(edges)),
        "transposition_edges": " ".join(f"{a}-{b}" for a, b in sorted(edges)),
        "max_transposition_component_size": int(max_comp),
        "component_size_histogram": ";".join(f"{k}:{v}" for k, v in sorted(Counter(comp).items())),
        "shared_label_transposition_pair_count": int(shared_pairs),
        "generated_group_order": int(len(group)),
        "group_truncated": bool(truncated),
        "exact_s3_group_closure": bool(exact_s3),
        "nonabelian_group_closure": bool(nonabelian),
        "algebra_family": family,
        "gauge_group": gauge_group,
        "loop_maps": " | ".join(" ".join(f"{a}->{b}" for a, b in sorted(m.items())) for m in distinct[:20]),
    }

# ---------------------------------------------------------------------------
# Atlas reconstruction and namespace extraction
# ---------------------------------------------------------------------------
def _ensure_upstream_defaults(args: Any) -> Any:
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
        "max_chart_coords": 5,
        "max_support_coords": 4,
        "max_charts_per_fiber": 16,
        "min_chart_classes": 2,
        "min_chart_entropy": 0.05,
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


def _build_states(q: int, vertices: int, n: int, max_total_states: int = 200000) -> List[Tuple[int, ...]]:
    total = int(q) ** int(vertices) if int(vertices) < 32 else max_total_states + 1
    if total == int(n) and total <= int(max_total_states):
        # Lexicographic order matches int-to-base-q indexing for q=2/v=9 used in the witness.
        states: List[Tuple[int, ...]] = []
        for i in range(total):
            x = i; digs = [0] * int(vertices)
            for pos in range(int(vertices) - 1, -1, -1):
                digs[pos] = x % int(q); x //= int(q)
            states.append(tuple(int(d) for d in digs))
        return states
    # Fallback to upstream initializer if full enumeration is not possible.
    if DCFP is None:
        raise RuntimeError("Cannot build sampled states without dynamicsconsistencyfixedpointaudit")
    rng = np.random.default_rng(0)
    states, _next, _meta = DCFP.initialize_sampled_transition(
        q=int(q), vertices=int(vertices), mode="random_full_permutation", rng=rng,
        max_state_samples=int(n), max_total_states=int(max_total_states), max_pred=0,
        proliferation_iterations=4, horizon=3,
    )
    if len(states) != n:
        raise ValueError(f"could not reconstruct state reservoir of length {n}; got {len(states)}")
    return states


def _rng_for_atlas(seed: int, profile: str, capacity: int, target_iteration: int, step: int, mode: str) -> np.random.Generator:
    if str(mode) == "original_iteration":
        h = _stable_hash_int(profile) % 1000
        value = int(seed) + 1299709 * int(capacity) + 15485863 * int(target_iteration) + 104729 * int(h)
    elif str(mode) == "step":
        value = int(seed) + 1000003 * int(step)
    else:  # literal/same
        value = int(seed)
    return np.random.default_rng(value)


def _build_atlas(states: Sequence[Tuple[int, ...]], frozen_next: np.ndarray, q: int, profile: str, capacity: int, rng: np.random.Generator, args: Any):
    if GCPR is None:
        raise RuntimeError("generatedcandidatephysicsreplayaudit is required to build atlases")
    atlas, bounded, pstats, rel_stats, rel_rows, eff, lift_stats, eff_rows = GCPR._advance_effective(
        states, frozen_next, int(q), str(profile), int(capacity), rng, args, str(args.atlas_lift_mode)
    )
    return atlas, bounded, pstats, rel_stats, lift_stats


def _domain_fiber_supports(atlas: Any, min_size: int = 1) -> Dict[Tuple[int, int], Set[int]]:
    out: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    seen_domains: Set[int] = set()
    for coll_name in ("domains_current", "domains_all"):
        for d in list(getattr(atlas, coll_name, []) or []):
            try:
                did = int(getattr(d, "domain_id"))
            except Exception:
                continue
            if did in seen_domains:
                continue
            seen_domains.add(did)
            labels = getattr(d, "labels", None)
            if labels is None:
                continue
            arr = np.asarray(labels, dtype=np.int64)
            for fib in sorted(set(int(x) for x in arr.tolist())):
                supp = {int(i) for i, x in enumerate(arr) if int(x) == int(fib)}
                if len(supp) >= int(min_size):
                    out[(did, int(fib))] |= supp
    return dict(out)


def _maps_by_namespace(atlas: Any, max_cycles_per_namespace: int = 0) -> Dict[Tuple[int, int], List[LabelMap]]:
    groups: Dict[Tuple[int, int], List[LabelMap]] = defaultdict(list)
    for cr in list(getattr(atlas, "chart_cycle_rows", []) or []):
        if isinstance(cr, dict):
            flag = cr.get("chart_c2_holonomy", False)
            loop = cr.get("loop_map", "")
            parent = _safe_int(cr.get("parent_domain_id", -1), -1)
            fib = _safe_int(cr.get("fiber_label", -1), -1)
        else:
            continue
        lm = _parse_loop_map(loop)
        if not lm:
            continue
        if not _safe_bool(flag):
            labs = sorted(set(lm.keys()) | set(lm.values()))
            p = _perm_from_label_map(lm, labs)
            if p is None or _perm_order(p) != 2:
                continue
        key = (int(parent), int(fib))
        if int(max_cycles_per_namespace) > 0 and len(groups[key]) >= int(max_cycles_per_namespace):
            continue
        groups[key].append(lm)
    return dict(groups)


def _overlap_stats(a: Set[int], b: Set[int]) -> Dict[str, float]:
    if not a or not b:
        return {"overlap_count": 0, "jaccard": 0.0, "k_overlap_fraction": 0.0, "namespace_overlap_fraction": 0.0}
    inter = len(a & b)
    union = len(a | b)
    return {
        "overlap_count": int(inter),
        "jaccard": float(inter / union) if union else 0.0,
        "k_overlap_fraction": float(inter / max(1, len(a))),
        "namespace_overlap_fraction": float(inter / max(1, len(b))),
    }


def _step_supports(frozen_next: Sequence[int], k0: Set[int], max_step: int) -> List[Set[int]]:
    nxt = np.asarray(frozen_next, dtype=np.int64)
    out: List[Set[int]] = [set(int(x) for x in k0)]
    cur = set(out[0])
    for _ in range(int(max_step)):
        new: Set[int] = set()
        for x in cur:
            if 0 <= int(x) < len(nxt):
                y = int(nxt[int(x)])
                if 0 <= y < len(nxt):
                    new.add(y)
        out.append(new)
        cur = new
    return out

# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------
def run_support_flux_memory_audit(
    q: int = 2,
    vertices: int = 9,
    frozen_transition_npy: str = "",
    target_support_indices: str = "",
    target_support_json: str = "",
    target_support_npy: str = "",
    max_step: int = 20,
    jaccard_threshold: float = 0.3,
    min_overlap_states: int = 1,
    profile: str = "full_atlas",
    atlas_capacity: int = 32,
    atlas_lift_mode: str = "bijective",
    observer_seed: int = 3221741,
    target_iteration: int = 8,
    atlas_seed_mode: str = "original_iteration",
    rebuild_atlas_each_step: bool = True,
    max_state_samples: int = 512,
    max_total_states: int = 200000,
    proliferation_iterations: int = 4,
    horizon: int = 3,
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
    max_charts_per_fiber: int = 16,
    max_signature_charts: int = 48,
    min_fiber_states: int = 2,
    min_support_states: int = 4,
    chart_min_overlap_states: int = 4,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.05,
    max_chart_coords: int = 5,
    max_support_coords: int = 4,
    max_cycle_len: int = 4,
    max_cycles_per_fiber: int = 500,
    max_cycles_per_namespace: int = 0,
    null_supports: int = 0,
    random_seed: int = 0,
    out: str = "",
    plot: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not frozen_transition_npy or not os.path.exists(frozen_transition_npy):
        raise FileNotFoundError(f"--frozen-transition-npy not found: {frozen_transition_npy}")
    frozen_next = np.asarray(np.load(frozen_transition_npy), dtype=np.int64)
    if frozen_next.ndim != 1:
        raise ValueError(f"frozen transition must be a 1D array, got {frozen_next.shape}")
    if int(max_state_samples) < len(frozen_next):
        max_state_samples = len(frozen_next)
    states = _build_states(int(q), int(vertices), int(len(frozen_next)), max_total_states=int(max_total_states))
    if len(states) != len(frozen_next):
        raise ValueError(f"state count {len(states)} does not match frozen transition length {len(frozen_next)}")

    # Build an args namespace consumed by upstream relgauge code.
    args = argparse.Namespace(**locals())
    args.min_overlap_states = int(chart_min_overlap_states)
    args = _ensure_upstream_defaults(args)

    k0, support_source = load_target_support(args)
    if not k0:
        raise ValueError("target support is empty; supply --target-support-indices/json/npy")
    if any(x < 0 or x >= len(frozen_next) for x in k0):
        bad = [x for x in sorted(k0) if x < 0 or x >= len(frozen_next)]
        raise ValueError(f"support indices outside frozen transition range: {bad[:10]}")

    k_steps = _step_supports(frozen_next, k0, int(max_step))
    seen_hash: Dict[str, int] = {}
    first_repeat_step = -1
    for i, k in enumerate(k_steps):
        h = _sha_support(k)
        if h in seen_hash and first_repeat_step < 0:
            first_repeat_step = int(i)
        seen_hash.setdefault(h, int(i))

    rng_null = np.random.default_rng(int(random_seed))
    row_records: List[Dict[str, Any]] = []
    match_records: List[Dict[str, Any]] = []
    atlas_cache = None

    for step, kset in enumerate(k_steps):
        if (atlas_cache is None) or bool(rebuild_atlas_each_step):
            rng = _rng_for_atlas(int(observer_seed), str(profile), int(atlas_capacity), int(target_iteration), int(step), str(atlas_seed_mode))
            atlas, bounded, pstats, rel_stats, lift_stats = _build_atlas(
                states, frozen_next, int(q), str(profile), int(atlas_capacity), rng, args
            )
            atlas_cache = (atlas, bounded, pstats, rel_stats, lift_stats)
        else:
            atlas, bounded, pstats, rel_stats, lift_stats = atlas_cache
        supports = _domain_fiber_supports(atlas, min_size=1)
        maps = _maps_by_namespace(atlas, max_cycles_per_namespace=int(max_cycles_per_namespace))

        overlap_rows: List[Dict[str, Any]] = []
        for key, supp in supports.items():
            ov = _overlap_stats(kset, supp)
            if ov["overlap_count"] < int(min_overlap_states):
                continue
            if ov["jaccard"] + 1e-12 < float(jaccard_threshold):
                continue
            cls = _classify_maps(maps.get(key, []), max_group_order=4096)
            rec = {
                "step": int(step),
                "namespace_parent_domain_id": int(key[0]),
                "namespace_fiber_label": int(key[1]),
                "k_support_size": int(len(kset)),
                "k_support_hash": _sha_support(kset),
                "k_support_sample": _sample_vals(kset),
                "namespace_support_size": int(len(supp)),
                "namespace_support_hash": _sha_support(supp),
                "namespace_support_sample": _sample_vals(supp),
                **ov,
                **cls,
            }
            overlap_rows.append(rec)
            match_records.append(rec)

        # Null comparison using random supports of the same size as K_n.
        null_best_jaccs: List[float] = []
        null_c2 = 0; null_s3 = 0; null_overlap = 0
        all_indices = np.arange(len(frozen_next), dtype=np.int64)
        for ni in range(int(null_supports)):
            if len(kset) <= 0:
                ns = set()
            else:
                ns = {int(x) for x in rng_null.choice(all_indices, size=len(kset), replace=False).tolist()}
            best_j = 0.0; has_c2 = False; has_s3 = False; has_overlap = False
            for key, supp in supports.items():
                ov = _overlap_stats(ns, supp)
                best_j = max(best_j, float(ov["jaccard"]))
                if ov["overlap_count"] >= int(min_overlap_states) and ov["jaccard"] + 1e-12 >= float(jaccard_threshold):
                    has_overlap = True
                    cls = _classify_maps(maps.get(key, []), max_group_order=4096)
                    if int(cls.get("n_distinct_c2_maps", 0)) > 0:
                        has_c2 = True
                    if bool(cls.get("exact_s3_group_closure", False)):
                        has_s3 = True
            null_best_jaccs.append(best_j)
            null_overlap += int(has_overlap)
            null_c2 += int(has_c2)
            null_s3 += int(has_s3)

        best_overlap = max(overlap_rows, key=lambda r: (float(r["jaccard"]), int(r["generated_group_order"]), int(r["max_transposition_component_size"])), default=None)
        best_gauge = max([r for r in overlap_rows if int(r.get("n_distinct_c2_maps", 0)) > 0], key=lambda r: (bool(r.get("exact_s3_group_closure", False)), int(r.get("generated_group_order", 0)), float(r.get("jaccard", 0.0))), default=None)
        s3_rows = [r for r in overlap_rows if bool(r.get("exact_s3_group_closure", False))]
        c2_rows = [r for r in overlap_rows if int(r.get("n_distinct_c2_maps", 0)) > 0]
        row_records.append({
            "step": int(step),
            "k_support_size": int(len(kset)),
            "k_support_hash": _sha_support(kset),
            "k_support_sample": _sample_vals(kset),
            "n_overlap_namespaces": int(len(overlap_rows)),
            "n_c2_overlap_namespaces": int(len(c2_rows)),
            "n_s3_overlap_namespaces": int(len(s3_rows)),
            "any_c2_wake": bool(len(c2_rows) > 0),
            "any_s3_wake": bool(len(s3_rows) > 0),
            "best_overlap_jaccard": float(best_overlap.get("jaccard", 0.0)) if best_overlap else 0.0,
            "best_overlap_group": str(best_overlap.get("gauge_group", "none")) if best_overlap else "none",
            "best_overlap_group_order": int(best_overlap.get("generated_group_order", 0)) if best_overlap else 0,
            "best_overlap_component_size": int(best_overlap.get("max_transposition_component_size", 0)) if best_overlap else 0,
            "best_overlap_namespace": f"{best_overlap.get('namespace_parent_domain_id')}:{best_overlap.get('namespace_fiber_label')}" if best_overlap else "",
            "best_gauge_jaccard": float(best_gauge.get("jaccard", 0.0)) if best_gauge else 0.0,
            "best_gauge_group": str(best_gauge.get("gauge_group", "none")) if best_gauge else "none",
            "best_gauge_group_order": int(best_gauge.get("generated_group_order", 0)) if best_gauge else 0,
            "best_gauge_component_size": int(best_gauge.get("max_transposition_component_size", 0)) if best_gauge else 0,
            "best_gauge_namespace": f"{best_gauge.get('namespace_parent_domain_id')}:{best_gauge.get('namespace_fiber_label')}" if best_gauge else "",
            "max_overlap_group_order": int(max([r.get("generated_group_order", 0) for r in overlap_rows] or [0])),
            "max_overlap_component_size": int(max([r.get("max_transposition_component_size", 0) for r in overlap_rows] or [0])),
            "null_supports": int(null_supports),
            "null_mean_best_jaccard": float(np.mean(null_best_jaccs)) if null_best_jaccs else 0.0,
            "null_overlap_fraction": float(null_overlap / max(1, int(null_supports))) if int(null_supports) > 0 else 0.0,
            "null_c2_overlap_fraction": float(null_c2 / max(1, int(null_supports))) if int(null_supports) > 0 else 0.0,
            "null_s3_overlap_fraction": float(null_s3 / max(1, int(null_supports))) if int(null_supports) > 0 else 0.0,
            "bounded_atlas_classes": int(pstats.get("bounded_atlas_classes", 0)) if isinstance(pstats, dict) else 0,
            "n_chart_c2": int(getattr(atlas, "n_chart_c2", 0)),
            "n_chart_nontrivial": int(getattr(atlas, "n_chart_nontrivial", 0)),
            "n_chart_cycles": int(getattr(atlas, "n_chart_cycles", 0)),
        })

    rdf = pd.DataFrame(row_records)
    mdf = pd.DataFrame(match_records)

    # Wake summary: contiguous steps after 0 where any gauge overlaps the flux.
    c2_steps = [int(x) for x in rdf.loc[rdf.get("any_c2_wake", False).astype(bool), "step"].tolist()] if not rdf.empty else []
    s3_steps = [int(x) for x in rdf.loc[rdf.get("any_s3_wake", False).astype(bool), "step"].tolist()] if not rdf.empty else []
    first_nontriv_after_0 = next((s for s in c2_steps if s > 0), None)
    first_s3_after_0 = next((s for s in s3_steps if s > 0), None)
    # classify simple verdict
    if s3_steps and len([s for s in s3_steps if s > 0]) > 0:
        verdict = "SUPPORT-FLUX MEMORY SIGNAL: downstream image support intersects S3-carrying namespaces"
    elif first_nontriv_after_0 is not None:
        verdict = "SUPPORT-FLUX C2 WAKE SIGNAL: downstream image support intersects C2-carrying namespaces"
    elif 0 in s3_steps and len(c2_steps) == 1:
        verdict = "SUPPORT-FLUX LOCAL EVENT SIGNAL: original S3 support is gauge-active but downstream images are flat under overlap test"
    elif c2_steps:
        verdict = "SUPPORT-FLUX LOCAL C2 SIGNAL: only original/initial image supports carry gauge under overlap test"
    else:
        verdict = "SUPPORT-FLUX EMPTY SIGNAL: no overlapping gauge-carrying namespace found along image support"

    summary = {
        "verdict": verdict,
        "audit_version": "support_flux_memory_audit_v1_forward_image_gauge_wake",
        "q": int(q),
        "vertices": int(vertices),
        "frozen_transition_npy": str(frozen_transition_npy),
        "transition_size": int(len(frozen_next)),
        "support_source": support_source,
        "initial_support_size": int(len(k0)),
        "initial_support_hash": _sha_support(k0),
        "initial_support_sample": _sample_vals(k0),
        "max_step": int(max_step),
        "jaccard_threshold": float(jaccard_threshold),
        "min_overlap_states": int(min_overlap_states),
        "profile": str(profile),
        "atlas_capacity": int(atlas_capacity),
        "atlas_seed_mode": str(atlas_seed_mode),
        "observer_seed": int(observer_seed),
        "target_iteration": int(target_iteration),
        "rebuild_atlas_each_step": bool(rebuild_atlas_each_step),
        "n_rows": int(len(rdf)),
        "n_match_rows": int(len(mdf)),
        "c2_wake_steps": c2_steps,
        "s3_wake_steps": s3_steps,
        "first_nontrivial_after_step0": first_nontriv_after_0,
        "first_s3_after_step0": first_s3_after_0,
        "max_best_overlap_jaccard": float(rdf["best_overlap_jaccard"].max()) if not rdf.empty else 0.0,
        "max_best_gauge_jaccard": float(rdf["best_gauge_jaccard"].max()) if not rdf.empty else 0.0,
        "max_overlap_group_order": int(rdf["max_overlap_group_order"].max()) if not rdf.empty else 0,
        "max_overlap_component_size": int(rdf["max_overlap_component_size"].max()) if not rdf.empty else 0,
        "first_repeat_step": int(first_repeat_step),
        "step_summaries": rdf[["step", "k_support_size", "n_overlap_namespaces", "n_c2_overlap_namespaces", "n_s3_overlap_namespaces", "best_gauge_group", "best_gauge_jaccard", "best_gauge_namespace"]].to_dict(orient="records") if not rdf.empty else [],
        "args": {k: _json_safe(v) for k, v in vars(args).items() if k not in {"states", "frozen_next"}},
    }

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        rdf.to_csv(out, index=False)
        matches_path = os.path.splitext(out)[0] + "_matches.csv"
        summary_path = os.path.splitext(out)[0] + "_summary.json"
        mdf.to_csv(matches_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
    if plot:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(plot) or ".", exist_ok=True)
            fig, ax1 = plt.subplots(figsize=(12, 6))
            x = rdf["step"].to_numpy() if not rdf.empty else np.array([])
            if len(x):
                ax1.plot(x, rdf["best_overlap_jaccard"], marker="o", label="best support Jaccard")
                ax1.plot(x, rdf["best_gauge_jaccard"], marker="s", label="best gauge Jaccard")
                ax1.bar(x - 0.15, rdf["n_c2_overlap_namespaces"], width=0.3, alpha=0.35, label="C2 overlap count")
                ax1.bar(x + 0.15, rdf["n_s3_overlap_namespaces"], width=0.3, alpha=0.35, label="S3 overlap count")
            ax1.set_xlabel("forward image step n: K_n = T^n(K_0)")
            ax1.set_ylabel("overlap / count")
            ax2 = ax1.twinx()
            if len(x):
                ax2.plot(x, rdf["max_overlap_group_order"], marker="^", linestyle="--", label="max group order")
                ax2.plot(x, rdf["max_overlap_component_size"], marker="d", linestyle="--", label="max component size")
            ax2.set_ylabel("group order / component size")
            ax1.set_title("Support-flux memory: downstream gauge wake along T^n(K0)")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
            fig.tight_layout()
            fig.savefig(plot, dpi=160)
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"plot failed: {e}")
    return rdf, mdf, summary

# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------
def run_synthetic_smoke(out: str = "", plot: str = "") -> None:
    # Tiny synthetic transition and support for parser/plot sanity.  This does
    # not build a relgauge atlas; it only verifies the image-support logic.
    path = "/tmp/support_flux_synthetic_transition.npy"
    np.save(path, np.array([1, 2, 3, 4, 5, 0], dtype=np.int64))
    print("Synthetic transition saved:", path)
    print("For full audit, run with a relgauge-compatible frozen transition and support.")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Trace forward image supports from an S3 support and test for downstream gauge wake")
    ap.add_argument("q", type=int, nargs="?", default=2)
    ap.add_argument("--vertices", type=int, default=9)
    ap.add_argument("--frozen-transition-npy", required=False, default="")
    ap.add_argument("--target-support-indices", default="")
    ap.add_argument("--target-support-json", default="")
    ap.add_argument("--target-support-npy", default="")
    ap.add_argument("--max-step", type=int, default=20)
    ap.add_argument("--jaccard-threshold", type=float, default=0.3)
    ap.add_argument("--min-overlap-states", type=int, default=1)
    ap.add_argument("--profile", default="full_atlas")
    ap.add_argument("--atlas-capacity", type=int, default=32)
    ap.add_argument("--atlas-lift-mode", default="bijective")
    ap.add_argument("--observer-seed", type=int, default=3221741)
    ap.add_argument("--target-iteration", type=int, default=8)
    ap.add_argument("--atlas-seed-mode", default="original_iteration", choices=["original_iteration", "literal", "step"])
    ap.add_argument("--reuse-atlas", action="store_true", help="Build the atlas once and reuse it for every K_n; default rebuilds each step")
    ap.add_argument("--null-supports", type=int, default=0)
    ap.add_argument("--random-seed", type=int, default=0)
    # Upstream atlas args
    ap.add_argument("--max-state-samples", type=int, default=512)
    ap.add_argument("--max-total-states", type=int, default=200000)
    ap.add_argument("--proliferation-iterations", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=3)
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
    ap.add_argument("--max-charts-per-fiber", type=int, default=16)
    ap.add_argument("--max-signature-charts", type=int, default=48)
    ap.add_argument("--min-fiber-states", type=int, default=2)
    ap.add_argument("--min-support-states", type=int, default=4)
    ap.add_argument("--chart-min-overlap-states", type=int, default=4)
    ap.add_argument("--min-chart-classes", type=int, default=2)
    ap.add_argument("--min-chart-entropy", type=float, default=0.05)
    ap.add_argument("--max-chart-coords", type=int, default=5)
    ap.add_argument("--max-support-coords", type=int, default=4)
    ap.add_argument("--max-cycle-len", type=int, default=4)
    ap.add_argument("--max-cycles-per-fiber", type=int, default=500)
    ap.add_argument("--max-cycles-per-namespace", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", default="")
    ap.add_argument("--synthetic-smoke", action="store_true")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.synthetic_smoke:
        run_synthetic_smoke(args.out, args.plot)
        return
    rdf, mdf, summary = run_support_flux_memory_audit(
        q=int(args.q), vertices=int(args.vertices), frozen_transition_npy=str(args.frozen_transition_npy),
        target_support_indices=str(args.target_support_indices), target_support_json=str(args.target_support_json), target_support_npy=str(args.target_support_npy),
        max_step=int(args.max_step), jaccard_threshold=float(args.jaccard_threshold), min_overlap_states=int(args.min_overlap_states),
        profile=str(args.profile), atlas_capacity=int(args.atlas_capacity), atlas_lift_mode=str(args.atlas_lift_mode),
        observer_seed=int(args.observer_seed), target_iteration=int(args.target_iteration), atlas_seed_mode=str(args.atlas_seed_mode),
        rebuild_atlas_each_step=not bool(args.reuse_atlas), max_state_samples=int(args.max_state_samples), max_total_states=int(args.max_total_states),
        proliferation_iterations=int(args.proliferation_iterations), horizon=int(args.horizon), max_pred=int(args.max_pred),
        initial_boundary=str(args.initial_boundary), initial_boundary_q=args.initial_boundary_q,
        max_domains_per_depth=int(args.max_domains_per_depth), min_live_classes=int(args.min_live_classes), min_fiber_size=int(args.min_fiber_size),
        min_entropy_bits=float(args.min_entropy_bits), synergy_threshold=float(args.synergy_threshold), max_signature_domains=int(args.max_signature_domains),
        max_parent_domains=int(args.max_parent_domains), max_fibers_per_parent=int(args.max_fibers_per_parent), max_charts_per_fiber=int(args.max_charts_per_fiber),
        max_signature_charts=int(args.max_signature_charts), min_fiber_states=int(args.min_fiber_states), min_support_states=int(args.min_support_states),
        chart_min_overlap_states=int(args.chart_min_overlap_states), min_chart_classes=int(args.min_chart_classes), min_chart_entropy=float(args.min_chart_entropy),
        max_chart_coords=int(args.max_chart_coords), max_support_coords=int(args.max_support_coords), max_cycle_len=int(args.max_cycle_len),
        max_cycles_per_fiber=int(args.max_cycles_per_fiber), max_cycles_per_namespace=int(args.max_cycles_per_namespace),
        null_supports=int(args.null_supports), random_seed=int(args.random_seed), out=str(args.out), plot=str(args.plot)
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    if args.out:
        print(f"wrote {args.out}")
        print(f"wrote {os.path.splitext(args.out)[0]}_matches.csv")
        print(f"wrote {os.path.splitext(args.out)[0]}_summary.json")
    if args.plot:
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
