"""
binarycompositegaugespectrumaudit.py

Composite gauge-spectrum audit for q=2 generated fiber-atlas dynamics.

The purpose of this module is to test the stricter q=2-first philosophy:
we do not posit q=3 or q=4 as primitive alphabets.  Instead, we ask whether
composed binary atlas sectors, produced by the iterated fiber-atlas pipeline,
carry richer effective automorphism groups (C2 x C2, C3/C6/C12 cyclic traces,
S3 ~= GL(2,2), affine S4 ~= AGL(2,2), or larger nonabelian structure).

The module reads atlas temporal maps, usually from
    generated_candidate_replay_q2_atlas_maps.csv
or
    iterated_fiber_atlas_q2_atlas_maps.csv
and groups them by generated-candidate id or by rule/profile/capacity/seed.
For each candidate it extracts deterministic atlas maps across replay/atlas
iterations, restricts them to a common closed finite core, treats bijective
closed maps as permutation generators, and computes the finite group closure
when small enough.

No target group is inserted.  The audit classifies the generated group
post-hoc.

Example
-------
python -m relgauge.binarycompositegaugespectrumaudit ^
  --atlas-maps-csv example_results/generated_candidate_replay_q2_atlas_maps.csv ^
  --replay-csv example_results/generated_candidate_replay_q2.csv ^
  --manifest-csv example_results/generated_candidate_replay_q2_manifest.csv ^
  --out example_results/binary_composite_gauge_spectrum_q2.csv ^
  --plot example_results/fig_binary_composite_gauge_spectrum_q2.png
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, Any

try:
    import numpy as np
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("binarycompositegaugespectrumaudit requires numpy and pandas") from e


# ---------------------------------------------------------------------------
# Small finite permutation/group helpers
# ---------------------------------------------------------------------------
def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha(obj: Any, n: int = 20) -> str:
    return hashlib.sha1(_json_dumps_compact(obj).encode("utf-8")).hexdigest()[:n]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _log2_int_if_power(n: int) -> int:
    return int(round(math.log2(n))) if _is_power_of_two(n) else -1


def _compose(p: Tuple[int, ...], q: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return p after q, permutations as tuples over 0..n-1."""
    return tuple(p[q[i]] for i in range(len(p)))


def _perm_inverse(p: Tuple[int, ...]) -> Tuple[int, ...]:
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def _perm_order(p: Tuple[int, ...], max_order: int = 100000) -> int:
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


def _cycle_type(p: Tuple[int, ...]) -> Tuple[int, ...]:
    n = len(p)
    seen = [False] * n
    lengths: List[int] = []
    for i in range(n):
        if seen[i]:
            continue
        cur = i
        length = 0
        while not seen[cur]:
            seen[cur] = True
            cur = p[cur]
            length += 1
        if length > 1:
            lengths.append(length)
    return tuple(sorted(lengths))


def _fixed_points(p: Tuple[int, ...]) -> int:
    return sum(1 for i, j in enumerate(p) if i == j)


def _permutation_from_map(mapping: Dict[int, int], core: Sequence[int]) -> Optional[Tuple[int, ...]]:
    core = list(core)
    idx = {x: i for i, x in enumerate(core)}
    image: List[int] = []
    for x in core:
        if x not in mapping:
            return None
        y = mapping[x]
        if y not in idx:
            return None
        image.append(idx[y])
    if len(set(image)) != len(core):
        return None
    return tuple(image)


def _close_group(generators: Sequence[Tuple[int, ...]], max_group_order: int = 4096) -> Tuple[Set[Tuple[int, ...]], bool]:
    if not generators:
        return set(), False
    n = len(generators[0])
    ident = tuple(range(n))
    gens: List[Tuple[int, ...]] = []
    for g in generators:
        if len(g) != n:
            continue
        if g not in gens:
            gens.append(g)
        inv = _perm_inverse(g)
        if inv not in gens:
            gens.append(inv)
    group: Set[Tuple[int, ...]] = {ident}
    q: deque[Tuple[int, ...]] = deque([ident])
    truncated = False
    while q:
        a = q.popleft()
        for g in gens:
            for h in (_compose(g, a), _compose(a, g)):
                if h not in group:
                    group.add(h)
                    if len(group) > max_group_order:
                        truncated = True
                        return group, truncated
                    q.append(h)
    return group, truncated


def _is_abelian(group: Sequence[Tuple[int, ...]], max_checks: int = 200000) -> bool:
    g = list(group)
    checks = 0
    for a in g:
        for b in g:
            checks += 1
            if checks > max_checks:
                # Avoid quadratic blowup; treat as unknown/non-certified.
                return False
            if _compose(a, b) != _compose(b, a):
                return False
    return True


def _orbits(group: Sequence[Tuple[int, ...]], n: int) -> List[List[int]]:
    seen = [False] * n
    out: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        orb = set([i])
        changed = True
        while changed:
            changed = False
            for g in group:
                for x in list(orb):
                    y = g[x]
                    if y not in orb:
                        orb.add(y); changed = True
        for x in orb:
            seen[x] = True
        out.append(sorted(orb))
    return sorted(out, key=lambda z: (len(z), z))


def _contains_s3_relation(group: Sequence[Tuple[int, ...]], element_orders: Dict[Tuple[int, ...], int]) -> bool:
    return _s3_relation_witness(group, element_orders) is not None


def _commutator(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return [a,b] = a b a^{-1} b^{-1}."""
    return _compose(_compose(_compose(a, b), _perm_inverse(a)), _perm_inverse(b))


def _noncommuting_witness(perms: Sequence[Tuple[int, ...]], max_checks: int = 200000) -> Optional[Tuple[int, int, int]]:
    """Return (i,j, order([g_i,g_j])) for the first noncommuting pair."""
    checks = 0
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            if i >= j:
                continue
            checks += 1
            if checks > max_checks:
                return None
            if _compose(a, b) != _compose(b, a):
                return (i, j, _perm_order(_commutator(a, b)))
    return None


def _s3_relation_witness(
    group: Sequence[Tuple[int, ...]],
    element_orders: Dict[Tuple[int, ...], int],
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Return a witness (s,r) with s^2=e, r^3=e, s r s^-1 = r^-1."""
    elems = list(group)
    order2 = [g for g in elems if element_orders.get(g) == 2]
    order3 = [g for g in elems if element_orders.get(g) == 3]
    for s in order2:
        s_inv = _perm_inverse(s)
        for r in order3:
            lhs = _compose(_compose(s, r), s_inv)
            rhs = _perm_inverse(r)
            if lhs == rhs:
                return (s, r)
    return None


def _s3_generator_relation_witness(
    generators: Sequence[Tuple[int, ...]],
    generator_orders: Sequence[int],
) -> Optional[Tuple[int, int]]:
    """Return generator indices (s_idx,r_idx) satisfying the S3 conjugation relation."""
    order2 = [(i, g) for i, (g, o) in enumerate(zip(generators, generator_orders)) if o == 2]
    order3 = [(i, g) for i, (g, o) in enumerate(zip(generators, generator_orders)) if o == 3]
    for si, s in order2:
        s_inv = _perm_inverse(s)
        for ri, r in order3:
            lhs = _compose(_compose(s, r), s_inv)
            rhs = _perm_inverse(r)
            if lhs == rhs:
                return (si, ri)
    return None


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def _auto_iteration_col(df: pd.DataFrame) -> str:
    for c in ["replay_iteration", "atlas_iteration", "iteration", "time"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find replay_iteration/atlas_iteration/iteration/time column in atlas maps CSV")


def _candidate_key_cols(df: pd.DataFrame) -> List[str]:
    if "candidate_id" in df.columns:
        return ["candidate_id"]
    cols = [c for c in ["rule_mode", "instance", "profile", "atlas_capacity", "initial_seed"] if c in df.columns]
    if len(cols) >= 3:
        return cols
    raise ValueError("Could not infer candidate/group columns; expected candidate_id or rule_mode/instance/profile/capacity")


def _candidate_id_from_row(row: pd.Series) -> str:
    if "candidate_id" in row and not pd.isna(row["candidate_id"]):
        return str(row["candidate_id"])
    rm = str(row.get("rule_mode", "mode"))
    inst = str(row.get("instance", "0"))
    prof = str(row.get("profile", "profile"))
    cap = str(row.get("atlas_capacity", row.get("capacity", "cap")))
    seed = str(row.get("initial_seed", row.get("seed", "seed")))
    return f"{rm}|inst={inst}|{prof}|cap={cap}|seed={seed}"


def _build_manifest_id(row: pd.Series) -> str:
    return _candidate_id_from_row(row)


def _load_optional(path: str) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None


def _derive_default_paths(atlas_maps_csv: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if atlas_maps_csv.endswith("_atlas_maps.csv"):
        base = atlas_maps_csv[:-len("_atlas_maps.csv")]
        out["replay_csv"] = base + ".csv"
        out["manifest_csv"] = base + "_manifest.csv"
        out["worldlines_csv"] = base + "_worldlines.csv"
        out["charges_csv"] = base + "_charges.csv"
    return out


def _summarize_replay(replay: Optional[pd.DataFrame]) -> pd.DataFrame:
    if replay is None or replay.empty or "candidate_id" not in replay.columns:
        return pd.DataFrame()
    rows = []
    for cid, g in replay.groupby("candidate_id", dropna=False):
        row = {"candidate_id": str(cid)}
        for c in ["rule_mode", "instance", "profile", "atlas_capacity", "initial_seed", "q", "vertices"]:
            if c in g.columns:
                row[c] = g[c].iloc[0]
        for c in ["chart_c2_count", "chart_nontrivial_count", "chart_c3_count", "max_chart_order", "bounded_atlas_classes", "rel_atlas_class_count", "temporal_relation_determinism", "bounded_atlas_fiber_entropy_bits"]:
            if c in g.columns:
                row["replay_max_" + c] = pd.to_numeric(g[c], errors="coerce").max()
                row["replay_mean_" + c] = pd.to_numeric(g[c], errors="coerce").mean()
        if "generated_c2_now" in g.columns:
            row["replay_any_generated_c2_now"] = bool(g["generated_c2_now"].astype(bool).any())
        if "replay_dynamic_sector_candidate" in g.columns:
            row["replay_dynamic_sector_candidate_fraction"] = float(g["replay_dynamic_sector_candidate"].astype(bool).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_manifest(manifest: Optional[pd.DataFrame]) -> pd.DataFrame:
    if manifest is None or manifest.empty:
        return pd.DataFrame()
    mf = manifest.copy()
    if "candidate_id" not in mf.columns:
        mf["candidate_id"] = mf.apply(_build_manifest_id, axis=1)
    keep = [c for c in [
        "candidate_id", "candidate_rank", "rule_mode", "instance", "profile", "atlas_capacity", "initial_seed",
        "initial_c2_count", "final_c2_count", "c2_generated_after_start_any", "gauge_generated_after_start_any",
        "first_c2_iteration", "c2_persistent_to_final", "effective_fixed_point_found", "effective_limit_cycle_found",
    ] if c in mf.columns]
    return mf[keep].copy()


def _summarize_worldlines(worldlines: Optional[pd.DataFrame]) -> pd.DataFrame:
    if worldlines is None or worldlines.empty or "candidate_id" not in worldlines.columns:
        return pd.DataFrame()
    rows = []
    for cid, g in worldlines.groupby("candidate_id", dropna=False):
        row = {"candidate_id": str(cid)}
        if "strict_generated_c2_worldline" in g.columns:
            row["strict_generated_c2_worldline_count"] = int(g["strict_generated_c2_worldline"].astype(bool).sum())
            row["any_strict_generated_c2_worldline"] = bool(g["strict_generated_c2_worldline"].astype(bool).any())
        if "worldline_family" in g.columns:
            row["worldline_families"] = ";".join(sorted(map(str, g["worldline_family"].dropna().unique())))
        if "time_span" in g.columns:
            row["max_worldline_span"] = pd.to_numeric(g["time_span"], errors="coerce").max()
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_charges(charges: Optional[pd.DataFrame]) -> pd.DataFrame:
    if charges is None or charges.empty or "candidate_id" not in charges.columns:
        return pd.DataFrame()
    rows = []
    for cid, g in charges.groupby("candidate_id", dropna=False):
        row = {"candidate_id": str(cid)}
        row["charge_endpoint_rows"] = int(len(g))
        if "endpoint_type" in g.columns:
            row["charge_endpoint_types"] = ";".join(sorted(map(str, g["endpoint_type"].dropna().unique())))
            row["has_charge_pair_endpoint"] = {"source", "sink"}.issubset(set(map(str, g["endpoint_type"].dropna().unique())))
        if "z2_endpoint" in g.columns:
            row["z2_endpoint_count"] = int(pd.to_numeric(g["z2_endpoint"], errors="coerce").fillna(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spectrum classification
# ---------------------------------------------------------------------------
@dataclass
class CandidateSpectrum:
    row: Dict[str, Any]
    generators: List[Tuple[int, ...]]
    group: Set[Tuple[int, ...]]


def _maps_by_iteration(g: pd.DataFrame, iteration_col: str, determinism_threshold: float, min_source_count: int) -> Dict[int, Dict[int, int]]:
    maps: Dict[int, Dict[int, int]] = {}
    for it, gg in g.groupby(iteration_col):
        # choose one target per source, preferring max determinism/source_count/majority_count
        local: Dict[int, int] = {}
        if "determinism_fraction" in gg.columns:
            gg = gg[pd.to_numeric(gg["determinism_fraction"], errors="coerce").fillna(1.0) >= determinism_threshold]
        if "source_count" in gg.columns:
            gg = gg[pd.to_numeric(gg["source_count"], errors="coerce").fillna(0) >= min_source_count]
        if gg.empty:
            continue
        sort_cols = []
        ascending = []
        for c in ["determinism_fraction", "majority_count", "source_count"]:
            if c in gg.columns:
                sort_cols.append(c); ascending.append(False)
        if sort_cols:
            gg = gg.sort_values(sort_cols, ascending=ascending)
        for _, r in gg.iterrows():
            try:
                s = int(r["source_atlas_class"]); t = int(r["target_atlas_class"])
            except Exception:
                continue
            if s not in local:
                local[s] = t
        if local:
            maps[int(it)] = local
    return maps


def _closed_core(maps: Dict[int, Dict[int, int]], mode: str = "intersection", max_prune_steps: int = 100) -> List[int]:
    if not maps:
        return []
    vals = list(maps.values())
    if mode == "union":
        core: Set[int] = set()
        for m in vals:
            core.update(m.keys()); core.update(m.values())
    else:
        core = set(vals[0].keys()) & set(vals[0].values())
        for m in vals[1:]:
            core &= set(m.keys())
            core &= set(m.values())
    for _ in range(max_prune_steps):
        old = set(core)
        for m in vals:
            core = {x for x in core if x in m and m[x] in core}
        if core == old:
            break
    return sorted(core)


def _classify_candidate(
    cid: str,
    g: pd.DataFrame,
    iteration_col: str,
    determinism_threshold: float,
    min_source_count: int,
    core_mode: str,
    min_core_size: int,
    max_generators: int,
    max_group_order: int,
) -> CandidateSpectrum:
    maps = _maps_by_iteration(g, iteration_col, determinism_threshold, min_source_count)
    iterations = sorted(maps)
    core = _closed_core(maps, mode=core_mode)
    row: Dict[str, Any] = {"candidate_id": cid, "map_iteration_count": len(maps), "iteration_min": min(iterations) if iterations else None, "iteration_max": max(iterations) if iterations else None}
    for c in ["rule_mode", "instance", "profile", "atlas_capacity", "initial_seed", "q", "vertices"]:
        if c in g.columns:
            row[c] = g[c].iloc[0]
    row["core_mode"] = core_mode
    row["core_size"] = int(len(core))
    row["effective_binary_dimension"] = _log2_int_if_power(len(core))
    row["effective_binary_dimension_floor"] = int(math.floor(math.log2(len(core)))) if len(core) > 0 else 0
    row["core_size_power_of_two"] = _is_power_of_two(len(core))

    generators: List[Tuple[int, ...]] = []
    generator_iterations: List[int] = []
    generator_orders: List[int] = []
    generator_cycle_types: List[Tuple[int, ...]] = []
    if len(core) >= min_core_size:
        for it in iterations:
            p = _permutation_from_map(maps[it], core)
            if p is None:
                continue
            if p not in generators:
                generators.append(p)
                generator_iterations.append(int(it))
                generator_orders.append(_perm_order(p))
                generator_cycle_types.append(_cycle_type(p))
            if len(generators) >= max_generators:
                break

    row["n_permutation_generators"] = int(len(generators))
    row["generator_iterations"] = " ".join(map(str, generator_iterations))
    row["generator_orders"] = " ".join(map(str, generator_orders))
    row["generator_order_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(Counter(generator_orders).items()))
    row["generator_cycle_types"] = ";".join(str(x) for x in generator_cycle_types)
    row["has_generator_order2"] = any(o == 2 for o in generator_orders)
    row["has_generator_order3"] = any(o == 3 for o in generator_orders)
    row["has_generator_order4"] = any(o == 4 for o in generator_orders)
    row["has_generator_order6"] = any(o == 6 for o in generator_orders)
    row["has_generator_order12"] = any(o == 12 for o in generator_orders)

    # Generator-level noncommutativity is the key distinction between a mere
    # cyclic/order-3 trace and a genuinely composed binary coordinate action.
    # Count all generator pairs, and also order-2/order-3 pairs that fail to
    # commute.  The latter is the local, data-driven precursor to an S3 test.
    order2_order3_pair_count = 0
    noncommuting_pair_count = 0
    order2_order3_noncommuting_count = 0
    noncommuting_order2_order3_witness = ""
    commutator_orders: List[int] = []
    for i, a in enumerate(generators):
        for j, b in enumerate(generators):
            if i >= j:
                continue
            oi = int(generator_orders[i]) if i < len(generator_orders) else 0
            oj = int(generator_orders[j]) if j < len(generator_orders) else 0
            is_23 = sorted([oi, oj]) == [2, 3]
            if is_23:
                order2_order3_pair_count += 1
            if _compose(a, b) != _compose(b, a):
                noncommuting_pair_count += 1
                co = _perm_order(_commutator(a, b))
                commutator_orders.append(int(co))
                if is_23:
                    order2_order3_noncommuting_count += 1
                    if not noncommuting_order2_order3_witness:
                        noncommuting_order2_order3_witness = f"{i} {j}"
    row["order2_order3_generator_pair_count"] = int(order2_order3_pair_count)
    row["order2_order3_generator_pair"] = bool(order2_order3_pair_count > 0)
    row["noncommuting_generator_pair_count"] = int(noncommuting_pair_count)
    row["noncommuting_generator_pair"] = bool(noncommuting_pair_count > 0)
    row["order2_order3_noncommuting_generator_pair_count"] = int(order2_order3_noncommuting_count)
    row["order2_order3_noncommuting_generator_pair"] = bool(order2_order3_noncommuting_count > 0)
    row["order2_order3_noncommuting_generator_witness"] = noncommuting_order2_order3_witness
    row["noncommuting_generator_commutator_orders"] = " ".join(map(str, sorted(commutator_orders)))

    gen_nc = _noncommuting_witness(generators) if len(generators) >= 2 else None
    row["noncommuting_generator_pair_indices"] = f"{gen_nc[0]} {gen_nc[1]}" if gen_nc else ""
    row["noncommuting_generator_commutator_order"] = int(gen_nc[2]) if gen_nc else 0
    gen_s3 = _s3_generator_relation_witness(generators, generator_orders) if generators else None
    row["s3_generator_relation_candidate"] = bool(gen_s3 is not None)
    row["s3_generator_relation_indices"] = f"{gen_s3[0]} {gen_s3[1]}" if gen_s3 else ""

    group: Set[Tuple[int, ...]] = set()
    truncated = False
    if generators:
        group, truncated = _close_group(generators, max_group_order=max_group_order)
    row["group_closure_truncated"] = bool(truncated)
    row["group_order"] = int(len(group)) if group else 0
    row["group_hash"] = _sha(sorted(group)) if group else ""

    element_orders: Dict[Tuple[int, ...], int] = {}
    element_order_counts: Counter = Counter()
    if group:
        for h in group:
            o = _perm_order(h, max_order=10_000)
            element_orders[h] = int(o)
            element_order_counts[int(o)] += 1
    row["element_order_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(element_order_counts.items()))
    row["has_order2"] = element_order_counts.get(2, 0) > 0
    row["has_order3"] = element_order_counts.get(3, 0) > 0
    row["has_order4"] = element_order_counts.get(4, 0) > 0
    row["has_order6"] = element_order_counts.get(6, 0) > 0
    row["has_order12"] = element_order_counts.get(12, 0) > 0
    row["max_element_order"] = max(element_order_counts.keys()) if element_order_counts else 0
    row["order2_order3_same_core"] = bool(row["has_order2"] and row["has_order3"])

    abelian = _is_abelian(list(group)) if group and not truncated else False
    row["abelian_group"] = bool(abelian) if group and not truncated else False
    row["nonabelian_group"] = bool(group and not truncated and not abelian and len(group) > 1)
    group_nc = _noncommuting_witness(list(group)) if group and not truncated and len(group) >= 2 else None
    row["noncommuting_group_pair"] = bool(group_nc is not None)
    row["noncommuting_group_commutator_order"] = int(group_nc[2]) if group_nc else 0
    row["cyclic_candidate"] = bool(group and not truncated and any(o == len(group) for o in element_orders.values()))
    row["cyclic_order3_trace"] = bool(row["cyclic_candidate"] and row["has_order3"])
    row["noncyclic_order3_trace"] = bool((not row["cyclic_candidate"]) and row["has_order3"])
    row["c2_candidate"] = bool(len(group) == 2 and row["has_order2"])
    row["c3_candidate"] = bool(len(group) == 3 and row["has_order3"] and row["cyclic_candidate"])
    row["c4_candidate"] = bool(len(group) == 4 and row["has_order4"] and row["cyclic_candidate"])
    row["c6_candidate"] = bool(len(group) == 6 and row["max_element_order"] == 6 and row["cyclic_candidate"])
    row["c12_candidate"] = bool(len(group) == 12 and row["max_element_order"] == 12 and row["cyclic_candidate"])

    elementary_2 = bool(group and not truncated and abelian and all(o in (1, 2) for o in element_orders.values()))
    row["elementary_abelian_2group"] = elementary_2
    row["elementary_abelian_2_rank"] = _log2_int_if_power(len(group)) if elementary_2 else 0
    row["c2_product_candidate"] = bool(elementary_2 and len(group) >= 4)

    contains_s3 = _contains_s3_relation(list(group), element_orders) if group and not truncated else False
    row["contains_s3_relation"] = bool(contains_s3)
    row["s3_relation_candidate"] = bool(contains_s3 or row["s3_generator_relation_candidate"])
    row["s3_gl2_candidate"] = bool((len(group) == 6 and row["nonabelian_group"] and row["has_order2"] and row["has_order3"]) or contains_s3 or row["s3_generator_relation_candidate"])
    row["composed_binary_mixing_candidate"] = bool(row["noncommuting_generator_pair"] or row["noncommuting_group_pair"] or row["s3_relation_candidate"] or row["order2_order3_noncommuting_generator_pair"])

    orbit_sizes: List[int] = []
    gl2_natural = False
    if group and not truncated and core:
        orbits = _orbits(list(group), len(core))
        orbit_sizes = sorted([len(o) for o in orbits])
        gl2_natural = bool(row["s3_gl2_candidate"] and len(core) == 4 and orbit_sizes == [1, 3])
    row["orbit_sizes"] = " ".join(map(str, orbit_sizes))
    row["gl2_2_natural_action_candidate"] = bool(gl2_natural)
    row["higher_gl_like_candidate"] = bool(row["s3_gl2_candidate"] and row["effective_binary_dimension_floor"] >= 3)
    row["composite_gauge_candidate"] = bool(row["c2_product_candidate"] or row["s3_gl2_candidate"] or row["has_order3"] or row["nonabelian_group"])

    return CandidateSpectrum(row=row, generators=generators, group=group)


# ---------------------------------------------------------------------------
# Public audit API
# ---------------------------------------------------------------------------
def run_binary_composite_gauge_spectrum_audit(
    atlas_maps_csv: str,
    replay_csv: str = "",
    manifest_csv: str = "",
    worldlines_csv: str = "",
    charges_csv: str = "",
    determinism_threshold: float = 1.0,
    min_source_count: int = 1,
    core_mode: str = "intersection",
    min_core_size: int = 2,
    max_candidates: int = 0,
    require_generated: bool = False,
    require_persistent: bool = False,
    max_generators: int = 16,
    max_group_order: int = 4096,
    window_size: int = 0,
    out: str = "",
    plot: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    defaults = _derive_default_paths(atlas_maps_csv)
    replay_csv = replay_csv or defaults.get("replay_csv", "")
    manifest_csv = manifest_csv or defaults.get("manifest_csv", "")
    worldlines_csv = worldlines_csv or defaults.get("worldlines_csv", "")
    charges_csv = charges_csv or defaults.get("charges_csv", "")

    maps_df = pd.read_csv(atlas_maps_csv)
    iteration_col = _auto_iteration_col(maps_df)
    key_cols = _candidate_key_cols(maps_df)

    replay_df = _load_optional(replay_csv)
    manifest_df = _load_optional(manifest_csv)
    worldlines_df = _load_optional(worldlines_csv)
    charges_df = _load_optional(charges_csv)

    replay_summary = _summarize_replay(replay_df)
    manifest_summary = _summarize_manifest(manifest_df)
    worldline_summary = _summarize_worldlines(worldlines_df)
    charge_summary = _summarize_charges(charges_df)

    allowed_ids: Optional[Set[str]] = None
    if require_generated or require_persistent:
        if manifest_summary.empty:
            # fall back to replay generated flags if possible
            if replay_summary.empty or "replay_any_generated_c2_now" not in replay_summary.columns:
                allowed_ids = set()
            else:
                tmp = replay_summary.copy()
                mask = pd.Series(True, index=tmp.index)
                if require_generated:
                    mask &= tmp["replay_any_generated_c2_now"].astype(bool)
                allowed_ids = set(map(str, tmp.loc[mask, "candidate_id"]))
        else:
            tmp = manifest_summary.copy()
            mask = pd.Series(True, index=tmp.index)
            if require_generated and "c2_generated_after_start_any" in tmp.columns:
                mask &= tmp["c2_generated_after_start_any"].astype(bool)
            if require_persistent and "c2_persistent_to_final" in tmp.columns:
                mask &= tmp["c2_persistent_to_final"].astype(bool)
            allowed_ids = set(map(str, tmp.loc[mask, "candidate_id"]))

    rows: List[Dict[str, Any]] = []
    gen_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []

    grouped = maps_df.groupby(key_cols, dropna=False)
    count = 0
    for key, g in grouped:
        if isinstance(key, tuple):
            if key_cols == ["candidate_id"]:
                cid_base = str(key[0])
            else:
                tmp_row = {c: v for c, v in zip(key_cols, key)}
                cid_base = _candidate_id_from_row(pd.Series(tmp_row))
        else:
            cid_base = str(key)
        if allowed_ids is not None and cid_base not in allowed_ids:
            continue

        # Whole-candidate mode, or local sliding windows over the temporal maps.
        subgroups: List[Tuple[str, pd.DataFrame, Optional[int], Optional[int]]] = []
        if window_size and window_size > 0:
            its = sorted(pd.to_numeric(g[iteration_col], errors="coerce").dropna().astype(int).unique())
            if len(its) >= window_size:
                for start_i in range(0, len(its) - window_size + 1):
                    win = its[start_i:start_i + window_size]
                    gg = g[g[iteration_col].astype(int).isin(win)].copy()
                    cid = f"{cid_base}|window={win[0]}-{win[-1]}"
                    subgroups.append((cid, gg, int(win[0]), int(win[-1])))
            else:
                subgroups.append((cid_base + "|window=all", g, None, None))
        else:
            subgroups.append((cid_base, g, None, None))

        for cid, gg, w0, w1 in subgroups:
            spec = _classify_candidate(
                cid=cid, g=gg, iteration_col=iteration_col,
                determinism_threshold=determinism_threshold, min_source_count=min_source_count,
                core_mode=core_mode, min_core_size=min_core_size,
                max_generators=max_generators, max_group_order=max_group_order,
            )
            row = dict(spec.row)
            row["base_candidate_id"] = cid_base
            row["window_start"] = w0
            row["window_end"] = w1
            rows.append(row)
            # generator rows
            for i, gen in enumerate(spec.generators):
                gen_rows.append({
                    "candidate_id": cid,
                    "base_candidate_id": cid_base,
                    "window_start": w0,
                    "window_end": w1,
                    "generator_index": i,
                    "generator_order": _perm_order(gen),
                    "cycle_type": " ".join(map(str, _cycle_type(gen))),
                    "fixed_points": _fixed_points(gen),
                    "permutation": " ".join(map(str, gen)) if len(gen) <= 128 else "",
                })
            # group element rows, capped in output for size
            for i, elem in enumerate(sorted(spec.group)[:512]):
                group_rows.append({
                    "candidate_id": cid,
                    "base_candidate_id": cid_base,
                    "window_start": w0,
                    "window_end": w1,
                    "element_index": i,
                    "element_order": _perm_order(elem),
                    "cycle_type": " ".join(map(str, _cycle_type(elem))),
                    "fixed_points": _fixed_points(elem),
                    "permutation": " ".join(map(str, elem)) if len(elem) <= 128 else "",
                })
            count += 1
            if max_candidates and count >= max_candidates:
                break
        if max_candidates and count >= max_candidates:
            break

    spectrum_df = pd.DataFrame(rows)
    generators_df = pd.DataFrame(gen_rows)
    group_df = pd.DataFrame(group_rows)

    # Merge optional metadata.  In windowed mode rows use candidate_id|window=...,
    # so join on base_candidate_id when available.
    for meta in [manifest_summary, replay_summary, worldline_summary, charge_summary]:
        if not meta.empty and "candidate_id" in meta.columns and not spectrum_df.empty:
            join_left = "base_candidate_id" if "base_candidate_id" in spectrum_df.columns else "candidate_id"
            meta2 = meta.copy().rename(columns={"candidate_id": join_left})
            overlap = [c for c in meta2.columns if c in spectrum_df.columns and c != join_left]
            meta2 = meta2.drop(columns=overlap) if overlap else meta2
            spectrum_df = spectrum_df.merge(meta2, on=join_left, how="left")

    # Summary.
    summary: Dict[str, Any] = {
        "verdict": "BINARY COMPOSITE GAUGE SPECTRUM COMPLETE",
        "audit_version": "binary_composite_gauge_spectrum_v2_noncommuting_generator_classification",
        "source_atlas_maps_csv": atlas_maps_csv,
        "source_replay_csv": replay_csv,
        "source_manifest_csv": manifest_csv,
        "iteration_col": iteration_col,
        "determinism_threshold": determinism_threshold,
        "core_mode": core_mode,
        "min_core_size": min_core_size,
        "window_size": int(window_size),
        "n_candidates": int(len(spectrum_df)),
        "any_composite_gauge_candidate": bool((spectrum_df.get("composite_gauge_candidate", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_nonabelian_group": bool((spectrum_df.get("nonabelian_group", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_noncommuting_generator_pair": bool((spectrum_df.get("noncommuting_generator_pair", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_noncommuting_group_pair": bool((spectrum_df.get("noncommuting_group_pair", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_order2_order3_same_core": bool((spectrum_df.get("order2_order3_same_core", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_order2_order3_generator_pair": bool((spectrum_df.get("order2_order3_generator_pair", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_order2_order3_noncommuting_generator_pair": bool((spectrum_df.get("order2_order3_noncommuting_generator_pair", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_order3": bool((spectrum_df.get("has_order3", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_cyclic_order3_trace": bool((spectrum_df.get("cyclic_order3_trace", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_noncyclic_order3_trace": bool((spectrum_df.get("noncyclic_order3_trace", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_s3_generator_relation_candidate": bool((spectrum_df.get("s3_generator_relation_candidate", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_s3_relation_candidate": bool((spectrum_df.get("s3_relation_candidate", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_s3_gl2_candidate": bool((spectrum_df.get("s3_gl2_candidate", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_gl2_2_natural_action_candidate": bool((spectrum_df.get("gl2_2_natural_action_candidate", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_c2_product_candidate": bool((spectrum_df.get("c2_product_candidate", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_c12_candidate": bool((spectrum_df.get("c12_candidate", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "any_composed_binary_mixing_candidate": bool((spectrum_df.get("composed_binary_mixing_candidate", pd.Series(dtype=bool)).astype(bool)).any()) if not spectrum_df.empty else False,
        "max_group_order": int(pd.to_numeric(spectrum_df.get("group_order", pd.Series([0])), errors="coerce").fillna(0).max()) if not spectrum_df.empty else 0,
        "max_effective_binary_dimension": int(pd.to_numeric(spectrum_df.get("effective_binary_dimension", pd.Series([0])), errors="coerce").fillna(0).max()) if not spectrum_df.empty else 0,
        "max_core_size": int(pd.to_numeric(spectrum_df.get("core_size", pd.Series([0])), errors="coerce").fillna(0).max()) if not spectrum_df.empty else 0,
    }

    # Verdict refinement.  Separate cyclic/order-3 traces from genuine
    # noncommuting binary-coordinate mixing.  This prevents a C3/C6/C12
    # cyclic orbit from being overread as S3/GL(2,2).
    if summary["any_s3_gl2_candidate"] or summary["any_gl2_2_natural_action_candidate"]:
        summary["verdict"] = "BINARY COMPOSITE GAUGE SPECTRUM SIGNAL: q2 composed atlas maps generate S3/GL(2,2)-like noncommuting structure"
    elif summary["any_composed_binary_mixing_candidate"] or summary["any_nonabelian_group"]:
        summary["verdict"] = "BINARY COMPOSITE GAUGE SPECTRUM NONCOMMUTING TRACE SIGNAL: q2 composed atlas maps show noncommuting generator structure without full S3 certification"
    elif summary["any_order3"]:
        summary["verdict"] = "BINARY COMPOSITE GAUGE SPECTRUM MEDIUM SIGNAL: q2 composed atlas maps generate cyclic/order-3 traces"
    elif summary["any_c2_product_candidate"]:
        summary["verdict"] = "BINARY COMPOSITE GAUGE SPECTRUM WEAK SIGNAL: q2 composed atlas maps generate multi-bit C2-product sectors"
    elif summary["n_candidates"] > 0:
        summary["verdict"] = "BINARY COMPOSITE GAUGE SPECTRUM FLAT/WEAK SIGNAL: generated q2 candidates remain C2/cyclic or partial under tested maps"

    def by_group(cols: List[str]) -> List[Dict[str, Any]]:
        if spectrum_df.empty or not all(c in spectrum_df.columns for c in cols):
            return []
        out_rows: List[Dict[str, Any]] = []
        for key, gg in spectrum_df.groupby(cols, dropna=False):
            r: Dict[str, Any] = {}
            if not isinstance(key, tuple): key = (key,)
            for c, v in zip(cols, key): r[c] = v
            r["n"] = int(len(gg))
            bool_cols = [
                "composite_gauge_candidate", "nonabelian_group", "noncommuting_generator_pair",
                "order2_order3_same_core", "order2_order3_generator_pair",
                "order2_order3_noncommuting_generator_pair", "has_order3",
                "cyclic_order3_trace", "noncyclic_order3_trace",
                "s3_generator_relation_candidate", "s3_relation_candidate", "s3_gl2_candidate",
                "gl2_2_natural_action_candidate", "c2_product_candidate", "c12_candidate",
                "composed_binary_mixing_candidate",
            ]
            for b in bool_cols:
                if b in gg.columns:
                    r[b + "_fraction"] = float(gg[b].astype(bool).mean())
            if "group_order" in gg.columns:
                r["max_group_order"] = int(pd.to_numeric(gg["group_order"], errors="coerce").fillna(0).max())
                r["mean_group_order"] = float(pd.to_numeric(gg["group_order"], errors="coerce").fillna(0).mean())
            if "core_size" in gg.columns:
                r["mean_core_size"] = float(pd.to_numeric(gg["core_size"], errors="coerce").fillna(0).mean())
            out_rows.append(r)
        return out_rows

    summary["by_mode"] = by_group(["rule_mode"]) if "rule_mode" in spectrum_df.columns else []
    summary["by_mode_profile"] = by_group(["rule_mode", "profile"]) if all(c in spectrum_df.columns for c in ["rule_mode", "profile"]) else []
    summary["by_capacity"] = by_group(["atlas_capacity"]) if "atlas_capacity" in spectrum_df.columns else []

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        spectrum_df.to_csv(out, index=False)
        stem, _ = os.path.splitext(out)
        generators_df.to_csv(stem + "_generators.csv", index=False)
        group_df.to_csv(stem + "_group_elements.csv", index=False)
        with open(stem + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if plot:
        _plot_spectrum(spectrum_df, plot)

    return spectrum_df, generators_df, group_df, summary


def _plot_spectrum(df: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "no rows", ha="center", va="center")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return
    label_cols = [c for c in ["rule_mode", "profile"] if c in df.columns]
    if label_cols:
        agg_spec = {
            "composite": ("composite_gauge_candidate", "mean"),
            "noncommuting": ("composed_binary_mixing_candidate", "mean"),
            "order3": ("has_order3", "mean"),
            "cyclic_o3": ("cyclic_order3_trace", "mean"),
            "s3": ("s3_gl2_candidate", "mean"),
            "c2prod": ("c2_product_candidate", "mean"),
            "group_order": ("group_order", "max"),
            "core_size": ("core_size", "mean"),
        }
        # Older CSVs may not have every v2 column; keep plotting robust.
        agg_spec = {k: v for k, v in agg_spec.items() if v[0] in df.columns}
        g = df.groupby(label_cols, dropna=False).agg(**agg_spec).reset_index()
        labels = ["\n".join(str(r[c]) for c in label_cols) for _, r in g.iterrows()]
    else:
        g = df.copy()
        labels = [str(i) for i in range(len(g))]
    x = np.arange(len(g))
    fig, ax1 = plt.subplots(figsize=(max(12, len(g) * 0.75), 6))
    width = 0.16
    bar_cols = ["composite", "noncommuting", "order3", "cyclic_o3", "s3", "c2prod"]
    width = min(0.14, 0.8 / max(1, len(bar_cols)))
    offset0 = -(len(bar_cols) - 1) / 2.0
    for i, col in enumerate(bar_cols):
        if col in g.columns:
            ax1.bar(x + (offset0 + i) * width, g[col].astype(float), width, label=col)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("fraction")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax2 = ax1.twinx()
    if "group_order" in g.columns:
        ax2.plot(x, g["group_order"].astype(float), marker="o", label="max group order")
    if "core_size" in g.columns:
        ax2.plot(x, g["core_size"].astype(float), marker="^", linestyle="--", label="mean core size")
    ax2.set_ylabel("order / classes")
    ax1.set_title("Binary composite gauge spectrum: q2 atlas-generated transport groups")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="best")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Classify richer gauge groups emerging from composed q=2 atlas maps.")
    p.add_argument("--atlas-maps-csv", required=True, help="CSV with source_atlas_class,target_atlas_class maps, e.g. generated_candidate_replay_q2_atlas_maps.csv")
    p.add_argument("--replay-csv", default="", help="Optional generated_candidate_replay CSV for metadata")
    p.add_argument("--manifest-csv", default="", help="Optional generated_candidate_replay manifest CSV")
    p.add_argument("--worldlines-csv", default="", help="Optional replay worldlines CSV")
    p.add_argument("--charges-csv", default="", help="Optional replay charges CSV")
    p.add_argument("--determinism-threshold", type=float, default=1.0)
    p.add_argument("--min-source-count", type=int, default=1)
    p.add_argument("--core-mode", choices=["intersection", "union"], default="intersection")
    p.add_argument("--min-core-size", type=int, default=2)
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--require-generated", action="store_true")
    p.add_argument("--require-persistent", action="store_true")
    p.add_argument("--max-generators", type=int, default=16)
    p.add_argument("--max-group-order", type=int, default=4096)
    p.add_argument("--window-size", type=int, default=0, help="Classify local windows of this many iterations instead of whole candidate; 0=whole candidate")
    p.add_argument("--out", default="example_results/binary_composite_gauge_spectrum_q2.csv")
    p.add_argument("--plot", default="example_results/fig_binary_composite_gauge_spectrum_q2.png")
    args = p.parse_args(argv)

    _df, _gens, _group, summary = run_binary_composite_gauge_spectrum_audit(
        atlas_maps_csv=args.atlas_maps_csv,
        replay_csv=args.replay_csv,
        manifest_csv=args.manifest_csv,
        worldlines_csv=args.worldlines_csv,
        charges_csv=args.charges_csv,
        determinism_threshold=float(args.determinism_threshold),
        min_source_count=int(args.min_source_count),
        core_mode=str(args.core_mode),
        min_core_size=int(args.min_core_size),
        max_candidates=int(args.max_candidates),
        require_generated=bool(args.require_generated),
        require_persistent=bool(args.require_persistent),
        max_generators=int(args.max_generators),
        max_group_order=int(args.max_group_order),
        window_size=int(args.window_size),
        out=args.out,
        plot=args.plot,
    )
    print(json.dumps(summary, indent=2))
    if args.out:
        print(f"wrote {args.out}")
        stem, _ = os.path.splitext(args.out)
        print(f"wrote {stem}_generators.csv")
        print(f"wrote {stem}_group_elements.csv")
        print(f"wrote {stem}_summary.json")
    if args.plot:
        print(f"wrote {args.plot}")


if __name__ == "__main__":  # pragma: no cover
    main()
