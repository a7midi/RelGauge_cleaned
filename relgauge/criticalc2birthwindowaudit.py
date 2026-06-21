"""
criticalc2birthwindowaudit.py

Critical C2-birth window / transposition-diversity provenance audit.

Why this module exists
----------------------
The q=2 iterated fiber-atlas runs show robust generation of C2 chart holonomy.
But nonabelian closure is not controlled by raw C2 abundance.  The one verified
v9 S3 event appeared when two same-namespace C2 chart-cycle flips were distinct
and shared one label, e.g. (0 1) and (0 2), at the first generated-C2 window.

This audit asks a more specific question:

    Which candidate/iteration/domain/fiber conditions produce
    more than one distinct C2 transposition in the same namespace?

It streams chart-cycle rows, filters to candidate C2-birth windows, groups maps
by candidate / iteration-window / parent_domain_id / fiber_label, computes
transposition diversity, and joins candidate/domain provenance from iterated and
domain CSVs.

Typical use
-----------
python -m relgauge.criticalc2birthwindowaudit ^
  --chart-cycles-csv example_results/iterated_fiber_atlas_q2_v9_chart_cycles.csv ^
  --iterated-csv example_results/iterated_fiber_atlas_q2_v9.csv ^
  --domains-csv example_results/iterated_fiber_atlas_q2_v9_domains.csv ^
  --require-generated --require-flat-start ^
  --focus-first-c2-window 1 ^
  --namespace-cols parent_domain_id,fiber_label ^
  --out example_results/critical_c2_birth_windows_q2_v9.csv ^
  --plot example_results/fig_critical_c2_birth_windows_q2_v9.png

The module does not insert S3.  It simply tests whether overlapping same-
namespace transpositions are present and records what provenance conditions
surround them.
"""
from __future__ import annotations

import argparse
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
    raise RuntimeError("criticalc2birthwindowaudit requires numpy and pandas") from e

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
        return float(v)
    except Exception:
        return float(default)


def _safe_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n", "nan", "", "none"}:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return bool(x)


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


def _candidate_id_from_fields(row: Any) -> str:
    if isinstance(row, dict):
        get = row.get
    else:
        get = row.get
    if "candidate_id" in getattr(row, "index", []) if not isinstance(row, dict) else "candidate_id" in row:
        val = get("candidate_id")
        try:
            if val is not None and not pd.isna(val):
                return str(val)
        except Exception:
            if val is not None:
                return str(val)
    rm = str(get("rule_mode", "mode"))
    inst = str(_safe_int(get("instance", 0)))
    prof = str(get("profile", "profile"))
    cap = str(_safe_int(get("atlas_capacity", get("capacity", 0))))
    seed = str(_safe_int(get("initial_seed", get("seed", 0))))
    return f"{rm}|inst={inst}|{prof}|cap={cap}|seed={seed}"


def _base_candidate_id(cid: Any) -> str:
    return str(cid).split("|window=")[0]


def _auto_iteration_col(cols: Iterable[str]) -> Optional[str]:
    colset = set(cols)
    for c in ["atlas_iteration", "replay_iteration", "iteration", "time"]:
        if c in colset:
            return c
    return None


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


def _compact_map(m: LabelMap, limit: int = 40) -> str:
    items = sorted((int(k), int(v)) for k, v in m.items())[: int(limit)]
    return " ".join(f"{a}->{b}" for a, b in items)


def _perm_from_label_map(m: LabelMap, core: Sequence[int]) -> Optional[Perm]:
    core = [int(x) for x in core]
    idx = {x: i for i, x in enumerate(core)}
    arr: List[int] = []
    for x in core:
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
    """Return p after q; permutations over 0..n-1."""
    return tuple(p[q[i]] for i in range(len(p)))


def _perm_inverse(p: Perm) -> Perm:
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def _perm_order(p: Perm, max_order: int = 100000) -> int:
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


def _cycle_type(p: Perm) -> Tuple[int, ...]:
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


def _close_group(generators: Sequence[Perm], max_group_order: int = 4096) -> Tuple[Set[Perm], bool]:
    if not generators:
        return set(), False
    n = len(generators[0])
    ident = tuple(range(n))
    gens: List[Perm] = []
    for g in generators:
        if len(g) != n:
            continue
        if g not in gens:
            gens.append(g)
        inv = _perm_inverse(g)
        if inv not in gens:
            gens.append(inv)
    group: Set[Perm] = {ident}
    q: deque[Perm] = deque([ident])
    while q:
        a = q.popleft()
        for g in gens:
            for h in (_compose(g, a), _compose(a, g)):
                if h not in group:
                    group.add(h)
                    if len(group) > int(max_group_order):
                        return group, True
                    q.append(h)
    return group, False


def _is_abelian(group: Sequence[Perm], max_checks: int = 100000) -> bool:
    g = list(group)
    checks = 0
    for a in g:
        for b in g:
            checks += 1
            if checks > int(max_checks):
                return False
            if _compose(a, b) != _compose(b, a):
                return False
    return True


def _is_single_transposition(m: LabelMap) -> Tuple[bool, Optional[Tuple[int, int]]]:
    labels = sorted(set(m.keys()) | set(m.values()))
    moved = []
    for x in labels:
        y = int(m.get(x, x))
        if y != x:
            moved.append((x, y))
    if len(moved) != 2:
        return False, None
    a, b = moved[0]
    c, d = moved[1]
    if a == d and b == c and a != b:
        return True, tuple(sorted((a, b)))
    return False, None


def _component_sizes(edges: Sequence[Tuple[int, int]]) -> List[int]:
    parent: Dict[int, int] = {}
    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for a, b in edges:
        union(int(a), int(b))
    comps: Dict[int, Set[int]] = defaultdict(set)
    for x in list(parent):
        comps[find(x)].add(x)
    return sorted([len(v) for v in comps.values()])


def _shared_disjoint_pair_counts(edges: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    shared = 0
    disjoint = 0
    e = [tuple(sorted((int(a), int(b)))) for a, b in edges]
    for i in range(len(e)):
        for j in range(i + 1, len(e)):
            inter = set(e[i]) & set(e[j])
            if len(inter) == 1:
                shared += 1
            elif len(inter) == 0:
                disjoint += 1
    return shared, disjoint

# ---------------------------------------------------------------------------
# Candidate/iteration provenance
# ---------------------------------------------------------------------------
def _load_iterated_meta(path: str, rule_modes: str = "", profiles: str = "", capacities: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[int, Dict[str, Any]]]]:
    if not path or not os.path.exists(path):
        return pd.DataFrame(), pd.DataFrame(), {}
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df, pd.DataFrame(), {}
    if "candidate_id" not in df.columns:
        df["candidate_id"] = df.apply(_candidate_id_from_fields, axis=1)
    df["base_candidate_id"] = df["candidate_id"].astype(str).map(_base_candidate_id)
    if rule_modes:
        allowed = {x.strip() for x in str(rule_modes).replace(";", ",").split(",") if x.strip()}
        if allowed and "rule_mode" in df.columns:
            df = df[df["rule_mode"].astype(str).isin(allowed)].copy()
    if profiles:
        allowed = {x.strip() for x in str(profiles).replace(";", ",").split(",") if x.strip()}
        if allowed and "profile" in df.columns:
            df = df[df["profile"].astype(str).isin(allowed)].copy()
    if capacities:
        allowed = {_safe_int(x, -999) for x in str(capacities).replace(";", ",").split(",") if str(x).strip()}
        if allowed and "atlas_capacity" in df.columns:
            df = df[df["atlas_capacity"].map(_safe_int).isin(allowed)].copy()
    itcol = "atlas_iteration" if "atlas_iteration" in df.columns else _auto_iteration_col(df.columns)
    if itcol is None:
        raise ValueError("iterated CSV needs atlas_iteration/replay_iteration/iteration column")
    df["atlas_iteration"] = pd.to_numeric(df[itcol], errors="coerce").fillna(-1).astype(int)
    # First C2 per candidate.
    rows = []
    iter_lookup: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for cid, g in df.groupby("base_candidate_id", dropna=False):
        cid = str(cid)
        g = g.sort_values("atlas_iteration")
        for _, r in g.iterrows():
            iter_lookup[cid][int(r["atlas_iteration"])] = r.to_dict()
        initial_c2 = _safe_int(g.get("initial_c2_count", pd.Series([0])).iloc[0] if "initial_c2_count" in g else 0)
        generated_mask = pd.Series(False, index=g.index)
        if "c2_generated_after_start" in g.columns:
            generated_mask |= g["c2_generated_after_start"].map(_safe_bool)
        if "n_chart_c2" in g.columns:
            generated_mask |= ((pd.to_numeric(g["n_chart_c2"], errors="coerce").fillna(0) > 0) & (initial_c2 == 0))
        first_c2 = -1
        if generated_mask.any():
            first_c2 = int(g.loc[generated_mask, "atlas_iteration"].min())
        elif "first_c2_iteration_so_far" in g.columns:
            vals = pd.to_numeric(g["first_c2_iteration_so_far"], errors="coerce").dropna()
            vals = vals[vals >= 0]
            if len(vals):
                first_c2 = int(vals.min())
        last = g.iloc[-1]
        row = {c: last.get(c) for c in ["candidate_id", "base_candidate_id", "rule_mode", "instance", "profile", "atlas_capacity", "initial_seed", "q", "vertices"] if c in g.columns}
        row.update({
            "base_candidate_id": cid,
            "initial_c2_count": int(initial_c2),
            "first_c2_iteration": int(first_c2),
            "generated_c2_candidate": bool(first_c2 >= 0 and initial_c2 == 0),
            "final_c2_count": _safe_int(last.get("n_chart_c2", 0)),
            "final_chart_c2_present": _safe_bool(last.get("chart_c2_present", False)),
            "c2_persistent_to_final": _safe_bool(last.get("c2_persistent_to_final", False)) if "c2_persistent_to_final" in last.index else bool(_safe_int(last.get("n_chart_c2", 0)) > 0),
        })
        rows.append(row)
    return df, pd.DataFrame(rows), iter_lookup


def _iter_prefix(row: Optional[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if row is None:
        return out
    keys = [
        "initial_c2_count", "initial_gauge_count", "n_chart_c2", "n_chart_nontrivial", "n_chart_cycles", "n_chart_valid_cycles",
        "chart_c2_present", "c2_generated_after_start", "gauge_generated_after_start", "c2_bearing_atlas_candidate",
        "first_c2_iteration_so_far", "bounded_atlas_classes", "bounded_atlas_fiber_entropy_bits", "full_signature_classes",
        "full_signature_entropy_bits", "dependency_edges", "dependency_beta1", "proliferation_nontriviality_score",
        "temporal_relation_determinism", "atlas_capacity", "profile", "rule_mode", "instance", "initial_seed",
    ]
    for k in keys:
        if k in row:
            out[f"{prefix}_{k}"] = row.get(k)
    return out


def _load_domain_lookup(path: str) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return out
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return out
    if "candidate_id" not in df.columns:
        df["candidate_id"] = df.apply(_candidate_id_from_fields, axis=1)
    df["base_candidate_id"] = df["candidate_id"].astype(str).map(_base_candidate_id)
    itcol = "atlas_iteration" if "atlas_iteration" in df.columns else _auto_iteration_col(df.columns)
    if itcol is None:
        return out
    for _, r in df.iterrows():
        cid = str(r["base_candidate_id"])
        it = _safe_int(r.get(itcol), -1)
        did = _safe_int(r.get("domain_id"), -999999)
        if did == -999999:
            continue
        d = r.to_dict()
        out[(cid, it, did)] = d
    return out

# ---------------------------------------------------------------------------
# Chart-cycle processing
# ---------------------------------------------------------------------------
def _parse_windowed_iteration(it: int, first_c2: int, namespace_window_size: int) -> Tuple[int, int]:
    if namespace_window_size <= 1:
        return int(it), int(it)
    if first_c2 >= 0:
        rel = int(it) - int(first_c2)
        bucket = math.floor(rel / int(namespace_window_size))
        start = int(first_c2) + int(bucket) * int(namespace_window_size)
    else:
        start = (int(it) // int(namespace_window_size)) * int(namespace_window_size)
    return int(start), int(start + int(namespace_window_size) - 1)


def _collect_order2_rows_from_chart_cycles(
    chart_cycles_csv: str,
    candidate_meta: pd.DataFrame,
    focus_first_c2_window: int = -1,
    namespace_cols: Sequence[str] = ("parent_domain_id", "fiber_label"),
    extra_group_cols: Sequence[str] = (),
    namespace_window_size: int = 1,
    require_generated: bool = False,
    require_flat_start: bool = False,
    require_persistent: bool = False,
    chart_cycle_chunk_size: int = 250000,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not chart_cycles_csv or not os.path.exists(chart_cycles_csv):
        raise FileNotFoundError(f"chart cycles CSV not found: {chart_cycles_csv}")
    meta = candidate_meta.copy() if candidate_meta is not None else pd.DataFrame()
    if meta.empty:
        allowed: Optional[Set[str]] = None
        meta_lookup: Dict[str, Dict[str, Any]] = {}
    else:
        meta_lookup = {str(r["base_candidate_id"]): r.to_dict() for _, r in meta.iterrows()}
        m = meta.copy()
        mask = pd.Series(True, index=m.index)
        if require_generated and "generated_c2_candidate" in m.columns:
            mask &= m["generated_c2_candidate"].map(_safe_bool)
        if require_flat_start and "initial_c2_count" in m.columns:
            mask &= pd.to_numeric(m["initial_c2_count"], errors="coerce").fillna(0).astype(int).eq(0)
        if require_persistent and "c2_persistent_to_final" in m.columns:
            mask &= m["c2_persistent_to_final"].map(_safe_bool)
        allowed = set(map(str, m.loc[mask, "base_candidate_id"]))
    stats = Counter()
    rows: List[Dict[str, Any]] = []
    reader: Iterable[pd.DataFrame]
    if int(chart_cycle_chunk_size) > 0:
        reader = pd.read_csv(chart_cycles_csv, chunksize=int(chart_cycle_chunk_size), low_memory=False)
    else:
        reader = [pd.read_csv(chart_cycles_csv, low_memory=False)]
    for chunk in reader:
        stats["raw_rows"] += int(len(chunk))
        if chunk.empty:
            continue
        if "candidate_id" not in chunk.columns:
            chunk = chunk.copy()
            chunk["candidate_id"] = chunk.apply(_candidate_id_from_fields, axis=1)
        chunk["base_candidate_id"] = chunk["candidate_id"].astype(str).map(_base_candidate_id)
        if allowed is not None:
            before = len(chunk)
            chunk = chunk[chunk["base_candidate_id"].isin(allowed)].copy()
            stats["skipped_candidate_filter"] += int(before - len(chunk))
        if chunk.empty:
            continue
        if "chart_loop_valid" in chunk.columns:
            before = len(chunk)
            chunk = chunk[chunk["chart_loop_valid"].map(_safe_bool)].copy()
            stats["skipped_invalid_loop"] += int(before - len(chunk))
        if "chart_c2_holonomy" in chunk.columns:
            before = len(chunk)
            chunk = chunk[chunk["chart_c2_holonomy"].map(_safe_bool)].copy()
            stats["skipped_non_c2_flag"] += int(before - len(chunk))
        if chunk.empty:
            continue
        itcol = _auto_iteration_col(chunk.columns)
        if itcol is None:
            raise ValueError("chart cycles CSV needs atlas_iteration/replay_iteration/iteration column")
        for idx, r in chunk.iterrows():
            cid = str(r["base_candidate_id"])
            md = meta_lookup.get(cid, {})
            first_c2 = _safe_int(md.get("first_c2_iteration", -1), -1)
            it = _safe_int(r.get(itcol), -1)
            if int(focus_first_c2_window) >= 0:
                if first_c2 < 0 or abs(int(it) - int(first_c2)) > int(focus_first_c2_window):
                    stats["skipped_birth_window"] += 1
                    continue
            lm = _parse_loop_map(r.get("loop_map", ""))
            if not lm:
                stats["skipped_no_loop_map"] += 1
                continue
            labels = sorted(set(lm.keys()) | set(lm.values()))
            p = _perm_from_label_map(lm, labels)
            if p is None:
                stats["skipped_not_permutation"] += 1
                continue
            order = _perm_order(p)
            if order != 2:
                stats["skipped_not_order2"] += 1
                continue
            w0, w1 = _parse_windowed_iteration(it, first_c2, int(namespace_window_size))
            row: Dict[str, Any] = {
                "candidate_id": cid,
                "source_candidate_id": str(r.get("candidate_id", cid)),
                "atlas_iteration": int(it),
                "window_start": int(w0),
                "window_end": int(w1),
                "first_c2_iteration": int(first_c2),
                "iteration_minus_first_c2": int(it - first_c2) if first_c2 >= 0 else 999999,
                "loop_map": _compact_map(lm),
                "loop_order": int(order),
                "loop_cycle_type": " ".join(map(str, _cycle_type(p))),
                "loop_labels": " ".join(map(str, labels)),
                "row_index": int(idx),
                "cycle_id": r.get("cycle_id", ""),
                "cycle_charts": r.get("cycle_charts", ""),
                "rule_mode": md.get("rule_mode", r.get("rule_mode", "")),
                "instance": md.get("instance", r.get("instance", "")),
                "profile": md.get("profile", r.get("profile", "")),
                "atlas_capacity": md.get("atlas_capacity", r.get("atlas_capacity", "")),
                "initial_seed": md.get("initial_seed", r.get("initial_seed", "")),
            }
            for col in list(namespace_cols) + list(extra_group_cols):
                row[col] = r.get(col, "")
            row["group_key"] = json.dumps([cid, int(w0)] + [str(row.get(c, "")) for c in list(namespace_cols) + list(extra_group_cols)], sort_keys=True)
            row["_map_tuple"] = tuple(sorted((int(k), int(v)) for k, v in lm.items()))
            is_trans, edge = _is_single_transposition(lm)
            row["is_single_transposition"] = bool(is_trans)
            row["transposition_edge"] = f"{edge[0]}-{edge[1]}" if edge else ""
            rows.append(row)
            stats["parsed_order2_rows"] += 1
    return rows, dict(stats)


def _analyze_namespace_group(rows: List[Dict[str, Any]], max_group_order: int = 4096) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    first = rows[0]
    # Deduplicate exact C2 maps.
    distinct: List[Dict[str, Any]] = []
    seen: Set[Tuple[Tuple[int, int], ...]] = set()
    dup = 0
    for r in rows:
        key = tuple(r.get("_map_tuple", ()))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        distinct.append(r)
    maps: List[LabelMap] = []
    union_labels: Set[int] = set()
    trans_edges: List[Tuple[int, int]] = []
    general_involutions = 0
    gen_rows: List[Dict[str, Any]] = []
    for gi, r in enumerate(distinct):
        lm = dict(r.get("_map_tuple", ()))
        lm = {int(k): int(v) for k, v in lm.items()}
        labels = sorted(set(lm.keys()) | set(lm.values()))
        union_labels.update(labels)
        maps.append(lm)
        is_trans, edge = _is_single_transposition(lm)
        if is_trans and edge is not None:
            trans_edges.append(edge)
        else:
            general_involutions += 1
        gen_rows.append({
            **{k: first.get(k) for k in ["candidate_id", "rule_mode", "instance", "profile", "atlas_capacity", "initial_seed", "window_start", "window_end", "first_c2_iteration"]},
            "atlas_iteration": r.get("atlas_iteration"),
            "parent_domain_id": first.get("parent_domain_id", ""),
            "fiber_label": first.get("fiber_label", ""),
            "generator_index": int(gi),
            "row_index": int(r.get("row_index", -1)),
            "cycle_id": r.get("cycle_id", ""),
            "cycle_charts": r.get("cycle_charts", ""),
            "loop_map": _compact_map(lm),
            "is_single_transposition": bool(is_trans),
            "transposition_edge": f"{edge[0]}-{edge[1]}" if edge else "",
            "union_core_size": int(len(union_labels)),
        })
    union_core = sorted(union_labels)
    perms: List[Perm] = []
    for lm in maps:
        ext = {x: x for x in union_core}
        ext.update(lm)
        p = _perm_from_label_map(ext, union_core)
        if p is not None:
            perms.append(p)
    group: Set[Perm] = set()
    truncated = False
    if perms:
        group, truncated = _close_group(perms, max_group_order=int(max_group_order))
    orders = Counter(_perm_order(g) for g in group) if group else Counter()
    comp_sizes = _component_sizes(trans_edges)
    shared, disjoint = _shared_disjoint_pair_counts(trans_edges)
    exact_s3 = bool((not truncated) and len(group) == 6 and (not _is_abelian(list(group))) and orders.get(3, 0) > 0)
    comp_ge3 = bool(comp_sizes and max(comp_sizes) >= 3)
    comp_ge4 = bool(comp_sizes and max(comp_sizes) >= 4)
    if exact_s3:
        family = "S3_critical_birth_window"
    elif shared > 0 or comp_ge3:
        family = "overlapping_transposition_near_miss"
    elif len(set(trans_edges)) >= 2 and shared == 0:
        family = "disjoint_C2_product_near_miss"
    elif len(distinct) >= 2:
        family = "multi_involution_no_overlap"
    else:
        family = "C2_single_or_duplicate_flip"
    row = {
        "candidate_id": first.get("candidate_id"),
        "rule_mode": first.get("rule_mode", ""),
        "instance": _safe_int(first.get("instance"), 0),
        "profile": first.get("profile", ""),
        "atlas_capacity": _safe_int(first.get("atlas_capacity"), 0),
        "initial_seed": _safe_int(first.get("initial_seed"), 0),
        "window_start": _safe_int(first.get("window_start"), -1),
        "window_end": _safe_int(first.get("window_end"), -1),
        "first_c2_iteration": _safe_int(first.get("first_c2_iteration"), -1),
        "iteration_minus_first_c2": _safe_int(first.get("iteration_minus_first_c2"), 999999),
        "parent_domain_id": _safe_int(first.get("parent_domain_id"), -1),
        "fiber_label": first.get("fiber_label", ""),
        "group_key": first.get("group_key", ""),
        "n_raw_order2_rows": int(len(rows)),
        "n_duplicate_order2_rows": int(dup),
        "duplicate_map_fraction": float(dup / max(1, len(rows))),
        "n_distinct_c2_maps": int(len(distinct)),
        "n_single_transpositions": int(len(trans_edges)),
        "n_distinct_single_transposition_edges": int(len(set(trans_edges))),
        "n_general_involutions": int(general_involutions),
        "support_label_count": int(len(union_core)),
        "union_core_labels": " ".join(map(str, union_core[:100])),
        "transposition_edges": " ".join(f"{a}-{b}" for a, b in sorted(set(trans_edges))),
        "transposition_component_sizes": " ".join(map(str, comp_sizes)),
        "max_transposition_component_size": int(max(comp_sizes) if comp_sizes else 0),
        "shared_element_transposition_pair_count": int(shared),
        "disjoint_transposition_pair_count": int(disjoint),
        "contains_s3_by_transposition_pair": bool(shared > 0),
        "contains_s3_by_transposition_graph": bool(comp_ge3),
        "contains_component_ge4": bool(comp_ge4),
        "generated_group_order": int(len(group)),
        "group_closure_truncated": bool(truncated),
        "generated_group_abelian": bool(_is_abelian(list(group)) if group and not truncated else False),
        "generated_group_nonabelian": bool(group and not truncated and not _is_abelian(list(group)) and len(group) > 1),
        "element_order_counts": ";".join(f"{k}:{v}" for k, v in sorted(orders.items())),
        "has_order3": bool(orders.get(3, 0) > 0),
        "exact_s3_group_closure": bool(exact_s3),
        "two_plus_distinct_maps": bool(len(distinct) >= 2),
        "two_plus_distinct_transpositions": bool(len(set(trans_edges)) >= 2),
        "two_plus_disjoint_transpositions_only": bool(len(set(trans_edges)) >= 2 and shared == 0),
        "transposition_diversity_score": float((len(set(trans_edges)) - 1) + 2 * shared + max(0, (max(comp_sizes) if comp_sizes else 0) - 2)),
        "algebra_family": family,
    }
    events: List[Dict[str, Any]] = []
    if shared > 0 or exact_s3 or len(set(trans_edges)) >= 2:
        events.append({
            **{k: row.get(k) for k in ["candidate_id", "rule_mode", "instance", "profile", "atlas_capacity", "initial_seed", "window_start", "window_end", "parent_domain_id", "fiber_label", "first_c2_iteration"]},
            "event_kind": "s3" if exact_s3 else ("shared_label_pair" if shared > 0 else "distinct_disjoint_near_miss"),
            "exact_s3_group_closure": bool(exact_s3),
            "shared_element_transposition_pair_count": int(shared),
            "transposition_edges": row["transposition_edges"],
            "max_transposition_component_size": row["max_transposition_component_size"],
            "generated_group_order": row["generated_group_order"],
        })
    return row, gen_rows, events

# ---------------------------------------------------------------------------
# Join provenance
# ---------------------------------------------------------------------------
def _add_provenance(rows: pd.DataFrame, iter_lookup: Dict[str, Dict[int, Dict[str, Any]]], domain_lookup: Dict[Tuple[str, int, int], Dict[str, Any]]) -> pd.DataFrame:
    if rows.empty:
        return rows
    outs: List[Dict[str, Any]] = []
    for _, rr in rows.iterrows():
        r = rr.to_dict()
        cid = str(r.get("candidate_id"))
        it = _safe_int(r.get("window_start"), -1)
        prev = it - 1
        currow = iter_lookup.get(cid, {}).get(it)
        prevrow = iter_lookup.get(cid, {}).get(prev)
        r.update(_iter_prefix(currow, "iter"))
        r.update(_iter_prefix(prevrow, "prev"))
        first = _safe_int(r.get("first_c2_iteration"), -1)
        r["iteration_minus_first_c2"] = int(it - first) if first >= 0 else 999999
        sig_delta = _safe_int(r.get("iter_full_signature_classes"), 0) - _safe_int(r.get("prev_full_signature_classes"), 0)
        fiber_delta = _safe_float(r.get("iter_bounded_atlas_fiber_entropy_bits"), 0.0) - _safe_float(r.get("prev_bounded_atlas_fiber_entropy_bits"), 0.0)
        prev_valid = _safe_float(r.get("prev_n_chart_valid_cycles"), 0.0)
        iter_valid = _safe_float(r.get("iter_n_chart_valid_cycles"), 0.0)
        drop_frac = max(0.0, (prev_valid - iter_valid) / max(1.0, prev_valid)) if prev_valid > 0 else 0.0
        first_window = bool(first >= 0 and abs(it - first) <= 1)
        c2_birth = bool(_safe_int(r.get("iter_n_chart_c2"), 0) > 0 and _safe_int(r.get("prev_n_chart_c2"), 0) == 0)
        rebound = max(0.0, float(sig_delta)) + 50.0 * max(0.0, float(fiber_delta)) + 100.0 * drop_frac + (25.0 if c2_birth else 0.0)
        r.update({
            "signature_class_delta_from_prev": int(sig_delta),
            "fiber_entropy_delta_from_prev": float(fiber_delta),
            "valid_cycle_drop_fraction_from_prev": float(drop_frac),
            "first_c2_birth_window": bool(first_window),
            "c2_birth_from_prev_window": bool(c2_birth),
            "compression_rebound_score": float(rebound),
        })
        did = _safe_int(r.get("parent_domain_id"), -999999)
        dom = domain_lookup.get((cid, it, did), {}) if did != -999999 else {}
        for k, v in dom.items():
            if k in {"candidate_id", "base_candidate_id"}:
                continue
            r[f"domain_{k}"] = v
        depth = _safe_int(dom.get("depth"), 0) if dom else 0
        live = _safe_float(dom.get("live_fiber_count"), 0.0) if dom else 0.0
        maxpred = _safe_float(dom.get("max_predictive_classes_per_fiber"), 0.0) if dom else 0.0
        entropy = _safe_float(dom.get("entropy_bits"), 0.0) if dom else 0.0
        r["s3_search_priority_score"] = float(
            5.0 * _safe_float(r.get("transposition_diversity_score"), 0.0)
            + 0.01 * rebound
            + 0.5 * max(0.0, depth - 2)
            + 0.05 * live
            + 0.02 * maxpred
            + 0.05 * entropy
            + (2.0 if c2_birth else 0.0)
        )
        outs.append(r)
    return pd.DataFrame(outs)

# ---------------------------------------------------------------------------
# Summary / plotting
# ---------------------------------------------------------------------------
def _hist_str(vals: Sequence[Any], max_items: int = 20) -> Dict[str, int]:
    c = Counter(str(v) for v in vals)
    return {k: int(v) for k, v in c.most_common(max_items)}


def _summarize(df: pd.DataFrame, gen_df: pd.DataFrame, events_df: pd.DataFrame, stats: Dict[str, int], args: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "verdict": "CRITICAL C2-BIRTH WINDOW AUDIT EMPTY",
            "audit_version": "critical_c2_birth_window_v1_transposition_diversity_provenance",
            "n_namespace_rows": 0,
            "parse_stats": stats,
            "args": args,
        }
    exact = df["exact_s3_group_closure"].map(_safe_bool)
    shared = df["contains_s3_by_transposition_pair"].map(_safe_bool)
    twoplus = df["two_plus_distinct_maps"].map(_safe_bool)
    comp_ge3 = pd.to_numeric(df["max_transposition_component_size"], errors="coerce").fillna(0) >= 3
    comp_ge4 = pd.to_numeric(df["max_transposition_component_size"], errors="coerce").fillna(0) >= 4
    if exact.any():
        verdict = "CRITICAL C2-BIRTH WINDOW S3 SIGNAL: overlapping C2 flips found in a generated birth/rebound window"
    elif shared.any() or comp_ge3.any():
        verdict = "CRITICAL C2-BIRTH WINDOW NEAR-S3 SIGNAL: shared-label transposition diversity found, exact closure not certified"
    elif twoplus.any():
        verdict = "CRITICAL C2-BIRTH WINDOW DIVERSITY NEAR-MISS: multiple C2 flips found, but no overlapping transpositions"
    else:
        verdict = "CRITICAL C2-BIRTH WINDOW C2 BASELINE: birth windows contain only single/duplicate flips"
    by_cols = [c for c in ["rule_mode", "profile", "atlas_capacity"] if c in df.columns]
    by: List[Dict[str, Any]] = []
    if by_cols:
        for key, g in df.groupby(by_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            r = {c: key[i] for i, c in enumerate(by_cols)}
            r.update({
                "n": int(len(g)),
                "exact_s3_fraction": float(g["exact_s3_group_closure"].map(_safe_bool).mean()),
                "shared_pair_fraction": float(g["contains_s3_by_transposition_pair"].map(_safe_bool).mean()),
                "two_plus_maps_fraction": float(g["two_plus_distinct_maps"].map(_safe_bool).mean()),
                "max_transposition_component_size": int(pd.to_numeric(g["max_transposition_component_size"], errors="coerce").fillna(0).max()),
                "mean_duplicate_map_fraction": float(pd.to_numeric(g["duplicate_map_fraction"], errors="coerce").fillna(0).mean()),
                "mean_compression_rebound_score": float(pd.to_numeric(g.get("compression_rebound_score", pd.Series([0]*len(g))), errors="coerce").fillna(0).mean()),
                "mean_s3_search_priority_score": float(pd.to_numeric(g.get("s3_search_priority_score", pd.Series([0]*len(g))), errors="coerce").fillna(0).mean()),
            })
            by.append(r)
    summary: Dict[str, Any] = {
        "verdict": verdict,
        "audit_version": "critical_c2_birth_window_v1_transposition_diversity_provenance",
        "n_namespace_rows": int(len(df)),
        "n_generator_rows": int(len(gen_df)) if gen_df is not None else 0,
        "n_event_rows": int(len(events_df)) if events_df is not None else 0,
        "parse_stats": stats,
        "algebra_family_counts": {str(k): int(v) for k, v in df["algebra_family"].value_counts(dropna=False).items()},
        "any_exact_s3_group_closure": bool(exact.any()),
        "any_shared_label_transposition_pair": bool(shared.any()),
        "any_two_plus_distinct_maps": bool(twoplus.any()),
        "any_transposition_component_ge3": bool(comp_ge3.any()),
        "any_transposition_component_ge4": bool(comp_ge4.any()),
        "exact_s3_fraction": float(exact.mean()),
        "shared_label_pair_fraction": float(shared.mean()),
        "two_plus_distinct_maps_fraction": float(twoplus.mean()),
        "max_distinct_maps": int(pd.to_numeric(df["n_distinct_c2_maps"], errors="coerce").fillna(0).max()),
        "max_transposition_component_size": int(pd.to_numeric(df["max_transposition_component_size"], errors="coerce").fillna(0).max()),
        "max_generated_group_order": int(pd.to_numeric(df["generated_group_order"], errors="coerce").fillna(0).max()),
        "mean_duplicate_map_fraction": float(pd.to_numeric(df["duplicate_map_fraction"], errors="coerce").fillna(0).mean()),
        "mean_compression_rebound_score": float(pd.to_numeric(df.get("compression_rebound_score", pd.Series([0]*len(df))), errors="coerce").fillna(0).mean()),
        "top_priority_rows": _json_safe(df.sort_values("s3_search_priority_score", ascending=False).head(10)[[c for c in ["candidate_id", "window_start", "parent_domain_id", "fiber_label", "algebra_family", "n_distinct_c2_maps", "max_transposition_component_size", "compression_rebound_score", "s3_search_priority_score"] if c in df.columns]].to_dict(orient="records")),
        "distinct_maps_per_namespace_histogram": _hist_str(df["n_distinct_c2_maps"].astype(int).tolist()),
        "component_size_histogram": _hist_str(df["max_transposition_component_size"].astype(int).tolist()),
        "iteration_minus_first_c2_histogram": _hist_str(df["iteration_minus_first_c2"].astype(int).tolist()) if "iteration_minus_first_c2" in df.columns else {},
        "by_mode_profile_capacity": by,
        "args": args,
    }
    return summary


def _plot_summary(df: pd.DataFrame, summary: Dict[str, Any], path: str) -> None:
    if not path:
        return
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        labels = ["exact S3", "shared pair", "2+ maps", "comp>=3", "comp>=4"]
        vals = [
            float(summary.get("exact_s3_fraction", 0.0)),
            float(summary.get("shared_label_pair_fraction", 0.0)),
            float(summary.get("two_plus_distinct_maps_fraction", 0.0)),
            float(df["max_transposition_component_size"].ge(3).mean()) if len(df) and "max_transposition_component_size" in df else 0.0,
            float(df["max_transposition_component_size"].ge(4).mean()) if len(df) and "max_transposition_component_size" in df else 0.0,
        ]
        ax.bar(labels, vals)
        ax.set_ylim(0, max(0.05, max(vals) * 1.25 if vals else 0.05))
        ax.set_ylabel("fraction of namespace groups")
        ax.set_title(str(summary.get("verdict", "critical C2 birth window audit")))
        ax.tick_params(axis='x', rotation=20)
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f"plot failed: {e}")

# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_critical_c2_birth_window_audit(
    chart_cycles_csv: str,
    iterated_csv: str = "",
    domains_csv: str = "",
    out: str = "",
    plot: str = "",
    namespace_cols: str = "parent_domain_id,fiber_label",
    extra_group_cols: str = "",
    focus_first_c2_window: int = 1,
    namespace_window_size: int = 1,
    require_generated: bool = True,
    require_flat_start: bool = True,
    require_persistent: bool = False,
    rule_modes: str = "",
    profiles: str = "",
    atlas_capacities: str = "",
    chart_cycle_chunk_size: int = 250000,
    max_group_order: int = 4096,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    ns_cols = [c.strip() for c in str(namespace_cols).replace(";", ",").split(",") if c.strip()]
    ex_cols = [c.strip() for c in str(extra_group_cols).replace(";", ",").split(",") if c.strip()]
    iter_df, meta_df, iter_lookup = _load_iterated_meta(iterated_csv, rule_modes=rule_modes, profiles=profiles, capacities=atlas_capacities)
    domain_lookup = _load_domain_lookup(domains_csv)
    order2_rows, stats = _collect_order2_rows_from_chart_cycles(
        chart_cycles_csv=chart_cycles_csv,
        candidate_meta=meta_df,
        focus_first_c2_window=int(focus_first_c2_window),
        namespace_cols=ns_cols,
        extra_group_cols=ex_cols,
        namespace_window_size=int(namespace_window_size),
        require_generated=bool(require_generated),
        require_flat_start=bool(require_flat_start),
        require_persistent=bool(require_persistent),
        chart_cycle_chunk_size=int(chart_cycle_chunk_size),
    )
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in order2_rows:
        by_group[str(r["group_key"])].append(r)
    rows: List[Dict[str, Any]] = []
    gen_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    for _gkey, rs in by_group.items():
        row, gens, events = _analyze_namespace_group(rs, max_group_order=int(max_group_order))
        rows.append(row)
        gen_rows.extend(gens)
        event_rows.extend(events)
    rdf = pd.DataFrame(rows)
    gdf = pd.DataFrame(gen_rows)
    edf = pd.DataFrame(event_rows)
    if not rdf.empty:
        rdf = _add_provenance(rdf, iter_lookup, domain_lookup)
        # Propagate selected provenance to generator/event rows.
        if not gdf.empty:
            join_cols = ["candidate_id", "window_start", "parent_domain_id", "fiber_label"]
            prov_cols = [c for c in ["compression_rebound_score", "s3_search_priority_score", "iteration_minus_first_c2", "domain_depth", "domain_n_labels", "domain_live_fiber_count"] if c in rdf.columns]
            if prov_cols:
                gdf = gdf.merge(rdf[join_cols + prov_cols], on=join_cols, how="left")
        if not edf.empty:
            join_cols = ["candidate_id", "window_start", "parent_domain_id", "fiber_label"]
            prov_cols = [c for c in ["compression_rebound_score", "s3_search_priority_score", "iteration_minus_first_c2", "domain_depth", "domain_n_labels", "domain_live_fiber_count"] if c in rdf.columns]
            if prov_cols:
                edf = edf.merge(rdf[join_cols + prov_cols], on=join_cols, how="left")
    args = dict(
        chart_cycles_csv=chart_cycles_csv,
        iterated_csv=iterated_csv,
        domains_csv=domains_csv,
        namespace_cols=ns_cols,
        extra_group_cols=ex_cols,
        focus_first_c2_window=int(focus_first_c2_window),
        namespace_window_size=int(namespace_window_size),
        require_generated=bool(require_generated),
        require_flat_start=bool(require_flat_start),
        require_persistent=bool(require_persistent),
        rule_modes=rule_modes,
        profiles=profiles,
        atlas_capacities=atlas_capacities,
        chart_cycle_chunk_size=int(chart_cycle_chunk_size),
        max_group_order=int(max_group_order),
    )
    summary = _summarize(rdf, gdf, edf, stats, args)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        rdf.to_csv(out, index=False)
        base = out[:-4] if out.lower().endswith(".csv") else out
        gdf.to_csv(base + "_generators.csv", index=False)
        edf.to_csv(base + "_events.csv", index=False)
        with open(base + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
    if plot:
        _plot_summary(rdf, summary, plot)
    return rdf, gdf, edf, summary

# ---------------------------------------------------------------------------
# Synthetic smoke
# ---------------------------------------------------------------------------
def _synthetic_smoke(out: str = "", plot: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # Synthetic rows equivalent to a candidate born flat at it=8 with one S3 namespace,
    # one disjoint near-miss, and one duplicate-only baseline.
    rows = []
    cid = "synthetic|inst=0|full_atlas|cap=32|seed=1"
    def add(it, pdid, fib, cycle, lm, mode="synthetic"):
        rows.append({
            "candidate_id": cid, "base_candidate_id": cid, "atlas_iteration": it,
            "first_c2_iteration": 8, "iteration_minus_first_c2": it-8,
            "loop_map": lm, "row_index": len(rows), "cycle_id": cycle, "cycle_charts": "1 2 3 4",
            "rule_mode": mode, "instance": 0, "profile": "full_atlas", "atlas_capacity": 32, "initial_seed": 1,
            "parent_domain_id": pdid, "fiber_label": fib,
            "group_key": json.dumps([cid, it, str(pdid), str(fib)]),
        })
    add(8, 75, 7, 73, "0->1 1->0")
    add(8, 75, 7, 139, "0->2 2->0")
    add(8, 76, 1, 1, "0->1 1->0")
    add(8, 76, 1, 2, "0->1 1->0")
    add(8, 77, 1, 3, "0->1 1->0")
    add(8, 77, 1, 4, "2->3 3->2")
    for r in rows:
        lm = _parse_loop_map(r["loop_map"])
        r["_map_tuple"] = tuple(sorted(lm.items()))
        trans, edge = _is_single_transposition(lm)
        r["is_single_transposition"] = trans
        r["transposition_edge"] = f"{edge[0]}-{edge[1]}" if edge else ""
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group_key"]].append(r)
    outrows, gens, evs = [], [], []
    for rs in by_group.values():
        row, g, e = _analyze_namespace_group(rs)
        # Add fake provenance.
        row.update({
            "iter_n_chart_c2": 2, "prev_n_chart_c2": 0,
            "iter_full_signature_classes": 394, "prev_full_signature_classes": 223,
            "iter_bounded_atlas_fiber_entropy_bits": 7.08, "prev_bounded_atlas_fiber_entropy_bits": 5.57,
            "iter_n_chart_valid_cycles": 1125, "prev_n_chart_valid_cycles": 9571,
            "compression_rebound_score": 300.0,
            "domain_depth": 4, "domain_n_labels": 82, "domain_live_fiber_count": 13,
            "s3_search_priority_score": 15.0 if row.get("exact_s3_group_closure") else 3.0,
        })
        outrows.append(row); gens.extend(g); evs.extend(e)
    rdf, gdf, edf = pd.DataFrame(outrows), pd.DataFrame(gens), pd.DataFrame(evs)
    summary = _summarize(rdf, gdf, edf, {"synthetic_rows": len(rows)}, {"synthetic_smoke": True})
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        rdf.to_csv(out, index=False)
        base = out[:-4] if out.lower().endswith(".csv") else out
        gdf.to_csv(base + "_generators.csv", index=False)
        edf.to_csv(base + "_events.csv", index=False)
        with open(base + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
    if plot:
        _plot_summary(rdf, summary, plot)
    return rdf, gdf, edf, summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Critical C2 birth-window transposition-diversity provenance audit")
    ap.add_argument("--chart-cycles-csv", default="")
    ap.add_argument("--iterated-csv", default="")
    ap.add_argument("--domains-csv", default="")
    ap.add_argument("--namespace-cols", default="parent_domain_id,fiber_label")
    ap.add_argument("--extra-group-cols", default="")
    ap.add_argument("--focus-first-c2-window", type=int, default=1)
    ap.add_argument("--namespace-window-size", type=int, default=1)
    ap.add_argument("--require-generated", action="store_true")
    ap.add_argument("--no-require-generated", dest="require_generated", action="store_false")
    ap.set_defaults(require_generated=True)
    ap.add_argument("--require-flat-start", action="store_true")
    ap.add_argument("--no-require-flat-start", dest="require_flat_start", action="store_false")
    ap.set_defaults(require_flat_start=True)
    ap.add_argument("--require-persistent", action="store_true")
    ap.add_argument("--rule-modes", default="")
    ap.add_argument("--profiles", default="")
    ap.add_argument("--atlas-capacities", default="")
    ap.add_argument("--chart-cycle-chunk-size", type=int, default=250000)
    ap.add_argument("--max-group-order", type=int, default=4096)
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", default="")
    ap.add_argument("--synthetic-smoke", action="store_true")
    args = ap.parse_args()
    if args.synthetic_smoke:
        rdf, gdf, edf, summary = _synthetic_smoke(args.out, args.plot)
    else:
        if not args.chart_cycles_csv:
            raise SystemExit("--chart-cycles-csv is required unless --synthetic-smoke is used")
        rdf, gdf, edf, summary = run_critical_c2_birth_window_audit(
            chart_cycles_csv=str(args.chart_cycles_csv),
            iterated_csv=str(args.iterated_csv),
            domains_csv=str(args.domains_csv),
            out=str(args.out),
            plot=str(args.plot),
            namespace_cols=str(args.namespace_cols),
            extra_group_cols=str(args.extra_group_cols),
            focus_first_c2_window=int(args.focus_first_c2_window),
            namespace_window_size=int(args.namespace_window_size),
            require_generated=bool(args.require_generated),
            require_flat_start=bool(args.require_flat_start),
            require_persistent=bool(args.require_persistent),
            rule_modes=str(args.rule_modes),
            profiles=str(args.profiles),
            atlas_capacities=str(args.atlas_capacities),
            chart_cycle_chunk_size=int(args.chart_cycle_chunk_size),
            max_group_order=int(args.max_group_order),
        )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    if args.out:
        print(f"wrote {args.out}")
        base = args.out[:-4] if args.out.lower().endswith(".csv") else args.out
        print(f"wrote {base}_generators.csv")
        print(f"wrote {base}_events.csv")
        print(f"wrote {base}_summary.json")
    if args.plot:
        print(f"wrote {args.plot}")


if __name__ == "__main__":  # pragma: no cover
    main()
