"""
Cloud Wilson/scaling and attractor evidence runner for SIQ/RelGauge on Modal.

Run from a project root that contains a local `relgauge/` Python package:

    python -m pip install modal
    python -m modal setup
    modal run cloud_wilson_scaling.py

This script runs real RelGauge computations, not summary readers. It produces six
sections of output:

1. q-specificity at horizon=4;
2. q=3 horizon hierarchy;
3. lift-mode/axiom test;
4. Wilson-loop scaling for q=3,h=4;
5. structural invariants of S3 fixed points;
6. q=6 coexistence scan.

The core object is basepoint-aware predictive-quotient isotropy. No raw label
mixing across chart bases is used.
"""
from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import subprocess
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import modal

APP_NAME = "siq-definitive-wilson-scaling"
REMOTE_ROOT = pathlib.Path("/root")
REMOTE_WORK = pathlib.Path("/tmp/siq_definitive")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "matplotlib")
    .add_local_dir("relgauge", remote_path="/root/relgauge")
)

app = modal.App(APP_NAME, image=image)

# ----------------------------- generic helpers -----------------------------

def _run(cmd: List[str], timeout: int = 21600, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    env["MPLBACKEND"] = "Agg"
    print("RUN:", " ".join(cmd), flush=True)
    cp = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=int(timeout),
    )
    print(cp.stdout[-10000:], flush=True)
    if cp.returncode != 0:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stdout}")
    return cp


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "" or str(x).lower() == "nan":
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "" or str(x).lower() == "nan":
            return default
        return int(float(x))
    except Exception:
        return default


def _state_count(q: int, vertices: int) -> int:
    return int(int(q) ** int(vertices))


def _parse_int_map(text: str, default: Dict[int, int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        a, b = part.split(":", 1)
        out[int(a)] = int(b)
    return out or dict(default)


def _parse_int_list(text: str, default: Iterable[int]) -> List[int]:
    vals: List[int] = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals or list(default)


def _entropy_from_counts(counts: Iterable[int]) -> float:
    vals = [int(x) for x in counts if int(x) > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log(c / total, 2) for c in vals))


def _to_base_digits(i: int, q: int, v: int) -> List[int]:
    digs = [0] * int(v)
    x = int(i)
    for k in range(int(v) - 1, -1, -1):
        digs[k] = x % int(q)
        x //= int(q)
    return digs


# ----------------------- parsing and aggregation helpers --------------------

def _summarize_iteration_outputs(outdir: pathlib.Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    attractor_path = outdir / "attractor_classification.csv"
    s3_path = outdir / "s3_rate_by_iteration.csv"
    attr_rows = _read_csv(attractor_path)
    s3_rows = _read_csv(s3_path)
    counts = Counter(str(r.get("gauge_at_fixed_point", "")).strip() for r in attr_rows if str(r.get("gauge_at_fixed_point", "")).strip())
    instances = int(cfg.get("instances", len(attr_rows) or 1))
    s3_fp_count = counts.get("S3", 0)
    c3_fp_count = counts.get("C3", 0)
    c2_fp_count = counts.get("C2", 0)
    flat_fp_count = counts.get("flat", 0)
    no_fp_count = max(0, instances - len(attr_rows))
    iter0 = [r for r in s3_rows if _safe_int(r.get("iteration"), -999) == 0]
    all_n_s3_0 = sum(_safe_int(r.get("n_s3")) for r in iter0)
    all_n_total_0 = sum(_safe_int(r.get("n_total")) for r in iter0)
    iter0_pooled_s3_rate = (all_n_s3_0 / all_n_total_0) if all_n_total_0 else 0.0
    iter_means: Dict[int, Dict[str, float]] = {}
    byit: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for r in s3_rows:
        byit[_safe_int(r.get("iteration"), 0)].append(r)
    for it, rows in byit.items():
        n_s3 = sum(_safe_int(r.get("n_s3")) for r in rows)
        n_total = sum(_safe_int(r.get("n_total")) for r in rows)
        iter_means[int(it)] = {
            "mean_s3_rate": sum(_safe_float(r.get("s3_rate")) for r in rows) / max(1, len(rows)),
            "pooled_s3_rate": (n_s3 / n_total) if n_total else 0.0,
            "n_s3": n_s3,
            "n_total": n_total,
        }
    max_s3_rate = max([_safe_float(r.get("s3_rate")) for r in s3_rows] + [0.0])
    max_mean_s3_rate = max([d["mean_s3_rate"] for d in iter_means.values()] + [0.0])
    return {
        "q": int(cfg.get("q")),
        "vertices": int(cfg.get("vertices")),
        "states": _state_count(int(cfg.get("q")), int(cfg.get("vertices"))),
        "horizon": int(cfg.get("horizon")),
        "lift": str(cfg.get("lift", "bijective")),
        "instances": instances,
        "classified_fixed_points": len(attr_rows),
        "no_fixed_point_count": int(no_fp_count),
        "attractor_counts": dict(counts),
        "s3_fp_count": int(s3_fp_count),
        "c3_fp_count": int(c3_fp_count),
        "c2_fp_count": int(c2_fp_count),
        "flat_fp_count": int(flat_fp_count),
        "s3_fp_rate": float(s3_fp_count / max(1, instances)),
        "c3_fp_rate": float(c3_fp_count / max(1, instances)),
        "c2_fp_rate": float(c2_fp_count / max(1, instances)),
        "flat_fp_rate": float(flat_fp_count / max(1, instances)),
        "iter0_s3_rate_pooled": float(iter0_pooled_s3_rate),
        "iter0_n_s3": int(all_n_s3_0),
        "iter0_n_total": int(all_n_total_0),
        "fp_over_iter0_ratio": float((s3_fp_count / max(1, instances)) / max(1e-12, iter0_pooled_s3_rate)),
        "max_s3_rate": float(max_s3_rate),
        "max_mean_s3_rate": float(max_mean_s3_rate),
        "iteration_summary": iter_means,
        "attractor_csv": str(attractor_path),
        "s3_rate_csv": str(s3_path),
    }


def _summarize_microscope_outputs(outdir: pathlib.Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    anatomy_rows = _read_csv(outdir / "fixed_point_anatomy.csv")
    rep_rows = _read_csv(outdir / "representation_decomposition.csv")
    wilson_rows = _read_csv(outdir / "wilson_loops.csv")
    s3_anatomy = [r for r in anatomy_rows if str(r.get("attractor_class")) == "S3"]

    # Weighted Wilson means by loop length.
    wilson_by_L: Dict[int, Dict[str, float]] = {}
    for L in [3, 4, 5]:
        rows = [r for r in wilson_rows if _safe_int(r.get("loop_length"), -1) == L]
        n = sum(_safe_int(r.get("n_loops"), 0) for r in rows)
        if n:
            mean = sum(_safe_float(r.get("mean_wilson_character")) * _safe_int(r.get("n_loops"), 0) for r in rows) / n
            fid = sum(_safe_float(r.get("fraction_identity")) * _safe_int(r.get("n_loops"), 0) for r in rows) / n
            ftr = sum(_safe_float(r.get("fraction_transposition")) * _safe_int(r.get("n_loops"), 0) for r in rows) / n
            f3 = sum(_safe_float(r.get("fraction_3cycle")) * _safe_int(r.get("n_loops"), 0) for r in rows) / n
        else:
            mean = float("nan"); fid = ftr = f3 = float("nan")
        wilson_by_L[L] = {
            "n_loops": int(n),
            "mean_wilson_character": float(mean),
            "delta_from_identity": float(2.0 - mean) if math.isfinite(mean) else float("nan"),
            "fraction_identity": float(fid),
            "fraction_transposition": float(ftr),
            "fraction_3cycle": float(f3),
        }

    # Independent S3-bearing fibers/supports per atlas.
    supports_by_instance: Dict[int, set] = defaultdict(set)
    rows_by_instance: Dict[int, int] = defaultdict(int)
    for r in rep_rows:
        inst = _safe_int(r.get("instance"), -1)
        rows_by_instance[inst] += 1
        if "support_hash" in r and r.get("support_hash"):
            key = (r.get("parent_domain_id", ""), r.get("fiber_label", ""), r.get("support_hash", ""))
        else:
            key = (r.get("parent_domain_id", ""), r.get("fiber_label", ""))
        supports_by_instance[inst].add(key)
    unique_support_counts = {int(k): len(v) for k, v in supports_by_instance.items() if k >= 0}

    hash_counts = Counter(str(r.get("transition_hash", "")) for r in s3_anatomy if str(r.get("transition_hash", "")))
    return {
        "n_anatomy_rows": len(anatomy_rows),
        "n_s3_anatomy_rows": len(s3_anatomy),
        "s3_transition_hash_counts": dict(hash_counts),
        "dominant_s3_transition_hash": hash_counts.most_common(1)[0][0] if hash_counts else "",
        "dominant_s3_transition_hash_count": hash_counts.most_common(1)[0][1] if hash_counts else 0,
        "wilson_by_loop_length": wilson_by_L,
        "representation_signature_counts": dict(Counter(str(r.get("representation_signature", "")) for r in rep_rows if str(r.get("representation_signature", "")))),
        "total_representation_rows": len(rep_rows),
        "unique_s3_supports_by_instance": unique_support_counts,
        "mean_unique_s3_supports": float(sum(unique_support_counts.values()) / max(1, len(unique_support_counts))),
        "max_unique_s3_supports": int(max(unique_support_counts.values()) if unique_support_counts else 0),
    }


def _analyze_transition_array(T: List[int], q: int, vertices: int) -> Dict[str, Any]:
    n = len(T)
    # Cycle structure.
    visited = [False] * n
    lengths: List[int] = []
    for s in range(n):
        if visited[s]:
            continue
        cur = s
        L = 0
        while 0 <= cur < n and not visited[cur]:
            visited[cur] = True
            L += 1
            cur = int(T[cur])
        lengths.append(L)
    cnt = Counter(lengths)
    # Digits and images.
    states = [_to_base_digits(i, q, vertices) for i in range(n)]
    img = [states[int(T[i])] for i in range(n)]
    clock_coords: List[int] = []
    coord_change_rates: Dict[str, float] = {}
    for c in range(vertices):
        deltas = [((img[i][c] - states[i][c]) % q) for i in range(n)]
        if all(d == 1 for d in deltas):
            clock_coords.append(c)
        coord_change_rates[str(c)] = sum(1 for d in deltas if d != 0) / max(1, n)
    det_pairs: List[Tuple[int, int]] = []
    for a in range(vertices):
        for b in range(a + 1, vertices):
            mapping: Dict[Tuple[int, int], Tuple[int, int]] = {}
            ok = True
            for i in range(n):
                k = (states[i][a], states[i][b])
                v = (img[i][a], img[i][b])
                if k in mapping and mapping[k] != v:
                    ok = False
                    break
                mapping[k] = v
            if ok:
                det_pairs.append((a, b))
    hamming = [sum(1 for c in range(vertices) if states[i][c] != img[i][c]) for i in range(n)]
    return {
        "is_permutation": sorted(int(x) for x in T) == list(range(n)),
        "n_cycles": int(len(lengths)),
        "cycle_length_counts": dict(sorted((str(k), int(v)) for k, v in cnt.items())),
        "max_cycle_length": int(max(lengths) if lengths else 0),
        "n_fixed_points": int(cnt.get(1, 0)),
        "cycle_structure_entropy": float(_entropy_from_counts(cnt.values())),
        "single_full_cycle": bool(len(lengths) == 1 and lengths[0] == n),
        "clock_coords": clock_coords,
        "n_clock_coords": int(len(clock_coords)),
        "deterministic_coordinate_pairs": det_pairs,
        "n_deterministic_coordinate_pairs": int(len(det_pairs)),
        "mean_hamming_distance": float(sum(hamming) / max(1, len(hamming))),
        "coord_change_rates": coord_change_rates,
    }


def _compute_s3_fixedpoint_structures(cfg: Dict[str, Any], attractor_rows: List[Dict[str, str]], limit: int = 12) -> List[Dict[str, Any]]:
    """Reconstruct S3 fixed-point transitions and compute anatomy beyond fixedpointmicroscope output."""
    import argparse
    import numpy as np
    from relgauge import iterationattractoraudit as IAA

    out: List[Dict[str, Any]] = []
    q = int(cfg["q"]); vertices = int(cfg["vertices"])
    args = argparse.Namespace(
        q=q, vertices=vertices, seed=int(cfg.get("seed", 2000006)), rule_mode=str(cfg.get("rule_mode", "random_full_permutation")),
        profile=str(cfg.get("profile", "full_atlas")), atlas_capacity=int(cfg.get("atlas_capacity", 32)),
        atlas_iterations=int(cfg.get("atlas_iterations", 8)), max_state_samples=int(cfg.get("max_state_samples", q ** vertices)),
        max_total_states=int(cfg.get("max_total_states", q ** vertices)), max_pred=0,
        proliferation_iterations=int(cfg.get("proliferation_iterations", 4)), horizon=int(cfg.get("horizon", 4)),
        atlas_lift_mode=str(cfg.get("lift", "bijective")), initial_boundary="sum_mod_q", initial_boundary_q=None,
        max_domains_per_depth=32, min_live_classes=2, min_fiber_size=2, min_entropy_bits=0.05, synergy_threshold=0.01,
        max_signature_domains=16, max_parent_domains=8, max_fibers_per_parent=6, max_charts_per_fiber=int(cfg.get("max_charts_per_fiber", 32)),
        max_signature_charts=48, min_fiber_states=6, min_support_states=3, min_overlap_states=3, min_chart_classes=2,
        min_chart_entropy=0.0, max_chart_coords=5, max_support_coords=4, max_cycle_len=5, max_cycles_per_fiber=500,
        max_loops_per_base=500, max_group_order=4096, max_domains_scan=20, max_fibers_per_domain_scan=3,
        max_predictive_coords=6, max_pair_targets=24, max_local_support_coords=4, chart_selection="balanced",
        include_current_basis=False, no_parent_label=False, no_boundary_sum=False, no_horizon1_coords=False,
        no_horizon2_coords=False, no_pair_horizon1=False, no_local_horizon1=False,
    )
    s3_rows = [r for r in attractor_rows if str(r.get("gauge_at_fixed_point")) == "S3"][: int(limit)]
    for r in s3_rows:
        inst = _safe_int(r.get("instance"), 0)
        fpit = _safe_int(r.get("first_fixed_point_iteration"), -1)
        if fpit < 0:
            continue
        init_seed = IAA._candidate_seed(int(args.seed), 0, inst)
        states, current_next, _meta = IAA._initial_transition(q, vertices, str(args.rule_mode), int(init_seed), args)
        T = current_next
        for it in range(fpit):
            _atlas, eff, _ps, _ls = IAA._atlas_and_eff(states, T, q, str(args.profile), int(args.atlas_capacity), int(it), int(init_seed), args)
            T = np.asarray(eff, dtype=np.int64)
        ana = _analyze_transition_array([int(x) for x in T], q, vertices)
        ana.update({
            "instance": int(inst),
            "seed": int(init_seed),
            "fixed_point_iteration": int(fpit),
            "q": q, "vertices": vertices,
        })
        try:
            ana["transition_hash_sha256_20"] = IAA._hash_arr(T)
        except Exception:
            pass
        out.append(ana)
    return out


# ------------------------------- remote jobs --------------------------------

@app.function(timeout=21600, cpu=4, memory=8192)
def run_iteration_job(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one q/v/h/lift iteration job, optionally with microscope/Wilson/anatomy."""
    job_id = str(cfg.get("job_id", f"q{cfg.get('q')}_v{cfg.get('vertices')}_h{cfg.get('horizon')}_{cfg.get('lift','bijective')}"))
    work = REMOTE_WORK / job_id
    work.mkdir(parents=True, exist_ok=True)
    iter_dir = work / "iteration"
    iter_dir.mkdir(parents=True, exist_ok=True)
    q = int(cfg["q"]); vertices = int(cfg["vertices"])
    max_samples = int(cfg.get("max_state_samples", q ** vertices))
    max_total = int(cfg.get("max_total_states", q ** vertices))
    cmd = [
        "python", "-m", "relgauge.iterationattractoraudit",
        "--experiments", "s3_by_iteration,attractors",
        "--q-values", str(q),
        "--vertices-map", f"{q}:{vertices}",
        "--instances", str(int(cfg.get("instances", 10))),
        "--atlas-iterations", str(int(cfg.get("atlas_iterations", 8))),
        "--horizon", str(int(cfg.get("horizon", 4))),
        "--rule-mode", str(cfg.get("rule_mode", "random_full_permutation")),
        "--profile", str(cfg.get("profile", "full_atlas")),
        "--atlas-capacity", str(int(cfg.get("atlas_capacity", 32))),
        "--seed", str(int(cfg.get("seed", 2000006))),
        "--atlas-lift-mode", str(cfg.get("lift", "bijective")),
        "--max-state-samples", str(max_samples),
        "--max-total-states", str(max_total),
        "--max-domains-scan", str(int(cfg.get("max_domains_scan", 20))),
        "--max-fibers-per-domain-scan", str(int(cfg.get("max_fibers_per_domain_scan", 3))),
        "--max-loops-per-base", str(int(cfg.get("max_loops_per_base", 500))),
        "--max-charts-per-fiber", str(int(cfg.get("max_charts_per_fiber", 32))),
        "--max-cycle-len", str(int(cfg.get("max_cycle_len", 5))),
        "--out-dir", str(iter_dir),
    ]
    _run(cmd, timeout=int(cfg.get("subprocess_timeout", 21600)))
    summary = _summarize_iteration_outputs(iter_dir, cfg)

    microscope_summary: Dict[str, Any] = {}
    s3_structures: List[Dict[str, Any]] = []
    if bool(cfg.get("do_microscope", False)):
        mic_dir = work / "microscope"
        mic_dir.mkdir(parents=True, exist_ok=True)
        mcmd = [
            "python", "-m", "relgauge.fixedpointmicroscopeaudit",
            "--experiments", "anatomy,representations,wilson",
            "--attractor-csv", str(iter_dir / "attractor_classification.csv"),
            "--q", str(q),
            "--vertices", str(vertices),
            "--instances", str(int(cfg.get("instances", 10))),
            "--atlas-iterations", str(int(cfg.get("atlas_iterations", 8))),
            "--horizon", str(int(cfg.get("horizon", 4))),
            "--rule-mode", str(cfg.get("rule_mode", "random_full_permutation")),
            "--profile", str(cfg.get("profile", "full_atlas")),
            "--atlas-capacity", str(int(cfg.get("atlas_capacity", 32))),
            "--seed", str(int(cfg.get("seed", 2000006))),
            "--atlas-lift-mode", str(cfg.get("lift", "bijective")),
            "--max-state-samples", str(max_samples),
            "--max-total-states", str(max_total),
            "--max-domains-scan", str(int(cfg.get("max_domains_scan", 20))),
            "--max-fibers-per-domain-scan", str(int(cfg.get("max_fibers_per_domain_scan", 3))),
            "--max-loops-per-base", str(int(cfg.get("max_loops_per_base", 500))),
            "--max-charts-per-fiber", str(int(cfg.get("max_charts_per_fiber", 32))),
            "--max-cycle-len", str(int(cfg.get("max_cycle_len", 5))),
            "--out-dir", str(mic_dir),
        ]
        _run(mcmd, timeout=int(cfg.get("subprocess_timeout", 21600)))
        microscope_summary = _summarize_microscope_outputs(mic_dir, cfg)
        try:
            s3_structures = _compute_s3_fixedpoint_structures(cfg, _read_csv(iter_dir / "attractor_classification.csv"), limit=int(cfg.get("structure_limit", 12)))
        except Exception as e:
            s3_structures = [{"error": str(e)}]
    return {
        "job_id": job_id,
        "kind": "iteration",
        "cfg": cfg,
        "summary": summary,
        "microscope": microscope_summary,
        "s3_structures": s3_structures,
        "remote_workdir": str(work),
    }


@app.function(timeout=21600, cpu=4, memory=8192)
def run_coexistence_job(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run q=6 coexistence scan."""
    job_id = str(cfg.get("job_id", "q6_coexistence"))
    work = REMOTE_WORK / job_id
    work.mkdir(parents=True, exist_ok=True)
    outdir = work / "coexistence"
    outdir.mkdir(parents=True, exist_ok=True)
    q = int(cfg.get("q", 3)); vertices = int(cfg.get("vertices", 6))
    cmd = [
        "python", "-m", "relgauge.fixedpointmicroscopeaudit",
        "--experiments", "coexistence",
        "--q", str(q),
        "--vertices", str(vertices),
        "--horizon", str(int(cfg.get("horizon", 4))),
        "--seed", str(int(cfg.get("seed", 2000006))),
        "--coexistence-q", str(int(cfg.get("coexistence_q", 6))),
        "--coexistence-vertices", str(int(cfg.get("coexistence_vertices", 4))),
        "--coexistence-instances", str(int(cfg.get("coexistence_instances", 10))),
        "--coexistence-iterations", str(int(cfg.get("coexistence_iterations", 8))),
        "--coexistence-max-state-samples", str(int(cfg.get("coexistence_max_state_samples", 1296))),
        "--coexistence-max-total-states", str(int(cfg.get("coexistence_max_total_states", 1296))),
        "--max-domains-scan", str(int(cfg.get("max_domains_scan", 20))),
        "--max-fibers-per-domain-scan", str(int(cfg.get("max_fibers_per_domain_scan", 3))),
        "--max-loops-per-base", str(int(cfg.get("max_loops_per_base", 500))),
        "--max-charts-per-fiber", str(int(cfg.get("max_charts_per_fiber", 32))),
        "--out-dir", str(outdir),
    ]
    _run(cmd, timeout=int(cfg.get("subprocess_timeout", 21600)))
    rows = _read_csv(outdir / "coexistence_scan.csv")
    # Summarize by atlas (instance, iteration).
    by_atlas: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in rows:
        key = (_safe_int(r.get("instance")), _safe_int(r.get("iteration")))
        d = by_atlas.setdefault(key, {"instance": key[0], "iteration": key[1], "n_c2": 0, "n_c3": 0, "n_s3": 0, "coexistence": False})
        d["n_c2"] = max(d["n_c2"], _safe_int(r.get("n_c2_fibers_in_atlas")))
        d["n_c3"] = max(d["n_c3"], _safe_int(r.get("n_c3_fibers_in_atlas")))
        d["n_s3"] = max(d["n_s3"], _safe_int(r.get("n_s3_fibers_in_atlas")))
        d["coexistence"] = d["coexistence"] or str(r.get("coexistence_detected", "")).lower() in {"true", "1", "yes"}
    atlas_rows = list(by_atlas.values())
    any_coexist = any(bool(r["coexistence"]) for r in atlas_rows)
    max_c2 = max([int(r["n_c2"]) for r in atlas_rows] + [0])
    max_c3 = max([int(r["n_c3"]) for r in atlas_rows] + [0])
    max_s3 = max([int(r["n_s3"]) for r in atlas_rows] + [0])
    coexist_types = []
    if any(r["n_c2"] > 0 and r["n_c3"] > 0 for r in atlas_rows):
        coexist_types.append("C2+C3")
    if any(r["n_c2"] > 0 and r["n_s3"] > 0 for r in atlas_rows):
        coexist_types.append("C2+S3")
    return {
        "job_id": job_id,
        "kind": "coexistence",
        "cfg": cfg,
        "summary": {
            "any_coexistence_detected": bool(any_coexist),
            "coexistence_types": coexist_types,
            "max_c2_fibers_in_atlas": int(max_c2),
            "max_c3_fibers_in_atlas": int(max_c3),
            "max_s3_fibers_in_atlas": int(max_s3),
            "n_rows": len(rows),
            "n_atlas_rows": len(atlas_rows),
        },
        "atlas_rows_sample": atlas_rows[:50],
        "remote_workdir": str(work),
    }


# ------------------------------ local reporting ----------------------------

def _write_rows(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Flatten nested dicts/lists to JSON strings.
    flat_rows: List[Dict[str, Any]] = []
    keys = set()
    for r in rows:
        rr: Dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (dict, list, tuple)):
                rr[k] = json.dumps(v, sort_keys=True, default=str)
            else:
                rr[k] = v
            keys.add(k)
        flat_rows.append(rr)
    fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)


def _section_row(label: str, res: Dict[str, Any]) -> Dict[str, Any]:
    s = res.get("summary", {})
    return {
        "label": label,
        "q": s.get("q"),
        "vertices": s.get("vertices"),
        "states": s.get("states"),
        "horizon": s.get("horizon"),
        "lift": s.get("lift"),
        "instances": s.get("instances"),
        "S3_fp": f"{s.get('s3_fp_count',0)}/{s.get('instances',0)}",
        "S3_fp_rate": s.get("s3_fp_rate"),
        "C3_fp": f"{s.get('c3_fp_count',0)}/{s.get('instances',0)}",
        "C3_fp_rate": s.get("c3_fp_rate"),
        "C2_fp": f"{s.get('c2_fp_count',0)}/{s.get('instances',0)}",
        "C2_fp_rate": s.get("c2_fp_rate"),
        "flat_fp": f"{s.get('flat_fp_count',0)}/{s.get('instances',0)}",
        "flat_fp_rate": s.get("flat_fp_rate"),
        "iter0_s3_rate": s.get("iter0_s3_rate_pooled"),
        "fp_over_iter0_ratio": s.get("fp_over_iter0_ratio"),
        "max_s3_rate": s.get("max_s3_rate"),
        "attractor_counts": s.get("attractor_counts"),
    }


def _wilson_scaling_row(label: str, res: Dict[str, Any]) -> Dict[str, Any]:
    s = res.get("summary", {})
    m = res.get("microscope", {})
    wil = m.get("wilson_by_loop_length", {})
    structs = res.get("s3_structures", [])
    # Aggregate structural invariants.
    good_structs = [x for x in structs if "error" not in x]
    n_struct = len(good_structs)
    n_full_cycle = sum(1 for x in good_structs if x.get("single_full_cycle"))
    mean_det_pairs = sum(int(x.get("n_deterministic_coordinate_pairs", 0)) for x in good_structs) / max(1, n_struct)
    mean_clock = sum(int(x.get("n_clock_coords", 0)) for x in good_structs) / max(1, n_struct)
    clock_any = sorted({c for x in good_structs for c in x.get("clock_coords", [])})
    row = _section_row(label, res)
    for L in [3, 4, 5]:
        d = wil.get(L) or wil.get(str(L)) or {}
        W = d.get("mean_wilson_character", float("nan"))
        row[f"W{L}"] = W
        row[f"delta{L}"] = d.get("delta_from_identity", (2.0 - W) if isinstance(W, (float, int)) and math.isfinite(W) else float("nan"))
        row[f"n_loops_L{L}"] = d.get("n_loops", 0)
    row.update({
        "dominant_s3_hash": m.get("dominant_s3_transition_hash", ""),
        "dominant_s3_hash_count": m.get("dominant_s3_transition_hash_count", 0),
        "mean_unique_s3_supports": m.get("mean_unique_s3_supports", 0),
        "max_unique_s3_supports": m.get("max_unique_s3_supports", 0),
        "n_s3_structures_analyzed": n_struct,
        "full_cycle_fraction": n_full_cycle / max(1, n_struct),
        "mean_n_clock_coords": mean_clock,
        "clock_coords_union": clock_any,
        "mean_n_deterministic_pairs": mean_det_pairs,
    })
    return row


def _plot_wilson_scaling(rows: List[Dict[str, Any]], path: pathlib.Path) -> None:
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        by_lift = defaultdict(list)
        for r in rows:
            if int(r.get("q", 0) or 0) == 3 and int(r.get("horizon", 0) or 0) == 4:
                by_lift[str(r.get("lift", ""))].append(r)
        for lift, rs in by_lift.items():
            if lift != "bijective":
                continue
            rs = sorted(rs, key=lambda x: int(x.get("vertices") or 0))
            xs = [int(r.get("vertices") or 0) for r in rs]
            for L in [3, 4, 5]:
                ys = [float(r.get(f"W{L}") if r.get(f"W{L}") not in {None, ""} else float("nan")) for r in rs]
                ax.plot(xs, ys, marker="o", label=f"W(L={L})")
        ax.set_xlabel("vertices v (q=3, h=4)")
        ax.set_ylabel("mean S3 standard character")
        ax.set_title("Finite chart-Wilson scaling")
        ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    except Exception as e:
        print("plot failed:", e)


@app.local_entrypoint()
def main(
    output_dir: str = "cloud_results/definitive_wilson_scaling",
    v_values: str = "6,7,8,9",
    instances_map: str = "6:20,7:20,8:10,9:10",
    qspec_instances: int = 10,
    q3_instances: int = 20,
    horizon_instances: int = 10,
    atlas_iterations: int = 8,
    horizon: int = 4,
    include_representative_by_v: bool = True,
    include_cross_checks: bool = False,
    quick: bool = False,
) -> None:
    """Launch the definitive cloud evidence run and write the six-section table."""
    tstamp = time.strftime("%Y%m%d_%H%M%S")
    outroot = pathlib.Path(output_dir) / tstamp
    outroot.mkdir(parents=True, exist_ok=True)

    v_list = _parse_int_list(v_values, [6, 7, 8, 9])
    default_inst_map = {6: 20, 7: 20, 8: 10, 9: 10}
    inst_map = _parse_int_map(instances_map, default_inst_map)
    if quick:
        qspec_instances = min(qspec_instances, 2)
        q3_instances = min(q3_instances, 3)
        horizon_instances = min(horizon_instances, 2)
        inst_map = {v: min(inst_map.get(v, 2), 2) for v in v_list}
        atlas_iterations = min(atlas_iterations, 3)

    jobs: Dict[str, Dict[str, Any]] = {}

    def add_iteration_job(key: str, q: int, vertices: int, h: int, lift: str, instances: int, do_microscope: bool = False) -> None:
        states = _state_count(q, vertices)
        jobs[key] = {
            "job_id": key,
            "q": int(q),
            "vertices": int(vertices),
            "horizon": int(h),
            "lift": str(lift),
            "instances": int(instances),
            "atlas_iterations": int(atlas_iterations),
            "max_state_samples": int(states),
            "max_total_states": int(states),
            "do_microscope": bool(do_microscope),
            "rule_mode": "random_full_permutation",
            "profile": "full_atlas",
            "atlas_capacity": 32,
            "seed": 2000006,
            "subprocess_timeout": 21600,
        }

    # Section 1: q-specificity.
    add_iteration_job("section1_q2_v9_h4_bij", 2, 9, horizon, "bijective", qspec_instances, False)
    add_iteration_job("section1_q3_v6_h4_bij", 3, 6, horizon, "bijective", q3_instances, True)
    add_iteration_job("section1_q4_v5_h4_bij", 4, 5, horizon, "bijective", qspec_instances, False)
    add_iteration_job("section1_q5_v4_h4_bij", 5, 4, horizon, "bijective", qspec_instances, False)

    # Section 2: q=3 horizon hierarchy.
    for h in [2, 3, 4, 5]:
        key = f"section2_q3_v6_h{h}_bij"
        if key not in jobs and not (h == horizon and "section1_q3_v6_h4_bij" in jobs):
            add_iteration_job(key, 3, 6, h, "bijective", horizon_instances, False)
        elif h != horizon:
            add_iteration_job(key, 3, 6, h, "bijective", horizon_instances, False)

    # Section 3: lift mode. Bijective is reused from section 1; representative is new.
    add_iteration_job("section3_q3_v6_h4_rep", 3, 6, horizon, "representative", q3_instances, False)

    # Section 4/5: Wilson scaling and structural invariants by v.
    for v in v_list:
        key = f"section4_q3_v{v}_h4_bij"
        if v == 6:
            # Reuse section1 q3 if possible; it already has microscope.
            continue
        add_iteration_job(key, 3, int(v), horizon, "bijective", inst_map.get(v, 10), True)
        if include_representative_by_v:
            add_iteration_job(f"section4_q3_v{v}_h4_rep", 3, int(v), horizon, "representative", inst_map.get(v, 10), False)

    # q=6 coexistence.
    coexist_cfg = {
        "job_id": "section6_q6_coexistence",
        "q": 3,
        "vertices": 6,
        "horizon": horizon,
        "coexistence_q": 6,
        "coexistence_vertices": 4,
        "coexistence_instances": 10 if not quick else 2,
        "coexistence_iterations": atlas_iterations,
        "coexistence_max_state_samples": 1296,
        "coexistence_max_total_states": 1296,
        "subprocess_timeout": 21600,
    }

    # Optional cross-checks.
    if include_cross_checks:
        add_iteration_job("cross_q3_v7_h3_bij", 3, 7, 3, "bijective", min(inst_map.get(7, 10), 10), False)
        add_iteration_job("cross_q4_v6_h4_bij", 4, 6, horizon, "bijective", 5 if not quick else 2, False)

    print(f"Launching {len(jobs)} iteration jobs plus coexistence job")
    futures: Dict[str, Any] = {key: run_iteration_job.spawn(cfg) for key, cfg in jobs.items()}
    coexist_future = run_coexistence_job.spawn(coexist_cfg)

    results: Dict[str, Dict[str, Any]] = {}
    for i, (key, fut) in enumerate(futures.items(), start=1):
        print(f"Waiting for job {i}/{len(futures)}: {key}")
        results[key] = fut.get()
        print(json.dumps({"key": key, "summary": results[key].get("summary", {})}, indent=2, sort_keys=True)[:4000])
    coexist_result = coexist_future.get()
    print("Coexistence:", json.dumps(coexist_result.get("summary", {}), indent=2, sort_keys=True))

    # Build section tables.
    section1_keys = ["section1_q2_v9_h4_bij", "section1_q3_v6_h4_bij", "section1_q4_v5_h4_bij", "section1_q5_v4_h4_bij"]
    section1 = [_section_row(k, results[k]) for k in section1_keys if k in results]

    section2: List[Dict[str, Any]] = []
    for h in [2, 3, 4, 5]:
        key = "section1_q3_v6_h4_bij" if h == horizon else f"section2_q3_v6_h{h}_bij"
        if key in results:
            section2.append(_section_row(f"h={h}", results[key]))

    section3_keys = ["section1_q3_v6_h4_bij", "section3_q3_v6_h4_rep"]
    section3 = [_section_row(k, results[k]) for k in section3_keys if k in results]

    section4: List[Dict[str, Any]] = []
    for v in v_list:
        key = "section1_q3_v6_h4_bij" if v == 6 else f"section4_q3_v{v}_h4_bij"
        if key in results:
            section4.append(_wilson_scaling_row(f"v={v},bij", results[key]))
        repkey = f"section4_q3_v{v}_h4_rep" if v != 6 else "section3_q3_v6_h4_rep"
        if include_representative_by_v and repkey in results:
            section4.append(_wilson_scaling_row(f"v={v},rep", results[repkey]))

    # Section 5 is structural subset of section4 bijective rows.
    section5 = [r for r in section4 if str(r.get("lift")) == "bijective"]

    section6 = [coexist_result.get("summary", {})]

    all_results = {
        "created_at": tstamp,
        "jobs": results,
        "coexistence": coexist_result,
        "section1_q_specific": section1,
        "section2_horizon_hierarchy": section2,
        "section3_lift_axiom_test": section3,
        "section4_wilson_scaling": section4,
        "section5_structural_invariants": section5,
        "section6_coexistence": section6,
    }
    (outroot / "definitive_results.json").write_text(json.dumps(all_results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_rows(outroot / "section1_q_specificity.csv", section1)
    _write_rows(outroot / "section2_horizon_hierarchy.csv", section2)
    _write_rows(outroot / "section3_lift_axiom_test.csv", section3)
    _write_rows(outroot / "section4_wilson_scaling.csv", section4)
    _write_rows(outroot / "section5_structural_invariants.csv", section5)
    _write_rows(outroot / "section6_coexistence.csv", section6)
    _plot_wilson_scaling(section4, outroot / "wilson_scaling.png")

    # Print compact summary table.
    print("\n=== SECTION 1: q-specificity ===")
    for r in section1:
        print(f"q={r['q']} v={r['vertices']} S3_fp={r['S3_fp']} ({_safe_float(r['S3_fp_rate']):.3f}) C2={r['C2_fp']} C3={r['C3_fp']} flat={r['flat_fp']}")
    print("\n=== SECTION 2: horizon hierarchy ===")
    for r in section2:
        print(f"{r['label']} S3_fp={r['S3_fp']} ({_safe_float(r['S3_fp_rate']):.3f}) C3={r['C3_fp']} flat={r['flat_fp']} iter0={_safe_float(r['iter0_s3_rate']):.3f}")
    print("\n=== SECTION 3: lift/axiom test ===")
    for r in section3:
        print(f"{r['lift']} S3_fp={r['S3_fp']} flat={r['flat_fp']} iter0={_safe_float(r['iter0_s3_rate']):.3f}")
    print("\n=== SECTION 4: Wilson scaling ===")
    for r in section4:
        print(f"{r['label']} states={r['states']} S3_fp={r['S3_fp']} W3={_safe_float(r.get('W3'), float('nan')):.3f} W4={_safe_float(r.get('W4'), float('nan')):.3f} W5={_safe_float(r.get('W5'), float('nan')):.3f} d4={_safe_float(r.get('delta4'), float('nan')):.3f} d5={_safe_float(r.get('delta5'), float('nan')):.3f}")
    print("\n=== SECTION 6: coexistence ===")
    print(json.dumps(section6[0], indent=2, sort_keys=True))
    print(f"\nWrote results under: {outroot}")


if __name__ == "__main__":
    # Modal invokes the local_entrypoint. This branch is intentionally empty.
    pass
