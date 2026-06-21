"""
observationmethodscaletest.py

Observation-method scale test for multi-observer S3 consensus.

Motivation
----------
A frozen q=2 transition table at the verified v9 critical event can support a
localized exact S3 consensus orbit.  The multi-observer consensus audit tests
whether many independent atlas RNG seeds recover that S3 under a *fixed* atlas
construction method.  This module performs the next stress test: vary the
observation method itself.

The audit fixes one frozen transition table and one target/support region, then
runs many independent observer reconstructions across a grid of atlas-resolution
parameters, such as

    profile, atlas capacity, max chart coordinates, max support coordinates,
    charts per fiber, entropy/support/overlap thresholds.

It reports which observation methods recover exact S3 at the exact target
namespace, which recover S3 on the same microstate support, and where the signal
fails.  This directly tests whether the S3 event is method-invariant above a
resolution threshold or depends on one special chart construction.

Typical use
-----------
python -m relgauge.observationmethodscaletest 2 ^
  --vertices 9 ^
  --iterated-csv example_results/repro_witness_search_q2_v9_fullperm_cap32.csv ^
  --frozen-transition-npy example_results/repro_witness_search_q2_v9_fullperm_cap32_transition_iter8_random_full_permutation_full_atlas_cap32_inst28_seed3221741.npy ^
  --target-rule-mode random_full_permutation ^
  --target-instance 28 ^
  --target-profile full_atlas ^
  --target-atlas-capacity 32 ^
  --target-seed 3221741 ^
  --target-iteration 8 ^
  --target-parent-domain 75 ^
  --target-fiber-label 7 ^
  --observer-profiles full_atlas,fiber_preserving,boundary_only ^
  --observer-atlas-capacities 16,32,64,128 ^
  --max-chart-coords-list 3,4,5,6 ^
  --max-support-coords-list 3,4,5 ^
  --max-charts-per-fiber-list 8,12,16,24 ^
  --min-chart-entropy-list 0,0.025,0.05,0.1 ^
  --min-support-states-list 2,4,8 ^
  --min-overlap-states-list 2,4,8 ^
  --observer-runs 16 ^
  --max-configs 200 ^
  --out example_results/observation_method_scale_q2_v9.csv ^
  --plot example_results/fig_observation_method_scale_q2_v9.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

try:
    import numpy as np
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("observationmethodscaletest requires numpy and pandas") from e

try:  # pragma: no cover - package import
    from . import generatedcandidatephysicsreplayaudit as GCPR
    from . import multiobserverconsensusaudit as MOC
except Exception:  # pragma: no cover - direct/local import
    try:
        import generatedcandidatephysicsreplayaudit as GCPR  # type: ignore
        import multiobserverconsensusaudit as MOC  # type: ignore
    except Exception:
        # Synthetic smoke tests do not need the package imports.  Real runs fail
        # later with a clearer message if these are unavailable.
        GCPR = None  # type: ignore
        MOC = None  # type: ignore


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


def _parse_int_list(text: Any, default: Sequence[int]) -> List[int]:
    vals: List[int] = []
    if text is not None:
        for part in str(text).replace(";", ",").split(","):
            p = part.strip()
            if not p:
                continue
            vals.append(int(float(p)))
    return vals or [int(x) for x in default]


def _parse_float_list(text: Any, default: Sequence[float]) -> List[float]:
    vals: List[float] = []
    if text is not None:
        for part in str(text).replace(";", ",").split(","):
            p = part.strip()
            if not p:
                continue
            vals.append(float(p))
    return vals or [float(x) for x in default]


def _parse_text_list(text: Any, default: Sequence[str]) -> List[str]:
    vals = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    return vals or [str(x) for x in default]


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


def _args_with_defaults(**kwargs: Any) -> argparse.Namespace:
    args = argparse.Namespace(**kwargs)
    return MOC._ensure_upstream_defaults(args)  # type: ignore[attr-defined]


def _reference_support_from_frozen(
    q: int,
    vertices: int,
    iterated_csv: str,
    frozen_transition_npy: str,
    target_candidate: str,
    target_rule_mode: str,
    target_instance: int,
    target_profile: str,
    target_atlas_capacity: int,
    target_seed: int,
    target_iteration: int,
    target_parent_domain: int,
    target_fiber_label: int,
    replay_rng_hash_mode: str,
    reference_observer_seed: int,
    reference_seed_mode: str,
    base_kwargs: Dict[str, Any],
) -> Tuple[pd.Series, Any, np.ndarray, Dict[str, Any], Set[int], Dict[str, Any]]:
    df = pd.read_csv(iterated_csv, low_memory=False)
    select_args = _args_with_defaults(
        q=int(q), vertices=int(vertices), iterated_csv=str(iterated_csv), frozen_transition_npy=str(frozen_transition_npy),
        target_candidate=str(target_candidate), target_rule_mode=str(target_rule_mode), target_instance=int(target_instance),
        target_profile=str(target_profile), target_atlas_capacity=int(target_atlas_capacity), target_seed=int(target_seed),
        target_iteration=int(target_iteration), target_parent_domain=int(target_parent_domain), target_fiber_label=int(target_fiber_label),
        replay_rng_hash_mode=str(replay_rng_hash_mode),
        **base_kwargs,
    )
    row_dict = MOC._select_target_row(df, select_args)  # type: ignore[attr-defined]
    states, frozen_next, meta = MOC._reconstruct_frozen_transition(  # type: ignore[attr-defined]
        row_dict, int(q), int(vertices), int(target_iteration), select_args
    )
    profile = str(meta.get("profile", target_profile))
    capacity = int(meta.get("atlas_capacity", target_atlas_capacity))
    base_seed = int(meta.get("initial_seed", target_seed))

    if int(reference_observer_seed) >= 0:
        ref_seed = int(reference_observer_seed)
        ref_seed_source = "manual_reference_seed"
    elif str(reference_seed_mode) == "observer_seed_start":
        ref_seed = int(base_kwargs.get("observer_seed_start", 0))
        ref_seed_source = "observer_seed_start"
    else:
        ph = MOC._stable_hash(profile) % 1000  # type: ignore[attr-defined]
        ref_seed = int(base_seed) + 1299709 * int(capacity) + 15485863 * int(target_iteration) + 104729 * int(ph)
        ref_seed_source = "original_iteration_seed_formula"

    ref_args = _args_with_defaults(**{**base_kwargs, **{
        "q": int(q), "vertices": int(vertices), "atlas_lift_mode": base_kwargs.get("atlas_lift_mode", "bijective")
    }})
    ref_rng = np.random.default_rng(int(ref_seed))
    ref_atlas, _bounded, _pstats, _rel, _rr, _eff, _ls, _er = GCPR._advance_effective(
        states, frozen_next, int(q), profile, int(capacity), ref_rng, ref_args, str(ref_args.atlas_lift_mode)
    )
    support = MOC._support_for_namespace(ref_atlas, int(target_parent_domain), int(target_fiber_label))  # type: ignore[attr-defined]
    ref_summary, _ref_groups = MOC._analyze_atlas_chart_cycles(  # type: ignore[attr-defined]
        ref_atlas,
        int(target_parent_domain), int(target_fiber_label),
        max_cycles=int(base_kwargs.get("max_spatial_cycles_per_namespace", 0)),
        max_group_order=int(base_kwargs.get("max_group_order", 4096)),
        target_support=support,
        min_support_overlap_fraction=float(base_kwargs.get("min_support_overlap_fraction", 0.5)),
        min_support_jaccard=float(base_kwargs.get("min_support_jaccard", 0.05)),
        min_support_overlap_states=int(base_kwargs.get("min_support_overlap_states", 1)),
    )
    ref_info = {
        "reference_observer_seed_used": int(ref_seed),
        "reference_observer_seed_source": ref_seed_source,
        "reference_profile": profile,
        "reference_atlas_capacity": int(capacity),
        "reference_target_support_size": int(len(support)),
        "reference_target_support_hash": MOC._support_hash(support),  # type: ignore[attr-defined]
        "reference_target_support_sample": MOC._support_sample(support),  # type: ignore[attr-defined]
        "reference_target_exact_s3": bool(ref_summary.get("target_exact_s3_group_closure", False)),
        "reference_target_n_distinct_c2_maps": int(ref_summary.get("target_n_distinct_c2_maps", 0)),
        "reference_target_max_component_size": int(ref_summary.get("target_max_transposition_component_size", 0)),
        "reference_target_generated_group_order": int(ref_summary.get("target_generated_group_order", 0)),
        "frozen_transition_npy": str(frozen_transition_npy),
        "frozen_transition_length": int(len(frozen_next)),
    }
    return pd.Series(row_dict), states, frozen_next, meta, support, ref_info


def _build_config_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    profiles = _parse_text_list(args.observer_profiles, [args.target_profile or "full_atlas"])
    caps = _parse_int_list(args.observer_atlas_capacities, [int(args.target_atlas_capacity or 32)])
    max_chart_coords = _parse_int_list(args.max_chart_coords_list, [int(args.max_chart_coords)])
    max_support_coords = _parse_int_list(args.max_support_coords_list, [int(args.max_support_coords)])
    max_charts_per_fiber = _parse_int_list(args.max_charts_per_fiber_list, [int(args.max_charts_per_fiber)])
    min_chart_entropy = _parse_float_list(args.min_chart_entropy_list, [float(args.min_chart_entropy)])
    min_support_states = _parse_int_list(args.min_support_states_list, [int(args.min_support_states)])
    min_overlap_states = _parse_int_list(args.min_overlap_states_list, [int(args.min_overlap_states)])
    min_fiber_states = _parse_int_list(args.min_fiber_states_list, [int(args.min_fiber_states)])
    min_chart_transition_det = _parse_float_list(args.min_chart_transition_determinism_list, [float(args.min_chart_transition_determinism)])

    grid: List[Dict[str, Any]] = []
    for prof, cap, mcc, msc, mcpf, ent, mss, mos, mfs, ctd in itertools.product(
        profiles, caps, max_chart_coords, max_support_coords, max_charts_per_fiber,
        min_chart_entropy, min_support_states, min_overlap_states, min_fiber_states, min_chart_transition_det,
    ):
        cfg = {
            "observer_profile": str(prof),
            "observer_atlas_capacity": int(cap),
            "max_chart_coords": int(mcc),
            "max_support_coords": int(msc),
            "max_charts_per_fiber": int(mcpf),
            "min_chart_entropy": float(ent),
            "min_support_states": int(mss),
            "min_overlap_states": int(mos),
            "min_fiber_states": int(mfs),
            "min_chart_transition_determinism": float(ctd),
        }
        grid.append(cfg)
    # Deterministic order: lower capacities/coords first unless shuffled externally by user.
    if int(args.max_configs) > 0:
        grid = grid[: int(args.max_configs)]
    return grid


def _run_one_config(
    config_index: int,
    cfg: Dict[str, Any],
    q: int,
    vertices: int,
    states: Any,
    frozen_next: np.ndarray,
    meta: Dict[str, Any],
    target_parent_domain: int,
    target_fiber_label: int,
    target_support: Set[int],
    ref_info: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    profile = str(cfg["observer_profile"])
    capacity = int(cfg["observer_atlas_capacity"])
    run_args_dict = vars(args).copy()
    for k in ["max_chart_coords", "max_support_coords", "max_charts_per_fiber", "min_chart_entropy", "min_support_states", "min_overlap_states", "min_fiber_states", "min_chart_transition_determinism"]:
        run_args_dict[k] = cfg[k]
    run_args = _args_with_defaults(**run_args_dict)

    for r in range(int(args.observer_runs)):
        if str(args.observer_seed_mode) == "candidate_offset":
            obs_seed = int(_safe_int(meta.get("initial_seed"), 0) + int(args.observer_seed_start) + r * int(args.observer_seed_stride))
        else:
            obs_seed = int(args.observer_seed_start + r * int(args.observer_seed_stride))
        rng = np.random.default_rng(int(obs_seed))
        atlas, bounded, pstats, rel_stats, rel_rows, eff, lift_stats, eff_rows = GCPR._advance_effective(
            states, frozen_next, int(q), str(profile), int(capacity), rng, run_args, str(run_args.atlas_lift_mode)
        )
        obs_summary, obs_group_rows = MOC._analyze_atlas_chart_cycles(  # type: ignore[attr-defined]
            atlas,
            int(target_parent_domain), int(target_fiber_label),
            max_cycles=int(args.max_spatial_cycles_per_namespace),
            max_group_order=int(args.max_group_order),
            target_support=target_support,
            min_support_overlap_fraction=float(args.min_support_overlap_fraction),
            min_support_jaccard=float(args.min_support_jaccard),
            min_support_overlap_states=int(args.min_support_overlap_states),
        )
        row = {
            "config_index": int(config_index),
            "observer_run": int(r),
            "observer_seed": int(obs_seed),
            "candidate_id": str(meta.get("candidate_id", "")),
            "target_iteration": int(args.target_iteration),
            "target_parent_domain": int(target_parent_domain),
            "target_fiber_label": int(target_fiber_label),
            **cfg,
            "bounded_atlas_classes": int(pstats.get("bounded_atlas_classes", 0)) if isinstance(pstats, dict) else 0,
            "bounded_atlas_fiber_entropy_bits": float(pstats.get("bounded_atlas_fiber_entropy_bits", 0.0)) if isinstance(pstats, dict) else 0.0,
            "temporal_relation_determinism": float(rel_stats.get("temporal_relation_determinism", 0.0)) if isinstance(rel_stats, dict) else 0.0,
            "chart_c2_count": int(getattr(atlas, "n_chart_c2", 0)),
            "chart_nontrivial_count": int(getattr(atlas, "n_chart_nontrivial", 0)),
            **obs_summary,
        }
        rows.append(row)
        if bool(args.save_group_rows):
            for gr in obs_group_rows:
                group_rows.append({"config_index": int(config_index), "observer_run": int(r), "observer_seed": int(obs_seed), **cfg, **gr})
    rdf = pd.DataFrame(rows)
    summary = _summarize_config(rdf, cfg, config_index, args, ref_info)
    return summary, group_rows


def _summarize_config(rdf: pd.DataFrame, cfg: Dict[str, Any], config_index: int, args: argparse.Namespace, ref_info: Dict[str, Any]) -> Dict[str, Any]:
    if rdf.empty:
        return {"config_index": int(config_index), **cfg, "observer_runs": 0, "verdict": "empty_config"}
    def frac(col: str) -> float:
        return float(rdf[col].map(_safe_bool).mean()) if col in rdf.columns else 0.0
    def maxnum(col: str, default: float = 0.0) -> float:
        return float(pd.to_numeric(rdf.get(col, pd.Series([default])), errors="coerce").fillna(default).max())
    def meannum(col: str, default: float = 0.0) -> float:
        return float(pd.to_numeric(rdf.get(col, pd.Series([default])), errors="coerce").fillna(default).mean())
    summary: Dict[str, Any] = {
        "config_index": int(config_index),
        **cfg,
        "observer_runs": int(len(rdf)),
        "target_namespace_observed_fraction": frac("target_namespace_observed"),
        "target_exact_s3_consensus_fraction": frac("target_exact_s3_group_closure"),
        "target_shared_pair_consensus_fraction": frac("target_shared_label_transposition_pair"),
        "support_overlap_observed_fraction": frac("support_overlap_observed"),
        "support_exact_s3_consensus_fraction": frac("support_exact_s3_group_closure"),
        "support_shared_pair_consensus_fraction": frac("support_shared_label_transposition_pair"),
        "support_two_plus_maps_consensus_fraction": frac("support_two_plus_distinct_maps"),
        "any_exact_s3_fraction": frac("any_exact_s3_group_closure"),
        "max_support_best_jaccard": maxnum("support_best_jaccard"),
        "mean_support_best_jaccard": meannum("support_best_jaccard"),
        "min_support_best_jaccard": float(pd.to_numeric(rdf.get("support_best_jaccard", pd.Series([0.0])), errors="coerce").fillna(0.0).min()),
        "max_support_best_target_overlap_fraction": maxnum("support_best_target_overlap_fraction"),
        "mean_support_best_target_overlap_fraction": meannum("support_best_target_overlap_fraction"),
        "max_target_transposition_component_size": int(maxnum("target_max_transposition_component_size")),
        "max_support_transposition_component_size": int(maxnum("support_max_transposition_component_size")),
        "max_any_transposition_component_size": int(maxnum("max_transposition_component_size")),
        "max_target_generated_group_order": int(maxnum("target_generated_group_order")),
        "max_support_generated_group_order": int(maxnum("support_max_generated_group_order")),
        "mean_target_distinct_c2_maps": meannum("target_n_distinct_c2_maps"),
        "mean_chart_c2_count": meannum("chart_c2_count"),
        "mean_chart_nontrivial_count": meannum("chart_nontrivial_count"),
        "mean_bounded_atlas_classes": meannum("bounded_atlas_classes"),
        "mean_bounded_atlas_fiber_entropy_bits": meannum("bounded_atlas_fiber_entropy_bits"),
        "reference_target_support_size": int(ref_info.get("reference_target_support_size", 0)),
        "reference_target_support_hash": str(ref_info.get("reference_target_support_hash", "")),
    }
    if summary["target_exact_s3_consensus_fraction"] >= float(args.min_consensus_fraction):
        verdict = "exact_namespace_s3_consensus"
    elif summary["support_exact_s3_consensus_fraction"] >= float(args.min_consensus_fraction):
        verdict = "support_s3_consensus"
    elif summary["support_exact_s3_consensus_fraction"] > 0:
        verdict = "weak_support_s3"
    elif summary["support_overlap_observed_fraction"] > 0:
        verdict = "support_seen_without_s3"
    elif summary["any_exact_s3_fraction"] > 0:
        verdict = "relocated_s3"
    else:
        verdict = "no_s3"
    summary["config_verdict"] = verdict
    return summary


def _summarize_all(cdf: pd.DataFrame, odf: pd.DataFrame, ref_info: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if cdf.empty:
        return {"verdict": "OBSERVATION-METHOD SCALE TEST EMPTY", "n_configs": 0}
    n = len(cdf)
    exact_consensus = cdf["target_exact_s3_consensus_fraction"] >= float(args.min_consensus_fraction)
    support_consensus = cdf["support_exact_s3_consensus_fraction"] >= float(args.min_consensus_fraction)
    any_consensus = (cdf["target_exact_s3_consensus_fraction"] >= float(args.min_consensus_fraction)) | (cdf["support_exact_s3_consensus_fraction"] >= float(args.min_consensus_fraction))
    exact_all = bool(exact_consensus.all()) if n else False
    support_all = bool(support_consensus.all()) if n else False
    if exact_all:
        verdict = "OBSERVATION-METHOD STRUCTURAL S3 INVARIANCE: exact target S3 survives all tested observation methods"
    elif support_all:
        verdict = "OBSERVATION-METHOD SUPPORT S3 INVARIANCE: support-level S3 survives all tested observation methods"
    elif any_consensus.any():
        verdict = "OBSERVATION-METHOD THRESHOLD S3 SIGNAL: S3 survives some tested observation methods but not all"
    elif (cdf["support_overlap_observed_fraction"] > 0).any():
        verdict = "OBSERVATION-METHOD SUPPORT-ONLY SIGNAL: target support is recovered but S3 is method-dependent or absent"
    else:
        verdict = "OBSERVATION-METHOD S3 ABSENT: target support/S3 not recovered under tested methods"

    # Best config sorted by support S3, exact S3, jaccard, component/order.
    sort_cols = [
        "support_exact_s3_consensus_fraction", "target_exact_s3_consensus_fraction",
        "max_support_best_jaccard", "max_support_transposition_component_size", "max_support_generated_group_order"
    ]
    best = cdf.sort_values(sort_cols, ascending=[False, False, False, False, False]).iloc[0].to_dict()
    # crude resolution threshold: among configs with S3 support consensus, report min coords/classes knobs.
    sig = cdf[support_consensus | exact_consensus].copy()
    threshold: Dict[str, Any] = {}
    if not sig.empty:
        for col in ["observer_atlas_capacity", "max_chart_coords", "max_support_coords", "max_charts_per_fiber", "min_support_states", "min_overlap_states", "min_chart_entropy"]:
            if col in sig.columns:
                threshold[f"min_{col}_among_consensus"] = float(pd.to_numeric(sig[col], errors="coerce").min())
                threshold[f"max_{col}_among_consensus"] = float(pd.to_numeric(sig[col], errors="coerce").max())
    summary = {
        "verdict": verdict,
        "audit_version": "observation_method_scale_test_v1_multiobserver_parameter_grid",
        "q": int(args.q),
        "vertices": int(args.vertices),
        "source_iterated_csv": str(args.iterated_csv),
        "frozen_transition_npy": str(args.frozen_transition_npy),
        "target_iteration": int(args.target_iteration),
        "target_parent_domain": int(args.target_parent_domain),
        "target_fiber_label": int(args.target_fiber_label),
        "n_configs": int(n),
        "observer_runs_per_config": int(args.observer_runs),
        "configs_exact_target_s3_consensus": int(exact_consensus.sum()),
        "configs_support_s3_consensus": int(support_consensus.sum()),
        "configs_any_s3_consensus": int(any_consensus.sum()),
        "exact_target_s3_config_fraction": float(exact_consensus.mean()) if n else 0.0,
        "support_s3_config_fraction": float(support_consensus.mean()) if n else 0.0,
        "max_target_exact_s3_consensus_fraction": float(pd.to_numeric(cdf["target_exact_s3_consensus_fraction"], errors="coerce").fillna(0).max()),
        "max_support_exact_s3_consensus_fraction": float(pd.to_numeric(cdf["support_exact_s3_consensus_fraction"], errors="coerce").fillna(0).max()),
        "min_support_exact_s3_consensus_fraction": float(pd.to_numeric(cdf["support_exact_s3_consensus_fraction"], errors="coerce").fillna(0).min()),
        "max_support_best_jaccard": float(pd.to_numeric(cdf["max_support_best_jaccard"], errors="coerce").fillna(0).max()),
        "max_support_transposition_component_size": int(pd.to_numeric(cdf["max_support_transposition_component_size"], errors="coerce").fillna(0).max()),
        "max_support_generated_group_order": int(pd.to_numeric(cdf["max_support_generated_group_order"], errors="coerce").fillna(0).max()),
        "config_verdict_counts": dict(Counter(map(str, cdf.get("config_verdict", pd.Series(dtype=str))))),
        "best_config": {k: _json_safe(v) for k, v in best.items()},
        "resolution_threshold_summary": threshold,
        "reference": _json_safe(ref_info),
        "args": {k: _json_safe(v) for k, v in vars(args).items() if k != "synthetic_smoke"},
    }
    return summary


def _write_outputs(cdf: pd.DataFrame, odf: pd.DataFrame, gdf: pd.DataFrame, summary: Dict[str, Any], out: str, plot: str) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        cdf.to_csv(out, index=False)
        opath = out[:-4] + "_observers.csv" if out.endswith(".csv") else out + "_observers.csv"
        odf.to_csv(opath, index=False)
        if not gdf.empty:
            gpath = out[:-4] + "_groups.csv" if out.endswith(".csv") else out + "_groups.csv"
            gdf.to_csv(gpath, index=False)
            print(f"wrote {gpath}")
        spath = out[:-4] + "_summary.json" if out.endswith(".csv") else out + "_summary.json"
        with open(spath, "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
        print(f"wrote {out}")
        print(f"wrote {opath}")
        print(f"wrote {spath}")
    if plot:
        try:
            import matplotlib.pyplot as plt
            if cdf.empty:
                return
            labels = [str(int(x)) for x in cdf["config_index"].tolist()]
            if len(labels) > 50:
                # show a downsampled view of first 50 configs for readability
                pdf = cdf.head(50).copy()
                labels = [str(int(x)) for x in pdf["config_index"].tolist()]
            else:
                pdf = cdf
            fig, ax1 = plt.subplots(figsize=(max(10, min(24, 0.38 * len(labels) + 5)), 5))
            ax2 = ax1.twinx()
            x = np.arange(len(labels))
            w = 0.28
            ax1.bar(x - w, pdf["target_exact_s3_consensus_fraction"].astype(float).to_numpy(), width=w, label="target S3")
            ax1.bar(x, pdf["support_exact_s3_consensus_fraction"].astype(float).to_numpy(), width=w, label="support S3")
            ax1.bar(x + w, pdf["support_overlap_observed_fraction"].astype(float).to_numpy(), width=w, label="support seen")
            ax2.plot(x, pdf["max_support_best_jaccard"].astype(float).to_numpy(), marker="o", label="best Jaccard")
            ax2.plot(x, pdf["max_support_generated_group_order"].astype(float).to_numpy(), marker="s", label="support order")
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("fraction")
            ax2.set_ylabel("Jaccard / group order")
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=60, ha="right")
            ax1.set_xlabel("config index")
            ax1.set_title(str(summary.get("verdict", "observation-method scale test")))
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            fig.tight_layout()
            os.makedirs(os.path.dirname(plot) or ".", exist_ok=True)
            fig.savefig(plot, dpi=160)
            print(f"wrote {plot}")
        except Exception as e:
            print(f"plot failed: {e}")


def _run_synthetic(out: str, plot: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    configs = []
    observers = []
    for ci, cap in enumerate([8, 16, 32, 64]):
        support_s3 = 1.0 if cap >= 32 else 0.0
        exact_s3 = 1.0 if cap >= 64 else 0.0
        cfg = {
            "config_index": ci,
            "observer_profile": "full_atlas",
            "observer_atlas_capacity": cap,
            "max_chart_coords": 5,
            "max_support_coords": 4,
            "max_charts_per_fiber": 16,
            "min_chart_entropy": 0.05,
            "min_support_states": 4,
            "min_overlap_states": 4,
            "min_fiber_states": 2,
            "observer_runs": 4,
            "target_exact_s3_consensus_fraction": exact_s3,
            "support_exact_s3_consensus_fraction": support_s3,
            "support_overlap_observed_fraction": 1.0,
            "support_shared_pair_consensus_fraction": support_s3,
            "support_two_plus_maps_consensus_fraction": support_s3,
            "any_exact_s3_fraction": support_s3,
            "max_support_best_jaccard": 1.0,
            "mean_support_best_jaccard": 1.0,
            "min_support_best_jaccard": 1.0,
            "max_support_best_target_overlap_fraction": 1.0,
            "mean_support_best_target_overlap_fraction": 1.0,
            "max_support_transposition_component_size": 3 if support_s3 else 2,
            "max_support_generated_group_order": 6 if support_s3 else 2,
            "config_verdict": "exact_namespace_s3_consensus" if exact_s3 else ("support_s3_consensus" if support_s3 else "support_seen_without_s3"),
        }
        configs.append(cfg)
        for r in range(4):
            observers.append({"config_index": ci, "observer_run": r, "observer_atlas_capacity": cap, "support_exact_s3_group_closure": bool(support_s3), "target_exact_s3_group_closure": bool(exact_s3)})
    cdf = pd.DataFrame(configs)
    odf = pd.DataFrame(observers)
    gdf = pd.DataFrame()
    args = argparse.Namespace(q=2, vertices=9, iterated_csv="synthetic", frozen_transition_npy="synthetic", target_iteration=8, target_parent_domain=75, target_fiber_label=7, observer_runs=4, min_consensus_fraction=0.5, synthetic_smoke=True)
    ref = {"reference_target_support_size": 17, "reference_target_support_hash": "synthetic"}
    summary = _summarize_all(cdf, odf, ref, args)
    _write_outputs(cdf, odf, gdf, summary, out, plot)
    return cdf, odf, gdf, summary


def run_observation_method_scale_test(
    q: int = 2,
    vertices: int = 9,
    iterated_csv: str = "",
    frozen_transition_npy: str = "",
    target_candidate: str = "",
    target_rule_mode: str = "",
    target_instance: int = -1,
    target_profile: str = "full_atlas",
    target_atlas_capacity: int = 32,
    target_seed: int = -1,
    target_iteration: int = 8,
    target_parent_domain: int = 75,
    target_fiber_label: int = 7,
    observer_profiles: str = "full_atlas",
    observer_atlas_capacities: str = "32",
    max_chart_coords_list: str = "5",
    max_support_coords_list: str = "4",
    max_charts_per_fiber_list: str = "16",
    min_chart_entropy_list: str = "0.05",
    min_support_states_list: str = "4",
    min_overlap_states_list: str = "4",
    min_fiber_states_list: str = "2",
    min_chart_transition_determinism_list: str = "0.98",
    observer_runs: int = 16,
    observer_seed_start: int = 0,
    observer_seed_stride: int = 1,
    observer_seed_mode: str = "independent",
    reference_observer_seed: int = -1,
    reference_seed_mode: str = "original_iteration",
    min_consensus_fraction: float = 0.5,
    max_configs: int = 0,
    max_spatial_cycles_per_namespace: int = 0,
    max_group_order: int = 4096,
    min_support_overlap_fraction: float = 0.5,
    min_support_jaccard: float = 0.05,
    min_support_overlap_states: int = 1,
    replay_rng_hash_mode: str = "stable",
    atlas_lift_mode: str = "bijective",
    proliferation_iterations: int = 4,
    horizon: int = 3,
    max_pred: int = 0,
    max_total_states: int = 200000,
    max_state_samples: int = 512,
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
    max_signature_charts: int = 48,
    min_chart_classes: int = 2,
    min_chart_entropy: float = 0.05,
    max_chart_coords: int = 5,
    max_support_coords: int = 4,
    max_charts_per_fiber: int = 16,
    min_fiber_states: int = 2,
    min_support_states: int = 4,
    min_overlap_states: int = 4,
    min_chart_transition_determinism: float = 0.98,
    save_group_rows: bool = False,
    out: str = "example_results/observation_method_scale_test.csv",
    plot: str = "",
    synthetic_smoke: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if synthetic_smoke:
        return _run_synthetic(out=out, plot=plot)
    if GCPR is None or MOC is None:
        raise RuntimeError("observationmethodscaletest must be run inside the relgauge package for real audits")
    if not iterated_csv or not os.path.exists(iterated_csv):
        raise ValueError("Provide --iterated-csv")
    if not frozen_transition_npy or not os.path.exists(frozen_transition_npy):
        raise ValueError("Provide --frozen-transition-npy from patched iteratedfiberatlasdynamicsaudit")

    args = argparse.Namespace(**locals())
    base_kwargs = {k: getattr(args, k) for k in [
        "atlas_lift_mode", "proliferation_iterations", "horizon", "max_pred", "max_total_states", "max_state_samples",
        "initial_boundary", "initial_boundary_q", "max_domains_per_depth", "min_live_classes", "min_fiber_size",
        "min_entropy_bits", "synergy_threshold", "max_signature_domains", "max_parent_domains", "max_fibers_per_parent",
        "max_signature_charts", "min_chart_classes", "min_chart_entropy", "max_chart_coords", "max_support_coords",
        "max_charts_per_fiber", "min_fiber_states", "min_support_states", "min_overlap_states", "min_chart_transition_determinism",
        "max_spatial_cycles_per_namespace", "max_group_order", "min_support_overlap_fraction", "min_support_jaccard",
        "min_support_overlap_states", "observer_seed_start",
    ]}
    target_row, states, frozen_next, meta, support, ref_info = _reference_support_from_frozen(
        int(q), int(vertices), str(iterated_csv), str(frozen_transition_npy), str(target_candidate), str(target_rule_mode),
        int(target_instance), str(target_profile), int(target_atlas_capacity), int(target_seed), int(target_iteration),
        int(target_parent_domain), int(target_fiber_label), str(replay_rng_hash_mode), int(reference_observer_seed),
        str(reference_seed_mode), base_kwargs,
    )
    if not support:
        raise RuntimeError("Reference target support is empty. Check target namespace, frozen transition, and reference atlas settings.")

    configs = _build_config_grid(args)
    config_summaries: List[Dict[str, Any]] = []
    observer_rows: List[pd.DataFrame] = []  # currently not filled from per-config function, but kept for future
    all_group_rows: List[Dict[str, Any]] = []
    # To avoid retaining every observer row generated internally, _run_one_config returns only config summary and optional group rows.
    # Re-run a lightweight observer dataframe per config by using summary-level outputs; detailed observer rows are omitted by design.
    detailed_observer_rows: List[Dict[str, Any]] = []
    for ci, cfg in enumerate(configs):
        print(f"obs-scale config={ci+1}/{len(configs)} profile={cfg['observer_profile']} cap={cfg['observer_atlas_capacity']} "
              f"chart={cfg['max_chart_coords']} support={cfg['max_support_coords']} charts/fiber={cfg['max_charts_per_fiber']} "
              f"entropy={cfg['min_chart_entropy']} min_support={cfg['min_support_states']} min_overlap={cfg['min_overlap_states']}")
        # Duplicate _run_one_config logic but keep observer rows for output.
        cfg_args_dict = vars(args).copy()
        for k in ["max_chart_coords", "max_support_coords", "max_charts_per_fiber", "min_chart_entropy", "min_support_states", "min_overlap_states", "min_fiber_states", "min_chart_transition_determinism"]:
            cfg_args_dict[k] = cfg[k]
        run_args = _args_with_defaults(**cfg_args_dict)
        obs_rows_cfg: List[Dict[str, Any]] = []
        for r in range(int(observer_runs)):
            if str(observer_seed_mode) == "candidate_offset":
                obs_seed = int(_safe_int(meta.get("initial_seed"), 0) + int(observer_seed_start) + r * int(observer_seed_stride))
            else:
                obs_seed = int(observer_seed_start + r * int(observer_seed_stride))
            rng = np.random.default_rng(int(obs_seed))
            atlas, bounded, pstats, rel_stats, rel_rows, eff, lift_stats, eff_rows = GCPR._advance_effective(
                states, frozen_next, int(q), str(cfg["observer_profile"]), int(cfg["observer_atlas_capacity"]), rng, run_args, str(run_args.atlas_lift_mode)
            )
            obs_summary, obs_group_rows = MOC._analyze_atlas_chart_cycles(  # type: ignore[attr-defined]
                atlas, int(target_parent_domain), int(target_fiber_label),
                max_cycles=int(max_spatial_cycles_per_namespace), max_group_order=int(max_group_order), target_support=support,
                min_support_overlap_fraction=float(min_support_overlap_fraction), min_support_jaccard=float(min_support_jaccard),
                min_support_overlap_states=int(min_support_overlap_states),
            )
            orow = {
                "config_index": int(ci), "observer_run": int(r), "observer_seed": int(obs_seed),
                "candidate_id": str(meta.get("candidate_id", "")), "target_iteration": int(target_iteration),
                "target_parent_domain": int(target_parent_domain), "target_fiber_label": int(target_fiber_label),
                **cfg,
                "bounded_atlas_classes": int(pstats.get("bounded_atlas_classes", 0)) if isinstance(pstats, dict) else 0,
                "bounded_atlas_fiber_entropy_bits": float(pstats.get("bounded_atlas_fiber_entropy_bits", 0.0)) if isinstance(pstats, dict) else 0.0,
                "temporal_relation_determinism": float(rel_stats.get("temporal_relation_determinism", 0.0)) if isinstance(rel_stats, dict) else 0.0,
                "chart_c2_count": int(getattr(atlas, "n_chart_c2", 0)),
                "chart_nontrivial_count": int(getattr(atlas, "n_chart_nontrivial", 0)),
                **obs_summary,
            }
            obs_rows_cfg.append(orow)
            detailed_observer_rows.append(orow)
            if bool(save_group_rows):
                for gr in obs_group_rows:
                    all_group_rows.append({"config_index": int(ci), "observer_run": int(r), "observer_seed": int(obs_seed), **cfg, **gr})
        cfg_df = pd.DataFrame(obs_rows_cfg)
        config_summaries.append(_summarize_config(cfg_df, cfg, ci, args, ref_info))

    cdf = pd.DataFrame(config_summaries)
    odf = pd.DataFrame(detailed_observer_rows)
    gdf = pd.DataFrame(all_group_rows)
    summary = _summarize_all(cdf, odf, ref_info, args)
    _write_outputs(cdf, odf, gdf, summary, out, plot)
    return cdf, odf, gdf, summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Observation-method scale test for frozen-transition multi-observer S3 consensus")
    ap.add_argument("q", type=int, nargs="?", default=2)
    ap.add_argument("--vertices", type=int, default=9)
    ap.add_argument("--iterated-csv", default="")
    ap.add_argument("--frozen-transition-npy", default="")
    ap.add_argument("--target-candidate", default="")
    ap.add_argument("--target-rule-mode", default="")
    ap.add_argument("--target-instance", type=int, default=-1)
    ap.add_argument("--target-profile", default="full_atlas")
    ap.add_argument("--target-atlas-capacity", type=int, default=32)
    ap.add_argument("--target-seed", type=int, default=-1)
    ap.add_argument("--target-iteration", type=int, default=8)
    ap.add_argument("--target-parent-domain", type=int, default=75)
    ap.add_argument("--target-fiber-label", type=int, default=7)
    ap.add_argument("--observer-profiles", default="full_atlas")
    ap.add_argument("--observer-atlas-capacities", default="32")
    ap.add_argument("--max-chart-coords-list", default="5")
    ap.add_argument("--max-support-coords-list", default="4")
    ap.add_argument("--max-charts-per-fiber-list", default="16")
    ap.add_argument("--min-chart-entropy-list", default="0.05")
    ap.add_argument("--min-support-states-list", default="4")
    ap.add_argument("--min-overlap-states-list", default="4")
    ap.add_argument("--min-fiber-states-list", default="2")
    ap.add_argument("--min-chart-transition-determinism-list", default="0.98")
    ap.add_argument("--observer-runs", type=int, default=16)
    ap.add_argument("--observer-seed-start", type=int, default=0)
    ap.add_argument("--observer-seed-stride", type=int, default=1)
    ap.add_argument("--observer-seed-mode", default="independent")
    ap.add_argument("--reference-observer-seed", type=int, default=-1)
    ap.add_argument("--reference-seed-mode", default="original_iteration")
    ap.add_argument("--min-consensus-fraction", type=float, default=0.5)
    ap.add_argument("--max-configs", type=int, default=0)
    ap.add_argument("--max-spatial-cycles-per-namespace", type=int, default=0)
    ap.add_argument("--max-group-order", type=int, default=4096)
    ap.add_argument("--min-support-overlap-fraction", type=float, default=0.5)
    ap.add_argument("--min-support-jaccard", type=float, default=0.05)
    ap.add_argument("--min-support-overlap-states", type=int, default=1)
    ap.add_argument("--replay-rng-hash-mode", default="stable")
    # upstream atlas defaults / knobs
    ap.add_argument("--atlas-lift-mode", default="bijective")
    ap.add_argument("--proliferation-iterations", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--max-pred", type=int, default=0)
    ap.add_argument("--max-total-states", type=int, default=200000)
    ap.add_argument("--max-state-samples", type=int, default=512)
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
    ap.add_argument("--min-chart-classes", type=int, default=2)
    ap.add_argument("--min-chart-entropy", type=float, default=0.05)
    ap.add_argument("--max-chart-coords", type=int, default=5)
    ap.add_argument("--max-support-coords", type=int, default=4)
    ap.add_argument("--max-charts-per-fiber", type=int, default=16)
    ap.add_argument("--min-fiber-states", type=int, default=2)
    ap.add_argument("--min-support-states", type=int, default=4)
    ap.add_argument("--min-overlap-states", type=int, default=4)
    ap.add_argument("--min-chart-transition-determinism", type=float, default=0.98)
    ap.add_argument("--save-group-rows", action="store_true")
    ap.add_argument("--out", default="example_results/observation_method_scale_test.csv")
    ap.add_argument("--plot", default="")
    ap.add_argument("--synthetic-smoke", action="store_true")
    args = ap.parse_args(argv)
    cdf, odf, gdf, summary = run_observation_method_scale_test(**vars(args))
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
