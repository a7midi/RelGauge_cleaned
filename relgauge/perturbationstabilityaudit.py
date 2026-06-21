"""
perturbationstabilityaudit.py

Perturbation stability test for structural S₃ consensus events.

Takes a frozen transition table (.npy) known to support structural S₃,
applies controlled perturbations at varying strengths and locations,
and measures whether S₃ consensus survives. The minimum perturbation
that destroys S₃ is a scalar quantity intrinsic to the event — the
framework's first candidate for a mass-like observable.

Perturbation modes:
  - random:  perturb k randomly chosen transition entries
  - inside:  perturb k entries whose SOURCE is in the S₃ support
  - outside: perturb k entries whose SOURCE is outside the S₃ support

For each (level, mode, trial), builds an atlas and checks for S₃ at the
target namespace and via support overlap.

Usage:
    python -m relgauge.perturbationstabilityaudit 2 \\
      --vertices 9 \\
      --frozen-transition-npy data/transition_iter8_inst28.npy \\
      --support-indices "29,30,44,47,61,62,156,159,188,191,221,222,253,254,256,284,287" \\
      --perturbation-levels 0,1,2,4,8,16,32,64,128 \\
      --perturbation-modes random,inside,outside \\
      --trials-per-level 10 \\
      --out example_results/perturbation_stability.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
    import pandas as pd
except Exception as e:
    raise RuntimeError("requires numpy and pandas") from e

try:
    from . import generatedcandidatephysicsreplayaudit as GCPR
    from . import multiobserverconsensusaudit as MOC
except Exception:
    try:
        import generatedcandidatephysicsreplayaudit as GCPR  # type: ignore
        import multiobserverconsensusaudit as MOC  # type: ignore
    except Exception:
        GCPR = None  # type: ignore
        MOC = None  # type: ignore


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
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _safe_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}


def _ensure_defaults(args: Any) -> Any:
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


def _parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    vals = []
    for p in str(text or "").replace(";", ",").split(","):
        p = p.strip()
        if p:
            vals.append(int(float(p)))
    return vals or list(default)


def _parse_text_list(text: str, default: Sequence[str]) -> List[str]:
    vals = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    return vals or list(default)


# ---------------------------------------------------------------------------
# Perturbation logic
# ---------------------------------------------------------------------------
def _perturb_transition(
    T: np.ndarray,
    n_perturb: int,
    mode: str,
    support_indices: Set[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return a perturbed copy of T and metadata about what was changed."""
    n = len(T)
    T_new = T.copy()

    if n_perturb <= 0:
        return T_new, {
            "n_perturbed": 0,
            "perturbed_indices": [],
            "n_inside_support": 0,
            "n_outside_support": 0,
        }

    support_list = sorted(support_indices)
    outside_list = sorted(set(range(n)) - support_indices)

    if mode == "inside":
        # Perturb entries whose source is in the support
        if n_perturb > len(support_list):
            candidates = np.array(support_list)
        else:
            candidates = rng.choice(support_list, size=min(n_perturb, len(support_list)), replace=False)
    elif mode == "outside":
        # Perturb entries whose source is outside the support
        if n_perturb > len(outside_list):
            candidates = np.array(outside_list)
        else:
            candidates = rng.choice(outside_list, size=min(n_perturb, len(outside_list)), replace=False)
    else:  # random
        candidates = rng.choice(n, size=min(n_perturb, n), replace=False)

    # For each selected index, reassign its transition to a random target
    # (different from current, to ensure actual perturbation)
    for idx in candidates:
        old_target = int(T_new[idx])
        new_target = int(rng.integers(0, n))
        attempts = 0
        while new_target == old_target and attempts < 20:
            new_target = int(rng.integers(0, n))
            attempts += 1
        T_new[idx] = new_target

    n_in = sum(1 for i in candidates if int(i) in support_indices)
    n_out = len(candidates) - n_in

    return T_new, {
        "n_perturbed": int(len(candidates)),
        "perturbed_indices": sorted(int(i) for i in candidates),
        "n_inside_support": int(n_in),
        "n_outside_support": int(n_out),
    }


# ---------------------------------------------------------------------------
# Atlas check for S₃
# ---------------------------------------------------------------------------
def _check_s3(
    states: np.ndarray,
    T: np.ndarray,
    q: int,
    profile: str,
    capacity: int,
    target_parent_domain: int,
    target_fiber_label: int,
    target_support: Set[int],
    rng: np.random.Generator,
    args: Any,
) -> Dict[str, Any]:
    """Build one atlas from T and check for S₃."""
    atlas, bounded, pstats, rel_stats, rel_rows, eff, lift_stats, eff_rows = \
        GCPR._advance_effective(
            states, T, int(q), str(profile), int(capacity),
            rng, args, str(args.atlas_lift_mode)
        )

    # Analyze chart cycles
    obs_summary, obs_groups = MOC._analyze_atlas_chart_cycles(
        atlas,
        int(target_parent_domain), int(target_fiber_label),
        max_cycles=0,
        max_group_order=4096,
        target_support=target_support,
        min_support_overlap_fraction=0.3,
        min_support_jaccard=0.05,
        min_support_overlap_states=1,
    )

    return {
        "chart_c2_count": int(getattr(atlas, "n_chart_c2", 0)),
        "chart_nontrivial_count": int(getattr(atlas, "n_chart_nontrivial", 0)),
        "target_exact_s3": bool(obs_summary.get("target_exact_s3_group_closure", False)),
        "target_group_order": int(obs_summary.get("target_generated_group_order", 0)),
        "target_n_distinct_c2": int(obs_summary.get("target_n_distinct_c2_maps", 0)),
        "target_max_component": int(obs_summary.get("target_max_transposition_component_size", 0)),
        "target_namespace_observed": bool(obs_summary.get("target_namespace_observed", False)),
        "any_s3": bool(obs_summary.get("any_exact_s3_group_closure", False)),
        "support_exact_s3": bool(obs_summary.get("support_exact_s3_group_closure", False)),
        "support_best_jaccard": float(obs_summary.get("support_best_jaccard", 0.0) or 0.0),
        "support_overlap_observed": bool(obs_summary.get("support_overlap_observed", False)),
        "support_best_overlap_count": int(obs_summary.get("support_best_overlap_count", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_perturbation_stability_audit(
    q: int = 2,
    vertices: int = 9,
    frozen_transition_npy: str = "",
    support_indices: str = "29,30,44,47,61,62,156,159,188,191,221,222,253,254,256,284,287",
    perturbation_levels: str = "0,1,2,4,8,16,32,64,128",
    perturbation_modes: str = "random,inside,outside",
    trials_per_level: int = 10,
    observers_per_trial: int = 1,
    observer_seed_base: int = 42,
    profile: str = "full_atlas",
    capacity: int = 32,
    target_parent_domain: int = 75,
    target_fiber_label: int = 7,
    out: str = "",
    plot: str = "",
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    if GCPR is None or MOC is None:
        raise RuntimeError("requires relgauge package")

    if not frozen_transition_npy or not os.path.exists(frozen_transition_npy):
        raise ValueError(f"Provide --frozen-transition-npy: {frozen_transition_npy}")

    # Load transition table
    T_orig = np.load(frozen_transition_npy)
    n_states = len(T_orig)
    n_bits = int(vertices)
    states = np.array([[int(b) for b in format(i, f'0{n_bits}b')] for i in range(n_states)])

    # Parse support indices
    support_set: Set[int] = set()
    for tok in str(support_indices).replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            support_set.add(int(float(tok)))

    # Parse levels and modes
    levels = _parse_int_list(perturbation_levels, [0, 1, 2, 4, 8, 16, 32, 64, 128])
    modes = _parse_text_list(perturbation_modes, ["random", "inside", "outside"])

    # Build args
    args = argparse.Namespace(
        q=q, vertices=vertices,
        atlas_lift_mode=kwargs.get("atlas_lift_mode", "bijective"),
        **{k: v for k, v in kwargs.items() if k != "atlas_lift_mode"},
    )
    _ensure_defaults(args)

    print(f"Frozen transition: {frozen_transition_npy} ({n_states} states)")
    print(f"Support: {len(support_set)} states: {sorted(support_set)}")
    print(f"Levels: {levels}")
    print(f"Modes: {modes}")
    print(f"Trials per (level, mode): {trials_per_level}")
    print(f"Observers per trial: {observers_per_trial}")

    # First: verify unperturbed baseline
    baseline_rng = np.random.default_rng(observer_seed_base)
    baseline = _check_s3(
        states, T_orig, q, profile, capacity,
        target_parent_domain, target_fiber_label, support_set,
        baseline_rng, args,
    )
    print(f"\nBaseline (unperturbed): S3={baseline['target_exact_s3']}, "
          f"C2={baseline['chart_c2_count']}, order={baseline['target_group_order']}")

    # Run perturbation grid
    rows: List[Dict[str, Any]] = []
    trial_seed = 100000

    for level in levels:
        for mode in modes:
            # Skip inside perturbation if level exceeds support size
            if mode == "inside" and level > len(support_set):
                continue

            for trial in range(trials_per_level):
                trial_seed += 1
                perturb_rng = np.random.default_rng(trial_seed)

                T_pert, perturb_meta = _perturb_transition(
                    T_orig, level, mode, support_set, perturb_rng,
                )

                # Check if transition actually changed
                n_changed = int(np.sum(T_pert != T_orig))

                # Run observer(s)
                s3_votes = 0
                support_s3_votes = 0
                c2_counts = []
                group_orders = []
                jaccards = []

                for obs in range(observers_per_trial):
                    obs_rng = np.random.default_rng(observer_seed_base + obs)
                    result = _check_s3(
                        states, T_pert, q, profile, capacity,
                        target_parent_domain, target_fiber_label, support_set,
                        obs_rng, args,
                    )
                    if result["target_exact_s3"]:
                        s3_votes += 1
                    if result["support_exact_s3"]:
                        support_s3_votes += 1
                    c2_counts.append(result["chart_c2_count"])
                    group_orders.append(result["target_group_order"])
                    jaccards.append(result["support_best_jaccard"])

                row = {
                    "perturbation_level": int(level),
                    "perturbation_mode": str(mode),
                    "trial": int(trial),
                    "trial_seed": int(trial_seed),
                    "n_actually_changed": int(n_changed),
                    "n_inside_support": int(perturb_meta["n_inside_support"]),
                    "n_outside_support": int(perturb_meta["n_outside_support"]),
                    "target_s3_fraction": float(s3_votes / max(1, observers_per_trial)),
                    "support_s3_fraction": float(support_s3_votes / max(1, observers_per_trial)),
                    "mean_c2_count": float(np.mean(c2_counts)),
                    "mean_group_order": float(np.mean(group_orders)),
                    "mean_support_jaccard": float(np.mean(jaccards)),
                    "target_s3_survived": bool(s3_votes == observers_per_trial),
                    "support_s3_survived": bool(support_s3_votes == observers_per_trial),
                }
                rows.append(row)

                status = "S3" if row["target_s3_survived"] else ("C2" if row["mean_c2_count"] > 0 else "FLAT")
                print(f"  level={level:3d} mode={mode:7s} trial={trial+1:2d}/{trials_per_level} "
                      f"changed={n_changed:3d} in={perturb_meta['n_inside_support']:2d} "
                      f"out={perturb_meta['n_outside_support']:3d} -> {status} "
                      f"c2={row['mean_c2_count']:.0f} order={row['mean_group_order']:.0f} "
                      f"jacc={row['mean_support_jaccard']:.3f}")

    df = pd.DataFrame(rows)

    # Aggregate by (level, mode)
    agg_rows: List[Dict[str, Any]] = []
    for (level, mode), g in df.groupby(["perturbation_level", "perturbation_mode"]):
        n_trials = len(g)
        agg = {
            "perturbation_level": int(level),
            "perturbation_mode": str(mode),
            "n_trials": int(n_trials),
            "s3_survival_rate": float(g["target_s3_survived"].astype(float).mean()),
            "support_s3_survival_rate": float(g["support_s3_survived"].astype(float).mean()),
            "mean_s3_fraction": float(g["target_s3_fraction"].mean()),
            "mean_support_s3_fraction": float(g["support_s3_fraction"].mean()),
            "mean_c2_count": float(g["mean_c2_count"].mean()),
            "mean_group_order": float(g["mean_group_order"].mean()),
            "mean_support_jaccard": float(g["mean_support_jaccard"].mean()),
            "mean_n_changed": float(g["n_actually_changed"].mean()),
        }
        agg_rows.append(agg)

    adf = pd.DataFrame(agg_rows)

    # Find critical perturbation level per mode
    critical: Dict[str, Any] = {}
    for mode in modes:
        mode_data = adf[adf["perturbation_mode"] == mode].sort_values("perturbation_level")
        # Critical level = first level where s3_survival_rate < 1.0
        below = mode_data[mode_data["s3_survival_rate"] < 1.0]
        if below.empty:
            critical[mode] = {"critical_level": "above_max_tested", "max_tested": int(levels[-1])}
        else:
            first_break = below.iloc[0]
            critical[mode] = {
                "critical_level": int(first_break["perturbation_level"]),
                "survival_at_critical": float(first_break["s3_survival_rate"]),
                "c2_at_critical": float(first_break["mean_c2_count"]),
            }
            # Also find level where S₃ is completely destroyed
            destroyed = mode_data[mode_data["s3_survival_rate"] == 0.0]
            if not destroyed.empty:
                critical[mode]["destruction_level"] = int(destroyed.iloc[0]["perturbation_level"])

    # Compute mass-like proxy: ratio of inside vs outside critical levels
    inside_crit = critical.get("inside", {}).get("critical_level", None)
    outside_crit = critical.get("outside", {}).get("critical_level", None)
    if isinstance(inside_crit, int) and isinstance(outside_crit, int) and outside_crit > 0:
        mass_ratio = float(inside_crit) / float(outside_crit)
    elif isinstance(inside_crit, str) and isinstance(outside_crit, str):
        mass_ratio = 1.0  # both above max
    else:
        mass_ratio = None

    summary = {
        "verdict": "",
        "audit_version": "perturbation_stability_v1_mass_proxy",
        "frozen_transition_npy": str(frozen_transition_npy),
        "n_states": int(n_states),
        "support_size": int(len(support_set)),
        "support_indices": sorted(support_set),
        "perturbation_levels": levels,
        "perturbation_modes": modes,
        "trials_per_level": int(trials_per_level),
        "observers_per_trial": int(observers_per_trial),
        "baseline_s3": bool(baseline["target_exact_s3"]),
        "baseline_c2_count": int(baseline["chart_c2_count"]),
        "baseline_group_order": int(baseline["target_group_order"]),
        "critical_levels": _json_safe(critical),
        "mass_like_ratio_inside_vs_outside": _json_safe(mass_ratio),
        "n_trial_rows": int(len(df)),
        "n_aggregate_rows": int(len(adf)),
    }

    # Verdict
    if all(isinstance(v.get("critical_level"), str) for v in critical.values()):
        summary["verdict"] = "S3 ROBUST: survives all tested perturbation levels"
    elif mass_ratio is not None and mass_ratio < 0.5:
        summary["verdict"] = f"S3 FRAGILE INSIDE: inside perturbations destroy S3 at lower threshold (ratio={mass_ratio:.2f})"
    elif mass_ratio is not None and mass_ratio > 2.0:
        summary["verdict"] = f"S3 FRAGILE OUTSIDE: outside perturbations destroy S3 at lower threshold (ratio={mass_ratio:.2f})"
    else:
        summary["verdict"] = "S3 PERTURBATION SENSITIVITY MEASURED"

    print(f"\n{'='*60}")
    print(f"VERDICT: {summary['verdict']}")
    print(f"{'='*60}")
    print(f"Baseline: S3={baseline['target_exact_s3']}, C2={baseline['chart_c2_count']}")
    for mode, crit in critical.items():
        print(f"  {mode}: critical_level={crit.get('critical_level')}")
    if mass_ratio is not None:
        print(f"Mass-like ratio (inside/outside): {mass_ratio:.3f}")

    # Write outputs
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df.to_csv(out, index=False)
        agg_path = out.replace(".csv", "_aggregate.csv")
        adf.to_csv(agg_path, index=False)
        sum_path = out.replace(".csv", "_summary.json")
        with open(sum_path, "w") as f:
            json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
        print(f"\nwrote {out}")
        print(f"wrote {agg_path}")
        print(f"wrote {sum_path}")

    if plot:
        _plot(adf, summary, plot)

    return df, adf, summary


def _plot(adf: pd.DataFrame, summary: Dict[str, Any], path: str):
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for mode in adf["perturbation_mode"].unique():
            md = adf[adf["perturbation_mode"] == mode].sort_values("perturbation_level")
            ax1.plot(md["perturbation_level"], md["s3_survival_rate"],
                     marker="o", label=f"{mode} (target S3)")
            ax1.plot(md["perturbation_level"], md["support_s3_survival_rate"],
                     marker="s", linestyle="--", alpha=0.5, label=f"{mode} (support S3)")

        ax1.set_xlabel("Perturbation level (entries changed)")
        ax1.set_ylabel("S₃ survival rate")
        ax1.set_title("S₃ stability under perturbation")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(fontsize=8)
        ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

        for mode in adf["perturbation_mode"].unique():
            md = adf[adf["perturbation_mode"] == mode].sort_values("perturbation_level")
            ax2.plot(md["perturbation_level"], md["mean_c2_count"],
                     marker="o", label=f"{mode} C2")
            ax2.plot(md["perturbation_level"], md["mean_group_order"],
                     marker="s", linestyle="--", alpha=0.5, label=f"{mode} order")

        ax2.set_xlabel("Perturbation level (entries changed)")
        ax2.set_ylabel("Mean C₂ count / group order")
        ax2.set_title("Gauge structure vs perturbation")
        ax2.legend(fontsize=8)

        fig.suptitle(summary.get("verdict", "Perturbation stability"))
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"wrote {path}")
    except Exception as e:
        print(f"Plot failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Perturbation stability audit: mass-like proxy for structural S₃"
    )
    ap.add_argument("q", type=int, nargs="?", default=2)
    ap.add_argument("--vertices", type=int, default=9)
    ap.add_argument("--frozen-transition-npy", required=True)
    ap.add_argument("--support-indices",
                     default="29,30,44,47,61,62,156,159,188,191,221,222,253,254,256,284,287")
    ap.add_argument("--perturbation-levels", default="0,1,2,4,8,16,32,64,128")
    ap.add_argument("--perturbation-modes", default="random,inside,outside")
    ap.add_argument("--trials-per-level", type=int, default=10)
    ap.add_argument("--observers-per-trial", type=int, default=1)
    ap.add_argument("--observer-seed-base", type=int, default=42)
    ap.add_argument("--profile", default="full_atlas")
    ap.add_argument("--capacity", type=int, default=32)
    ap.add_argument("--target-parent-domain", type=int, default=75)
    ap.add_argument("--target-fiber-label", type=int, default=7)

    # Atlas knobs
    ap.add_argument("--atlas-lift-mode", default="bijective")
    ap.add_argument("--proliferation-iterations", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--max-chart-coords", type=int, default=5)
    ap.add_argument("--max-support-coords", type=int, default=4)
    ap.add_argument("--max-charts-per-fiber", type=int, default=16)
    ap.add_argument("--min-chart-entropy", type=float, default=0.05)
    ap.add_argument("--min-support-states", type=int, default=4)
    ap.add_argument("--min-overlap-states", type=int, default=4)

    ap.add_argument("--out", default="example_results/perturbation_stability.csv")
    ap.add_argument("--plot", default="")

    args = ap.parse_args()

    run_perturbation_stability_audit(
        q=args.q,
        vertices=args.vertices,
        frozen_transition_npy=args.frozen_transition_npy,
        support_indices=args.support_indices,
        perturbation_levels=args.perturbation_levels,
        perturbation_modes=args.perturbation_modes,
        trials_per_level=args.trials_per_level,
        observers_per_trial=args.observers_per_trial,
        observer_seed_base=args.observer_seed_base,
        profile=args.profile,
        capacity=args.capacity,
        target_parent_domain=args.target_parent_domain,
        target_fiber_label=args.target_fiber_label,
        out=args.out,
        plot=args.plot,
        atlas_lift_mode=args.atlas_lift_mode,
        proliferation_iterations=args.proliferation_iterations,
        horizon=args.horizon,
        max_chart_coords=args.max_chart_coords,
        max_support_coords=args.max_support_coords,
        max_charts_per_fiber=args.max_charts_per_fiber,
        min_chart_entropy=args.min_chart_entropy,
        min_support_states=args.min_support_states,
        min_overlap_states=args.min_overlap_states,
    )


if __name__ == "__main__":
    main()
