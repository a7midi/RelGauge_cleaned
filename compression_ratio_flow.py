"""
compression_ratio_flow.py

Track how the compression ratio (fiber_size / n_chart_labels) evolves
across self-observation iterations. Look for fixed-point convergence
that might determine natural alphabet sizes (q values).

The hypothesis: iterated self-observation compression converges to
specific ratios, and those ratios determine the effective observation
alphabet, hence the maximum holonomy group.
"""
import sys, os, json, hashlib, argparse
import numpy as np

sys.path.insert(0, ".")

def stable_hash(text):
    return int(hashlib.sha1(str(text).encode()).hexdigest()[:8], 16)

def run_compression_flow(q, vertices, n_instances, capacity, iterations, seed_base, out_dir):
    from relgauge import dynamicsconsistencyfixedpointaudit as DCFP
    from relgauge import fiberchartconnectionaudit as FCA
    from relgauge import generatedcandidatephysicsreplayaudit as GCPR

    os.makedirs(out_dir, exist_ok=True)
    n_states = q ** vertices
    states = np.array([
        [int(x) for x in np.base_repr(i, q).zfill(vertices)]
        for i in range(n_states)
    ], dtype=np.int64)

    all_flows = []

    for inst in range(n_instances):
        seed = seed_base + 7919 * inst
        print(f"\n{'='*60}")
        print(f"q={q} v={vertices} instance={inst} seed={seed}")
        print(f"{'='*60}")

        rng0 = np.random.default_rng(seed)
        _, current_next, _ = DCFP.initialize_sampled_transition(
            q=q, vertices=vertices, mode="random_full_permutation", rng=rng0,
            max_state_samples=n_states, max_total_states=200000, max_pred=0,
            proliferation_iterations=4, horizon=3,
        )
        current_next = np.asarray(current_next, dtype=np.int64)

        flow = {"q": q, "vertices": vertices, "instance": inst, "seed": seed, "iterations": []}

        args = argparse.Namespace(
            proliferation_iterations=4, horizon=3,
            initial_boundary="sum_mod_q", initial_boundary_q=None,
            max_domains_per_depth=32, min_live_classes=2,
            min_fiber_size=2, min_entropy_bits=0.05,
            synergy_threshold=0.01, max_signature_domains=16,
            max_parent_domains=8, max_fibers_per_parent=6,
            max_charts_per_fiber=16, max_signature_charts=48,
            min_fiber_states=4, min_support_states=3,
            min_overlap_states=3, min_chart_classes=2,
            min_chart_entropy=0.05, max_chart_coords=5,
            max_support_coords=4, max_cycle_len=4,
            max_cycles_per_fiber=500, max_state_samples=n_states,
            max_total_states=200000, max_pred=0,
            atlas_lift_mode="bijective",
        )

        for it in range(iterations + 1):
            profile = "full_atlas"
            ph = stable_hash(profile) % 1000
            iseed = seed + 1299709 * capacity + 15485863 * it + 104729 * ph
            rng = np.random.default_rng(iseed)

            try:
                atlas, _bounded, _pstats, _rel, _rel_rows, eff, _lift, _eff_rows = (
                    GCPR._advance_effective(
                        states, current_next, q, profile, capacity, rng, args, "bijective"
                    )
                )
            except Exception as e:
                print(f"  iter {it}: atlas failed: {e}")
                break

            # Extract compression ratios from all domains/fibers
            ratios = []
            fiber_sizes = []
            chart_label_counts = []

            domains = []
            for attr in ["domains_current", "domains_all"]:
                for d in list(getattr(atlas, attr, []) or []):
                    domains.append(d)

            seen_domains = set()
            for domain in domains:
                did = int(getattr(domain, "domain_id", -1))
                if did in seen_domains:
                    continue
                seen_domains.add(did)
                labels = np.asarray(getattr(domain, "labels", []), dtype=np.int64)
                if len(labels) == 0:
                    continue
                vals, counts = np.unique(labels, return_counts=True)
                for v, c in zip(vals, counts):
                    fiber_size = int(c)
                    if fiber_size < 4:
                        continue
                    # Build charts for this fiber
                    try:
                        charts = FCA.build_charts_for_domain_fiber(
                            domain, int(v), states, current_next, q, 3,
                            max_chart_coords=5, max_support_coords=4,
                            max_charts_per_fiber=16, min_chart_classes=2,
                            min_chart_entropy=0.05, min_support_states=3,
                        )
                    except:
                        charts = []

                    for ch in charts:
                        nl = int(getattr(ch, "n_labels", 0))
                        ns = int(getattr(ch, "n_support", 0))
                        if nl > 0 and ns > 0:
                            ratio = ns / nl
                            ratios.append(ratio)
                            fiber_sizes.append(ns)
                            chart_label_counts.append(nl)

            if ratios:
                median_ratio = float(np.median(ratios))
                mean_ratio = float(np.mean(ratios))
                min_ratio = float(np.min(ratios))
                max_ratio = float(np.max(ratios))
                # Count how many charts have each label count
                label_hist = {}
                for nl in chart_label_counts:
                    label_hist[nl] = label_hist.get(nl, 0) + 1
            else:
                median_ratio = mean_ratio = min_ratio = max_ratio = 0.0
                label_hist = {}

            c2 = int(getattr(atlas, "n_chart_c2", 0))
            nontrivial = int(getattr(atlas, "n_chart_nontrivial", 0))

            iter_data = {
                "iteration": it,
                "n_charts": len(ratios),
                "median_ratio": round(median_ratio, 4),
                "mean_ratio": round(mean_ratio, 4),
                "min_ratio": round(min_ratio, 4),
                "max_ratio": round(max_ratio, 4),
                "n_c2": c2,
                "n_nontrivial": nontrivial,
                "label_histogram": {str(k): v for k, v in sorted(label_hist.items())},
                "n_domains": len(seen_domains),
            }
            flow["iterations"].append(iter_data)

            print(f"  iter {it}: charts={len(ratios):3d}  median_ratio={median_ratio:.3f}  "
                  f"mean_ratio={mean_ratio:.3f}  labels={dict(sorted(label_hist.items()))}  "
                  f"c2={c2}  nontrivial={nontrivial}")

            current_next = np.asarray(eff, dtype=np.int64)

        all_flows.append(flow)

    # Save results
    results_path = os.path.join(out_dir, f"compression_flow_q{q}_v{vertices}.json")
    with open(results_path, "w") as f:
        json.dump(all_flows, f, indent=2)
    print(f"\nWrote {results_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Compression Ratio Flow: q={q}, v={vertices}, {n_instances} instances", fontsize=14)

        # Plot 1: median ratio per iteration
        ax = axes[0, 0]
        for flow in all_flows:
            iters = [d["iteration"] for d in flow["iterations"]]
            medians = [d["median_ratio"] for d in flow["iterations"]]
            ax.plot(iters, medians, alpha=0.4, linewidth=1)
        # Plot mean across instances
        max_it = max(len(f["iterations"]) for f in all_flows)
        for it_idx in range(max_it):
            vals = [f["iterations"][it_idx]["median_ratio"]
                    for f in all_flows if it_idx < len(f["iterations"]) and f["iterations"][it_idx]["median_ratio"] > 0]
            if vals:
                ax.scatter(it_idx, np.mean(vals), color="red", s=30, zorder=5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Median compression ratio")
        ax.set_title("Median (fiber_size / n_labels) per iteration")
        ax.axhline(y=2, color="blue", linestyle="--", alpha=0.3, label="ratio=2 (q=2 natural)")
        ax.axhline(y=3, color="green", linestyle="--", alpha=0.3, label="ratio=3 (q=3 natural)")
        ax.legend(fontsize=8)

        # Plot 2: label count histogram evolution
        ax = axes[0, 1]
        for it_idx in [0, 2, 4, 8]:
            label_counts = {}
            for flow in all_flows:
                if it_idx < len(flow["iterations"]):
                    for k, v in flow["iterations"][it_idx]["label_histogram"].items():
                        label_counts[int(k)] = label_counts.get(int(k), 0) + v
            if label_counts:
                ks = sorted(label_counts.keys())
                ax.bar([x + it_idx * 0.15 for x in ks],
                       [label_counts[k] for k in ks],
                       width=0.12, alpha=0.6, label=f"iter {it_idx}")
        ax.set_xlabel("n_labels per chart")
        ax.set_ylabel("Count")
        ax.set_title("Chart label count distribution")
        ax.legend(fontsize=8)

        # Plot 3: C2 and nontrivial holonomy per iteration
        ax = axes[1, 0]
        for flow in all_flows:
            iters = [d["iteration"] for d in flow["iterations"]]
            c2s = [d["n_c2"] for d in flow["iterations"]]
            nts = [d["n_nontrivial"] for d in flow["iterations"]]
            ax.plot(iters, c2s, alpha=0.3, color="blue", linewidth=1)
            ax.plot(iters, nts, alpha=0.3, color="red", linewidth=1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Count")
        ax.set_title("C₂ (blue) vs nontrivial (red) holonomy")

        # Plot 4: ratio convergence - final vs initial
        ax = axes[1, 1]
        initial_ratios = []
        final_ratios = []
        for flow in all_flows:
            if len(flow["iterations"]) >= 2:
                r0 = flow["iterations"][0]["median_ratio"]
                rf = flow["iterations"][-1]["median_ratio"]
                if r0 > 0 and rf > 0:
                    initial_ratios.append(r0)
                    final_ratios.append(rf)
        if initial_ratios:
            ax.scatter(initial_ratios, final_ratios, alpha=0.5)
            lim = max(max(initial_ratios), max(final_ratios)) * 1.1
            ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
            ax.set_xlabel("Initial median ratio")
            ax.set_ylabel("Final median ratio")
            ax.set_title("Ratio convergence: initial vs final")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"compression_flow_q{q}_v{vertices}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Track compression ratio flow across self-observation iterations")
    ap.add_argument("--q", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--vertices", type=int, default=0, help="0 = auto-select per q")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--capacity", type=int, default=32)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--seed-base", type=int, default=2000006)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    # Auto-select vertices to keep state space ~500-1000
    auto_vertices = {2: 9, 3: 6, 4: 5, 5: 4, 6: 4, 7: 3, 8: 3}

    for q in args.q:
        v = args.vertices if args.vertices > 0 else auto_vertices.get(q, 4)
        print(f"\n{'#'*60}")
        print(f"# q={q}, vertices={v}, states={q**v}")
        print(f"{'#'*60}")
        run_compression_flow(q, v, args.instances, args.capacity, args.iterations, args.seed_base, args.out_dir)
