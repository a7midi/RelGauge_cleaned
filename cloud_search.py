import modal
import json

app = modal.App("siq-basepoint-search")

# Package your relgauge code into the cloud image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "matplotlib")
)

mount = modal.Mount.from_local_dir("relgauge", remote_path="/root/relgauge")
data_mount = modal.Mount.from_local_dir("raw", remote_path="/root/raw")

@app.function(image=image, mounts=[mount, data_mount], timeout=3600, cpu=2, memory=2048)
def search_candidate(vertices, seed, instance, capacity, max_charts, max_cycle_len, min_overlap):
    import sys, os
    sys.path.insert(0, "/root")
    os.makedirs("/root/results", exist_ok=True)

    import numpy as np
    from relgauge import dynamicsconsistencyfixedpointaudit as DCFP
    from relgauge import basepointawareholonomyaudit as BPA
    import argparse

    args = argparse.Namespace(
        q=2, vertices=vertices, iterated_csv="",
        frozen_transition_npy="",
        target_rule_mode="random_full_permutation",
        target_instance=instance, target_profile="full_atlas",
        target_atlas_capacity=capacity, target_seed=seed,
        target_iteration=-1, target_parent_domain=-1, target_fiber_label=-1,
        rule_modes="", profiles="", atlas_capacities="",
        max_candidates=1, atlas_iterations=12,
        require_generated=False, stop_at_first_nonabelian=True,
        max_state_samples=512, max_total_states=200000, max_pred=0,
        proliferation_iterations=4, horizon=3,
        initial_boundary="sum_mod_q", initial_boundary_q=None,
        max_domains_per_depth=32, min_live_classes=2,
        min_fiber_size=2, min_entropy_bits=0.05,
        synergy_threshold=0.01, max_signature_domains=16,
        max_parent_domains=8, max_fibers_per_parent=6,
        max_charts_per_fiber=max_charts, max_signature_charts=48,
        min_fiber_states=4, min_support_states=4,
        min_overlap_states=min_overlap, min_chart_classes=2,
        min_chart_entropy=0.05, max_chart_coords=5,
        max_support_coords=4, max_cycle_len=max_cycle_len,
        max_cycles_per_fiber=500, max_loops_per_base=1000,
        max_group_order=4096, max_domains_scan=0,
        max_fibers_per_domain_scan=0, atlas_lift_mode="bijective",
        include_trivial=False, out="", plot="",
    )
    BPA._ensure_upstream_defaults(args)

    # Initialize candidate
    rng0 = np.random.default_rng(seed)
    states, current_next, init_meta = DCFP.initialize_sampled_transition(
        q=2, vertices=vertices, mode="random_full_permutation", rng=rng0,
        max_state_samples=512, max_total_states=200000, max_pred=0,
        proliferation_iterations=4, horizon=3,
    )
    states = np.asarray(states, dtype=np.int64)
    current_next = np.asarray(current_next, dtype=np.int64)

    best_order = 0
    best_result = None

    for it in range(13):
        try:
            atlas, eff = BPA._build_atlas_from_transition(
                states, current_next, 2,
                {"rule_mode": "random_full_permutation", "instance": instance,
                 "profile": "full_atlas", "atlas_capacity": capacity,
                 "initial_seed": seed},
                it, args
            )
            gr, lr = BPA._analyze_atlas_basepoints(
                atlas, states, current_next, 2,
                {"rule_mode": "random_full_permutation", "instance": instance,
                 "profile": "full_atlas", "atlas_capacity": capacity,
                 "initial_seed": seed},
                it, args
            )
            for row in gr:
                order = int(row.get("generated_group_order", 0))
                if order > best_order:
                    best_order = order
                    best_result = row
                if row.get("nonabelian"):
                    return {"found": True, "vertices": vertices, "seed": seed,
                            "instance": instance, "capacity": capacity,
                            "iteration": it, "result": row}
            current_next = np.asarray(eff, dtype=np.int64)
        except Exception as e:
            pass

    return {"found": False, "vertices": vertices, "seed": seed,
            "instance": instance, "capacity": capacity,
            "best_order": best_order, "best_result": best_result}


@app.local_entrypoint()
def main():
    import itertools

    configs = []
    # v=10: 200 candidates across capacities
    for inst in range(100):
        for cap in [32, 64]:
            seed = 1000003 + 7919 * inst
            configs.append((10, seed, inst, cap, 24, 5, 3))

    # v=11: 100 candidates
    for inst in range(50):
        for cap in [32, 64]:
            seed = 2000003 + 7919 * inst
            configs.append((11, seed, inst, cap, 24, 5, 3))

    print(f"Launching {len(configs)} candidates across cloud workers...")

    results = []
    for batch_start in range(0, len(configs), 50):
        batch = configs[batch_start:batch_start + 50]
        batch_results = list(search_candidate.starmap(batch))
        for r in batch_results:
            if r["found"]:
                print(f"\n{'='*60}")
                print(f"NONABELIAN FOUND!")
                print(json.dumps(r, indent=2, default=str))
                print(f"{'='*60}\n")
            else:
                v = r["vertices"]
                s = r["seed"]
                bo = r.get("best_order", 0)
                print(f"  v={v} seed={s} best_order={bo}")
        results.extend(batch_results)

    # Summary
    found = [r for r in results if r["found"]]
    print(f"\nTotal candidates: {len(results)}")
    print(f"Nonabelian found: {len(found)}")

    with open("cloud_search_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Wrote cloud_search_results.json")