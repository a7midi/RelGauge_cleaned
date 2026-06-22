import numpy as np
from relgauge import iterationattractoraudit as IAA
import argparse

args = argparse.Namespace(
    q=3, vertices=6, seed=2000006, rule_mode="random_full_permutation",
    profile="full_atlas", atlas_capacity=32, atlas_iterations=8,
    max_state_samples=729, max_total_states=729, max_pred=0,
    proliferation_iterations=4, horizon=4, atlas_lift_mode="bijective",
    initial_boundary="sum_mod_q", initial_boundary_q=None,
    max_domains_per_depth=32, min_live_classes=2, min_fiber_size=2,
    min_entropy_bits=0.05, synergy_threshold=0.01,
    max_signature_domains=16, max_parent_domains=8,
    max_fibers_per_parent=6, max_charts_per_fiber=32,
    max_signature_charts=48, min_fiber_states=6,
    min_support_states=3, min_overlap_states=3,
    min_chart_classes=2, min_chart_entropy=0.0,
    max_chart_coords=5, max_support_coords=4,
    max_cycle_len=5, max_cycles_per_fiber=500,
    max_loops_per_base=500, max_group_order=4096,
    max_domains_scan=20, max_fibers_per_domain_scan=3,
)

seed = IAA._candidate_seed(2000006, 0, 1)  # instance 1
print(f"Reconstructing instance 1, seed={seed}")

states, T0, meta = IAA._initial_transition(3, 6, "random_full_permutation", seed, args)
print(f"T0 hash: {IAA._hash_arr(T0)}")

atlas, T1, ps, ls = IAA._atlas_and_eff(states, T0, 3, "full_atlas", 32, 0, seed, args)
print(f"T1 hash: {IAA._hash_arr(T1)}")

atlas2, T2, ps2, ls2 = IAA._atlas_and_eff(states, T1, 3, "full_atlas", 32, 1, seed, args)
print(f"T2 hash: {IAA._hash_arr(T2)}")

print(f"T1 == T2 (fixed point verified): {np.array_equal(T1, T2)}")
print(f"T* length: {len(T1)}")
print(f"T* is permutation: {sorted(T1.tolist()) == list(range(len(T1)))}")

np.save("results/s3_fixed_point_T_star.npy", T1)
print("Saved results/s3_fixed_point_T_star.npy")

# Cycle structure
visited = [False] * 729
cycles = []
for s in range(729):
    if visited[s]:
        continue
    cyc = []
    c = s
    while not visited[c]:
        visited[c] = True
        cyc.append(c)
        c = int(T1[c])
    cycles.append(len(cyc))

print(f"\nCycle structure of T*:")
print(f"  Number of cycles: {len(cycles)}")
print(f"  Longest cycle: {max(cycles)}")
print(f"  Fixed points (1-cycles): {sum(1 for c in cycles if c == 1)}")
print(f"  Top cycle lengths: {sorted(cycles, reverse=True)[:10]}")
print(f"  Total states in cycles: {sum(cycles)}")
