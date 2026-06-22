"""Compare S3 vs flat 729-cycle fixed points."""
import numpy as np
from collections import Counter
from relgauge import iterationattractoraudit as IAA
import argparse

# Load S3 fixed point
T_s3 = np.load("results/s3_fixed_point_T_star.npy")

# Reconstruct a flat 729-cycle fixed point (instance 5)
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

seed_flat = IAA._candidate_seed(2000006, 0, 5)
states, T0, _ = IAA._initial_transition(3, 6, "random_full_permutation", seed_flat, args)
_, T_flat, _, _ = IAA._atlas_and_eff(states, T0, 3, "full_atlas", 32, 0, seed_flat, args)

# Verify flat is also a single cycle
visited = set()
c = 0
while c not in visited:
    visited.add(c)
    c = int(T_flat[c])
print(f"Flat cycle length: {len(visited)}")

S = np.array([[int(c) for c in np.base_repr(i, base=3).zfill(6)] for i in range(729)])

for name, T in [("S3", T_s3), ("FLAT", T_flat)]:
    img = S[T]
    print(f"\n{'='*50}")
    print(f"  {name} CYCLE ANALYSIS")
    print(f"{'='*50}")

    # Per-coordinate delta distributions
    print("\n  Per-coordinate deltas:")
    for coord in range(6):
        deltas = (img[:, coord] - S[:, coord]) % 3
        cnt = Counter(int(d) for d in deltas)
        print(f"    x[{coord}]: {dict(sorted(cnt.items()))}")

    # Check for clock coordinate (always +1)
    clocks = []
    for coord in range(6):
        deltas = (img[:, coord] - S[:, coord]) % 3
        if all(d == 1 for d in deltas):
            clocks.append(coord)
    print(f"\n  Clock coordinates (always +1): {clocks}")

    # Deterministic coordinate pairs
    print("\n  Deterministic coordinate pairs:")
    det_pairs = []
    for i in range(6):
        for j in range(i + 1, 6):
            mapping = {}
            ok = True
            for s in range(729):
                key = (int(S[s, i]), int(S[s, j]))
                val = (int(img[s, i]), int(img[s, j]))
                if key in mapping and mapping[key] != val:
                    ok = False
                    break
                mapping[key] = val
            if ok:
                det_pairs.append((i, j))
                print(f"    (x[{i}], x[{j}]) -> (T(x)[{i}], T(x)[{j}]) DETERMINISTIC")
    print(f"    Total: {len(det_pairs)}/15")

    # Hamming distance distribution
    hamming = [sum(1 for c in range(6) if S[s, c] != img[s, c]) for s in range(729)]
    hcnt = Counter(hamming)
    print(f"\n  Hamming distance: {dict(sorted(hcnt.items()))}")
    print(f"  Mean Hamming: {np.mean(hamming):.3f}")

    # Coordinate change rates
    print("\n  Coordinate change rates:")
    for coord in range(6):
        n_change = sum(1 for s in range(729) if S[s, coord] != img[s, coord])
        print(f"    x[{coord}]: {n_change}/729 = {n_change/729:.4f}")

    # Sum mod 3 behavior
    sum_deltas = [(sum(img[s]) - sum(S[s])) % 3 for s in range(729)]
    scnt = Counter(sum_deltas)
    print(f"\n  Sum(x) mod 3 delta: {dict(sorted(scnt.items()))}")

print("\n" + "=" * 50)
print("COMPARISON COMPLETE")
print("=" * 50)
