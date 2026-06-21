"""
Verify whether the S3 witness chart cycles share a common base chart
or are at different base charts (which would make S3 a label artifact).

This is the critical test for the reviewer's objection that the two
transpositions (0,2) and (0,1) are at different charts and collapse
to a single C2 when conjugated to a common base.
"""
import numpy as np
import hashlib
import argparse
import sys
import os

sys.path.insert(0, ".")

# Find the .npy file
npy_candidates = [
    "data/transition_iter8_inst28_seed3221741.npy",
    "example_results/repro_witness_search_q2_v9_fullperm_cap32_transition_iter8_random_full_permutation_full_atlas_cap32_inst28_seed3221741.npy",
]
npy_path = None
for p in npy_candidates:
    if os.path.exists(p):
        npy_path = p
        break

if npy_path is None:
    print("ERROR: Cannot find frozen transition .npy file")
    print("Looked in:", npy_candidates)
    sys.exit(1)

print(f"Loading: {npy_path}")
T = np.load(npy_path)
states = np.array([[int(b) for b in format(i, "09b")] for i in range(512)])
print(f"States: {states.shape}")

# Build atlas with reference parameters
seed = 3221741
cap = 32
profile = "full_atlas"
it = 8
ph = int(hashlib.sha1(profile.encode()).hexdigest()[:8], 16) % 1000
iseed = seed + 1299709 * cap + 15485863 * it + 104729 * ph
rng = np.random.default_rng(iseed)

args = argparse.Namespace(
    proliferation_iterations=4, horizon=3,
    initial_boundary="sum_mod_q", initial_boundary_q=None,
    max_domains_per_depth=32, min_live_classes=2,
    min_fiber_size=2, min_entropy_bits=0.05,
    synergy_threshold=0.01, max_signature_domains=16,
    max_parent_domains=8, max_fibers_per_parent=6,
    max_charts_per_fiber=16, max_signature_charts=48,
    min_fiber_states=8, min_support_states=4,
    min_overlap_states=4, min_chart_classes=2,
    min_chart_entropy=0.05, max_chart_coords=5,
    max_support_coords=4, max_cycle_len=4,
    max_cycles_per_fiber=500,
)

from relgauge.iteratedfiberatlasdynamicsaudit import _one_atlas_pass

print("Building atlas...")
atlas = _one_atlas_pass(states, T, 2, rng, args)
print(f"Chart cycles: {atlas.n_chart_cycles}, C2: {atlas.n_chart_c2}")

print("\n" + "=" * 70)
print("ALL C2 CHART CYCLES AT DOMAIN 75, FIBER 7")
print("=" * 70)

target_cycles = []
for cr in atlas.chart_cycle_rows:
    if cr.get("parent_domain_id") == 75 and cr.get("fiber_label") == 7:
        if cr.get("chart_c2_holonomy"):
            target_cycles.append(cr)
            charts_str = cr.get("cycle_charts", "")
            loop_map = cr.get("loop_map", "")
            
            # Parse the chart sequence
            chart_list = []
            if charts_str:
                for tok in str(charts_str).split():
                    try:
                        chart_list.append(int(tok))
                    except ValueError:
                        pass
            
            base_chart = chart_list[0] if chart_list else "?"
            print(f"  charts={charts_str:20s}  base_chart={base_chart}  loop_map={loop_map}")

print(f"\nTotal C2 cycles: {len(target_cycles)}")

# Analyze base charts
base_charts = {}
for cr in target_cycles:
    charts_str = cr.get("cycle_charts", "")
    loop_map = cr.get("loop_map", "")
    chart_list = []
    if charts_str:
        for tok in str(charts_str).split():
            try:
                chart_list.append(int(tok))
            except ValueError:
                pass
    base = chart_list[0] if chart_list else -1
    if base not in base_charts:
        base_charts[base] = []
    base_charts[base].append(loop_map)

print("\n" + "=" * 70)
print("GROUPING BY BASE CHART")
print("=" * 70)
for base, maps in sorted(base_charts.items()):
    unique_maps = sorted(set(maps))
    print(f"\n  Base chart {base}: {len(maps)} cycles")
    for m in unique_maps:
        count = maps.count(m)
        print(f"    loop_map = {m}  ({count} routes)")

# Critical question
all_bases = sorted(base_charts.keys())
print("\n" + "=" * 70)
print("CRITICAL ANALYSIS")
print("=" * 70)

# Check: do (0->2, 2->0) and (0->1, 1->0) come from the same base chart?
map_02_bases = set()
map_01_bases = set()
for base, maps in base_charts.items():
    for m in maps:
        if "0->2" in m and "2->0" in m:
            map_02_bases.add(base)
        if "0->1" in m and "1->0" in m:
            map_01_bases.add(base)

print(f"\n  (0<->2) transposition found at base charts: {sorted(map_02_bases)}")
print(f"  (0<->1) transposition found at base charts: {sorted(map_01_bases)}")
print(f"  Shared base charts: {sorted(map_02_bases & map_01_bases)}")

if map_02_bases & map_01_bases:
    print("\n  >>> BOTH transpositions appear at the SAME base chart.")
    print("  >>> The S3 is NOT a base-chart labeling artifact.")
    print("  >>> The reviewer's objection does not apply.")
else:
    print("\n  >>> The two transpositions appear at DIFFERENT base charts.")
    print("  >>> The reviewer's objection MAY be valid.")
    print("  >>> Need to check whether conjugation collapses them to one C2.")

print("\nDone.")
