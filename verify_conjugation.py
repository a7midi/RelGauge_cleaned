"""
Decisive test: extract the overlap transport g_{1,5} and conjugate
the (0,1) loop at chart 5 back to chart 1's coordinates.

If the conjugated loop equals (0,2), then S3 collapses to C2
and the reviewer is right. If it's different, S3 may survive.
"""
import numpy as np
import hashlib
import argparse
import sys
import os
import re

sys.path.insert(0, ".")

npy_candidates = [
    "data/transition_iter8_inst28_seed3221741.npy",
    "example_results/repro_witness_search_q2_v9_fullperm_cap32_transition_iter8_random_full_permutation_full_atlas_cap32_inst28_seed3221741.npy",
]
npy_path = None
for p in npy_candidates:
    if os.path.exists(p):
        npy_path = p
        break

if not npy_path:
    print("ERROR: Cannot find .npy file")
    sys.exit(1)

T = np.load(npy_path)
states = np.array([[int(b) for b in format(i, "09b")] for i in range(512)])

seed = 3221741; cap = 32; profile = "full_atlas"; it = 8
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

# Find ALL chart cycle rows for domain 75, fiber 7 (not just C2)
print("\n" + "=" * 70)
print("ALL CHART CYCLES AT DOMAIN 75, FIBER 7 (including non-C2)")
print("=" * 70)
all_cycles_75_7 = []
for cr in atlas.chart_cycle_rows:
    if cr.get("parent_domain_id") == 75 and cr.get("fiber_label") == 7:
        all_cycles_75_7.append(cr)
        charts = cr.get("cycle_charts", "")
        lmap = cr.get("loop_map", "")
        valid = cr.get("chart_loop_valid", "")
        c2 = cr.get("chart_c2_holonomy", "")
        print(f"  charts={charts:20s} map={lmap:20s} valid={valid} c2={c2}")

print(f"\nTotal cycles in namespace: {len(all_cycles_75_7)}")

# Now look at the OVERLAP MAPS (transport maps between charts)
# These should be in the chart cycle data or computable from chart labels
print("\n" + "=" * 70)
print("EXTRACTING OVERLAP MAPS BETWEEN CHARTS")
print("=" * 70)

# Find the domain and its charts
target_domain = None
for d in list(getattr(atlas, "domains_current", [])) + list(getattr(atlas, "domains_all", [])):
    if getattr(d, "domain_id", -1) == 75:
        target_domain = d
        break

if target_domain is None:
    print("Domain 75 not found!")
    sys.exit(1)

domain_labels = np.array(getattr(target_domain, "labels", []))
fiber_7_states = np.where(domain_labels == 7)[0]
print(f"Fiber 7 support: {len(fiber_7_states)} states")
print(f"Fiber 7 state indices: {sorted(fiber_7_states.tolist())}")

# Get chart data - look for chart labels on fiber 7 states
# The charts assign labels to the fiber states
# We need to find what label each chart assigns to each state in fiber 7
print("\n--- Chart label assignments on fiber 7 states ---")

# Look through chart_cycle_rows for information about chart structure
# The overlap maps g_{i,j} tell us how chart i's labels relate to chart j's labels
# These are implicit in the cycle construction

# Let's extract the overlap maps from ALL cycles (not just C2)
# A cycle [c1, c2, c3, c4] with loop map H means:
#   g_{c1,c2} o g_{c2,c3} o g_{c3,c4} = H (automorphism of c1's labels)

# To find the individual transport g_{1,5}, we need to look at 2-chart paths
# or decompose the cycle data

# Alternative approach: look at what the chart_cycle_rows contain
# about intermediate transport steps
print("\nExamining cycle row fields:")
if all_cycles_75_7:
    sample = all_cycles_75_7[0]
    for k, v in sorted(sample.items()):
        print(f"  {k}: {v}")

# Try to find overlap/transport maps from the atlas structure
print("\n--- Looking for overlap maps in atlas ---")
# Check if the atlas has overlap data stored
for attr in ["overlap_maps", "chart_overlaps", "transport_maps", "edge_maps"]:
    val = getattr(atlas, attr, None)
    if val is not None:
        print(f"  Found: atlas.{attr} = {type(val)}")

# Try to reconstruct from the FCA module directly
print("\n--- Attempting direct chart reconstruction ---")
try:
    from relgauge import fiberchartconnectionaudit as FCA
    
    # The FCA module should have functions to build charts and compute overlaps
    # Let's see what's available
    fca_funcs = [f for f in dir(FCA) if not f.startswith("_") and callable(getattr(FCA, f, None))]
    print(f"  FCA functions: {fca_funcs[:20]}")
    
except Exception as e:
    print(f"  FCA import failed: {e}")

# MANUAL APPROACH: reconstruct what each chart does on fiber 7 states
# by examining the cycle compositions
print("\n" + "=" * 70)
print("MANUAL TRANSPORT RECONSTRUCTION")
print("=" * 70)

# From cycle [1, 5, 8, 7] with map (0->2, 2->0):
#   g_{1,5} o g_{5,8} o g_{8,7} o g_{7,1} = (0 2)  [at chart 1]
# 
# From cycle [5, 8, 7, 13] with map (0->1, 1->0):
#   g_{5,8} o g_{8,7} o g_{7,13} o g_{13,5} = (0 1)  [at chart 5]
#
# From cycle [7, 8, 12, 13] with map (0->1, 1->0):
#   g_{7,8} o g_{8,12} o g_{12,13} o g_{13,7} = (0 1)  [at chart 7]

# To conjugate loop at chart 5 to chart 1, we need g_{1,5}
# From the first cycle: g_{7,1} = (g_{1,5} o g_{5,8} o g_{8,7})^{-1} o (0 2)
# This is underdetermined without knowing individual transports

# KEY QUESTION: Do charts 1 and 5 describe the SAME states with the SAME
# or DIFFERENT label assignments?

# Let's check: what states does each chart cover?
# And what labels does each chart assign?
print("\nLooking for per-chart label data in cycle rows...")
for cr in all_cycles_75_7[:3]:
    charts_str = cr.get("cycle_charts", "")
    chart_ids = [int(x) for x in charts_str.split() if x.isdigit()]
    print(f"  Cycle charts: {chart_ids}")
    
    # Check for any per-chart fields
    for k, v in cr.items():
        if "label" in str(k).lower() or "support" in str(k).lower() or "state" in str(k).lower():
            if k not in ("fiber_label", "parent_domain_id"):
                print(f"    {k}: {str(v)[:80]}")

# Try to access chart objects directly from the atlas
print("\n--- Searching for chart objects in atlas ---")
charts_found = False
for attr in dir(atlas):
    if "chart" in attr.lower() and not attr.startswith("_"):
        val = getattr(atlas, attr, None)
        if val is not None and attr not in ("chart_cycle_rows", "n_chart_c2", "n_chart_c3", 
                                             "n_chart_nontrivial", "n_chart_cycles", 
                                             "n_chart_valid_cycles", "max_chart_order"):
            print(f"  atlas.{attr} = {type(val)}")
            if isinstance(val, (list, tuple)) and len(val) > 0:
                print(f"    length: {len(val)}")
                if hasattr(val[0], "__dict__"):
                    print(f"    item attrs: {list(val[0].__dict__.keys())[:10]}")
                charts_found = True

if not charts_found:
    print("  No direct chart objects found in atlas.")
    print("  The chart label data may only exist during FCA computation")
    print("  and not be stored in the atlas output.")
    print()
    print("  To resolve the reviewer's objection, we need to either:")
    print("  1. Modify FCA to store per-chart label arrays, or")
    print("  2. Add a basepoint-aware holonomy classifier, or")
    print("  3. Re-derive the overlap maps from the chart construction logic")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The two transpositions are at DIFFERENT base charts:
  (0<->2) at chart 1 only
  (0<->1) at charts 5 and 7 only

The reviewer's objection is structurally valid: loop maps at different
base charts use different label coordinate systems. Whether the S3
collapses to C2 under conjugation depends on the specific overlap
transport g_{1,5}.

TO RESOLVE THIS:
- Extract the actual overlap map between charts 1 and 5
- Conjugate (0<->1) from chart 5's labels to chart 1's labels  
- If the result is (0<->2): S3 is an artifact, only C2 exists
- If the result is different: S3 may survive

The chart label data is needed but may not be stored in the atlas
output structure. The FCA module constructs charts during the atlas
pass but may not retain per-chart label arrays afterward.
""")
