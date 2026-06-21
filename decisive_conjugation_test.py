"""
DECISIVE TEST: Extract per-chart labels, compute transport g_{1,5},
conjugate the loop maps, and determine if S3 is real or artifact.
"""
import numpy as np
import hashlib
import argparse
import sys
import os

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

# Step 1: Examine chart_rows structure
print("\n" + "=" * 70)
print("STEP 1: CHART ROW STRUCTURE")
print("=" * 70)

chart_rows = atlas.chart_rows
print(f"Total chart_rows: {len(chart_rows)}")

if chart_rows:
    sample = chart_rows[0]
    if isinstance(sample, dict):
        print(f"Chart row keys: {sorted(sample.keys())}")
    elif hasattr(sample, "__dict__"):
        print(f"Chart row attrs: {sorted(sample.__dict__.keys())}")
    else:
        print(f"Chart row type: {type(sample)}")
        print(f"Chart row value: {sample}")

# Step 2: Find charts for domain 75, fiber 7
print("\n" + "=" * 70)
print("STEP 2: CHARTS AT DOMAIN 75, FIBER 7")
print("=" * 70)

target_charts = []
for i, cr in enumerate(chart_rows):
    row = cr if isinstance(cr, dict) else (cr.__dict__ if hasattr(cr, "__dict__") else {})
    pdid = row.get("parent_domain_id", row.get("domain_id", -1))
    flab = row.get("fiber_label", -1)
    if pdid == 75 and flab == 7:
        chart_id = row.get("chart_id", row.get("chart_index", i))
        target_charts.append((chart_id, row))

print(f"Charts at domain 75, fiber 7: {len(target_charts)}")

for chart_id, row in target_charts[:20]:
    # Try to find label data
    labels = row.get("labels", row.get("chart_labels", row.get("label_array", None)))
    support = row.get("support", row.get("support_states", row.get("support_indices", None)))
    coords = row.get("coords", row.get("chart_coords", row.get("coordinate_subset", None)))
    n_classes = row.get("n_classes", row.get("n_chart_classes", "?"))
    
    print(f"\n  Chart {chart_id}:")
    print(f"    n_classes = {n_classes}")
    print(f"    coords = {coords}")
    if labels is not None:
        if hasattr(labels, "__len__"):
            print(f"    labels type={type(labels).__name__}, len={len(labels)}")
            if len(labels) <= 30:
                print(f"    labels = {list(labels)}")
            else:
                print(f"    labels[:20] = {list(labels[:20])}")
        else:
            print(f"    labels = {labels}")
    else:
        print(f"    labels = NOT STORED")
    
    if support is not None:
        if hasattr(support, "__len__"):
            print(f"    support type={type(support).__name__}, len={len(support)}")
            if len(support) <= 30:
                print(f"    support = {list(support)}")
        else:
            print(f"    support = {support}")
    else:
        print(f"    support = NOT STORED")
    
    # Print ALL fields for first chart
    if chart_id == target_charts[0][0]:
        print(f"    ALL FIELDS:")
        for k, v in sorted(row.items()):
            vstr = str(v)
            if len(vstr) > 100:
                vstr = vstr[:100] + "..."
            print(f"      {k}: {vstr}")

# Step 3: Try using FCA directly to get chart transports
print("\n" + "=" * 70)
print("STEP 3: DIRECT FCA CHART TRANSPORT")
print("=" * 70)

try:
    from relgauge import fiberchartconnectionaudit as FCA
    
    # Try to build charts for domain 75, fiber 7 directly
    domain_labels = None
    for d in list(getattr(atlas, "domains_current", [])) + list(getattr(atlas, "domains_all", [])):
        if getattr(d, "domain_id", -1) == 75:
            domain_labels = np.array(getattr(d, "labels", []))
            break
    
    if domain_labels is not None:
        fiber_7_mask = (domain_labels == 7)
        fiber_7_states_idx = np.where(fiber_7_mask)[0]
        print(f"Fiber 7 states: {len(fiber_7_states_idx)}")
        
        # Try to call build_charts_for_domain_fiber
        import inspect
        sig = inspect.signature(FCA.build_charts_for_domain_fiber)
        print(f"build_charts_for_domain_fiber signature: {sig}")
        
        # Try chart_transport_between
        sig2 = inspect.signature(FCA.chart_transport_between)
        print(f"chart_transport_between signature: {sig2}")
        
        # Try to get FiberChart objects
        print(f"\nFiberChart fields: {[f for f in dir(FCA.FiberChart) if not f.startswith('_')]}")
        
        # Try ChartTransport
        if hasattr(FCA, "ChartTransport"):
            print(f"ChartTransport fields: {[f for f in dir(FCA.ChartTransport) if not f.startswith('_')]}")
    
except Exception as e:
    print(f"FCA exploration failed: {e}")
    import traceback
    traceback.print_exc()

# Step 4: If chart_rows have label data, compute transport manually
print("\n" + "=" * 70)
print("STEP 4: MANUAL TRANSPORT FROM CHART LABELS")
print("=" * 70)

# Get label arrays for charts that appear in the C2 cycles
# Charts 1, 5, 7, 8, 9, 10, 12, 13 are involved
needed_charts = {1, 5, 7, 8, 9, 10, 12, 13}
chart_label_arrays = {}

for chart_id, row in target_charts:
    if chart_id in needed_charts:
        labels = row.get("labels", row.get("chart_labels", row.get("label_array", None)))
        support = row.get("support", row.get("support_states", row.get("support_indices", None)))
        
        if labels is not None and hasattr(labels, "__len__") and len(labels) > 0:
            chart_label_arrays[chart_id] = {
                "labels": np.array(labels) if not isinstance(labels, np.ndarray) else labels,
                "support": np.array(support) if support is not None and hasattr(support, "__len__") else None,
            }
            print(f"  Chart {chart_id}: got {len(labels)} labels")

if chart_label_arrays:
    print(f"\nGot label arrays for charts: {sorted(chart_label_arrays.keys())}")
    
    # For each pair of charts, compute the transport on shared support
    fiber_7_set = set([29, 30, 44, 47, 61, 62, 156, 159, 188, 191, 221, 222, 253, 254, 256, 284, 287])
    
    for c_from in [1]:
        for c_to in [5, 7]:
            if c_from in chart_label_arrays and c_to in chart_label_arrays:
                labels_from = chart_label_arrays[c_from]["labels"]
                labels_to = chart_label_arrays[c_to]["labels"]
                
                # The labels might be indexed by state or by support position
                # Check if they're full-state-space arrays or support-only
                print(f"\n  Transport chart {c_from} -> chart {c_to}:")
                print(f"    Chart {c_from} labels length: {len(labels_from)}")
                print(f"    Chart {c_to} labels length: {len(labels_to)}")
                
                if len(labels_from) == 512:
                    # Full state space labels
                    transport = {}
                    for s in fiber_7_set:
                        a = int(labels_from[s])
                        b = int(labels_to[s])
                        if a >= 0 and b >= 0:  # both charts cover this state
                            if a not in transport:
                                transport[a] = {}
                            if b not in transport[a]:
                                transport[a][b] = 0
                            transport[a][b] += 1
                    
                    print(f"    Transport map (label_from -> label_to : count):")
                    for a in sorted(transport):
                        for b in sorted(transport[a]):
                            print(f"      {a} -> {b} : {transport[a][b]} states")
                    
                    # Determine the majority transport
                    g = {}
                    for a in sorted(transport):
                        best_b = max(transport[a], key=transport[a].get)
                        g[a] = best_b
                    print(f"    Majority transport g_{{{c_from},{c_to}}}: {g}")
                    
                    # Now conjugate (0<->1) at chart c_to to chart c_from
                    # Conjugation: g_inv . loop . g
                    g_inv = {v: k for k, v in g.items()}
                    loop = {0: 1, 1: 0}  # the (0<->1) transposition at chart c_to
                    
                    conjugated = {}
                    for a in sorted(g):
                        b = g[a]  # a at c_from -> b at c_to
                        c = loop.get(b, b)  # apply loop at c_to
                        d = g_inv.get(c, c)  # transport back to c_from
                        conjugated[a] = d
                    
                    print(f"    Conjugated loop (at chart {c_from}): {conjugated}")
                    
                    # Check if conjugated loop equals (0<->2)
                    is_02 = (conjugated.get(0) == 2 and conjugated.get(2) == 0)
                    is_01 = (conjugated.get(0) == 1 and conjugated.get(1) == 0)
                    print(f"    Is (0<->2)? {is_02}")
                    print(f"    Is (0<->1)? {is_01}")
                    print(f"    Is identity? {all(conjugated.get(k,k)==k for k in conjugated)}")
else:
    print("  No chart label arrays found in chart_rows.")
    print("  Need to rebuild charts with label storage enabled.")
    print()
    print("  Try: look at atlas.chart_rows[0] to see what fields exist.")
    if target_charts:
        cid, row = target_charts[0]
        print(f"\n  First chart row keys: {sorted(row.keys())}")

print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)
if chart_label_arrays and 1 in chart_label_arrays and 5 in chart_label_arrays:
    print("Transport computed. See conjugation results above.")
    print("If conjugated (0<->1) at chart 5 = (0<->2) at chart 1: S3 IS ARTIFACT")
    print("If conjugated (0<->1) at chart 5 != (0<->2) at chart 1: S3 MAY BE REAL")
else:
    print("Could not extract chart label arrays.")
    print("The chart_rows may not store per-state labels.")
    print("Need to modify FCA to retain label arrays during chart construction.")
