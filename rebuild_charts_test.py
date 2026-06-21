"""
Rebuild charts via FCA.build_charts_for_domain_fiber to get actual
label arrays, compute transport g_{1,5}, and test conjugation.
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
# States as tuples (what FCA expects)
states_tuples = [tuple(int(x) for x in row) for row in states]

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
from relgauge import fiberchartconnectionaudit as FCA

print("Building atlas...")
atlas = _one_atlas_pass(states, T, 2, rng, args)

# Find domain 75
target_domain = None
for d in list(getattr(atlas, "domains_current", [])) + list(getattr(atlas, "domains_all", [])):
    if getattr(d, "domain_id", -1) == 75:
        target_domain = d
        break

if target_domain is None:
    print("ERROR: Domain 75 not found")
    sys.exit(1)

print(f"Domain 75 found. n_labels={getattr(target_domain, 'n_labels', '?')}")

# Rebuild charts for fiber 7 using FCA directly
print("\nRebuilding charts for domain 75, fiber 7...")
try:
    fiber_charts = FCA.build_charts_for_domain_fiber(
        domain=target_domain,
        fiber_label=7,
        states=states_tuples,
        next_idx=T.tolist(),
        q=2,
        horizon=3,
        max_chart_coords=5,
        max_support_coords=4,
        max_charts_per_fiber=16,
        min_chart_classes=2,
        min_chart_entropy=0.05,
        min_support_states=4,
    )
    print(f"Got {len(fiber_charts)} FiberChart objects")
except Exception as e:
    print(f"build_charts_for_domain_fiber failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Try alternative calling conventions
    print("\nTrying alternative...")
    try:
        import inspect
        src = inspect.getsource(FCA.build_charts_for_domain_fiber)
        print("Source preview:")
        for line in src.split("\n")[:30]:
            print(f"  {line}")
    except:
        pass
    sys.exit(1)

# Examine FiberChart objects
print("\n" + "=" * 70)
print("FIBERCHART OBJECTS")
print("=" * 70)

for i, fc in enumerate(fiber_charts):
    attrs = {}
    for attr in dir(fc):
        if not attr.startswith("_"):
            val = getattr(fc, attr, None)
            if not callable(val):
                attrs[attr] = val
    
    print(f"\n  Chart {i}:")
    for k, v in sorted(attrs.items()):
        vstr = str(v)
        if len(vstr) > 120:
            vstr = vstr[:120] + "..."
        print(f"    {k}: {vstr}")

# Try to get label arrays
print("\n" + "=" * 70)
print("EXTRACTING LABEL ARRAYS")
print("=" * 70)

chart_labels = {}
for i, fc in enumerate(fiber_charts):
    labels = getattr(fc, "labels", None)
    if labels is None:
        labels = getattr(fc, "chart_labels", None)
    if labels is None:
        labels = getattr(fc, "label_array", None)
    if labels is None:
        # Try to find any array-like attribute
        for attr in dir(fc):
            if not attr.startswith("_"):
                val = getattr(fc, attr, None)
                if isinstance(val, (np.ndarray, list, tuple)) and len(val) > 5:
                    print(f"  Chart {i}: found array attr '{attr}' with {len(val)} elements")
    
    support = getattr(fc, "support", None)
    if support is None:
        support = getattr(fc, "support_states", None)
    if support is None:
        support = getattr(fc, "support_indices", None)
    
    if labels is not None:
        chart_labels[i] = {
            "labels": np.array(labels) if not isinstance(labels, np.ndarray) else labels,
            "support": np.array(support) if support is not None else None,
        }
        print(f"  Chart {i}: labels={list(labels)[:20]}{'...' if len(labels)>20 else ''}")
    elif support is not None:
        print(f"  Chart {i}: no labels but support={list(support)[:20]}{'...' if len(support)>20 else ''}")
    else:
        print(f"  Chart {i}: no labels, no support found")

# Try using chart_transport_between
print("\n" + "=" * 70)
print("COMPUTING TRANSPORTS")
print("=" * 70)

if len(fiber_charts) >= 6:
    # Charts 1 and 5 in the cycle notation correspond to local chart indices
    # The cycle chart IDs might be local indices within this fiber's charts
    for ci in range(min(len(fiber_charts), 14)):
        for cj in range(ci+1, min(len(fiber_charts), 14)):
            try:
                transport = FCA.chart_transport_between(
                    fiber_charts[ci], fiber_charts[cj],
                    min_overlap_states=4,
                )
                if transport is not None:
                    # Examine transport object
                    t_attrs = {a: getattr(transport, a) for a in dir(transport) 
                              if not a.startswith("_") and not callable(getattr(transport, a))}
                    if t_attrs:
                        fwd = t_attrs.get("forward_map", t_attrs.get("map", t_attrs.get("transport", None)))
                        if fwd is not None:
                            print(f"  Transport {ci}->{cj}: {fwd}")
                        else:
                            print(f"  Transport {ci}->{cj}: attrs={t_attrs}")
            except Exception as e:
                pass  # Many pairs won't have valid transports

print("\nDone. Check output above for transport g_{1,5} and conjugation results.")
