#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
# Minimal metadata for the frozen transition target. This is not a summary; it is
# the candidate row needed by the actual relgauge modules.
cp data/repro_witness_search_minimal.csv results/repro_witness_search.csv

echo "[quick 1/2] Frozen-transition multi-observer consensus"
python -m relgauge.multiobserverconsensusaudit 2 \
  --vertices 9 \
  --iterated-csv results/repro_witness_search.csv \
  --frozen-transition-npy data/transition_iter8_inst28_seed3221741.npy \
  --target-rule-mode random_full_permutation \
  --target-instance 28 \
  --target-profile full_atlas \
  --target-atlas-capacity 32 \
  --target-seed 3221741 \
  --target-iteration 8 \
  --target-parent-domain 75 \
  --target-fiber-label 7 \
  --observer-runs 64 \
  --observer-seed-start 0 \
  --observer-seed-stride 1 \
  --max-state-samples 512 \
  --max-chart-coords 5 \
  --max-support-coords 4 \
  --max-charts-per-fiber 16 \
  --min-chart-entropy 0.05 \
  --min-support-overlap-fraction 0.5 \
  --min-support-jaccard 0.05 \
  --min-support-overlap-states 1 \
  --out results/multi_observer_consensus.csv \
  --plot results/fig_multi_observer_consensus.png

echo "[quick 2/2] Observation-method scale test"
python -m relgauge.observationmethodscaletest 2 \
  --vertices 9 \
  --iterated-csv results/repro_witness_search.csv \
  --frozen-transition-npy data/transition_iter8_inst28_seed3221741.npy \
  --target-rule-mode random_full_permutation \
  --target-instance 28 \
  --target-profile full_atlas \
  --target-atlas-capacity 32 \
  --target-seed 3221741 \
  --target-iteration 8 \
  --target-parent-domain 75 \
  --target-fiber-label 7 \
  --observer-profiles full_atlas,fiber_preserving,boundary_only \
  --observer-atlas-capacities 16,32,64,128 \
  --max-chart-coords-list 5 \
  --max-support-coords-list 4 \
  --max-charts-per-fiber-list 16 \
  --min-chart-entropy-list 0,0.025,0.05,0.1 \
  --min-support-states-list 4 \
  --min-overlap-states-list 4 \
  --observer-runs 16 \
  --max-state-samples 512 \
  --min-support-overlap-fraction 0.5 \
  --min-support-jaccard 0.05 \
  --out results/observation_method_scale.csv \
  --plot results/fig_observation_method_scale.png

echo "Quick reproduction complete. Key summaries:"
python - <<'PY'
import json
for path in ['results/multi_observer_consensus_summary.json','results/observation_method_scale_summary.json']:
    with open(path) as f: s=json.load(f)
    print(path)
    for k in ['verdict','target_exact_s3_consensus_fraction','support_exact_s3_consensus_fraction','any_exact_s3_fraction','support_s3_config_fraction','n_configs']:
        if k in s: print(f'  {k}: {s[k]}')
PY
