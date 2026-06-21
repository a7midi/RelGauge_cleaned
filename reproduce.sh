#!/usr/bin/env bash
set -euo pipefail
mkdir -p results

echo "[1/5] Generate/replay witness search (core positive chain; expected runtime: tens of minutes on a laptop)"
python -m relgauge.iteratedfiberatlasdynamicsaudit 2 \
  --vertices 9 \
  --rule-modes random_full_permutation \
  --instances 29 \
  --atlas-capacities 32 \
  --profiles full_atlas \
  --atlas-iterations 12 \
  --save-transition-at 8 \
  --seed 2000006 \
  --out results/repro_witness_search.csv

echo "[1/5 check] Compare regenerated transition to bundled frozen transition"
python - <<'PY'
import glob, numpy as np, sys
regen=glob.glob('results/repro_witness_search_transition_iter8_random_full_permutation_full_atlas_cap32_inst28_seed3221741.npy')
if not regen:
    raise SystemExit('Regenerated transition file not found')
a=np.load(regen[0]); b=np.load('data/transition_iter8_inst28_seed3221741.npy')
print('transition_arrays_equal:', bool((a==b).all()), 'shape:', a.shape)
if a.shape!=b.shape or not (a==b).all():
    raise SystemExit('Regenerated transition does not match bundled frozen transition')
PY

echo "[2/5] Critical C2-birth window / S3 detection"
python -m relgauge.criticalc2birthwindowaudit \
  --chart-cycles-csv results/repro_witness_search_chart_cycles.csv \
  --iterated-csv results/repro_witness_search.csv \
  --require-generated \
  --require-flat-start \
  --focus-first-c2-window 1 \
  --namespace-cols parent_domain_id,fiber_label \
  --out results/critical_birth.csv

echo "[3/5] Frozen-transition multi-observer consensus"
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
  --out results/multi_observer_consensus.csv

echo "[4/5] Observation-method scale test"
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
  --min-chart-entropy-list 0,0.025,0.05,0.1 \
  --observer-runs 16 \
  --out results/observation_method_scale.csv

echo "[5/5] Lift mode comparison"
for MODE in bijective permutation conservative information_conserving representative; do
  python -m relgauge.iteratedfiberatlasdynamicsaudit 2 \
    --vertices 9 \
    --rule-modes random_full_permutation \
    --instances 29 \
    --atlas-capacities 32 \
    --profiles full_atlas \
    --atlas-iterations 12 \
    --seed 2000006 \
    --atlas-lift-mode "$MODE" \
    --out "results/lift_mode_${MODE}.csv"
done

echo "Core positive-result reproduction chain complete."
