#!/usr/bin/env bash
set -euo pipefail
python -c "from relgauge.iteratedfiberatlasdynamicsaudit import run_iterated_fiber_atlas_dynamics_audit; print('OK iterated')"
python -c "from relgauge.multiobserverconsensusaudit import run_multiobserver_consensus_audit; print('OK multiobserver')"
python -c "from relgauge.observationmethodscaletest import run_observation_method_scale_test; print('OK observation_scale')"
python -c "from relgauge.perturbationstabilityaudit import run_perturbation_stability_audit; print('OK perturbation')"
