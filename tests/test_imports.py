def test_imports():
    from relgauge.iteratedfiberatlasdynamicsaudit import run_iterated_fiber_atlas_dynamics_audit
    from relgauge.multiobserverconsensusaudit import run_multiobserver_consensus_audit
    from relgauge.observationmethodscaletest import run_observation_method_scale_test
    from relgauge.perturbationstabilityaudit import run_perturbation_stability_audit
    assert callable(run_iterated_fiber_atlas_dynamics_audit)
    assert callable(run_multiobserver_consensus_audit)
    assert callable(run_observation_method_scale_test)
    assert callable(run_perturbation_stability_audit)
