# SIQ / RelGauge real computational package

This package is the **actual finite-atlas computation package** used for the paper's core experiments. It includes the minimal `relgauge` import dependency chain needed to run the paper commands, not JSON-summary wrappers.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # pinned versions
```

Optional editable install:

```bash
pip install -e .
```

## Smoke/import test

```bash
bash test_imports.sh
```

This verifies that the actual experiment entry points import:

```python
from relgauge.iteratedfiberatlasdynamicsaudit import run_iterated_fiber_atlas_dynamics_audit
from relgauge.multiobserverconsensusaudit import run_multiobserver_consensus_audit
from relgauge.observationmethodscaletest import run_observation_method_scale_test
from relgauge.perturbationstabilityaudit import run_perturbation_stability_audit
```

## Quick reproduction: fresh computation from the frozen transition

```bash
bash reproduce_quick.sh
```

This runs the real `relgauge.multiobserverconsensusaudit` and `relgauge.observationmethodscaletest` modules against `data/transition_iter8_inst28_seed3221741.npy`. It does not read precomputed JSON summaries. It should reproduce the target/support-level S3 consensus results from the frozen transition.

## Core positive-result reproduction chain

```bash
bash reproduce.sh
```

This runs the actual core positive-result commands: witness generation, regenerated-transition comparison, critical birth-window S3 detection, multi-observer consensus, observation-method scale, and lift-mode comparison. It does not rerun the entire exploratory/negative-control audit ecosystem from the paper; those larger controls are documented in the manuscript and archived result artifacts. The first step can take tens of minutes and writes large CSVs.

## Included computational modules

Only the import dependency chain required for the paper is included:

- `boundaryproliferationaudit.py`
- `dynamicsconsistencyfixedpointaudit.py`
- `fiberatlasfixedpointaudit.py`
- `fiberchartconnectionaudit.py`
- `generatedcandidatephysicsreplayaudit.py`
- `binarycompositegaugespectrumaudit.py`
- `iteratedfiberatlasdynamicsaudit.py`
- `criticalc2birthwindowaudit.py`
- `multiobserverconsensusaudit.py`
- `observationmethodscaletest.py`
- `perturbationstabilityaudit.py`
- dependency modules: `observerboundarygeometry.py`, `exactclosureholonomyaudit.py`

Everything else from the exploratory `relgauge` tree is intentionally excluded.

## Data

- `data/transition_iter8_inst28_seed3221741.npy`: frozen 512-entry transition table for the verified S3 event.
- `data/reference_support.json`: metadata for the 17-microstate support.
- `data/repro_witness_search_minimal.csv`: one candidate metadata row used by the quick frozen-transition tests.

## Reproducibility note

The quick path computes the multi-observer and observation-method scale results freshly from the frozen transition. The full path regenerates the witness and compares the regenerated transition table against the bundled frozen transition.


## Environment and expected runtimes

Validated environment:

```text
Python 3.13.5
numpy 2.3.5
pandas 2.2.3
matplotlib 3.10.8
```

The pinned Python package versions are in `requirements.txt` and `requirements-lock.txt`; a conda-style file is provided as `environment.yml`.

Approximate runtime on a contemporary laptop/workstation:

- `bash test_imports.sh`: seconds.
- `bash reproduce_quick.sh`: roughly 10-30 minutes; it recomputes the frozen-transition multi-observer and observation-method scale results from `data/transition_iter8_inst28_seed3221741.npy`.
- `bash reproduce.sh`: roughly 30-90 minutes depending on CPU and disk speed; it regenerates witness CSVs and chart cycles and then runs the core positive-result chain.

## Scope of this package

This repository is the real computational package for the central witness and consensus tests. It intentionally includes only the dependency chain required for the paper's core positive result. It is not a complete archive of every exploratory audit used during development. In particular, broad negative controls such as certified C3/C2 interfaces, paired returns, moving carriers, endpoint/schedule-cut crossing phases, and large v=10 scaling controls are reported in the manuscript from their archived outputs, but they are not all rerun by `reproduce.sh`.
