# Environment and runtimes

Validated environment used for this artifact build:

- Python 3.13.5
- numpy 2.3.5
- pandas 2.2.3
- matplotlib 3.10.8

Pinned package versions are in `requirements.txt` and `requirements-lock.txt`.
A conda-style environment file is provided as `environment.yml`.

Approximate runtimes vary by CPU and disk speed:

- `bash test_imports.sh`: seconds.
- `python -m pytest -q`: seconds to under a minute.
- `bash reproduce_quick.sh`: usually roughly 10-30 minutes. This recomputes the 64-run frozen-transition multi-observer and 48-configuration observation-method scale tests from `data/transition_iter8_inst28_seed3221741.npy`.
- `bash reproduce.sh`: roughly 30-90 minutes on a contemporary laptop/workstation. This regenerates witness CSVs and chart cycles before running the core positive-result chain.

The package does not rerun every exploratory negative-control audit from the research log. It is scoped to the central witness and consensus computations.
