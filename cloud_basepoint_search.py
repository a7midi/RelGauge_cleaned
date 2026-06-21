"""
Cloud basepoint-isotropy search for SIQ/RelGauge on Modal.

Run from a project directory that contains a local `relgauge/` Python package:

    pip install modal
    python -m modal setup
    modal run cloud_basepoint_search.py --vertices 10 --total-instances 200 --batch-size 10

This script does REAL computation in each cloud worker:
  1. runs relgauge.iteratedfiberatlasdynamicsaudit for a batch of candidate instances;
  2. runs relgauge.basepointawareholonomyaudit on the generated iterated CSV;
  3. returns summaries and writes aggregate local JSON/CSV files.

It does not read precomputed JSON summaries.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import textwrap
import time
from typing import Any, Dict, List

import modal

APP_NAME = "siq-basepoint-search"
REMOTE_ROOT = pathlib.Path("/root")
REMOTE_RESULTS = pathlib.Path("/tmp/siq_basepoint_results")

# Modal 1.x: add local source files through the Image, not modal.Mount.
# This expects a local directory named `relgauge` next to this script / project root.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "matplotlib")
    .add_local_dir("relgauge", remote_path="/root/relgauge")
)

app = modal.App(APP_NAME, image=image)


def _run(cmd: List[str], cwd: str | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    env["MPLBACKEND"] = "Agg"
    print("RUN:", " ".join(cmd), flush=True)
    cp = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    print(cp.stdout[-8000:], flush=True)
    if cp.returncode != 0:
        raise RuntimeError(f"command failed with return code {cp.returncode}: {' '.join(cmd)}\n{cp.stdout}")
    return cp


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _csv_head(path: pathlib.Path, max_rows: int = 25) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr):
            if i >= max_rows:
                break
            rows.append(dict(row))
    return rows


@app.function(timeout=18000, cpu=2)
def run_batch(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one candidate batch in a Modal worker."""
    vertices = int(cfg.get("vertices", 10))
    q = int(cfg.get("q", 2))
    batch_id = int(cfg.get("batch_id", 0))
    start_instance = int(cfg.get("start_instance", 0))
    instances = int(cfg.get("instances", 10))
    # Preserve the internal seed sequence across batches.  iteratedfiberatlasdynamicsaudit uses:
    # init_seed = seed + 1000003*(mode_index+1) + 7919*instance.
    seed_base = int(cfg.get("seed_base", 2000006)) + 7919 * start_instance

    rule_modes = str(cfg.get("rule_modes", "random_full_permutation"))
    profiles = str(cfg.get("profiles", "full_atlas"))
    atlas_capacities = str(cfg.get("atlas_capacities", "32,64,128"))
    atlas_iterations = int(cfg.get("atlas_iterations", 12))
    max_state_samples = int(cfg.get("max_state_samples", 1024 if vertices <= 10 else 2048))
    max_total_states = int(cfg.get("max_total_states", 200000))
    max_candidates = int(cfg.get("max_candidates", 0))
    max_charts_per_fiber = int(cfg.get("max_charts_per_fiber", 24))
    max_signature_charts = int(cfg.get("max_signature_charts", 64))
    max_cycle_len = int(cfg.get("max_cycle_len", 5))
    max_cycles_per_fiber = int(cfg.get("max_cycles_per_fiber", 1000))
    max_loops_per_base = int(cfg.get("max_loops_per_base", 5000))
    min_overlap_states = int(cfg.get("min_overlap_states", 3))
    min_support_states = int(cfg.get("min_support_states", 3))
    min_fiber_states = int(cfg.get("min_fiber_states", 6))
    max_domains_scan = int(cfg.get("max_domains_scan", 0))
    max_fibers_per_domain_scan = int(cfg.get("max_fibers_per_domain_scan", 0))

    work = REMOTE_RESULTS / f"v{vertices}_batch{batch_id:04d}_start{start_instance}_n{instances}"
    work.mkdir(parents=True, exist_ok=True)
    iter_csv = work / f"iterated_v{vertices}_batch{batch_id:04d}.csv"
    iso_csv = work / f"basepoint_v{vertices}_batch{batch_id:04d}.csv"

    t0 = time.time()
    iter_cmd = [
        "python", "-m", "relgauge.iteratedfiberatlasdynamicsaudit", str(q),
        "--vertices", str(vertices),
        "--rule-modes", rule_modes,
        "--instances", str(instances),
        "--atlas-capacities", atlas_capacities,
        "--profiles", profiles,
        "--atlas-iterations", str(atlas_iterations),
        "--proliferation-iterations", str(cfg.get("proliferation_iterations", 4)),
        "--horizon", str(cfg.get("horizon", 3)),
        "--max-state-samples", str(max_state_samples),
        "--max-total-states", str(max_total_states),
        "--max-charts-per-fiber", str(max_charts_per_fiber),
        "--max-signature-charts", str(max_signature_charts),
        "--max-cycle-len", str(max_cycle_len),
        "--max-cycles-per-fiber", str(max_cycles_per_fiber),
        "--min-overlap-states", str(min_overlap_states),
        "--min-support-states", str(min_support_states),
        "--min-fiber-states", str(min_fiber_states),
        "--seed", str(seed_base),
        "--atlas-lift-mode", str(cfg.get("atlas_lift_mode", "bijective")),
        "--out", str(iter_csv),
        "--plot", "",
    ]
    _run(iter_cmd, timeout=int(cfg.get("iterated_timeout", 14400)))

    iso_cmd = [
        "python", "-m", "relgauge.basepointawareholonomyaudit", str(q),
        "--vertices", str(vertices),
        "--iterated-csv", str(iter_csv),
        "--rule-modes", rule_modes,
        "--profiles", profiles,
        "--atlas-capacities", atlas_capacities,
        "--atlas-iterations", str(atlas_iterations),
        "--max-candidates", str(max_candidates),
        "--max-state-samples", str(max_state_samples),
        "--max-total-states", str(max_total_states),
        "--max-charts-per-fiber", str(max_charts_per_fiber),
        "--max-signature-charts", str(max_signature_charts),
        "--max-cycle-len", str(max_cycle_len),
        "--max-cycles-per-fiber", str(max_cycles_per_fiber),
        "--max-loops-per-base", str(max_loops_per_base),
        "--min-overlap-states", str(min_overlap_states),
        "--min-support-states", str(min_support_states),
        "--min-fiber-states", str(min_fiber_states),
        "--max-domains-scan", str(max_domains_scan),
        "--max-fibers-per-domain-scan", str(max_fibers_per_domain_scan),
        "--out", str(iso_csv),
        "--plot", "",
    ]
    if bool(cfg.get("stop_at_first_nonabelian", True)):
        iso_cmd.insert(-4, "--stop-at-first-nonabelian")
    _run(iso_cmd, timeout=int(cfg.get("isotropy_timeout", 14400)))

    summary_path = pathlib.Path(str(iso_csv).replace(".csv", "_summary.json"))
    summary = _load_json(summary_path)
    found = bool(summary.get("any_nonabelian_basepoint_isotropy") or summary.get("any_exact_s3_basepoint_isotropy"))
    result: Dict[str, Any] = {
        "vertices": vertices,
        "batch_id": batch_id,
        "start_instance": start_instance,
        "instances": instances,
        "seed_base_used": seed_base,
        "elapsed_sec": round(time.time() - t0, 3),
        "found_nonabelian": found,
        "summary": summary,
        "top_rows": _csv_head(iso_csv, max_rows=50 if found else 5),
        "work_dir": str(work),
    }
    return result


@app.local_entrypoint()
def main(
    vertices: int = 10,
    total_instances: int = 200,
    batch_size: int = 10,
    seed_base: int = 2000006,
    rule_modes: str = "random_full_permutation",
    profiles: str = "full_atlas",
    atlas_capacities: str = "32,64,128",
    atlas_iterations: int = 12,
    max_state_samples: int = 0,
    max_charts_per_fiber: int = 24,
    max_cycle_len: int = 5,
    min_overlap_states: int = 3,
    min_support_states: int = 3,
    min_fiber_states: int = 6,
    max_domains_scan: int = 0,
    max_fibers_per_domain_scan: int = 0,
    stop_at_first_nonabelian: bool = True,
):
    """Fan out batches and stop/report if a nonabelian basepoint isotropy is found."""
    if max_state_samples <= 0:
        max_state_samples = 1024 if vertices <= 10 else (2048 if vertices == 11 else 4096)

    batches = []
    batch_id = 0
    for start in range(0, int(total_instances), int(batch_size)):
        n = min(int(batch_size), int(total_instances) - start)
        cfg = dict(
            vertices=vertices,
            batch_id=batch_id,
            start_instance=start,
            instances=n,
            seed_base=seed_base,
            rule_modes=rule_modes,
            profiles=profiles,
            atlas_capacities=atlas_capacities,
            atlas_iterations=atlas_iterations,
            max_state_samples=max_state_samples,
            max_charts_per_fiber=max_charts_per_fiber,
            max_cycle_len=max_cycle_len,
            min_overlap_states=min_overlap_states,
            min_support_states=min_support_states,
            min_fiber_states=min_fiber_states,
            max_domains_scan=max_domains_scan,
            max_fibers_per_domain_scan=max_fibers_per_domain_scan,
            stop_at_first_nonabelian=stop_at_first_nonabelian,
        )
        batches.append(run_batch.spawn(cfg))
        batch_id += 1

    outdir = pathlib.Path("cloud_results") / f"v{vertices}_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []
    found_results: List[Dict[str, Any]] = []
    for i, fut in enumerate(batches, start=1):
        print(f"Waiting for batch {i}/{len(batches)}...", flush=True)
        res = fut.get()
        all_results.append(res)
        (outdir / f"batch_{res['batch_id']:04d}.json").write_text(json.dumps(res, indent=2, sort_keys=True))
        s = res.get("summary", {})
        print(
            f"batch={res['batch_id']} start={res['start_instance']} found={res['found_nonabelian']} "
            f"max_order={s.get('max_group_order') or s.get('max_basepoint_group_order')} "
            f"nonab={s.get('any_nonabelian_basepoint_isotropy')} s3={s.get('any_exact_s3_basepoint_isotropy')}",
            flush=True,
        )
        if res.get("found_nonabelian"):
            found_results.append(res)
            print("\nFOUND NONABELIAN BASEPOINT ISOTROPY")
            print(json.dumps(res, indent=2, sort_keys=True)[:20000])
            # We do not cancel existing workers here; Modal jobs will finish.  Stop after collection if desired.

    (outdir / "all_results.json").write_text(json.dumps(all_results, indent=2, sort_keys=True))
    (outdir / "found_results.json").write_text(json.dumps(found_results, indent=2, sort_keys=True))
    print(f"\nWrote local results to: {outdir}")
    if found_results:
        print(f"FOUND {len(found_results)} batch(es) with nonabelian isotropy.")
    else:
        print("No nonabelian basepoint isotropy found in this run.")
