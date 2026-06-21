from __future__ import annotations
import glob, json, pathlib, sys

paths = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/*basepoint*_summary.json"))
print(f"Found {len(paths)} summary files")
found = []
for p in paths:
    try:
        s = json.loads(pathlib.Path(p).read_text())
    except Exception as e:
        print("BAD", p, e)
        continue
    ok = bool(s.get("any_nonabelian_basepoint_isotropy") or s.get("any_exact_s3_basepoint_isotropy"))
    print(f"{p}: nonabelian={s.get('any_nonabelian_basepoint_isotropy')} s3={s.get('any_exact_s3_basepoint_isotropy')} max_order={s.get('max_group_order') or s.get('max_basepoint_group_order')}")
    if ok:
        found.append((p, s))
print("\nFOUND", len(found), "positive summaries")
for p, s in found:
    print("\n===", p, "===")
    print(json.dumps(s, indent=2, sort_keys=True)[:12000])
