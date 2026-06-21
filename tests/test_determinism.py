from pathlib import Path
from relgauge.multiobserverconsensusaudit import _stable_hash


def test_stable_hash_reproducible():
    assert _stable_hash('full_atlas') == _stable_hash('full_atlas')
    assert _stable_hash('full_atlas') != _stable_hash('fiber_preserving')


def test_no_process_randomized_hash_seed_paths():
    root = Path('relgauge')
    bad = []
    for path in root.glob('*.py'):
        text = path.read_text()
        for token in ('abs(hash(', 'hash(str(profile))', 'python_hash'):
            if token in text:
                bad.append((str(path), token))
    assert not bad, f'Process-randomized hash seed paths found: {bad}'
