"""Run from repository root: python -m tools.benchmark."""
import argparse
import contextlib
import csv
import io
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

import cube_assistant as app
from cube_solver import captured_to_facelets_URFDLB, solve_moves
from kociemba.pykociemba.facecube import FaceCube
from tools.cube_cases import impossible_cases, rotated_capture, scrambled


def measured_search(captured):
    """Instrument the production algorithm without changing its filter order."""
    faces = ['U', 'D', 'F', 'B', 'L', 'R']
    examined = full_checks = 0
    start = perf_counter()
    for rotations in product(range(4), repeat=6):
        examined += 1
        candidate = {f: app.rotate_face_n_times(captured[f], n)
                     for f, n in zip(faces, rotations)}
        if not app._early_filter_impossible_combos(candidate)[0]:
            continue
        full_checks += 1
        if app.is_valid_cube_silent(candidate):
            return candidate, examined, full_checks, perf_counter() - start
    return None, examined, full_checks, perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error('samples and repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for i in range(args.samples):
        state, _ = scrambled(args.seed + i)
        cases.append((f'legal_{i:03}', True, state,
                      rotated_capture(state, args.seed + 1000 + i)))
    for name, state in impossible_cases().items():
        cases.append((name, False, state, state))
    rows = []
    for name, expected, original, capture in cases:
        timings = []
        with contextlib.redirect_stdout(io.StringIO()):
            production, _ = app.try_all_orientations(capture)
        for _ in range(args.repeats):
            normalized, examined, checks, elapsed = measured_search(capture)
            timings.append(elapsed * 1000)
            assert normalized == production, 'Instrumented/production mismatch'
        found = normalized is not None
        assert found == expected, f'Unexpected accept/reject: {name}'
        solver_ok = False
        if found:
            assert FaceCube(captured_to_facelets_URFDLB(normalized)).toCubieCube().verify() == 0
            solve_moves(normalized)
            solver_ok = True
        rows.append(dict(case=name, expected_valid=expected, found_valid=found,
                         matches_original=normalized == original,
                         solver_accepted=solver_ok, candidates=examined,
                         full_checks=checks, full_checks_without_early_filter=examined,
                         median_ms=round(median(timings), 3)))
    with (args.output / 'benchmark.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    platform=platform.platform(), python=platform.python_version(),
                    seed=args.seed, repeats=args.repeats, legal_samples=args.samples,
                    packages={p: version(p) for p in ['numpy','opencv-python','Pillow','glfw','PyOpenGL','kociemba','pytest']},
                    limitations=['Synthetic states; no webcam accuracy or GUI timing.',
                                 'Without-early-filter counts are a calculated baseline, not measured runtime.',
                                 'Recovery of a valid candidate need not uniquely recover the photographed cube.'])
    (args.output / 'environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    legal = [r for r in rows if r['expected_valid']]
    invalid = [r for r in rows if not r['expected_valid']]
    total = sum(r['candidates'] for r in rows)
    full = sum(r['full_checks'] for r in rows)
    text = f'''# Synthetic-State Benchmark

- Run time (UTC): {metadata['timestamp_utc']}
- Environment: {metadata['platform']}; Python {metadata['python']}
- Seed: {args.seed}; 20-move scrambles; median of {args.repeats} runs per case
- Original baseline commit: `1525488e1bb9db244bf51927bad663dd533cf952`

| Metric | Result |
|---|---:|
| Valid candidate found from rotated legal inputs | {sum(r['found_valid'] for r in legal)}/{len(legal)} |
| Exact match to the original state | {sum(r['matches_original'] for r in legal)}/{len(legal)} |
| Impossible states rejected | {sum(not r['found_valid'] for r in invalid)}/{len(invalid)} |
| Orientation-search time for legal inputs: median of case medians | {median(r['median_ms'] for r in legal):.3f} ms |
| Total orientation candidates examined | {total} |
| Full invariant checks after early filtering | {full} |
| Checks if every candidate received full validation (calculated baseline) | {total} |
| Full invariant checks avoided by early filtering | {(1-full/total)*100:.2f}% |
| Real webcam recognition accuracy | Not measured |
| Actual GUI/OpenGL playback | Not verified in this environment |

These results describe a small synthetic regression experiment, not product accuracy or a guarantee for all inputs.
Runtime without the early filter was not measured. The reduction in check counts is not a measured speedup ratio.
Finding the first valid candidate does not establish unique recovery of a photographed state.
See [benchmark.csv](benchmark.csv) for individual cases and [environment.json](environment.json) for the environment and dependencies.

## Reproduce

```bash
python -m pytest -q --junitxml=results/pytest.xml
python -m tools.benchmark --samples {args.samples} --repeats {args.repeats} --seed {args.seed}
```
'''
    (args.output / 'BENCHMARK.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
