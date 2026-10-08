# Automated Validation

Run on 2026-10-08 using Linux/Python 3.12.14: **46 passed, 0 failed, 0 skipped**.
Command: `python -m pytest -q --junitxml=results/pytest.xml`.
See the [JUnit XML](pytest.xml) for individual cases and exact execution times.

| Check | Cases | Result |
|---|---:|---|
| Solved-state validation and empty solution | 1 | Passed |
| Legal scrambles, geometric-engine agreement, and solution replay | 20 | Passed |
| Single corner twist, single edge flip, and parity mismatch | 3 | Rejection verified |
| Turn/inverse and four-turn recovery for each face | 6 | Passed |
| Capture rotation normalization, external validation, and replay | 6 | Passed |
| Color counts, unknown labels, duplicate capture, and undo | 1 | Passed |
| Synthetic HSV color-patch classification | 8 | Passed |
| Incomplete solver-input rejection | 1 | Passed |

## Scope

The tests call the project's actual classification, validation, and move functions. No GUI windows or webcams are opened. Legal states are generated with Kociemba's cubie model and compared with the project's coordinate-based rotation implementation.

The fixtures use the internal `kociemba.pykociemba` interface, so that package is pinned to version 1.2.1.

Search performance is measured separately in the [benchmark report](BENCHMARK.md). Real webcam recognition, OpenGL rendering, and Windows device compatibility were not verified in this environment. Passing these 46 cases is regression evidence, not a correctness proof for all inputs.
