# Synthetic-State Benchmark

- Run time (UTC): 2026-10-08T13:05:56.630528+00:00
- Environment: Linux-6.18.44-x86_64-with-glibc2.39; Python 3.12.14
- Seed: 20261008; 20-move scrambles; median of 3 runs per case
- Original baseline commit: `1525488e1bb9db244bf51927bad663dd533cf952`

| Metric | Result |
|---|---:|
| Valid candidate found from rotated legal inputs | 20/20 |
| Exact match to the original state | 20/20 |
| Impossible states rejected | 3/3 |
| Orientation-search time for legal inputs: median of case medians | 14.229 ms |
| Total orientation candidates examined | 47423 |
| Full invariant checks after early filtering | 1620 |
| Checks if every candidate received full validation (calculated baseline) | 47423 |
| Full invariant checks avoided by early filtering | 96.58% |
| Real webcam recognition accuracy | Not measured |
| Actual GUI/OpenGL playback | Not verified in this environment |

These results describe a small synthetic regression experiment, not product accuracy or a guarantee for all inputs.
Runtime without the early filter was not measured. The reduction in check counts is not a measured speedup ratio.
Finding the first valid candidate does not establish unique recovery of a photographed state.
See [benchmark.csv](benchmark.csv) for individual cases and [environment.json](environment.json) for the environment and dependencies.

## Reproduce

```bash
python -m pytest -q --junitxml=results/pytest.xml
python -m tools.benchmark --samples 20 --repeats 3 --seed 20261008
```
