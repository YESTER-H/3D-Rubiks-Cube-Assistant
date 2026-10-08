# Applying the Portfolio Files

The additions were prepared against original commit `1525488e1bb9db244bf51927bad663dd533cf952`. The three original application Python files are unchanged.

If you received these materials as a ZIP:

1. Extract the archive.
2. Copy `README.md`, `requirements*.txt`, `pytest.ini`, `assets/`, `docs/`, `results/`, `tests/`, and `tools/` into your working repository.
3. Add `.pytest_cache/` to `.gitignore`.
4. Run the validation command below and review the README and synthetic demo.
5. Review, commit, and upload the changes with your Git client. Preserve and merge any local changes you made separately.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The ZIP excludes `.git`, virtual environments, caches, and credentials. Results under `results/` describe the recorded execution environment. Rerunning the benchmark updates the existing result files; archive them first if you want to preserve previous runs.

See [DEMO.md](DEMO.md) for adding real-camera recordings and recognition measurements.
