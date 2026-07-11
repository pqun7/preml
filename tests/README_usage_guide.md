# Usage Guide Tests

Run the Usage Guide validation tests with:

```powershell
pytest tests/test_usage_guide_execution.py -v
```

Run the full guide execution validation:

This can take a few minutes because it validates every runnable example.

```powershell
$env:RUN_FULL_USAGE_GUIDE_VALIDATION="1"
pytest tests/test_usage_guide_execution.py -vv -s
```

Optional tuning:

- `USAGE_GUIDE_SUBSET_BLOCKS` controls subset size.
- `USAGE_GUIDE_SUBSET_WORKERS` controls parallel workers.
- `USAGE_GUIDE_SUBSET_TIMEOUT` controls per-block timeout.
- `FULL_VALIDATION_WORKERS` controls full validation workers.
- `FULL_VALIDATION_TIMEOUT` controls full validation timeout.
- `CLEAN_OUTPUT=1` forces a clean validation run.
