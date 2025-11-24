Part 1 Solution (Note that both the report and the code are expected to be revised for the final submission.)

### File Overview

- **`common.py`**
- **`config.py`**
- **`basic_model.py`**
- **`risky_debt_model.py`**
- **`tests/`**
  - `test_common.py`: common utilities
  - `test_basic_model_unit.py`: basic model unit tests
  - `test_basic_model_integration.py`: basic model integration tests (`@pytest.mark.slow`)
  - `test_risky_debt_unit.py`: risky-debt model unit tests
  - `test_risky_debt_integration.py`: risky-debt model integration tests (`@pytest.mark.slow`)



## Running Tests

Run all tests:

```bash
python -m pytest -q
```

Example output:

```text
.......................
================================================================================ warnings summary =================================================================================
tests/test_basic_model_integration.py:9
  ... PytestUnknownMarkWarning: Unknown pytest.mark.slow ...
tests/test_risky_debt_integration.py:9
  ... PytestUnknownMarkWarning: Unknown pytest.mark.slow ...

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
23 passed, 2 warnings in 8.52s
```


## Running the Models

Run each model directly:

```bash
python basic_model.py
python risky_debt_model.py
```
