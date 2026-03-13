# Testing Patterns

**Analysis Date:** 2026-03-13

## Test Framework

**Runner:**
- pytest 8.1.1+
- Config: `pyproject.toml` in `[tool.pytest.ini_options]` section
- Key settings:
  - `--strict-markers` - enforce marker registration
  - `--color=yes` - colored output
  - `--disable-pytest-warnings` - suppress warnings

**Assertion Library:**
- Standard pytest assertions: `assert` statements
- Custom matchers via fixture classes for tensor/float comparisons

**Run Commands:**
```bash
pytest tests/                          # Run all tests
pytest tests/test_model.py             # Run specific test file
pytest tests/test_model.py::test_name  # Run specific test
pytest -v tests/                       # Verbose output
pytest --tb=short tests/               # Shorter traceback format
pytest -k "test_config"                # Run tests matching pattern
```

## Test File Organization

**Location:**
- Colocated with source: `tests/` directory mirrors `litgpt/` structure
- Example: `litgpt/adapter.py` → `tests/test_adapter.py`
- Subpackages: `litgpt/generate/` → `tests/generate/test_*.py`

**Naming:**
- Files: `test_*.py` (e.g., `test_model.py`, `test_adapter.py`)
- Functions: `test_*` (e.g., `test_config()`, `test_adapter_filter()`)
- Test organization: logical grouping within file

**Structure:**
```
tests/
├── conftest.py              # Shared fixtures
├── test_config.py           # Config module tests
├── test_model.py            # Model tests
├── test_adapter.py          # Adapter tests
├── test_full.py             # Full training tests
├── generate/                # Generation submodule tests
│   ├── __init__.py
│   ├── test_adapter.py
│   ├── test_main.py
│   └── utils.py            # Shared test utilities
├── convert/                 # Conversion tests
├── ext_thunder/             # Thunder compiler tests
└── data/                    # Test fixtures
    └── _fixtures/
        ├── alpaca.json
        ├── dolly.json
        └── longform_*.json
```

## Test Structure

**Suite Organization:**
```python
# Imports organized: standard, third-party, local
import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
import yaml

import litgpt.finetune.full as module
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca


# Test functions (no class wrapping typical)
def test_config():
    config = Config()
    assert config.name == ""
    assert config.block_size == 4096


@pytest.mark.parametrize("config", config_module.configs, ids=[c["name"] for c in config_module.configs])
def test_short_and_hf_names_are_equal_unless_on_purpose(config):
    # Implementation
    pass
```

**Patterns:**
- Individual test functions, not classes (though class-based tests also used)
- Fixtures injected as parameters
- Setup in fixture functions, not setUp methods
- Teardown via yield in fixtures

## Mocking

**Framework:** `unittest.mock` and `unittest.mock.Mock`

**Patterns:**
```python
from unittest import mock
from unittest.mock import Mock

# Patch module-level imports
@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_full_script(tmp_path, fake_checkpoint_dir, monkeypatch, alpaca_path):
    # Create mock
    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **__: torch.tensor([3, 2, 1])

    # Patch at module level
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)
    monkeypatch.setattr(module, "load_checkpoint", Mock())

    # Use in test
    module.setup(...)


# Patch sys.argv
with mock.patch("sys.argv", ["full.py", str(fake_checkpoint_dir)]):
    module.setup(...)


# Using Mock to check calls
fabric.save(save_path, {"model": model}, filter={"model": adapter_filter})
```

**What to Mock:**
- External dependencies (tokenizers, model loaders)
- Heavy computations (set as Mock with return_value)
- File I/O and environment variables
- PyTorch model checkpoints

**What NOT to Mock:**
- Core model architecture (test actual forward passes)
- Config objects (use real configs with small dimensions)
- Torch operations (test tensor logic with real ops)

## Fixtures and Factories

**Test Data:**
```python
# Fixture classes for flexible matching
class TensorLike:
    def __eq__(self, other):
        return isinstance(other, torch.Tensor)

class FloatLike:
    def __eq__(self, other):
        return not isinstance(other, int) and isinstance(other, float)


# Fixtures providing test data
@pytest.fixture()
def fake_checkpoint_dir(tmp_path):
    os.chdir(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints" / "tmp"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "lit_model.pth").touch()
    (checkpoint_dir / "model_config.yaml").touch()
    (checkpoint_dir / "tokenizer.json").touch()
    (checkpoint_dir / "tokenizer_config.json").touch()
    return checkpoint_dir


@pytest.fixture()
def mock_tokenizer():
    return MockTokenizer()


# Mock tokenizer implementation
class MockTokenizer:
    """A dummy tokenizer that encodes each character as its ASCII code."""

    bos_id = 0
    eos_id = 1

    def encode(self, text: str, bos: Optional[bool] = None, eos: bool = False, max_length: int = -1) -> torch.Tensor:
        output = []
        if bos:
            output.append(self.bos_id)
        output.extend([ord(c) for c in text])
        if eos:
            output.append(self.eos_id)
        output = output[:max_length] if max_length > 0 else output
        return torch.tensor(output)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(chr(int(t)) for t in tokens.tolist())
```

**Location:**
- `tests/conftest.py` - Global shared fixtures
- Fixture-specific utilities: `tests/generate/utils.py`
- Data files: `tests/data/_fixtures/` for JSON/test data

## Coverage

**Requirements:** No explicit coverage requirements enforced

**View Coverage:**
```bash
pytest --cov=litgpt tests/              # Generate coverage report
pytest --cov=litgpt --cov-report=html tests/  # HTML report
```

## Test Types

**Unit Tests:**
- Scope: Individual functions/classes
- Approach: Small, focused, fast
- Examples: `test_config()`, `test_adapter_filter()`, `test_rope_*()`
- Location: Directly in test_module.py

**Integration Tests:**
- Scope: Multiple components working together
- Approach: End-to-end workflows
- Examples: `test_full_script()`, `test_adapter_script()` - full training loop
- Setup: Use real configs with small dimensions, mocked data
- Pattern: Create model, load checkpoint, train for N steps, validate output

**E2E Tests:**
- Framework: Not formal pytest-based, but integration tests serve this purpose
- Examples: Full training workflows that write checkpoints and validate output directory structure
- Validation: Check files created, log output contains expected strings

## Test Markers and Selection

**Built-in Markers Used:**
- `@pytest.mark.parametrize` - Run test with multiple parameter combinations
- `@pytest.mark.skipif` - Skip based on conditions (environment variables)
- `@pytest.mark.xfail` - Expected to fail (known issues)

**Custom Markers:**
From `utils.py`:
```python
class _RunIf:
    """Decorator to conditionally skip tests based on environment."""
    min_cuda_gpus: int = 0
    standalone: bool = False

# Usage
@pytest.mark.skipif(condition, reason="message")
@_RunIf(min_cuda_gpus=1)
def test_something():
    pass
```

**Environment-Based Selection:**
- `PL_RUN_STANDALONE_TESTS=1` - Run standalone tests (GPU tests typically)
- `RUN_ONLY_CUDA_TESTS=1` - Only run CUDA-enabled tests
- Tests filtered by `pytest_collection_modifyitems` in conftest.py

## Common Patterns

**Async Testing:**
Not applicable - tests are synchronous, though they may invoke distributed training setups

**Error Testing:**
```python
# Using pytest.raises context manager
def test_nonexisting_name():
    with pytest.raises(ValueError, match="'invalid-model-name' is not a supported config name"):
        Config.from_name("invalid-model-name")


def test_from_hf_name_with_org_string():
    with pytest.raises(ValueError, match="'UnknownOrg/...' is not a supported config name"):
        Config.from_name("UnknownOrg/TinyLlama-1.1B-...")
```

**Parametrized Tests:**
```python
@pytest.mark.parametrize("config", config_module.configs, ids=[c["name"] for c in config_module.configs])
def test_applies_to_all_configs(config):
    # Test runs once per config in config_module.configs
    pass

# Multiple parameters
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[pytest.mark.xfail(raises=AssertionError, strict=False), _RunIf(min_cuda_gpus=1)],
        ),
    ],
)
def test_against_reference(device, dtype):
    pass
```

**Setup/Teardown:**
```python
@pytest.fixture(autouse=True)
def restore_default_dtype():
    # Setup (implicit)
    yield  # Test runs here
    # Teardown
    torch.set_default_dtype(torch.float32)


@pytest.fixture(autouse=True)
def destroy_process_group():
    yield

    import torch.distributed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
```

**Capture Output:**
```python
from io import StringIO
from contextlib import redirect_stdout

def test_full_script(tmp_path):
    stdout = StringIO()
    with redirect_stdout(stdout):
        module.setup(...)

    logs = stdout.getvalue()
    assert "expected text" in logs
    assert logs.count("pattern") == 6
```

**File System Testing:**
```python
def test_checkpoint_creation(tmp_path):
    out_dir = tmp_path / "out"
    module.setup(..., out_dir=out_dir)

    # Check directory structure created
    out_dir_contents = set(os.listdir(out_dir))
    checkpoint_dirs = {"step-000002", "step-000004", "step-000006", "final"}
    assert checkpoint_dirs.issubset(out_dir_contents)

    # Verify files exist in each checkpoint
    for checkpoint_dir in checkpoint_dirs:
        assert set(os.listdir(out_dir / checkpoint_dir)) == {
            "lit_model.pth",
            "model_config.yaml",
            "tokenizer_config.json",
            "tokenizer.json",
            "hyperparameters.yaml",
            "prompt_style.yaml",
        }
```

---

*Testing analysis: 2026-03-13*
