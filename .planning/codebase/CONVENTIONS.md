# Coding Conventions

**Analysis Date:** 2026-03-13

## Naming Patterns

**Files:**
- Lowercase with underscores: `model.py`, `tokenizer.py`, `utils.py`
- Test files: `test_*.py` (e.g., `test_model.py`, `test_adapter.py`)
- Module directories: lowercase (e.g., `generate/`, `finetune/`, `data/`)

**Functions:**
- snake_case for function names: `init_out_dir()`, `find_resume_path()`, `reset_parameters()`
- Private functions: underscore prefix `_init_weights()`, `_load_from_state_dict()`
- Filter functions use verb pattern: `adapter_filter()`, `num_parameters()`

**Variables:**
- snake_case for variables: `max_seq_length`, `input_pos`, `lm_head`, `out_dir`
- Configuration parameters: snake_case with full names (not abbreviated): `padded_vocab_size` not `pv_size`
- Loop variables and short-lived variables: single letters when appropriate (`i`, `k`, `v`, `q`)
- Tensor/tensor-like objects: descriptive snake_case (`cos`, `sin`, `mask_cache`, `adapter_kv_cache`)

**Classes:**
- PascalCase for class names: `Config`, `GPT`, `Block`, `CausalSelfAttention`, `LoRALinear`
- Dataclass configs also PascalCase: `Config` inherits from `BaseConfig`
- Private/internal implementation classes: still PascalCase but often prefixed (not Python convention, just clarity)

**Types:**
- Use `typing_extensions.Self` for self-referencing type hints
- Type hints are comprehensive: `def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> Union[torch.Tensor, List[torch.Tensor]]:`
- Literal types for constrained values: `Literal["csv", "tensorboard", "wandb", "mlflow", "litlogger"]`

## Code Style

**Formatting:**
- Line length: 120 characters (configured in `pyproject.toml` via ruff)
- Indentation: 4 spaces (Python standard)
- No single-letter type variables; use descriptive names

**Linting:**
- Tool: Ruff
- Config: `pyproject.toml` in `[tool.ruff]` section
- Line length rule (E501) disabled - long lines allowed
- Lambda assignment (E731) disabled - lambdas allowed
- Ambiguous variables (E741) TODO comment
- Unused variables (F841) TODO comment
- Docstring convention: Google style via `lint.pydocstyle.convention = "google"`

**Pre-commit Hooks:**
- `ruff check --fix` - auto-fix linting issues
- `ruff format` - auto-format with ruff formatter
- `codespell` - spell checking with auto-fix
- `prettier` - JSON/YAML/TOML formatting (print width 140)
- Trailing whitespace removal, YAML/TOML validation

## Import Organization

**Order:**
1. Standard library imports (math, os, sys, etc.)
2. Third-party imports (torch, lightning, transformers, etc.)
3. Local imports (from litgpt.*)

**Patterns:**
```python
# Standard library first
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Third-party libraries
import torch
import torch.nn as nn
from typing_extensions import Self

# Local imports
from litgpt.config import Config
from litgpt.model import GPT
```

**Path Aliases:**
- Direct relative imports used: `from litgpt.config import Config`
- No special alias configuration; absolute imports from package root
- TYPE_CHECKING guard for circular imports: `if TYPE_CHECKING: from litgpt import GPT`

## Error Handling

**Patterns:**
- Explicit exception types for different error conditions
- Informative error messages with context (variable values, constraints)
- Use specific exceptions: `ValueError`, `TypeError`, `RuntimeError`, `NotImplementedError`, `NotADirectoryError`

**Examples:**
```python
# Check preconditions with assert or raise
assert config.padded_vocab_size is not None

# Value validation with descriptive messages
if value > self.config.block_size:
    raise ValueError(
        f"Cannot attend to {value}, block size is only {self.config.block_size}."
        " This is likely because the input text exceeds the supported context length of this model."
    )

# Type checking
if not isinstance(module, GroupedTopkRouter):
    raise TypeError("Expected GroupedTopkRouter instance")

# Missing feature
raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")
```

## Logging

**Framework:** Standard `logging` module + `print()` statements for simple output

**Patterns:**
- Use `logging.getLogger()` for module-level loggers
- Suppress verbose framework warnings with logger filters:
  ```python
  pattern = re.compile(".*Profiler function .* will be ignored")
  logging.getLogger("torch._dynamo.variables.torch").addFilter(lambda record: not pattern.search(record.getMessage()))
  ```
- Disable specific loggers for noisy output:
  ```python
  logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
  ```
- Use `print()` for training/inference progress output (captured in logs/console)
- Warnings via `warnings.warn()` for user-facing issues:
  ```python
  print(
      f"Warning: KV cache has length {self.mask_cache.shape[-1]} < {value} = max_seq_length. Call 'set_kv_cache' before doing any forwards!"
  )
  ```

## Comments

**When to Comment:**
- Explain WHY, not WHAT (code shows what, comment shows why)
- Non-obvious algorithm choices: "For MHA this is a no-op"
- References to papers or external logic: "# Trigger resetting the rope-cache"
- Compatibility notes: "# for compatibility with older checkpoints"
- Temporary workarounds: "# Note that inferring..."

**JSDoc/Docstring:**
- Google-style docstrings for public functions/classes
- One-line summary followed by blank line, then detailed description
- Args/Returns sections only when needed
- Example from `litgpt/model.py`:
  ```python
  def max_seq_length(self, value: int) -> None:
      """
      When doing inference, the sequences used might be shorter than the model's context length.
      This allows setting a smaller number to avoid allocating unused memory
      """
  ```

## Function Design

**Size:**
- Functions should be focused and reasonably sized
- Typical range: 10-50 lines for utility functions
- Longer functions (100+ lines) acceptable for complex algorithms (e.g., forward passes)

**Parameters:**
- Use type hints consistently
- Optional parameters with sensible defaults
- Avoid excessive parameters; use config objects when many parameters needed
- Example: `def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, input_pos_maxp1: Optional[int] = None) -> Union[torch.Tensor, List[torch.Tensor]]:`

**Return Values:**
- Use specific return types: `-> int`, `-> Optional[Path]`, `-> Union[torch.Tensor, List[torch.Tensor]]`
- Tuple returns for multiple values: `-> Tuple[torch.Tensor, torch.Tensor]`
- Return early pattern common:
  ```python
  if not resume or isinstance(resume, Path):
      return resume
  ```

## Module Design

**Exports:**
- Public API defined in `__all__` at module level:
  ```python
  __all__ = ["LLM", "GPT", "Config", "PromptStyle", "Tokenizer"]
  ```
- Single responsibility: each module focuses on one concept
- Example: `litgpt/adapter.py` contains Adapter-specific Config, GPT, Block, CausalSelfAttention classes

**Barrel Files:**
- `__init__.py` imports main classes and manages logging:
  ```python
  from litgpt.api import LLM
  from litgpt.config import Config
  from litgpt.model import GPT
  from litgpt.prompts import PromptStyle
  from litgpt.tokenizer import Tokenizer
  ```
- Can suppress verbose dependency warnings at package initialization level

## Dataclasses

**Pattern:**
- Use `@dataclass` decorator for configuration objects
- Inherit from parent configs: `class Config(BaseConfig):`
- Add specific attributes in subclass:
  ```python
  @dataclass
  class Config(BaseConfig):
      adapter_prompt_length: int = 10
      adapter_start_layer: int = 2
  ```
- Type hints on all fields

## Type Hints

**Usage:**
- Comprehensive type hints throughout codebase
- Use `Union` for multiple return types
- Use `Optional` for nullable values
- Use `Literal` for constrained string values
- Import `Self` from `typing_extensions` for self-references:
  ```python
  @classmethod
  def from_name(cls, name: str, **kwargs: Any) -> Self:
      return cls(Config.from_name(name, **kwargs))
  ```

---

*Convention analysis: 2026-03-13*
