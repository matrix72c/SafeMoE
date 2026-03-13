# Codebase Structure

**Analysis Date:** 2026-03-13

## Directory Layout

```
safemoe/
├── litgpt/                         # Main package source
│   ├── __init__.py                # Public API exports (LLM, GPT, Config, etc.)
│   ├── __main__.py                # CLI entry point and command router
│   ├── api.py                     # LLM high-level Python API
│   ├── model.py                   # GPT transformer architecture
│   ├── config.py                  # Model configuration and parameter management
│   ├── args.py                    # Training/eval/logging argument dataclasses
│   ├── tokenizer.py               # Tokenizer wrapper
│   ├── prompts.py                 # Prompt style templates and formatting
│   ├── types.py                   # Type definitions (LoggerChoice, etc.)
│   ├── constants.py               # Feature availability flags
│   ├── utils.py                   # Shared utility functions
│   ├── parser_config.py           # CLI parser command registration
│   ├── lora.py                    # LoRA (Low-Rank Adaptation) layers
│   ├── adapter.py                 # Adapter fine-tuning layers
│   ├── adapter_v2.py              # Adapter v2 fine-tuning layers
│   ├── pretrain.py                # Pretraining script and setup
│   │
│   ├── data/                       # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── base.py               # DataModule base class and SFTDataset
│   │   ├── alpaca.py             # Alpaca dataset
│   │   ├── alpaca_2k.py          # Alpaca 2k variant
│   │   ├── alpaca_gpt4.py        # Alpaca GPT-4 variant
│   │   ├── deita.py              # DEITA dataset
│   │   ├── flan.py               # FLAN dataset
│   │   ├── json_data.py          # Generic JSON data loader
│   │   ├── lit_data.py           # LitData streaming dataset
│   │   ├── lima.py               # LIMA dataset
│   │   ├── longform.py           # Longform dataset
│   │   ├── microllama.py         # MicroLlama dataset
│   │   ├── openwebtext.py        # OpenWebText dataset
│   │   ├── text_files.py         # Plain text file loader
│   │   ├── tinyllama.py          # TinyLlama dataset
│   │   ├── tinystories.py        # TinyStories dataset
│   │   └── prepare_*.py          # Data preparation scripts
│   │
│   ├── finetune/                  # Fine-tuning implementations
│   │   ├── __init__.py
│   │   ├── full.py               # Full parameter fine-tuning
│   │   ├── lora.py               # LoRA fine-tuning
│   │   ├── adapter.py            # Adapter fine-tuning
│   │   ├── adapter_v2.py         # Adapter v2 fine-tuning
│   │   └── lora_legacy.py        # Legacy LoRA implementation
│   │
│   ├── generate/                  # Inference/generation implementations
│   │   ├── __init__.py
│   │   ├── base.py               # Base generation with KV cache and sampling
│   │   ├── full.py               # Full batch generation
│   │   ├── adapter.py            # Generation with LoRA/Adapter models
│   │   ├── adapter_v2.py         # Generation with Adapter v2 models
│   │   ├── sequentially.py       # Sequential multi-token generation
│   │   ├── tp.py                 # Tensor parallel generation
│   │   └── speculative_decoding.py # Speculative decoding (draft-verify)
│   │
│   ├── chat/                      # Interactive chat interface
│   │   ├── __init__.py
│   │   └── base.py               # Chat main function and REPL
│   │
│   ├── eval/                      # Evaluation harness
│   │   ├── __init__.py
│   │   └── evaluate.py           # Language model evaluation
│   │
│   ├── deploy/                    # Deployment/serving
│   │   ├── __init__.py
│   │   └── serve.py              # LitServe HTTP API server
│   │
│   └── scripts/                   # Utility and conversion scripts
│       ├── __init__.py
│       ├── download.py           # Download models from HuggingFace Hub
│       ├── merge_lora.py         # Merge LoRA weights into base model
│       ├── convert_hf_checkpoint.py      # HuggingFace → LitGPT conversion
│       ├── convert_lit_checkpoint.py     # LitGPT → HuggingFace conversion
│       └── convert_pretrained_checkpoint.py # Generic checkpoint conversion
│
├── tests/                         # Test suite
│   ├── conftest.py               # Pytest fixtures and configuration
│   ├── test_*.py                 # Unit tests (test_api.py, test_config.py, etc.)
│   ├── generate/                 # Generation tests
│   │   ├── __init__.py
│   │   ├── test_main.py
│   │   ├── test_adapter.py
│   │   ├── test_tp.py
│   │   ├── test_sequentially.py
│   │   └── utils.py              # Test utilities
│   └── ...
│
├── config_hub/                    # Pre-configured model configs
│   └── *.yaml                     # Model configuration files for various architectures
│
├── checkpoints/                   # Downloaded/saved model checkpoints
│   └── ...
│
├── tutorials/                     # Example notebooks and documentation
│   └── ...
│
├── extensions/                    # Extension modules (if any)
│   └── ...
│
├── scripts/                       # Top-level utility scripts
│   └── ...
│
├── .planning/                     # GSD planning artifacts
│   └── codebase/                 # Generated architecture/structure docs
│
├── .github/                       # GitHub workflows and CI/CD
│   └── workflows/
│
├── .claude/                       # Claude-specific configuration
│
├── .devcontainer/                # Dev container setup
│
├── .lightning/                    # Lightning Studio configuration
│
├── pyproject.toml                # Python project metadata and dependencies
├── uv.lock                        # Dependency lock file (uv package manager)
├── .pre-commit-config.yaml       # Pre-commit hook configuration
├── README.md                      # Project documentation
└── LICENSE                        # Apache License 2.0
```

## Directory Purposes

**litgpt/:**
- Purpose: Main package containing all source code
- Contains: All Python modules for training, inference, fine-tuning, and utilities
- Key files: `__init__.py` (exports public API), `__main__.py` (CLI), `api.py` (Python API)

**litgpt/data/:**
- Purpose: Dataset implementations and data loading infrastructure
- Contains: DataModule subclasses for different datasets, data preparation scripts
- Key files: `base.py` (DataModule base class), dataset-specific files (alpaca.py, openwebtext.py, etc.)

**litgpt/finetune/:**
- Purpose: Fine-tuning implementations with different adaptation strategies
- Contains: setup() functions for each fine-tuning method
- Key files: `full.py` (full parameter tuning), `lora.py` (LoRA), `adapter.py` (Adapter), `adapter_v2.py` (Adapter v2)

**litgpt/generate/:**
- Purpose: Inference and text generation implementations
- Contains: Different generation strategies for various hardware/efficiency configurations
- Key files: `base.py` (standard generation), `sequentially.py` (multi-token), `tp.py` (tensor parallel), `speculative_decoding.py` (draft-verify)

**litgpt/chat/:**
- Purpose: Interactive chat interface for conversing with models
- Contains: REPL loop, prompt formatting, streaming output
- Key files: `base.py` (main chat function)

**litgpt/eval/:**
- Purpose: Model evaluation against benchmarks
- Contains: Integration with lm-evaluation-harness
- Key files: `evaluate.py` (evaluation runner)

**litgpt/deploy/:**
- Purpose: Production model serving via HTTP API
- Contains: LitServe-based REST endpoints
- Key files: `serve.py` (HTTP server)

**litgpt/scripts/:**
- Purpose: Utility scripts for common tasks
- Contains: Model conversion, weight merging, downloading
- Key files: `convert_hf_checkpoint.py`, `merge_lora.py`, `download.py`

**tests/:**
- Purpose: Comprehensive test coverage for all components
- Contains: Unit tests, integration tests, test utilities
- Key files: `conftest.py` (pytest configuration), test files organized by component

**config_hub/:**
- Purpose: Pre-configured model architecture files
- Contains: YAML files with model configs for Llama, Phi, Mistral, etc.
- Key files: Model YAML files (e.g., `llama-13b.yaml`)

**checkpoints/:**
- Purpose: Cache directory for downloaded model checkpoints
- Contains: Pre-trained model weights and configuration
- Key files: Generated dynamically from downloads

**tutorials/:**
- Purpose: Documentation and example notebooks
- Contains: Jupyter notebooks, guides
- Key files: Various tutorial notebooks

## Key File Locations

**Entry Points:**
- `litgpt/__main__.py`: CLI entry point, routes commands to handlers
- `litgpt/api.py`: Python API (LLM class) for programmatic use
- `litgpt/__init__.py`: Public package exports (LLM, GPT, Config, Tokenizer, PromptStyle)

**Configuration:**
- `litgpt/config.py`: Model architecture configuration (Config dataclass)
- `litgpt/args.py`: Training/eval/logging argument classes (TrainArgs, EvalArgs, LogArgs)
- `config_hub/*.yaml`: Pre-defined model configurations

**Core Logic:**
- `litgpt/model.py`: GPT transformer implementation (GPT, Block, CausalSelfAttention classes)
- `litgpt/tokenizer.py`: Tokenizer wrapper
- `litgpt/prompts.py`: Prompt template system

**Testing:**
- `tests/conftest.py`: Pytest fixtures (fake_checkpoint_dir, tensor_like, float_like)
- `tests/test_api.py`: API layer tests
- `tests/test_config.py`: Configuration tests
- `tests/generate/`: Generation strategy tests

## Naming Conventions

**Files:**
- Module files: lowercase with underscores (e.g., `fine_tune.py`, `speculative_decoding.py`)
- Configuration files: lowercase with hyphens in YAML (e.g., `llama-7b.yaml`)
- Test files: `test_*.py` prefix (pytest convention)

**Directories:**
- Package directories: lowercase (e.g., `litgpt/`, `finetune/`, `generate/`)
- Built-in subdirectories: lowercase snake_case (e.g., `config_hub/`, `checkpoints/`)

**Classes:**
- Core classes: PascalCase (GPT, Block, Config, LLM, DataModule)
- Utility/adapter classes: PascalCase with descriptive names (LoRA, CausalSelfAttention, GroupedTopkRouter)

**Functions:**
- Setup/entry functions: `setup()` (litgpt/pretrain.py, litgpt/finetune/*.py, litgpt/generate/*.py)
- Main functions: `main()` (litgpt/chat/base.py, litgpt/generate/*.py)
- Utility functions: lowercase with underscores (check_valid_checkpoint_dir, auto_download_checkpoint, etc.)
- Sampling functions: `sample()`, `sample_top_p()`, `multinomial_num_samples_1()`

**Variables:**
- Model parameters: snake_case (n_layer, n_embd, vocab_size, block_size)
- Tensor shapes: descriptive snake_case (input_ids, target_ids, logits, hidden_states)
- Configuration objects: mixed (model_config, train_args, eval_args)

**Types:**
- Type aliases: PascalCase or descriptive lowercase
- Literal types: descriptive names (e.g., `Literal["LayerNorm", "RMSNorm"]`)

## Where to Add New Code

**New Fine-tuning Method:**
- Primary code: `litgpt/finetune/{method_name}.py` with `setup()` entry point
- Register command: Add to `litgpt/__main__.py::PARSER_DATA` dictionary
- Register parser: Add to `litgpt/parser_config.py::parser_commands()`
- Add corresponding generation: `litgpt/generate/{method_name}.py` for inference
- Tests: `tests/test_{method_name}.py`

**New Dataset:**
- Implementation: `litgpt/data/{dataset_name}.py` extending `DataModule`
- Implement: `connect()`, `train_dataloader()`, `val_dataloader()`
- Register: Auto-registered if added to package, optionally expose in `litgpt/data/__init__.py`
- Tests: `tests/test_data_{dataset_name}.py` or in existing data test files

**New Generation Strategy:**
- Implementation: `litgpt/generate/{strategy_name}.py` with `main()` function
- Register command: Add to `litgpt/__main__.py::PARSER_DATA`
- Register parser: Add to `litgpt/parser_config.py::parser_commands()`
- Base on existing: Study `litgpt/generate/base.py` for interface
- Tests: `tests/generate/test_{strategy_name}.py`

**New Model Architecture Feature:**
- Configuration: Add parameter to `Config` dataclass in `litgpt/config.py`
- Implementation: Add logic to `litgpt/model.py` (GPT, Block, or CausalSelfAttention classes)
- Example configs: Add YAML files to `config_hub/`
- Tests: Add to `tests/test_config.py` or create new test file

**Utilities/Helpers:**
- Shared helpers: `litgpt/utils.py` if broadly used
- Specific utilities: Co-locate with the module that uses them (e.g., conversion utilities in scripts/)
- Tests: `tests/test_utils.py` for utils, or in specific test files

**Data Preprocessing:**
- Preparation scripts: `litgpt/data/prepare_{dataset_name}.py`
- Large dataset handling: Use LitData streaming format when available

## Special Directories

**checkpoints/:**
- Purpose: Cache for downloaded model checkpoints
- Generated: Yes, created at runtime by auto_download_checkpoint()
- Committed: No, listed in .gitignore

**config_hub/:**
- Purpose: Pre-configured model architecture definitions
- Generated: No, committed to repo
- Committed: Yes, essential for model loading

**.planning/codebase/:**
- Purpose: GSD mapping artifacts (this document and ARCHITECTURE.md)
- Generated: Yes, created by /gsd:map-codebase
- Committed: Yes, updated via /gsd:execute-phase

**.claude/:**
- Purpose: Claude-specific project configuration
- Generated: No, managed externally
- Committed: Yes

**.github/workflows/:**
- Purpose: CI/CD pipeline definitions
- Generated: No, manually maintained
- Committed: Yes

**.devcontainer/:**
- Purpose: Dev environment configuration for containerized development
- Generated: No, manually maintained
- Committed: Yes

---

*Structure analysis: 2026-03-13*
