# Technology Stack

**Analysis Date:** 2026-03-13

## Languages

**Primary:**
- Python 3.10+ - All source code and scripts
- YAML - Configuration files, workflow definitions
- JSON - Model configurations, data files, tokenizer configs

**Secondary:**
- Shell - Deployment and utility scripts in `scripts/` directory

## Runtime

**Environment:**
- Python (CPython)
- Minimum version: Python 3.10
- Supported versions: 3.10, 3.11, 3.12, 3.13, 3.14

**Package Manager:**
- pip (standard)
- Lockfile: `uv.lock` (using `uv` tool for fast dependency resolution)

## Frameworks

**Core ML:**
- PyTorch 2.7+ - Deep learning framework, neural network implementations
- Lightning 2.6.1+ - Training orchestration and fabric, distributed training
- Transformers 4.51.3-4.57 - HuggingFace model loading and utilities

**Data Processing:**
- tokenizers 0.21+ - Fast tokenization for LLMs
- safetensors 0.4.3+ - Safe model serialization/deserialization
- huggingface-hub 0.30-1.4 - Model downloading and management from HuggingFace Hub
- psutil 7.1.3 - System monitoring and resource tracking
- tqdm 4.66+ - Progress bars for long-running operations

**Testing:**
- pytest 8.1.1+ - Test framework
- pytest-benchmark 5.1+ - Performance benchmarking
- pytest-dependency 0.6+ - Test dependency management
- pytest-rerunfailures 14+ - Flaky test rerunning
- pytest-timeout 2.3.1+ - Test timeout enforcement

**Build/Dev:**
- setuptools 68.2.2+ - Package building
- wheel 0.41.2+ - Wheel distribution format
- jsonargparse 4.37-4.41 - CLI argument parsing with type signatures

## Key Dependencies

**Critical:**
- torch 2.7+ - Why it matters: Core tensor computations, GPU acceleration required for model training/inference
- lightning 2.6.1+ - Why it matters: Distributed training orchestration, multi-GPU/TPU support
- transformers 4.51.3+ - Why it matters: Pre-trained model architectures, tokenizer configurations
- huggingface-hub 0.30+ - Why it matters: Downloads models from HuggingFace, manages model versioning and access control

**Infrastructure:**
- bitsandbytes 0.42-0.50 (platform-specific) - 8-bit and 4-bit quantization for model compression
- litdata 0.2.59 - Optimized data streaming and loading
- litserve 0.2+ - Model serving with OpenAI-compatible API support
- lm-eval 0.4.2-0.4.9.1 - Language model evaluation harness
- tensorboard 2.14+ - Training metrics visualization
- torchmetrics 1.3.1+ - ML-specific metric computations

**Conditional/Optional:**
- lightning-thunder 0.2.dev20250119+ - Compiler optimization (Linux only)
- litlogger 0.1.7+ - Lightning.ai native experiment tracking
- wandb - Weights & Biases experiment tracking (optional)
- mlflow - MLflow experiment tracking (optional)
- pandas 1.9+ - Data manipulation for dataset preparation
- pyarrow 15.0.2+ - Apache Arrow data format support
- requests 2.31+ - HTTP client for downloading remote data
- sentencepiece 0.2+ - Subword tokenization for llama-based models
- zstandard 0.22+ - Zstd compression for data preprocessing
- uvloop 0.2+ - High-performance async event loop (non-Windows)

## Configuration

**Environment:**
- HF_TOKEN: HuggingFace API token for accessing gated/private models
- WANDB_RUN_NAME: Weights & Biases run name (optional)
- WANDB_RUN_GROUP: Weights & Biases run grouping (optional)
- LIGHTNING_ARTIFACTS_DIR: Directory for Lightning artifacts (optional)
- TOKENIZERS_PARALLELISM: Tokenizer parallelism setting (set to "false" in evaluation)
- DATA_OPTIMIZER_GLOBAL_RANK: Distributed data optimization rank (used in data preprocessing)
- DATA_OPTIMIZER_NUM_WORKERS: Number of workers for data optimization

**Build:**
- `pyproject.toml` - Main project configuration, dependencies, build settings
- `ruff.toml` config section - Code linting and formatting rules (target Python 3.8+)
- `.pre-commit-config.yaml` - Git pre-commit hooks for code quality

## Platform Requirements

**Development:**
- Linux or macOS preferred (some dependencies like lightning-thunder are Linux-only)
- 16GB+ RAM recommended for model development
- NVIDIA GPU with CUDA support recommended (most features are GPU-optimized)
- Python 3.10+ installed
- pip or uv for dependency management

**Production:**
- **Serving**: Deployable via LitServe on any platform supporting PyTorch
  - Supports NVIDIA GPUs (CUDA), AMD GPUs (ROCm), CPU, and Apple Silicon (MPS)
  - Port configurable (default 8000)
  - OpenAI-compatible API endpoints when using `openai_spec=True`
- **Training**: Requires GPU (NVIDIA CUDA or AMD ROCm) for practical use
- **Inference**: Can run on CPU but GPU strongly recommended

---

*Stack analysis: 2026-03-13*
