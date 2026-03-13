# External Integrations

**Analysis Date:** 2026-03-13

## APIs & External Services

**Model Repository:**
- HuggingFace Hub - Central model and dataset repository
  - SDK/Client: `huggingface-hub` (0.30-1.4)
  - Auth: `HF_TOKEN` environment variable
  - Used in: `litgpt/scripts/download.py`, model loading via `litgpt.api.LLM.load()`
  - Purpose: Download pre-trained LLMs, access gated models, manage model versioning
  - Endpoint: https://huggingface.co/api/models/
  - File transfer: Optional `hf-transfer` plugin for fast downloads

**Experiment Tracking (Optional):**
- Weights & Biases (WandB) - Cloud experiment tracking
  - SDK/Client: `wandb` (optional dependency)
  - Auth: `WANDB_API_KEY` environment variable (standard WandB setup)
  - Used in: `litgpt/utils.py` get_logger function
  - Purpose: Log training metrics, model checkpoints, hyperparameters
  - Config: `WANDB_RUN_NAME`, `WANDB_RUN_GROUP` environment variables
  - Integration: `WandbLogger` via Lightning

- MLflow - Local/remote experiment tracking
  - SDK/Client: `mlflow` (optional dependency)
  - Used in: `litgpt/utils.py` get_logger function
  - Purpose: Track experiments, log metrics, manage model registry
  - Integration: `MLFlowLogger` via Lightning

- Lightning.ai (LitLogger) - Native Lightning experiment tracking
  - SDK/Client: `litlogger>=0.1.7` (optional dependency)
  - Used in: `litgpt/utils.py` get_logger function
  - Purpose: Lightning.ai-native experiment tracking and model logging
  - Config parameters: `teamspace`, `metadata`, `log_model`, `save_logs`, `checkpoint_name`
  - Integration: `LitLogger` via `lightning.pytorch.loggers`

**Evaluation Framework:**
- EleutherAI LM Evaluation Harness - Standardized LLM evaluation
  - SDK/Client: `lm-eval>=0.4.2,<0.4.9.1`
  - Used in: `litgpt/eval/evaluate.py`
  - Purpose: Benchmark models on standard evaluation datasets
  - Source: GitHub EleutherAI/lm-evaluation-harness
  - Note: Pinned below 0.4.9.1 due to trust_remote_code issues

**Model Serving:**
- LitServe - Lightning's model serving framework
  - SDK/Client: `litserve>0.2`
  - Used in: `litgpt/deploy/serve.py`
  - Purpose: Deploy models as REST APIs with optional streaming support
  - Features: OpenAI-compatible API endpoints via `OpenAISpec`
  - Endpoints provided: `/v1/chat/completions` (OpenAI-compatible), custom `/predict` endpoint
  - Transport: HTTP/REST with optional streaming via Server-Sent Events
  - Default port: 8000

**Data Streaming & Optimization:**
- LitData - Lightning's data streaming platform
  - SDK/Client: `litdata==0.2.59`
  - Used in: `litgpt/data/openwebtext.py`, `litgpt/data/tinystories.py`, `litgpt/data/tinyllama.py`, `litgpt/data/text_files.py`, `litgpt/data/lit_data.py`, `litgpt/data/prepare_slimpajama.py`, `litgpt/data/prepare_starcoder.py`
  - Purpose: Optimize and stream large datasets efficiently for training
  - Classes: `StreamingDataset`, `StreamingDataLoader`, `TokensLoader`, `DataProcessor`
  - Features: Distributed data optimization, chunked loading

## Data Storage

**Databases:**
- None detected - Project uses local/remote file storage only
- Local filesystem: Training data, model checkpoints, logs
- Remote support: S3-compatible paths (s3://) for data and models

**File Storage:**
- Local filesystem (default) - Model checkpoints, training data, logs stored locally
- S3-compatible storage - Optional remote data paths supported
  - Used in: `litgpt/data/openwebtext.py`, `litgpt/data/tinyllama.py`
  - Pattern: `if str(self.data_path).startswith("s3://")`
  - Client: Handled via LitData or Lightning Fabric

**Caching:**
- None explicitly detected
- PyTorch/HuggingFace caching directories used for model caching (standard locations)

## Authentication & Identity

**Auth Provider:**
- Custom token-based authentication via HuggingFace Hub
  - Implementation: Environment variable `HF_TOKEN`
  - Scope: Used to access private/gated models on HuggingFace
  - Used in: `litgpt/scripts/download.py`, `litgpt/data/lima.py`, `litgpt/utils.py`
  - Error handling: Raises error if gated model accessed without valid token

- WandB authentication (if using WandB)
  - Implementation: Standard WandB environment variables (`WANDB_API_KEY`)

- Lightning.ai authentication (if using LitLogger)
  - Implementation: Built into Lightning authentication flow

## Monitoring & Observability

**Error Tracking:**
- None detected - No external error tracking service integrated

**Logs:**
- TensorBoard (default)
  - SDK/Client: `tensorboard>=2.14`
  - Location: `{out_dir}/logs/tensorboard/`
  - Used in: `litgpt/utils.py`, `litgpt/pretrain.py`
  - Default logger for pretraining

- CSV logging (lightweight)
  - Default for most scripts
  - Location: `{out_dir}/lightning_logs/`

- Multi-logger support:
  - TensorBoard: Visualization dashboard
  - WandB: Cloud-based dashboard
  - MLflow: Experiment registry and tracking
  - LitLogger: Lightning.ai native tracking
  - Selection via `--logger_name` argument

## CI/CD & Deployment

**Hosting:**
- GitHub (code repository)
- GitHub Actions (CI/CD pipelines)

**CI Pipeline:**
- `.github/workflows/cpu-tests.yml` - CPU unit tests
- `.github/workflows/check-links.yml` - Documentation link verification
- `.github/workflows/publish-pkg.yml` - PyPI package publishing
- `.github/workflows/mkdocs-deploy.yml` - Documentation deployment
- Dependabot configuration (`.github/dependabot.yml`) for automatic dependency updates

**Deployment Options:**
- LitServe (recommended for production serving)
  - Serves on HTTP (default port 8000)
  - Supports OpenAI-compatible API via `--openai_spec` flag
  - Streaming responses via `--stream` flag
- Docker support via `.devcontainer/` configuration

## Environment Configuration

**Required env vars:**
- `HF_TOKEN` - (Optional but required for gated models) HuggingFace Hub authentication token
- `LIGHTNING_ARTIFACTS_DIR` - (Optional) Directory for Lightning training artifacts and logs
- `RANK` - (Auto-set in distributed training) Current process rank
- `WORLD_SIZE` - (Auto-set in distributed training) Total number of processes

**Optional env vars:**
- `WANDB_RUN_NAME` - Custom WandB run name
- `WANDB_RUN_GROUP` - WandB run grouping
- `TOKENIZERS_PARALLELISM` - Set to "false" to disable tokenizer parallelism warnings
- `DATA_OPTIMIZER_GLOBAL_RANK` - Distributed data optimization rank
- `DATA_OPTIMIZER_NUM_WORKERS` - Data optimization worker count

**Secrets location:**
- Not committed to repository (follows best practices)
- Expected to be set via CI/CD secrets (GitHub Actions) or local development environment
- No `.env` file used (relies on environment or CI/CD injection)

## Webhooks & Callbacks

**Incoming:**
- None detected - Project does not expose webhook endpoints

**Outgoing:**
- None detected - Project does not send webhooks to external systems
- One-directional integrations: Pull data from HuggingFace Hub, push metrics to WandB/MLflow/LitLogger

## Remote Data Sources

**Datasets via HTTP:**
- Alpaca dataset: `https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json`
  - Downloaded via `requests` library in `litgpt/data/alpaca.py`
  - Used for supervised fine-tuning tasks

**HuggingFace Datasets:**
- LIMA, TinyStories, SlimPajama, StarCoder, OpenWebText
  - Accessed via `huggingface-hub` API
  - Some require authentication (`HF_TOKEN`)
  - Streamed efficiently via LitData when available

## Data Format Standards

**Model Checkpoints:**
- SafeTensors format (.safetensors) - Primary serialization format
  - Faster loading, safer (doesn't execute arbitrary code)
  - Used via `safetensors` library

- PyTorch format (.pt, .pth) - Legacy support
  - Loaded via torch.load() with proper security measures

**Tokenizer Configs:**
- JSON format - Tokenizer configuration and chat templates
  - Location: `{checkpoint_dir}/tokenizer_config.json`
  - Parsed for chat templates in OpenAI-spec serving

**Training Configs:**
- YAML format - Model and training configurations
  - Used in `config_hub/` directory

---

*Integration audit: 2026-03-13*
