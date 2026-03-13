# Architecture

**Analysis Date:** 2026-03-13

## Pattern Overview

**Overall:** Modular GPT-based LLM framework with pluggable data, model, and training strategies

**Key Characteristics:**
- Transformer-based decoder-only architecture with configurable blocks
- Separation between model definition, training, inference, and fine-tuning
- Command-based CLI interface using JSON argument parsing (jsonargparse)
- Lightning Fabric for distributed training with flexible precision and strategy support
- Data-agnostic design with pluggable DataModule implementations
- Multiple fine-tuning approaches (LoRA, Adapter, Adapter-v2, Full)
- Inference strategies including tensor parallelism and speculative decoding

## Layers

**Model Layer:**
- Purpose: Define and instantiate the core GPT transformer architecture
- Location: `litgpt/model.py`, `litgpt/config.py`
- Contains: `GPT` class (decoder-only transformer with embedding, transformer blocks, language modeling head), `Block` (single transformer layer with attention and MLP), configuration management
- Depends on: PyTorch, torch.nn modules
- Used by: API layer, training scripts, inference scripts

**API/Inference Layer:**
- Purpose: High-level Python interface for loading models and generating text
- Location: `litgpt/api.py`, `litgpt/generate/` directory
- Contains: `LLM` class (main public API), generation strategies (base, sequential, tensor_parallel, adapter, adapter_v2, speculative_decoding)
- Depends on: Model layer, tokenizer, prompt styles, checkpoint utilities
- Used by: CLI, chat interface, external Python applications

**Training/Fine-tuning Layer:**
- Purpose: Implement different training paradigms (pretraining, full fine-tuning, parameter-efficient adaptation)
- Location: `litgpt/pretrain.py`, `litgpt/finetune/` directory
- Contains: `setup()` functions for pretrain, finetune_full, finetune_lora, finetune_adapter, finetune_adapter_v2
- Depends on: Model layer, data layer, Lightning Fabric, arguments configuration
- Used by: CLI entry point, orchestrated training workflows

**Data Layer:**
- Purpose: Abstract data loading and preprocessing for different datasets
- Location: `litgpt/data/` directory
- Contains: `DataModule` base class (extends LightningDataModule), dataset-specific implementations (Alpaca, LitData, OpenWebText, etc.), `SFTDataset` for supervised fine-tuning
- Depends on: Tokenizer, prompt styles
- Used by: Training scripts, evaluation

**Chat/Inference Interface:**
- Purpose: Interactive chat interface for model interaction
- Location: `litgpt/chat/base.py`
- Contains: `generate()` function for streaming generation, prompt processing, model initialization
- Depends on: API layer, model, tokenizer, prompts
- Used by: CLI chat command

**Deployment Layer:**
- Purpose: Serve models via HTTP API
- Location: `litgpt/deploy/serve.py`
- Contains: LitServe-based REST API endpoints
- Depends on: API layer, model loading utilities
- Used by: Production serving

**Configuration Management:**
- Purpose: Handle model and experiment configuration
- Location: `litgpt/config.py`, `litgpt/args.py`, `litgpt/parser_config.py`
- Contains: `Config` dataclass (model architecture parameters), `TrainArgs`/`EvalArgs`/`LogArgs` (experiment parameters), parser command registry
- Depends on: dataclasses, YAML serialization
- Used by: All training and inference scripts

**CLI Entry Point:**
- Purpose: Command dispatch and argument routing
- Location: `litgpt/__main__.py`
- Contains: PARSER_DATA mapping of commands to handler functions, main() dispatcher
- Depends on: jsonargparse CLI, all command implementations
- Used by: `litgpt` command-line tool

## Data Flow

**Pretraining Flow:**

1. User runs: `litgpt pretrain --model_name=... --data=TinyLlama`
2. `litgpt/__main__.py::main()` parses args and calls `litgpt/pretrain.py::setup()`
3. Setup function:
   - Loads or creates `Config` (model architecture)
   - Instantiates `DataModule` with tokenizer connection
   - Creates Lightning `Fabric` with precision/strategy
   - Initializes `GPT` model
   - Creates optimizer and learning rate scheduler
   - Launches training loop with gradient accumulation, checkpointing, evaluation
4. Data flows: `DataModule.train_dataloader()` → tokenized sequences → Model forward pass → loss computation → backward pass → optimizer step
5. Checkpoints saved to `out_dir/step-{step}/` with model weights, config, and optimizer state

**Fine-tuning Flow (LoRA Example):**

1. User runs: `litgpt finetune_lora --checkpoint_dir=path/to/base --data=Alpaca`
2. `litgpt/__main__.py` → `litgpt/finetune/lora.py::setup()`
3. Setup function:
   - Loads base `GPT` model from checkpoint
   - Wraps model with `LoRA` parameter-efficient layers using `lora.py` classes
   - Freezes base weights, only LoRA matrices trainable
   - Follows similar training loop to pretraining but with fewer parameters updated
4. Data flow: Supervised fine-tuning dataset (instruction + response) → prompt masking applied → model with LoRA adapters → loss on response tokens only
5. Saves LoRA checkpoint separate from base model

**Inference/Generation Flow:**

1. User calls: `llm = LLM.load("checkpoint_dir")` or `LLM.load("model_name")`
2. `litgpt/api.py::LLM.__init__()`:
   - Auto-downloads checkpoint if needed (HuggingFace Hub)
   - Loads `Config`, `GPT` model, `Tokenizer`, optional `PromptStyle`
   - Sets up `Fabric` for inference device/precision
   - Initializes KV cache if needed
3. User calls: `llm.generate(prompt_text, ...)`
4. Generation flow:
   - Tokenize prompt with `Tokenizer`
   - Apply `PromptStyle` template
   - Initialize position indices and KV cache
   - Loop: `generate/base.py::next_token()` → model forward → sample logits (top_k, top_p, temperature) → append token
   - Detokenize and return/stream result
5. Generation strategy affects how tokens are computed (sequential, tensor parallel, speculative)

**Chat Interactive Flow:**

1. User runs: `litgpt chat --checkpoint_dir=...`
2. `litgpt/chat/base.py::main()`:
   - Loads model via `LLM.load()`
   - Enters interactive loop reading user prompts
   - For each prompt: `process_prompt()` → tokenize and format → `generate()` → stream tokens back to user
   - Continue until user exits

**Evaluation Flow:**

1. User runs: `litgpt evaluate --checkpoint_dir=... --eval_dataset=openwebtext`
2. `litgpt/eval/evaluate.py::convert_and_evaluate()`:
   - Loads model checkpoint
   - Loads evaluation dataset
   - Runs language model evaluation harness
   - Computes metrics (accuracy, perplexity, etc.)
   - Saves results

**State Management:**

- Model state: Stored as PyTorch checkpoint (lit_model.pth) with architecture config (model_config.yaml)
- KV cache: Initialized on first forward pass for inference, resized based on sequence length
- Training state: Optimizer state, learning rate scheduler state, iteration counters saved in checkpoint for resumption
- Tokenizer: Loaded from checkpoint directory or downloaded from HuggingFace Hub
- Prompt styles: Loaded from checkpoint or defined in code, applied to format user prompts

## Key Abstractions

**GPT Model Abstraction:**
- Purpose: Encapsulates transformer architecture with configurable layers, attention types, and normalization
- Location: `litgpt/model.py`
- Examples: `Block` (transformer layer), `CausalSelfAttention` (self-attention with optional rotary embeddings)
- Pattern: Modular nn.Module composition, configuration-driven instantiation

**DataModule Abstraction:**
- Purpose: Unified interface for different data sources
- Location: `litgpt/data/base.py`
- Examples: Alpaca, LitData, OpenWebText, TinyLlama dataset classes
- Pattern: Subclass `DataModule`, implement `connect()` and `train_dataloader()`/`val_dataloader()`

**Fine-tuning Adapter Patterns:**
- LoRA: Adds learnable low-rank matrices to specific layers (litgpt/lora.py)
- Adapter: Bottleneck layers inserted between transformer blocks (litgpt/adapter.py)
- Adapter-v2: Improved adapter with better initialization (litgpt/adapter_v2.py)
- Full: No adaptation, all parameters fine-tuned (litgpt/finetune/full.py)

**Generation Strategies:**
- Base: Standard autoregressive generation with KV cache
- Sequential: Token-by-token on single device
- Tensor Parallel: Distribute model across multiple GPUs
- Speculative: Draft-verify approach for faster generation
- Pattern: All inherit from `generate_fn` interface in `litgpt/generate/base.py`

**Configuration Management:**
- Purpose: Externalize all hyperparameters and architecture choices
- Examples: `Config` for model architecture, `TrainArgs` for training parameters
- Pattern: Dataclass-based configuration with YAML serialization, loaded from checkpoint directory

## Entry Points

**CLI Entry Point:**
- Location: `litgpt/__main__.py::main()`
- Triggers: `litgpt` command (registered in pyproject.toml as console script)
- Responsibilities: Parse command (download, chat, pretrain, finetune_*, generate_*, evaluate, serve, etc.), route to appropriate handler function

**Python API Entry Point:**
- Location: `litgpt/api.py::LLM` class
- Triggers: `from litgpt import LLM; llm = LLM.load(...)`
- Responsibilities: Load model from checkpoint/Hub, expose `generate()`, `finetune()`, `trainer_setup()` methods

**Pretraining Entry Point:**
- Location: `litgpt/pretrain.py::setup()`
- Triggers: `litgpt pretrain --model_name=... --data=...`
- Responsibilities: Initialize training loop, manage checkpoints and logging

**Fine-tuning Entry Points:**
- Locations: `litgpt/finetune/lora.py::setup()`, `litgpt/finetune/full.py::setup()`, `litgpt/finetune/adapter.py::setup()`, `litgpt/finetune/adapter_v2.py::setup()`
- Triggers: `litgpt finetune_lora --checkpoint_dir=...`, etc.
- Responsibilities: Load base model, wrap with adaptation strategy, run training loop

**Generation Entry Points:**
- Locations: `litgpt/generate/base.py::main()`, `litgpt/generate/sequentially.py::main()`, etc.
- Triggers: `litgpt generate --checkpoint_dir=... --prompt="..."`
- Responsibilities: Load model, generate tokens using specified strategy

**Chat Entry Point:**
- Location: `litgpt/chat/base.py::main()`
- Triggers: `litgpt chat --checkpoint_dir=...`
- Responsibilities: Load model, enter interactive prompt loop

## Error Handling

**Strategy:** Explicit validation with informative error messages

**Patterns:**
- Checkpoint validation: `check_valid_checkpoint_dir()` ensures model_config.yaml, lit_model.pth, tokenizer files exist
- Config validation: Config dataclass `__post_init__()` validates architecture parameters (padding, layer counts, etc.)
- Device validation: `check_nvlink_connectivity()` warns about potential performance issues
- Precision validation: `get_default_supported_precision()` determines compatible precision for hardware
- Argument validation: TrainArgs `__post_init__()` validates warmup steps, batch sizes, scheduling parameters
- File size warnings: `check_file_size_on_cpu_and_warn()` alerts when loading large models to CPU only

## Cross-Cutting Concerns

**Logging:**
- TensorBoard, CSV, WandbLogger, MLFlowLogger via Lightning Fabric
- LitLogger for experiment tracking (optional)
- Configured via LogArgs and logger_name parameter
- Training metrics logged via RunningMean and ThroughputMonitor

**Validation:**
- During data loading: SFTDataset masks prompt tokens when mask_prompt=True
- During generation: Max sequence length enforced, stop_tokens detection
- Checkpoint loading: Config compatibility checked

**Authentication:**
- HuggingFace Hub access tokens passed to auto_download_checkpoint()
- Checkpoint directory resolution via extend_checkpoint_dir() for built-in models

**Distributed Training:**
- Lightning Fabric handles device placement and FSDP strategy setup
- Gradient accumulation computed from global_batch_size / (num_devices * num_nodes * micro_batch_size)
- All-reduce synchronization during backward pass on accumulated gradients

**Model Conversion:**
- HuggingFace checkpoint → LitGPT format: `scripts/convert_hf_checkpoint.py`
- LitGPT → HuggingFace format: `scripts/convert_lit_checkpoint.py`
- Other pretrained → LitGPT: `scripts/convert_pretrained_checkpoint.py`
- LoRA merge: `scripts/merge_lora.py` combines LoRA weights into base model

**Tokenization:**
- Tokenizer loaded from checkpoint or HuggingFace Hub
- Supports multiple tokenizer formats (BPE via tokenizers library)
- Applied during data loading and inference

---

*Architecture analysis: 2026-03-13*
