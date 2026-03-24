"""safemoe/evaluate.py — Full checkpoint evaluation and routing analysis"""

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import lightning as L
import torch
import yaml
from lightning.fabric.plugins.precision.fsdp import FSDPPrecision
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader

from litgpt.model import GPT, Block
from litgpt.utils import check_valid_checkpoint_dir, parse_devices
from safemoe.config import SafeMoEConfig
from safemoe.observability import RoutingObservabilityCollector
from safemoe.pretrain import validate


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ensure_checkpoint_dir(ckpt_dir: Path) -> Path:
    """Validate a checkpoint directory before attempting to read model files."""
    ckpt_dir = Path(ckpt_dir)
    try:
        check_valid_checkpoint_dir(
            ckpt_dir,
            raise_error=True,
            verbose=False,
            ignore_tokenizer_files=True,
        )
    except FileNotFoundError as exc:
        if ckpt_dir == Path("."):
            raise FileNotFoundError(
                f"{exc} checkpoint_dir resolved to the current working directory (`.`). "
                "If you used a temporary shell assignment such as "
                '`CKPT_DIR=... python -m safemoe evaluate "$CKPT_DIR"` or '
                '`CKPT_DIR=... python -m safemoe evaluate "$CKPT_DIR" --ablated "$CKPT_DIR/ablated"`, '
                "the shell expands `$CKPT_DIR` before that assignment applies. "
                "Export `CKPT_DIR` first or pass the checkpoint path literally."
            ) from exc
        raise
    return ckpt_dir


def _find_results_root(ckpt_dir: Path) -> Path:
    """Return the directory that owns the aggregated evaluation ledger."""
    ckpt_dir = Path(ckpt_dir).resolve()
    for candidate in (ckpt_dir, *ckpt_dir.parents):
        if candidate.name == "checkpoints":
            return candidate
    return ckpt_dir.parent


def _load_results_ledger(results_path: Path) -> dict[str, Any]:
    if not results_path.exists():
        return {}
    payload = json.loads(results_path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {results_path}, got {type(payload).__name__}")
    return payload


def _merge_dicts(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _checkpoint_relpath(ckpt_dir: Path, results_root: Path) -> str:
    ckpt_dir = Path(ckpt_dir).resolve()
    results_root = Path(results_root).resolve()
    try:
        return ckpt_dir.relative_to(results_root).as_posix()
    except ValueError:
        return ckpt_dir.name


def _load_checkpoint_run_name(ckpt_dir: Path) -> Optional[str]:
    hyperparameters_path = Path(ckpt_dir) / "hyperparameters.yaml"
    if not hyperparameters_path.is_file():
        return None
    payload = yaml.safe_load(hyperparameters_path.read_text())
    if not isinstance(payload, dict):
        return None
    run_name = payload.get("run_name")
    if isinstance(run_name, str) and run_name:
        return run_name
    return None


def _checkpoint_key(ckpt_dir: Path, results_root: Path) -> str:
    ckpt_dir = Path(ckpt_dir).resolve()
    checkpoint_run_name = _load_checkpoint_run_name(ckpt_dir)
    if checkpoint_run_name:
        return checkpoint_run_name
    return _checkpoint_relpath(ckpt_dir, results_root)


def _checkpoint_reference(ckpt_dir: Path, results_root: Optional[Path] = None) -> str:
    resolved = Path(ckpt_dir).resolve()
    checkpoint_run_name = _load_checkpoint_run_name(resolved)
    if checkpoint_run_name:
        return checkpoint_run_name
    if results_root is None:
        return str(resolved)
    return _checkpoint_relpath(resolved, results_root)


def _write_checkpoint_metrics(ckpt_dir: Path, metrics: dict[str, Any]) -> Path:
    """Merge one checkpoint's metrics into checkpoints/results.json."""
    ckpt_dir = Path(ckpt_dir)
    results_root = _find_results_root(ckpt_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    results_path = results_root / 'results.json'
    ledger = _load_results_ledger(results_path)
    record_key = _checkpoint_key(ckpt_dir, results_root)
    existing_record = ledger.get(record_key, {})
    if existing_record and not isinstance(existing_record, dict):
        raise TypeError(f"Expected mapping for results record {record_key!r}")

    checkpoint_relpath = _checkpoint_relpath(ckpt_dir, results_root)
    checkpoint_run_name = _load_checkpoint_run_name(ckpt_dir)
    key_parts = Path(record_key).parts
    record = _merge_dicts(
        {
            'checkpoint_dir': str(ckpt_dir.resolve()),
            'checkpoint_relpath': checkpoint_relpath,
            'checkpoint_name': ckpt_dir.name,
            'run_name': checkpoint_run_name or (key_parts[0] if key_parts else ckpt_dir.name),
            'metrics': {},
        },
        existing_record,
    )
    record['metrics'] = _merge_dicts(record.get('metrics', {}), metrics)
    record['updated_at'] = datetime.now(timezone.utc).isoformat()
    ledger[record_key] = record
    results_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    return results_path


class _EvalFSDPPrecision(FSDPPrecision):
    """Keep eval outputs in the model precision instead of upcasting giant logits tensors."""

    def convert_output(self, data):
        return data


def _make_eval_fabric(
    devices: int | str,
    num_nodes: int,
    accelerator: str,
    precision: Optional[str],
) -> L.Fabric:
    use_multi_device = isinstance(devices, int) and devices * num_nodes > 1
    strategy = (
        FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
        if use_multi_device
        else "auto"
    )
    if precision is None:
        precision = "bf16-true" if accelerator != "cpu" else "32-true"
    plugins = _EvalFSDPPrecision(precision) if use_multi_device else None
    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        accelerator=accelerator,
        strategy=strategy,
        precision=None if plugins is not None else precision,
        plugins=plugins,
    )
    fabric.launch()
    return fabric


def _load_model(ckpt_dir: Path, fabric: Optional[L.Fabric] = None) -> tuple:
    """Load a SafeMoE model from a checkpoint directory.

    Returns (model: GPT, config: SafeMoEConfig).
    When a fabric is provided, the checkpoint is loaded through the fabric so
    multi-GPU evaluation can use sharded model setup.
    """
    ckpt_dir = _ensure_checkpoint_dir(ckpt_dir)
    raw = yaml.safe_load((ckpt_dir / "model_config.yaml").read_text())
    config = SafeMoEConfig(**{k: v for k, v in raw.items() if not isinstance(v, dict)})

    if fabric is None:
        model = GPT(config)
    else:
        with fabric.init_module(empty_init=True):
            model = GPT(config)
        model = fabric.setup(model)

    checkpoint_path = ckpt_dir / "lit_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint 'model' entry must be a state_dict mapping")
        if fabric is None:
            model.load_state_dict(state_dict)
        else:
            # Under FSDP the wrapped module holds local shards; let the strategy
            # restore the full checkpoint instead of calling load_state_dict directly.
            fabric.load(checkpoint_path, {"model": model}, weights_only=False)
    else:
        if fabric is None:
            model.load_state_dict(checkpoint)
        else:
            fabric.load_raw(checkpoint_path, model, weights_only=False)
    model.eval()
    return model, config


class _TokenizerModelNameAlias:
    """Expose a stable tokenizer cache name while delegating to a real tokenizer when present."""

    def __init__(self, tokenizer: object | None, model_name: str) -> None:
        self._tokenizer = tokenizer
        self.model_name = model_name

    def __getattr__(self, name: str) -> object:
        if self._tokenizer is None:
            raise AttributeError(name)
        return getattr(self._tokenizer, name)


class _DatasetBatchIterable:
    """Sequentially batch indexable datasets without going through DataLoader internals."""

    def __init__(self, dataset: object, batch_size: int, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        dataset_length = len(self.dataset)
        step = self.batch_size
        limit = dataset_length if not self.drop_last else dataset_length - (dataset_length % step)
        for start in range(0, limit, step):
            stop = min(start + step, dataset_length)
            if self.drop_last and stop - start < step:
                break
            # litdata StreamingDataset returns tensors backed by storage that can
            # segfault under downstream tensor ops unless copied into ordinary tensors.
            yield torch.stack([self.dataset[index].clone() for index in range(start, stop)])

    def __len__(self) -> int:
        dataset_length = len(self.dataset)
        if self.drop_last:
            return dataset_length // self.batch_size
        return (dataset_length + self.batch_size - 1) // self.batch_size


def _resolve_manifest_checkpoint_dir(ckpt_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    for base_dir in (ckpt_dir, ckpt_dir.parent):
        resolved = base_dir / candidate
        if resolved.exists():
            return resolved
    return candidate


def _load_eval_tokenizer(ckpt_dir: Path) -> tuple[object | None, str | None]:
    from litgpt.tokenizer import Tokenizer

    hp_path = ckpt_dir / "hyperparameters.yaml"
    if hp_path.exists():
        hp = yaml.safe_load(hp_path.read_text())
        if not isinstance(hp, dict):
            raise TypeError(f"Expected mapping in {hp_path}, got {type(hp).__name__}")
        tokenizer_dir = hp.get("tokenizer_dir")
        if tokenizer_dir:
            tokenizer = Tokenizer(Path(tokenizer_dir))
            return tokenizer, tokenizer.model_name

    manifest_path = ckpt_dir / "intervention_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise TypeError(f"Expected mapping in {manifest_path}, got {type(manifest).__name__}")
        base_checkpoint_dir = manifest.get("base_checkpoint_dir")
        if isinstance(base_checkpoint_dir, str) and base_checkpoint_dir:
            resolved_dir = _resolve_manifest_checkpoint_dir(ckpt_dir, base_checkpoint_dir)
            if resolved_dir.exists():
                tokenizer = Tokenizer(resolved_dir)
                return tokenizer, tokenizer.model_name
            return None, resolved_dir.stem

        base_checkpoint_name = manifest.get("base_checkpoint_name")
        if isinstance(base_checkpoint_name, str) and base_checkpoint_name:
            return None, base_checkpoint_name

    surgery_path = ckpt_dir / "surgery_metadata.json"
    if surgery_path.exists():
        surgery_meta = json.loads(surgery_path.read_text())
        if not isinstance(surgery_meta, dict):
            raise TypeError(f"Expected mapping in {surgery_path}, got {type(surgery_meta).__name__}")
        base_checkpoint = surgery_meta.get("base_checkpoint")
        if isinstance(base_checkpoint, str) and base_checkpoint:
            base_dir = _resolve_manifest_checkpoint_dir(ckpt_dir, base_checkpoint)
            if base_dir.exists():
                tokenizer = Tokenizer(base_dir)
                return tokenizer, tokenizer.model_name
        base_checkpoint_name = surgery_meta.get("base_checkpoint_name")
        if isinstance(base_checkpoint_name, str) and base_checkpoint_name:
            return None, base_checkpoint_name

    tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    if all((ckpt_dir / filename).exists() for filename in tokenizer_files):
        tokenizer = Tokenizer(ckpt_dir)
        return tokenizer, tokenizer.model_name
    return None, None


def _get_eval_streaming_num_workers() -> int:
    """Return the worker count for standalone evaluation data loaders.

    Default to 0 to avoid litdata multiprocessing worker crashes on networked
    storage during standalone evaluation runs. Allow explicit override for
    users who want to trade safety for throughput.
    """
    raw_value = os.environ.get("SAFEMOE_EVAL_NUM_WORKERS")
    if raw_value is None:
        return 0

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "SAFEMOE_EVAL_NUM_WORKERS must be an integer >= 0, "
            f"got {raw_value!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            "SAFEMOE_EVAL_NUM_WORKERS must be an integer >= 0, "
            f"got {raw_value!r}"
        )
    return value


def _get_val_loaders(ckpt_dir: Path, config: SafeMoEConfig, data_mock=None) -> dict:
    """Return {"D_std": DataLoader, "D_harmful": DataLoader}.

    If data_mock is provided, delegate to data_mock.val_dataloaders().
    Otherwise construct a SafeDataModule from checkpoint hyperparameters.
    """
    if data_mock is not None:
        return data_mock.val_dataloaders()

    from safemoe.data.datamodule import SafeDataModule

    cache_dir = Path("data/.cache")
    datasets_cfg = {}

    hp_path = ckpt_dir / "hyperparameters.yaml"
    if hp_path.exists():
        hp = yaml.safe_load(hp_path.read_text())
        if not isinstance(hp, dict):
            raise TypeError(f"Expected mapping in {hp_path}, got {type(hp).__name__}")
        data_cfg = hp.get("data")
        if isinstance(data_cfg, dict):
            init_args = data_cfg.get("init_args")
            if isinstance(init_args, dict):
                if "cache_dir" in init_args:
                    cache_dir = Path(init_args["cache_dir"])
                datasets_cfg = init_args.get("datasets", datasets_cfg)

    if not datasets_cfg:
        raise ValueError(
            f"Cannot construct SafeDataModule for evaluation: "
            f"no datasets found in {hp_path}. "
            "Pass a data_mock or ensure hyperparameters.yaml has data.init_args.datasets."
        )

    tokenizer, tokenizer_name = _load_eval_tokenizer(ckpt_dir)
    if tokenizer_name is not None:
        tokenizer = _TokenizerModelNameAlias(tokenizer, tokenizer_name)

    loader = SafeDataModule(cache_dir=cache_dir, datasets=datasets_cfg)
    loader.num_workers = _get_eval_streaming_num_workers()
    loader.connect(tokenizer=tokenizer, batch_size=1, max_seq_length=config.block_size)
    if hasattr(loader, "val_datasets"):
        return {
            split_name: _DatasetBatchIterable(dataset, batch_size=loader.batch_size, drop_last=False)
            for split_name, dataset in loader.val_datasets().items()
        }
    return loader.val_dataloaders()


def _prepare_eval_loader(fabric: L.Fabric, loader: object) -> object:
    if isinstance(loader, DataLoader):
        return fabric.setup_dataloaders(loader)
    return loader



def _teardown_eval_session(session: Optional["_EvalSession"]) -> None:
    if session is None:
        return

    for loader in session.val_loaders.values():
        shutdown = getattr(loader, "shutdown", None)
        if callable(shutdown):
            shutdown()

    strategy = getattr(session.fabric, "strategy", None)
    # Fabric FSDP teardown reaches unsupported checkpoint_io, so eval relies on
    # explicit process-group teardown instead.
    if not isinstance(strategy, FSDPStrategy):
        teardown = getattr(strategy, "teardown", None)
        if callable(teardown):
            teardown()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()



def _ensure_surgery_checkpoint(ckpt_dir: Path) -> Path:
    """If *ckpt_dir* is a base checkpoint (no harmful experts), run surgery using
    parameters stored in its ``hyperparameters.yaml`` and return the surgery
    output directory.  If the checkpoint already has harmful experts, returns
    *ckpt_dir* unchanged.  Surgery is idempotent: re-running it reuses the
    existing output when the parameters match.
    """
    raw = yaml.safe_load((ckpt_dir / "model_config.yaml").read_text())
    config = SafeMoEConfig(**{k: v for k, v in raw.items() if not isinstance(v, dict)})
    if config.harmful_expert_indices:
        return ckpt_dir

    hp_path = ckpt_dir / "hyperparameters.yaml"
    if not hp_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {ckpt_dir} has no harmful_expert_indices and no "
            "hyperparameters.yaml to derive surgery parameters from. "
            "Run surgery manually first."
        )
    hp = yaml.safe_load(hp_path.read_text())
    if not isinstance(hp, dict):
        raise TypeError(f"Expected mapping in {hp_path}")
    num_harmful_experts = hp.get("num_harmful_experts") or 0
    num_harmful_attn_heads = hp.get("num_harmful_attn_heads") or 0
    epsilon = hp.get("epsilon")
    seed = hp.get("seed", 42)

    if not num_harmful_experts or epsilon is None:
        raise ValueError(
            f"Cannot auto-run surgery for {ckpt_dir}: "
            "hyperparameters.yaml missing num_harmful_experts or epsilon."
        )

    print(
        f"Checkpoint has no harmful experts — running surgery "
        f"(experts={num_harmful_experts}, attn_heads={num_harmful_attn_heads}, "
        f"seed={seed}, epsilon={epsilon}) …"
    )
    from safemoe import surgery as _surgery_mod
    surgery_output = _surgery_mod.setup(
        base_checkpoint=ckpt_dir,
        num_harmful_experts=num_harmful_experts,
        num_harmful_attn_heads=num_harmful_attn_heads,
        seed=seed,
        epsilon=epsilon,
    )
    return surgery_output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class _EvalSession:
    fabric: L.Fabric
    model: Any
    config: SafeMoEConfig
    val_loaders: dict[str, object]


def _create_eval_session(
    ckpt_dir: Path,
    data_mock=None,
    accelerator: str = "auto",
    devices: int | str = 1,
    num_nodes: int = 1,
    precision: Optional[str] = None,
) -> _EvalSession:
    ckpt_dir = Path(ckpt_dir)
    fabric = _make_eval_fabric(devices=devices, num_nodes=num_nodes, accelerator=accelerator, precision=precision)
    model, config = _load_model(ckpt_dir, fabric=fabric)
    val_loaders = _get_val_loaders(ckpt_dir, config, data_mock)
    return _EvalSession(fabric=fabric, model=model, config=config, val_loaders=val_loaders)


def _compute_perplexity_metrics(session: _EvalSession) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split in ["D_std", "D_harmful"]:
        loader = _prepare_eval_loader(session.fabric, session.val_loaders[split])
        loss = validate(session.fabric, session.model, loader, max_iters=sys.maxsize, verbose=False)
        metrics[f"val_ppl_{split}"] = math.exp(loss.item())
    return metrics


def _aggregate_routing_results(fabric: L.Fabric, routing_results: dict[str, float]) -> dict[str, float]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return routing_results

    aggregated: dict[str, float] = {}
    split_names = {
        key.removeprefix("dispatch_count_")
        for key in routing_results
        if key.startswith("dispatch_count_")
    }
    for split_name in split_names:
        dispatch_count = int(routing_results[f"dispatch_count_{split_name}"])
        harmful_dispatches = int(round(routing_results[f"routing_harmful_frac_{split_name}"] * dispatch_count))
        dispatch_tensor = torch.tensor(dispatch_count, device=fabric.device, dtype=torch.long)
        harmful_tensor = torch.tensor(harmful_dispatches, device=fabric.device, dtype=torch.long)
        torch.distributed.all_reduce(dispatch_tensor)
        torch.distributed.all_reduce(harmful_tensor)
        total_dispatches = int(dispatch_tensor.item())
        aggregated[f"dispatch_count_{split_name}"] = total_dispatches
        aggregated[f"routing_harmful_frac_{split_name}"] = harmful_tensor.item() / max(total_dispatches, 1)
    return aggregated


def _compute_routing_metrics(session: _EvalSession) -> dict[str, float]:
    collector = RoutingObservabilityCollector(session.model, session.config.harmful_expert_indices)
    split_runners = {
        split_name: (
            lambda loader=loader: validate(
                session.fabric,
                session.model,
                _prepare_eval_loader(session.fabric, loader),
                max_iters=sys.maxsize,
                verbose=False,
            )
        )
        for split_name, loader in session.val_loaders.items()
    }
    routing_results = collector.collect_splits(split_runners)
    routing_results = _aggregate_routing_results(session.fabric, routing_results)

    routing_metrics = dict(routing_results)
    if {
        "routing_harmful_frac_D_std",
        "routing_harmful_frac_D_harmful",
    }.issubset(routing_metrics):
        routing_metrics["routing_margin"] = (
            routing_metrics["routing_harmful_frac_D_harmful"]
            - routing_metrics["routing_harmful_frac_D_std"]
        )
    return routing_metrics


def _evaluate_checkpoint(
    ckpt_dir: Path,
    data_mock=None,
    accelerator: str = "auto",
    devices: int | str = 1,
    num_nodes: int = 1,
    precision: Optional[str] = None,
) -> tuple[_EvalSession, dict[str, dict[str, float]]]:
    session = _create_eval_session(
        ckpt_dir,
        data_mock=data_mock,
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
    )
    checkpoint_metrics: dict[str, dict[str, float]] = {
        "perplexity": _compute_perplexity_metrics(session),
        "routing": _compute_routing_metrics(session),
    }
    return session, checkpoint_metrics


def _persist_checkpoint_metrics(
    session: _EvalSession,
    ckpt_dir: Path,
    checkpoint_metrics: dict[str, dict[str, float]],
) -> Optional[Path]:
    if session.fabric.global_rank != 0:
        return None
    return _write_checkpoint_metrics(ckpt_dir, checkpoint_metrics)


def _build_combined_metrics_record(
    original_ckpt_dir: Path,
    original_metrics: dict[str, dict[str, float]],
    ablated_ckpt_dir: Optional[Path] = None,
    ablated_metrics: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "original": original_metrics,
    }
    if ablated_ckpt_dir is None or ablated_metrics is None:
        return combined

    combined["ablated"] = {
        "checkpoint": _checkpoint_reference(ablated_ckpt_dir, _find_results_root(original_ckpt_dir)),
        "metrics": ablated_metrics,
    }
    return combined


def _persist_combined_metrics(
    session: _EvalSession,
    original_ckpt_dir: Path,
    original_metrics: dict[str, dict[str, float]],
    ablated_ckpt_dir: Optional[Path] = None,
    ablated_metrics: Optional[dict[str, dict[str, float]]] = None,
) -> Optional[Path]:
    if session.fabric.global_rank != 0:
        return None
    return _write_checkpoint_metrics(
        original_ckpt_dir,
        _build_combined_metrics_record(
            original_ckpt_dir,
            original_metrics,
            ablated_ckpt_dir=ablated_ckpt_dir,
            ablated_metrics=ablated_metrics,
        ),
    )


def _print_summary_table(
    original_metrics: dict[str, dict[str, float]],
    ablated_metrics: Optional[dict[str, dict[str, float]]] = None,
) -> None:
    has_ablated = ablated_metrics is not None
    if has_ablated:
        print(f"\n{'Metric':<16} {'Split':<14} {'Original':>12} {'Ablated':>12}")
        print("-" * 57)
    else:
        print(f"\n{'Metric':<16} {'Split':<14} {'Original':>12}")
        print("-" * 44)

    rows = [
        ("Perplexity", "D_std", "perplexity", "val_ppl_D_std", 2),
        ("Perplexity", "D_harmful", "perplexity", "val_ppl_D_harmful", 2),
        ("Routing frac", "D_std", "routing", "routing_harmful_frac_D_std", 4),
        ("Routing frac", "D_harmful", "routing", "routing_harmful_frac_D_harmful", 4),
        ("Routing margin", "overall", "routing", "routing_margin", 4),
    ]

    for label, split, group, key, digits in rows:
        original_value = original_metrics[group][key]
        if has_ablated:
            ablated_value = ablated_metrics[group][key]
            print(
                f"{label:<16} {split:<14} "
                f"{original_value:>12.{digits}f} {ablated_value:>12.{digits}f}"
            )
        else:
            print(f"{label:<16} {split:<14} {original_value:>12.{digits}f}")


def evaluate_checkpoint(
    original_ckpt_dir: Path,
    ablated_ckpt_dir: Optional[Path] = None,
    data_mock=None,
    accelerator: str = "auto",
    devices: int | str = 1,
    num_nodes: int = 1,
    precision: Optional[str] = None,
) -> dict:
    """Evaluate all checkpoint metrics and persist them to checkpoints/results.json."""
    original_ckpt_dir = Path(original_ckpt_dir)
    ablated_ckpt_dir = Path(ablated_ckpt_dir) if ablated_ckpt_dir is not None else None

    original_session: Optional[_EvalSession] = None
    ablated_session: Optional[_EvalSession] = None
    try:
        original_session, original_metrics = _evaluate_checkpoint(
            original_ckpt_dir,
            data_mock=data_mock,
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
        )
        ablated_metrics: Optional[dict[str, dict[str, float]]] = None
        results_path: Optional[Path] = None

        if ablated_ckpt_dir is not None:
            ablated_session, ablated_metrics = _evaluate_checkpoint(
                ablated_ckpt_dir,
                data_mock=data_mock,
                accelerator=accelerator,
                devices=devices,
                num_nodes=num_nodes,
                precision=precision,
            )
            results_path = _persist_combined_metrics(
                original_session,
                original_ckpt_dir,
                original_metrics,
                ablated_ckpt_dir=ablated_ckpt_dir,
                ablated_metrics=ablated_metrics,
            )
        else:
            results_path = _persist_checkpoint_metrics(original_session, original_ckpt_dir, original_metrics)

        _print_summary_table(original_metrics, ablated_metrics)
        if results_path is not None:
            print(f"\nResults written to {results_path}")
        return _build_combined_metrics_record(
            original_ckpt_dir,
            original_metrics,
            ablated_ckpt_dir=ablated_ckpt_dir,
            ablated_metrics=ablated_metrics,
        )
    finally:
        _teardown_eval_session(ablated_session)
        _teardown_eval_session(original_session)


def evaluate_routing(
    ckpt_dir: Path,
    data_mock=None,
    accelerator: str = "auto",
    devices: int | str = 1,
    num_nodes: int = 1,
    precision: Optional[str] = None,
) -> dict:
    """Evaluate routing metrics and persist the full checkpoint results."""
    ckpt_dir = Path(ckpt_dir)
    session: Optional[_EvalSession] = None
    try:
        session, checkpoint_metrics = _evaluate_checkpoint(
            ckpt_dir,
            data_mock=data_mock,
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
        )
        routing_metrics = checkpoint_metrics["routing"]
        results_path = _persist_checkpoint_metrics(session, ckpt_dir, checkpoint_metrics)
        _print_summary_table(checkpoint_metrics)
        if results_path is not None:
            print(f"\nResults written to {results_path}")
        return routing_metrics
    finally:
        _teardown_eval_session(session)


def evaluate_cli(
    original: Path,
    ablated: Optional[Path] = None,
    accelerator: str = "auto",
    devices: int | str = "auto",
    num_nodes: int = 1,
    precision: Optional[str] = None,
) -> None:
    """CLI: python -m safemoe evaluate <ckpt_dir> [--ablated <path>] [--devices auto].

    All supported evaluation metrics are always computed and persisted.
    When *ablated* is omitted the ablated checkpoint is auto-located at
    ``<ckpt_dir>/ablated/``.  If that directory has no ``lit_model.pth`` the
    ablation is run automatically before evaluation begins.
    """
    original = Path(original)
    original = _ensure_surgery_checkpoint(original)
    devices = parse_devices(devices)

    if ablated is not None:
        ablated = Path(ablated)
    else:
        candidate = original / "ablated"
        if not (candidate / "lit_model.pth").exists():
            print(f"No ablated checkpoint at {candidate} — running ablation first …")
            from safemoe.ablate import ablate
            ablate(original)
        ablated = candidate

    original_session: Optional[_EvalSession] = None
    ablated_session: Optional[_EvalSession] = None
    try:
        original_session, original_metrics = _evaluate_checkpoint(
            original,
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
        )
        ablated_metrics: Optional[dict[str, dict[str, float]]] = None
        results_path: Optional[Path] = None

        if ablated is not None:
            ablated_session, ablated_metrics = _evaluate_checkpoint(
                ablated,
                accelerator=accelerator,
                devices=devices,
                num_nodes=num_nodes,
                precision=precision,
            )
            results_path = _persist_combined_metrics(
                original_session,
                original,
                original_metrics,
                ablated_ckpt_dir=ablated,
                ablated_metrics=ablated_metrics,
            )
        else:
            results_path = _persist_checkpoint_metrics(original_session, original, original_metrics)

        _print_summary_table(original_metrics, ablated_metrics)
        if results_path is not None:
            print(f"\nResults written to {results_path}")
    finally:
        _teardown_eval_session(ablated_session)
        _teardown_eval_session(original_session)

