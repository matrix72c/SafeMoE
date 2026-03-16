"""safemoe/evaluate.py — Per-split perplexity evaluation and routing attribution
analysis (EVAL-01, EVAL-02)."""

import json
import math
import sys
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import yaml
from torch.utils.data import DataLoader

from litgpt.model import GPT
from safemoe.config import SafeMoEConfig
from safemoe.masking import HarmfulParamRegistry
from safemoe.model import SafeMoELayer
from safemoe.pretrain import validate


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_model(ckpt_dir: Path) -> tuple:
    """Load a SafeMoE model from a checkpoint directory.

    Returns (model: GPT, config: SafeMoEConfig).
    The model is loaded from lit_model.pth, set to eval(), and returned raw
    (not wrapped by fabric) so callers can choose their own fabric setup.
    """
    ckpt_dir = Path(ckpt_dir)
    raw = yaml.safe_load((ckpt_dir / "model_config.yaml").read_text())
    config = SafeMoEConfig(**{k: v for k, v in raw.items() if not isinstance(v, dict)})

    model = GPT(config)
    state = torch.load(ckpt_dir / "lit_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, config


def _get_val_loaders(ckpt_dir: Path, config: SafeMoEConfig, data_mock=None) -> dict:
    """Return {"D_std": DataLoader, "D_harmful": DataLoader}.

    If data_mock is provided, delegate to data_mock.val_dataloaders().
    Otherwise construct a real MultiDataLoader from checkpoint config attributes.
    """
    if data_mock is not None:
        return data_mock.val_dataloaders()

    # Real path: construct MultiDataLoader from saved config
    from safemoe.data.datamodule import MultiDataLoader

    x = getattr(config, "x", 0)
    y = getattr(config, "y", 25)
    loader = MultiDataLoader(x=x, y=y)
    loader.connect(tokenizer=None, batch_size=4, max_seq_length=config.block_size)
    loader.setup()
    return loader.val_dataloaders()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_perplexity(
    original_ckpt_dir: Path,
    ablated_ckpt_dir: Optional[Path] = None,
    data_mock=None,
    out_dir: Optional[Path] = None,
) -> dict:
    """Load original (and optionally ablated) checkpoints, run validate() per split,
    print a comparison table, write results.json, and return the results dict.

    Returns:
        {"original": {"val_ppl_D_std": float, "val_ppl_D_harmful": float},
         "ablated":  {...},   # only when ablated_ckpt_dir provided
         "delta":    {...}}   # only when ablated_ckpt_dir provided
    """
    original_ckpt_dir = Path(original_ckpt_dir)

    model_orig, config = _load_model(original_ckpt_dir)
    fabric = L.Fabric(devices=1, accelerator="auto")
    fabric.launch()
    model_orig = fabric.setup(model_orig)

    val_loaders = _get_val_loaders(original_ckpt_dir, config, data_mock)

    # Evaluate original model on each split
    orig_ppls = {}
    for split in ["D_std", "D_harmful"]:
        loader = fabric.setup_dataloaders(val_loaders[split])
        loss = validate(fabric, model_orig, loader, max_iters=sys.maxsize, verbose=False)
        orig_ppls[f"val_ppl_{split}"] = math.exp(loss.item())

    result: dict = {"original": orig_ppls}

    if ablated_ckpt_dir is not None:
        model_abl, _ = _load_model(Path(ablated_ckpt_dir))
        model_abl = fabric.setup(model_abl)

        # Re-acquire val loaders (they may be exhausted after first pass)
        val_loaders_abl = _get_val_loaders(original_ckpt_dir, config, data_mock)

        abl_ppls = {}
        for split in ["D_std", "D_harmful"]:
            loader = fabric.setup_dataloaders(val_loaders_abl[split])
            loss = validate(fabric, model_abl, loader, max_iters=sys.maxsize, verbose=False)
            abl_ppls[f"val_ppl_{split}"] = math.exp(loss.item())

        delta = {k: abl_ppls[k] - orig_ppls[k] for k in orig_ppls}
        result["ablated"] = abl_ppls
        result["delta"] = delta

        # Print comparison table
        print(f"\n{'Split':<14} {'Original':>10} {'Ablated':>10} {'Delta':>10}")
        print("-" * 48)
        for split in ["D_std", "D_harmful"]:
            key = f"val_ppl_{split}"
            orig_v = orig_ppls[key]
            abl_v = abl_ppls[key]
            dlt_v = delta[key]
            sign = "+" if dlt_v >= 0 else ""
            print(f"{split:<14} {orig_v:>10.2f} {abl_v:>10.2f} {sign}{dlt_v:>9.2f}")
    else:
        print(f"\n{'Split':<14} {'Original':>10}")
        print("-" * 26)
        for split in ["D_std", "D_harmful"]:
            key = f"val_ppl_{split}"
            print(f"{split:<14} {orig_ppls[key]:>10.2f}")

    # Write results.json
    write_dir = Path(out_dir) if out_dir is not None else original_ckpt_dir
    write_dir.mkdir(parents=True, exist_ok=True)
    results_path = write_dir / "results.json"
    results_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {results_path}")

    return result


def routing_attribution(ckpt_dir: Path, data_mock=None) -> dict:
    """Install forward hooks on SafeMoELayer, run validate() per split to collect
    expert dispatch indices, compute harmful expert fraction per split.

    Writes routing_attribution.json and TensorBoard scalars to ckpt_dir.

    Returns:
        {"routing_harmful_frac_D_std": float, "routing_harmful_frac_D_harmful": float}
    """
    ckpt_dir = Path(ckpt_dir)

    model, config = _load_model(ckpt_dir)
    fabric = L.Fabric(devices=1, accelerator="auto")
    fabric.launch()
    model = fabric.setup(model)

    val_loaders = _get_val_loaders(ckpt_dir, config, data_mock)
    harmful_indices_set = set(config.harmful_expert_indices)
    safemoe_layers = [m for m in model.modules() if isinstance(m, SafeMoELayer)]

    routing_results = {}

    for split in ["D_std", "D_harmful"]:
        dispatch_all: list = []

        def hook(module, inp, out, _dispatch=dispatch_all):
            if hasattr(module, "_last_indices"):
                _dispatch.extend(module._last_indices.flatten().tolist())

        handles = [layer.register_forward_hook(hook) for layer in safemoe_layers]

        loader = fabric.setup_dataloaders(val_loaders[split])
        validate(fabric, model, loader, max_iters=sys.maxsize, verbose=False)

        for h in handles:
            h.remove()

        harmful_count = sum(1 for idx in dispatch_all if idx in harmful_indices_set)
        frac = harmful_count / max(len(dispatch_all), 1)
        routing_results[f"routing_harmful_frac_{split}"] = float(frac)

    # Write routing_attribution.json
    ra_path = ckpt_dir / "routing_attribution.json"
    ra_path.write_text(json.dumps(routing_results, indent=2))

    # TensorBoard histogram (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = ckpt_dir / "runs" / "routing_analysis"
        writer = SummaryWriter(log_dir=str(log_dir))
        for key, val in routing_results.items():
            writer.add_scalar(key, val, global_step=0)
        writer.close()
        print(f"Routing metrics logged to {log_dir}")
    except ImportError:
        print("TensorBoard not available — routing_attribution.json only")

    # Print routing fractions table
    print(f"\n{'Split':<14} {'Harmful Expert Frac':>20}")
    print("-" * 36)
    for split in ["D_std", "D_harmful"]:
        key = f"routing_harmful_frac_{split}"
        print(f"{split:<14} {routing_results[key]:>20.4f}")

    print(f"\nRouting attribution written to {ra_path}")
    return routing_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def setup(
    original: Path,
    ablated: Optional[Path] = None,
    routing: bool = False,
) -> None:
    """CLI: python -m safemoe evaluate --original <ckpt_dir> [--ablated <path>] [--routing]."""
    original = Path(original)
    if routing:
        routing_attribution(original)
    evaluate_perplexity(original, ablated_ckpt_dir=Path(ablated) if ablated else None)
