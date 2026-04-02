"""safemoe/__main__.py — CLI entry point for python -m safemoe {pretrain,ablate,evaluate,surgery}.

Usage:
    python -m safemoe pretrain --help
    python -m safemoe pretrain --config safemoe/configs/safemoe-tinystories.yaml
    python -m safemoe ablate <ckpt_dir>
    python -m safemoe evaluate <ckpt_dir> [--ablated <path>]
    python -m safemoe surgery --help
"""
from jsonargparse import CLI

from litgpt.safemoe.ablate import setup as ablate_fn
from litgpt.safemoe.evaluate import evaluate_cli as evaluate_fn
from litgpt.safemoe.pretrain import setup as pretrain_fn
from litgpt.safemoe.surgery import setup as surgery_fn

PARSER_DATA = {
    "pretrain": pretrain_fn,
    "ablate": ablate_fn,
    "evaluate": evaluate_fn,
    "surgery": surgery_fn,
}


def main() -> None:
    CLI(PARSER_DATA)


if __name__ == "__main__":
    main()
