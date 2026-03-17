"""safemoe/__main__.py — CLI entry point for python -m safemoe {pretrain,ablate,evaluate}.

Usage:
    python -m safemoe pretrain --help
    python -m safemoe pretrain --config safemoe/configs/safemoe-tinystories.yaml
    python -m safemoe ablate <ckpt_dir>
    python -m safemoe evaluate <ckpt_dir> [--ablated <path>] [--routing]
"""
from jsonargparse import CLI

from safemoe.pretrain import setup as pretrain_fn
from safemoe.ablate import setup as ablate_fn
from safemoe.evaluate import setup as evaluate_fn

PARSER_DATA = {
    "pretrain": pretrain_fn,
    "ablate": ablate_fn,
    "evaluate": evaluate_fn,
}


def main() -> None:
    CLI(PARSER_DATA)


if __name__ == "__main__":
    main()
