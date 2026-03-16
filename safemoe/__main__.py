"""safemoe/__main__.py — CLI entry point for python -m safemoe pretrain.

Usage:
    python -m safemoe pretrain --help
    python -m safemoe pretrain --config safemoe/configs/safemoe-tinystories.yaml
"""
from jsonargparse import CLI

from safemoe.pretrain import setup as pretrain_fn

PARSER_DATA = {"pretrain": pretrain_fn}


def main() -> None:
    CLI(PARSER_DATA)


if __name__ == "__main__":
    main()
