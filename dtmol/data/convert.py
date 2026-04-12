"""CLI entry point for converting datasets to unified LMDB format.

Usage:
    python -m dtmol.data.convert --source <name> --input <path> --output <path> --split-strategy <scaffold|random>
"""

from __future__ import annotations

import argparse
import logging
import sys

from dtmol.data.converters.base import BaseConverter

logger = logging.getLogger(__name__)

# Registry of available converters (populated by individual converter modules)
CONVERTER_REGISTRY: dict[str, type[BaseConverter]] = {}


def register_converter(name: str, cls: type[BaseConverter]) -> None:
    """Register a converter class under the given source name."""
    import sys
    CONVERTER_REGISTRY[name] = cls
    # When running as `python -m dtmol.data.convert`, the __main__ module has its
    # own CONVERTER_REGISTRY. Sync registrations to __main__ so main() can see them.
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(main_mod, "CONVERTER_REGISTRY"):
        main_registry = getattr(main_mod, "CONVERTER_REGISTRY")
        if main_registry is not CONVERTER_REGISTRY:
            main_registry[name] = cls


def _import_converters() -> None:
    """Import all converter modules to trigger registration."""
    # Each converter module calls register_converter() at import time.
    # Add imports here as converters are implemented:
    from dtmol.data.converters import pdbbind  # noqa: F401
    from dtmol.data.converters import qm9  # noqa: F401
    from dtmol.data.converters import ani2x  # noqa: F401
    from dtmol.data.converters import spice2  # noqa: F401
    from dtmol.data.converters import misato  # noqa: F401
    from dtmol.data.converters import pdb_apo  # noqa: F401
    from dtmol.data.converters import irc  # noqa: F401


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert datasets to unified LMDB format for dtmol training.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Dataset source name (e.g. pdbbind, qm9, ani2x, spice2, misato, pdb_apo, irc).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory for unified LMDB files.",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="random",
        choices=["random", "scaffold"],
        help="Split strategy for train/valid/test (default: random).",
    )
    # MISATO-specific arguments
    parser.add_argument(
        "--dedup-list",
        type=str,
        default=None,
        help="Path to file with PDB IDs to skip (one per line), for deduplication with PDBBind.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        default=False,
        help="Disable deduplication even if --dedup-list is provided.",
    )
    parser.add_argument(
        "--affinity-index",
        type=str,
        default=None,
        help="Path to PDBBind index file for merging binding affinities (MISATO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _import_converters()

    if args.source not in CONVERTER_REGISTRY:
        available = ", ".join(sorted(CONVERTER_REGISTRY)) or "(none registered)"
        logger.error("Unknown source %r. Available: %s", args.source, available)
        sys.exit(1)

    converter = CONVERTER_REGISTRY[args.source]()

    # Build extra kwargs for converters that accept them (e.g. MISATO)
    extra_kwargs: dict[str, object] = {}
    if args.dedup_list is not None:
        extra_kwargs["dedup_list"] = args.dedup_list
    if args.no_dedup:
        extra_kwargs["no_dedup"] = True
    if args.affinity_index is not None:
        extra_kwargs["affinity_index"] = args.affinity_index

    converter.convert(args.input, args.output, args.split_strategy, **extra_kwargs)
    logger.info("Done.")


if __name__ == "__main__":
    main()
