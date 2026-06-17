"""`vizaudit-overlay` console-script entrypoint."""

from __future__ import annotations

import argparse
import logging

from vizaudit.overlay.config import load_config
from vizaudit.overlay.dataset_watcher import resolve_dataset_root
from vizaudit.overlay.session import run_session


def _parse_connect(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"--connect must be HOST:PORT, got {value!r}")
    return host, int(port)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guided data-collection overlay: shows per-episode placement targets in Rerun."
    )
    parser.add_argument("--config", required=True, help="Path to a pattern/object YAML config.")
    parser.add_argument(
        "--connect",
        required=True,
        type=_parse_connect,
        help="Shared Rerun gRPC server address, e.g. 127.0.0.1:9876.",
    )
    parser.add_argument(
        "--dataset.repo_id", dest="repo_id", required=True, help="Dataset repo id being recorded."
    )
    parser.add_argument(
        "--dataset.root",
        dest="root",
        default=None,
        help="Override dataset root (matches lerobot-record's --dataset.root).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = load_config(args.config)
    dataset_root = resolve_dataset_root(args.repo_id, args.root)
    host, port = args.connect

    try:
        run_session(config, dataset_root, host, port)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped.")


if __name__ == "__main__":
    main()
