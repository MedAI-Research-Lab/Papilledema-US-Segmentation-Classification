"""Fresh-release evaluation using the original, fully validated test gate.

The engine returns state/test_access_opened.json while the historical receipt
validator requires protocol/test_access.json. This entry point supplies a
byte-identical alias after the existing 20-lock/current-code/current-config
checks pass. It changes no prediction, threshold, ROI, statistic or receipt
validation rule. Use the historical recovery script only in its original archive.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
from typing import Mapping, Any

from . import protocol
from .config import DEFAULT_CONFIG_PATH, load_config

HISTORICAL_CODE_SHA256 = "914ec1505a0d417b8b6766b3663160673b94d5ce107da68f81d3142bbf493776"


def create_gate_alias(cfg: Mapping[str, Any]) -> Path:
    """Revalidate the canonical gate, then create or verify its immutable alias."""
    root = protocol.output_root(cfg).resolve()
    canonical = (root / "state/test_access_opened.json").resolve()
    alias = (root / "protocol/test_access.json").resolve()
    if not canonical.is_relative_to(root) or not alias.is_relative_to(root):
        raise protocol.ProtocolGateError("Test-gate paths must remain inside the configured output")
    if protocol.code_fingerprint()["sha256"] == HISTORICAL_CODE_SHA256:
        raise protocol.ProtocolGateError("This is a fresh-release entry point, not historical archive recovery")
    # All prepare, preflight and development receipts still undergo the original
    # hash and artifact checks before any test gate can be opened or reused.
    protocol.verify_stage_receipt(cfg, "prepare")
    if alias.exists() and not canonical.is_file():
        raise protocol.ProtocolGateError("An orphan compatibility gate cannot open test access")
    opened = protocol.open_test_access(cfg).resolve()
    if opened != canonical:
        raise protocol.ProtocolGateError("Unexpected canonical test gate returned by the protocol")
    payload = canonical.read_bytes()
    if alias.exists():
        if not alias.is_file() or alias.read_bytes() != payload:
            raise protocol.ProtocolGateError("Refusing to overwrite a divergent test-gate alias")
    else:
        alias.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation never replaces an existing alias. A partial file
        # after interruption remains fail-closed at the next invocation.
        with alias.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    if alias.read_bytes() != payload or canonical.read_bytes() != payload:
        raise protocol.ProtocolGateError("Test-gate bytes changed during alias verification")
    return alias


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("a+b")
    except OSError as error:
        raise protocol.ProtocolGateError("A study launcher already owns the execution lock") from error
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise protocol.ProtocolGateError("A study launcher already owns the execution lock") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise protocol.ProtocolGateError("A study launcher already owns the execution lock") from error
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


@contextmanager
def gate_compatibility():
    from . import engine
    previous = engine.open_test_access
    engine.open_test_access = create_gate_alias
    try:
        yield
    finally:
        engine.open_test_access = previous


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "summarize", "audit"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    from .__main__ import main as original_main
    delegated = [args.command, "--config", args.config, "--model", args.model]
    if args.seed is not None:
        delegated += ["--seed", str(args.seed)]
    if args.dry_run:
        return original_main([*delegated, "--dry-run"])
    # Establish identity before creating even an orchestration lock directory.
    protocol.verify_stage_receipt(cfg, "prepare")
    with ExitStack() as stack:
        folder = protocol.output_root(cfg) / "orchestration"
        for name in ("five_seed_core_exclusive.lock", "clean_study_exclusive.lock"):
            stack.enter_context(_exclusive_lock(folder / name))
        stack.enter_context(gate_compatibility())
        return original_main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
