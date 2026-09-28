"""Command-line workflow for the strict predicted-ROI experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("YOLO_OFFLINE", "true")

from .config import DEFAULT_CONFIG_PATH, load_config, verify_locked_sources
from .protocol import (
    ProtocolGateError,
    assert_prerequisites,
    audit_state,
    select_models_and_seeds,
    stage_receipt_path,
    verify_stage_receipt,
)


COMMANDS = (
    "prepare",
    "preflight",
    "train-segmenters",
    "build-rois",
    "train-classifiers",
    "lock",
    "evaluate",
    "summarize",
    "lock-ablations",
    "evaluate-ablations",
    "audit",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False), flush=True)


def _runner_for(stage: str) -> Callable[..., Mapping[str, Any]]:
    # Heavy clinical dependencies are imported only after configuration and
    # prerequisite gates have passed.
    from . import engine

    names = {
        "prepare": "prepare",
        "preflight": "preflight",
        "train-segmenters": "train_segmenter",
        "build-rois": "build_rois",
        "train-classifiers": "train_classifier",
        "lock": "lock_validation",
        "evaluate": "evaluate",
        "summarize": "summarize",
        "lock-ablations": "lock_ablations",
        "evaluate-ablations": "evaluate_ablations",
    }
    function_name = names[stage]
    runner = getattr(engine, function_name, None)
    if not callable(runner):
        raise RuntimeError(f"Engine does not implement required runner: {function_name}")
    return runner


def _verify_runner_result(
    cfg: Mapping[str, Any],
    stage: str,
    result: Mapping[str, Any] | None,
    *,
    model: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise RuntimeError(f"{stage} runner must return a mapping with receipt/artifacts/metadata")
    receipt = verify_stage_receipt(cfg, stage, model=model, seed=seed)
    returned_receipt = result.get("receipt")
    expected_receipt = stage_receipt_path(cfg, stage, model=model, seed=seed)
    if isinstance(returned_receipt, Mapping):
        if (
            returned_receipt.get("stage") != stage
            or returned_receipt.get("model") != model
            or returned_receipt.get("seed") != seed
        ):
            raise RuntimeError(f"{stage} runner returned a receipt with the wrong scope")
    elif returned_receipt is not None and Path(returned_receipt).resolve() != expected_receipt.resolve():
        raise RuntimeError(f"{stage} runner returned the wrong receipt path: {returned_receipt}")
    return {**dict(result), "verified_receipt": str(expected_receipt), "status": "complete"}


def _dry_run_scope(cfg: Mapping[str, Any], stage: str, model: str | None, seed: int | None) -> dict[str, Any]:
    try:
        assert_prerequisites(cfg, stage, model=model, seed=seed)
    except ProtocolGateError as error:
        return {"stage": stage, "model": model, "seed": seed, "ready": False, "reason": str(error)}
    return {"stage": stage, "model": model, "seed": seed, "ready": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m predicted_roi_study",
        description=(
            "Four-segmenter strict predicted-ROI study. Commands are deliberately staged; "
            "there is no automatic all-through-test command."
        ),
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Locked JSON protocol")
    parser.add_argument("--model", default="all", help="One predeclared model or all")
    parser.add_argument("--seed", type=int, help="One of the five predeclared outer split seeds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report gates without starting a model fit or opening test access",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command in {
        "prepare",
        "summarize",
        "lock-ablations",
        "evaluate-ablations",
        "audit",
    } and (args.model != "all" or args.seed is not None):
        parser.error(f"{args.command} is global; do not pass --model or --seed")
    if args.command == "preflight" and args.seed is not None:
        parser.error("preflight is model-scoped; do not pass --seed")
    if args.command in {
        "train-segmenters",
        "build-rois",
        "train-classifiers",
        "lock",
    } and args.seed is None:
        parser.error(
            f"{args.command} requires one explicit --seed in the clean-run protocol"
        )

    if args.command == "audit":
        _print(audit_state(cfg))
        return 0

    if args.command == "prepare":
        if args.dry_run:
            _print(
                {
                    "stage": "prepare",
                    "ready": True,
                    "source_hashes": verify_locked_sources(cfg),
                    "action": "verify and byte-copy five immutable patient split CSV files; no clinical training",
                }
            )
        else:
            result = _runner_for("prepare")(cfg)
            _print(_verify_runner_result(cfg, "prepare", result))
        return 0

    if args.command in {"summarize", "lock-ablations", "evaluate-ablations"}:
        if args.dry_run:
            _print(_dry_run_scope(cfg, args.command, None, None))
            return 0
        assert_prerequisites(cfg, args.command)
        result = _runner_for(args.command)(cfg)
        _print(_verify_runner_result(cfg, args.command, result))
        return 0

    models, seeds = select_models_and_seeds(cfg, args.model, args.seed)
    if args.command == "preflight":
        outputs: list[dict[str, Any]] = []
        runner = None if args.dry_run else _runner_for("preflight")
        for model in models:
            if args.dry_run:
                outputs.append(_dry_run_scope(cfg, "preflight", model, None))
                continue
            assert_prerequisites(cfg, "preflight", model=model)
            result = runner(cfg, model)
            outputs.append(_verify_runner_result(cfg, "preflight", result, model=model))
        _print(outputs)
        return 0

    outputs = []
    runner = None if args.dry_run else _runner_for(args.command)
    for seed in seeds:
        for model in models:
            if args.dry_run:
                outputs.append(_dry_run_scope(cfg, args.command, model, seed))
                continue
            assert_prerequisites(cfg, args.command, model=model, seed=seed)
            result = runner(cfg, model, seed)
            outputs.append(_verify_runner_result(cfg, args.command, result, model=model, seed=seed))
    _print(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
