"""Command-line interface for the locked three-class predicted-ROI study.

The CLI deliberately mirrors the protocol stages exposed by :mod:`engine`.
Unit commands require explicit model/seed/strategy selectors (or ``all``),
while the ``run-core`` command executes the complete gated workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from .config import (
    CLASSIFIER_STRATEGIES,
    DEFAULT_CONFIG_PATH,
    EXPECTED_MODELS,
    EXPECTED_SEEDS,
    ConfigError,
    load_config,
    resolve_project_path,
)


MODEL_CHOICES = ("all", *EXPECTED_MODELS)
SEED_CHOICES = ("all", *(str(seed) for seed in EXPECTED_SEEDS))
STRATEGY_CHOICES = ("all", *CLASSIFIER_STRATEGIES)


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Locked study configuration JSON (default: package configuration).",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print a full Python traceback when a stage fails.",
    )


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)


def _add_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", required=True, choices=SEED_CHOICES)


def _add_strategy(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", required=True, choices=STRATEGY_CHOICES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m threeclass_roi_study",
        description=(
            "Leakage-safe, frozen-upstream three-class predicted-ROI study "
            "with a global validation-lock test gate."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-config", help="Validate the locked scientific configuration."
    )
    _add_runtime_options(validate)
    validate.add_argument(
        "--verify-frozen-sources",
        action="store_true",
        help="Also hash-check frozen source anchors and split files.",
    )

    plan = subparsers.add_parser(
        "plan", help="Print the exact locked execution matrix without running it."
    )
    _add_runtime_options(plan)

    prepare = subparsers.add_parser(
        "prepare", help="Create provenance snapshots and the output namespace."
    )
    _add_runtime_options(prepare)

    import_upstream = subparsers.add_parser(
        "import-upstream",
        help="Audit and receipt frozen development-only ROI artifacts.",
    )
    _add_runtime_options(import_upstream)
    _add_model(import_upstream)
    _add_seed(import_upstream)

    preflight = subparsers.add_parser(
        "preflight", help="Run GPU, gradient, shape, and outside-ROI checks."
    )
    _add_runtime_options(preflight)
    _add_model(preflight)

    train = subparsers.add_parser(
        "train", help="Train new three-logit classifiers using development data only."
    )
    _add_runtime_options(train)
    _add_model(train)
    _add_seed(train)
    _add_strategy(train)

    lock = subparsers.add_parser(
        "lock", help="Calibrate on validation and write an immutable validation lock."
    )
    _add_runtime_options(lock)
    _add_model(lock)
    _add_seed(lock)
    _add_strategy(lock)

    open_test = subparsers.add_parser(
        "open-test",
        help="Open global test access only after all 40 validation locks verify.",
    )
    _add_runtime_options(open_test)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate locked classifiers after the global test gate."
    )
    _add_runtime_options(evaluate)
    _add_model(evaluate)
    _add_seed(evaluate)
    _add_strategy(evaluate)

    summarize = subparsers.add_parser(
        "summarize", help="Generate five-seed Q1 tables, figures, and claim limits."
    )
    _add_runtime_options(summarize)

    audit = subparsers.add_parser(
        "audit", help="Verify every receipt, artifact hash, table, and gate."
    )
    _add_runtime_options(audit)

    run_core = subparsers.add_parser(
        "run-core", help="Execute the complete 4-model, 5-seed, 2-strategy workflow."
    )
    _add_runtime_options(run_core)

    return parser


def _selected_models(value: str, cfg: dict[str, Any]) -> tuple[str, ...]:
    return tuple(cfg["models"]) if value == "all" else (value,)


def _selected_seeds(value: str, cfg: dict[str, Any]) -> tuple[int, ...]:
    return (
        tuple(int(seed) for seed in cfg["split_seeds"])
        if value == "all"
        else (int(value),)
    )


def _selected_strategies(value: str, cfg: dict[str, Any]) -> tuple[str, ...]:
    return (
        tuple(cfg["classifier"]["strategy_order"])
        if value == "all"
        else (value,)
    )


def _plan(cfg: dict[str, Any]) -> dict[str, Any]:
    models = tuple(cfg["models"])
    seeds = tuple(int(seed) for seed in cfg["split_seeds"])
    strategies = tuple(cfg["classifier"]["strategy_order"])
    return {
        "status": "validated",
        "study_id": cfg["study_id"],
        "protocol_version": cfg["protocol_version"],
        "analysis_status": cfg["analysis_status"],
        "models": list(models),
        "seeds": list(seeds),
        "strategies": list(strategies),
        "development_import_units": len(models) * len(seeds),
        "validation_locks_required_before_test": (
            len(models) * len(seeds) * len(strategies)
        ),
        "evaluation_units": len(models) * len(seeds) * len(strategies),
        "output": str(resolve_project_path(cfg["output"])),
        "test_gate": "closed_until_all_validation_locks_verify",
        "confirmatory_claim_allowed": False,
        "prior_test_use_disclosure": cfg["prior_test_use"]["required_disclosure"],
    }


def _execute(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    command = args.command
    if command == "validate-config":
        return {
            "status": "validated",
            "study_id": cfg["study_id"],
            "protocol_version": cfg["protocol_version"],
            "config_sha256": cfg["config_sha256"],
            "frozen_sources_verified": bool(args.verify_frozen_sources),
        }
    if command == "plan":
        return _plan(cfg)

    # Heavy framework imports are intentionally deferred so --help and config
    # inspection remain fast and work even on non-GPU orchestration hosts.
    from . import engine
    from .protocol import open_test_access

    if command == "prepare":
        return engine.prepare(cfg)
    if command == "open-test":
        sentinel = open_test_access(cfg)
        return {"status": "complete", "test_access_sentinel": str(sentinel)}
    if command == "summarize":
        return engine.summarize(cfg)
    if command == "audit":
        return engine.audit(cfg)
    if command == "run-core":
        return engine.run_core(cfg)

    models = _selected_models(args.model, cfg)
    results: list[dict[str, Any]] = []
    if command == "preflight":
        for model in models:
            results.append(
                {"model": model, "result": engine.preflight(cfg, model)}
            )
    elif command == "import-upstream":
        for model in models:
            for seed in _selected_seeds(args.seed, cfg):
                results.append(
                    {
                        "model": model,
                        "seed": seed,
                        "result": engine.import_upstream(cfg, model, seed),
                    }
                )
    elif command in {"train", "lock", "evaluate"}:
        operation = {
            "train": engine.train_classifier,
            "lock": engine.lock_validation,
            "evaluate": engine.evaluate,
        }[command]
        for model in models:
            for seed in _selected_seeds(args.seed, cfg):
                for strategy in _selected_strategies(args.strategy, cfg):
                    results.append(
                        {
                            "model": model,
                            "seed": seed,
                            "strategy": strategy,
                            "result": operation(cfg, model, seed, strategy),
                        }
                    )
    else:  # pragma: no cover - argparse prevents this branch.
        raise RuntimeError(f"Unhandled command: {command}")
    return {"status": "complete", "command": command, "results": results}


def _json_safe(value: Any) -> Any:
    """Return strict-JSON-safe output without hiding scientific non-finites."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        observed = float(value)
        return observed if math.isfinite(observed) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(stream: Any, payload: Any) -> None:
    json.dump(
        _json_safe(payload),
        stream,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
        allow_nan=False,
    )
    stream.write("\n")
    stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(
            args.config,
            verify_frozen_sources=bool(
                getattr(args, "verify_frozen_sources", False)
            ),
        )
        result = _execute(args, cfg)
        _write_json(sys.stdout, result)
        return 0
    except KeyboardInterrupt:
        _write_json(
            sys.stderr,
            {"status": "interrupted", "command": args.command},
        )
        return 130
    except Exception as exc:  # The runner relies on a non-zero process status.
        _write_json(
            sys.stderr,
            {
                "status": "failed",
                "command": args.command,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        if args.traceback:
            traceback.print_exc(file=sys.stderr)
        return 2 if isinstance(exc, ConfigError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
