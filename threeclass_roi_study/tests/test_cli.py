from __future__ import annotations

import io
import json

from threeclass_roi_study.__main__ import _write_json, build_parser


def test_parser_exposes_all_protocol_stages() -> None:
    parser = build_parser()
    commands = {
        action.dest: set(action.choices)
        for action in parser._actions
        if action.dest == "command"
    }["command"]
    assert commands == {
        "validate-config",
        "plan",
        "prepare",
        "import-upstream",
        "preflight",
        "train",
        "lock",
        "open-test",
        "evaluate",
        "summarize",
        "audit",
        "run-core",
    }


def test_unit_selectors_accept_locked_values_and_all() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--model",
            "all",
            "--seed",
            "9103",
            "--strategy",
            "standardized_resnet18",
        ]
    )
    assert args.command == "evaluate"
    assert args.model == "all"
    assert args.seed == "9103"
    assert args.strategy == "standardized_resnet18"


def test_unit_selectors_are_required() -> None:
    parser = build_parser()
    try:
        parser.parse_args(["train"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("train unexpectedly accepted missing unit selectors")


def test_cli_output_is_strict_json_when_metric_is_nonfinite() -> None:
    stream = io.StringIO()
    _write_json(stream, {"finite": 0.5, "undefined": float("nan")})
    assert json.loads(stream.getvalue()) == {"finite": 0.5, "undefined": None}
    assert "NaN" not in stream.getvalue()
