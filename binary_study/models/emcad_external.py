"""Load separately licensed, hash-pinned EMCAD source supplied by the user.

This release contains neither EMCAD's decoder nor its distributed PVT source.
Set EMCAD_SOURCE_DIR to the root of the licensed upstream checkout. Importing
this module does not read external source, download files or construct a model.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType

UPSTREAM_COMMIT = "26c9c31f731f749b62c5fe83f44376dac75f3aa8"
SOURCE_FILES = {
    "lib/pvtv2.py": "9711ee6600e3e5274b927a7aecd706ad459173b1c86867aa56522fad5ec3f5f3",
    "lib/decoders.py": "4b36981117376ea005aed8844f4ea9c846f432286695b85c1530797a20157a18",
}
_MODULES: dict[tuple[str, str], ModuleType] = {}
_IMPORT_LOCK = RLock()


def verified_sources() -> dict[str, Path]:
    """Check both upstream files before permitting either file to execute."""
    configured = os.environ.get("EMCAD_SOURCE_DIR")
    if not configured:
        raise RuntimeError(
            "EMCAD requires a separately licensed upstream checkout. Set "
            "EMCAD_SOURCE_DIR to its root, containing lib/pvtv2.py and "
            f"lib/decoders.py at commit {UPSTREAM_COMMIT}. See THIRD_PARTY_NOTICES.md."
        )
    root = Path(configured).expanduser().resolve(strict=True)
    paths = {}
    for relative, expected in SOURCE_FILES.items():
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise RuntimeError(f"EMCAD source must be a regular file inside its checkout: {relative}")
        observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(
                f"EMCAD source SHA256 mismatch for {relative}: expected {expected}, found {observed}. "
                "No upstream module has been executed by this verification."
            )
        paths[relative] = candidate
    return paths


def _module(relative: str) -> ModuleType:
    with _IMPORT_LOCK:
        path = verified_sources()[relative]
        cache_key = (str(path), SOURCE_FILES[relative])
        if cache_key in _MODULES:
            return _MODULES[cache_key]
        identifier = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        module_name = f"_study_external_emcad_{path.stem}_{identifier}"
        specification = importlib.util.spec_from_file_location(module_name, path)
        if specification is None or specification.loader is None:
            raise ImportError(f"Cannot create an import specification for EMCAD {relative}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        try:
            specification.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        _MODULES[cache_key] = module
        return module


def pvt_v2_b0(**kwargs):
    """Construct the pinned upstream four-stage encoder without source copying."""
    return _module("lib/pvtv2.py").pvt_v2_b0(**kwargs)


def build_emcad_decoder(**kwargs):
    return _module("lib/decoders.py").EMCAD(**kwargs)
