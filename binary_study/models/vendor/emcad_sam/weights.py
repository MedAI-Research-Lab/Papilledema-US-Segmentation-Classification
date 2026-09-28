"""Official-weight cache with recorded and checked SHA256; never random fallback."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
import uuid

WEIGHT_SPECS = {
    "emcad_joint": ("pvt_v2_b0.pth", "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth"),
    "sam2_unet": ("sam2_hiera_tiny.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"),
}
LOCK_PATH = Path(__file__).with_name("weights_lock.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_weight(name, cache_dir=None):
    filename, url = WEIGHT_SPECS[name]
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[3] / "weights"
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8")) if LOCK_PATH.exists() else {}
    expected = lock.get(name, {}).get("sha256")
    sidecar = target.with_suffix(target.suffix + ".provenance.json")
    if expected is None and sidecar.exists():
        expected = json.loads(sidecar.read_text(encoding="utf-8"))["sha256"]
    if not target.exists():
        temporary = folder / (filename + "." + uuid.uuid4().hex + ".partial")
        try:
            with urlopen(Request(url, headers={"User-Agent": "USG-research-weights"}), timeout=180) as response:
                with temporary.open("wb") as destination:
                    for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        destination.write(chunk)
            actual = sha256(temporary)
            if expected and actual != expected:
                raise RuntimeError(f"Official weight SHA256 mismatch for {name}: {actual} != {expected}")
            temporary.replace(target)
        except Exception as exc:
            if temporary.exists():
                temporary.unlink()
            raise RuntimeError(f"Could not obtain official pretrained {name} weights from {url}; no random fallback") from exc
    actual = sha256(target)
    if expected and actual != expected:
        raise RuntimeError(f"Cached weight SHA256 mismatch: {target}")
    provenance = {"model": name, "url": url, "path": str(target), "sha256": actual,
                  "bytes": target.stat().st_size, "verified_utc": datetime.now(timezone.utc).isoformat(),
                  "hash_origin": "project-pinned official download" if name in lock else "first official download observation"}
    sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return target, provenance


if __name__ == "__main__":
    observed = {}
    for model_name in WEIGHT_SPECS:
        _, record = official_weight(model_name)
        observed[model_name] = {k: record[k] for k in ("url", "sha256", "bytes")}
        print(json.dumps(record), flush=True)
    LOCK_PATH.write_text(json.dumps(observed, indent=2), encoding="utf-8")
