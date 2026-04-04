#!/usr/bin/env python3
"""
compute_checksums.py
====================
Compute SHA-256 checksums for all model artifacts listed in
models/model_registry.yaml and write them back in-place.

Usage:
    python scripts/compute_checksums.py

Requires: pyyaml  (pip install pyyaml)
"""

import hashlib
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY  = REPO_ROOT / "models" / "model_registry.yaml"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not REGISTRY.exists():
        print(f"ERROR: Registry not found at {REGISTRY}")
        sys.exit(1)

    with open(REGISTRY) as f:
        registry = yaml.safe_load(f)

    updated = 0
    for model_name, model_info in registry["models"].items():
        artifact_rel = model_info.get("artifact")
        if not artifact_rel:
            continue
        artifact_path = REPO_ROOT / artifact_rel
        if not artifact_path.exists():
            print(f"  SKIP  {model_name}: artifact not found ({artifact_path})")
            continue
        checksum = sha256_file(artifact_path)
        model_info.setdefault("checksums", {})["sha256_artifact"] = checksum
        print(f"  OK    {model_name}: {checksum[:16]}...")
        updated += 1

    # Also checksum the shared training dataset
    train_data_rel = "data/processed/train_balanced.csv"
    train_data_path = REPO_ROOT / train_data_rel
    if train_data_path.exists():
        train_sha = sha256_file(train_data_path)
        for model_info in registry["models"].values():
            model_info.setdefault("checksums", {})["sha256_train_data"] = train_sha
        print(f"  OK    train_balanced.csv: {train_sha[:16]}...")
    else:
        print(f"  SKIP  train_balanced.csv not found (run preprocessing first)")

    from datetime import datetime, timezone
    registry["_checksums_last_updated"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    with open(REGISTRY, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n✅ Checksums written for {updated} model(s) → {REGISTRY.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
