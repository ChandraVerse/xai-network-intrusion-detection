"""
compute_checksums.py
--------------------
Computes SHA-256 checksums of trained model artifacts and training data,
then writes them back into models/model_registry.yaml.

Run after training all three models:
    python scripts/compute_checksums.py

Updates:
    models/model_registry.yaml  →  checksums.sha256_artifact
                                    checksums.sha256_train_data
                                    status: validated
                                    trained_at: <ISO timestamp>
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REGISTRY_PATH = Path("models/model_registry.yaml")
TRAIN_DATA_PATH = Path("data/processed/train_balanced.csv")

ARTIFACT_MAP = {
    "random_forest": Path("models/random_forest.pkl"),
    "xgboost":       Path("models/xgboost_model.pkl"),
    "lstm":          Path("models/lstm_model.h5"),
}


def sha256(path: Path) -> str | None:
    """Return hex SHA-256 digest of file at path, or None if file missing."""
    if not path.exists():
        log.warning("File not found — skipping checksum: %s", path)
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not REGISTRY_PATH.exists():
        log.error("Registry not found: %s", REGISTRY_PATH)
        return

    with open(REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)

    train_checksum = sha256(TRAIN_DATA_PATH)
    log.info("Train data SHA-256 : %s", train_checksum or "MISSING")

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for key, artifact_path in ARTIFACT_MAP.items():
        artifact_checksum = sha256(artifact_path)
        entry = registry["models"][key]

        entry["checksums"]["sha256_artifact"]   = artifact_checksum
        entry["checksums"]["sha256_train_data"] = train_checksum
        entry["trained_at"] = now_iso

        if artifact_checksum:
            entry["status"] = "trained"
            log.info("%s  →  %s  (status: trained)", key, artifact_checksum[:16] + "...")
        else:
            log.warning("%s  →  artifact missing, status unchanged", key)

    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    log.info("Registry updated → %s", REGISTRY_PATH)


if __name__ == "__main__":
    main()
