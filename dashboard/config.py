"""Dashboard configuration for XAI-NIDS.

Centralises all path constants, default settings, and environment-aware
configuration so ``app.py`` imports from one place.

Environment overrides (optional):
    XAI_NIDS_ROOT        Override the auto-detected repo root path.
    XAI_NIDS_LOG_LEVEL   Set logging level (DEBUG / INFO / WARNING).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root (auto-detected; override via XAI_NIDS_ROOT env var)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # dashboard/
ROOT = Path(os.environ.get("XAI_NIDS_ROOT", str(_THIS_DIR.parent)))

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
MODELS_DIR      = ROOT / "models"
DATA_PROC_DIR   = ROOT / "data" / "processed"
DATA_SAMPLES_DIR= ROOT / "data" / "samples"
SHAP_DATA_DIR   = ROOT / "data" / "shap"
REPORTS_DIR     = ROOT / "reports"

# ---------------------------------------------------------------------------
# Model artifact file paths
# ---------------------------------------------------------------------------
RF_MODEL_PATH    = MODELS_DIR / "random_forest.pkl"
XGB_MODEL_PATH   = MODELS_DIR / "xgboost_model.pkl"
# LSTM is stored as a tar.gz archive containing a Keras SavedModel.
# Extract with tarfile before loading via tf.keras.models.load_model().
LSTM_MODEL_PATH  = MODELS_DIR / "lstm_model.tar.gz"
LSTM_EXTRACT_DIR = MODELS_DIR / "lstm_extracted"
SCALER_PATH      = DATA_PROC_DIR / "scaler.pkl"
ENCODER_PATH     = DATA_PROC_DIR / "label_encoder.pkl"
FEATURE_NAMES_PATH = DATA_PROC_DIR / "feature_names.json"
LABEL_MAP_PATH   = DATA_PROC_DIR / "label_map.json"
SUMMARY_PATH     = MODELS_DIR / "metrics_summary.json"

# ---------------------------------------------------------------------------
# Dashboard defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL        = "Random Forest"
DEFAULT_SPEED        = 3       # flows per batch
DEFAULT_DELAY        = 0.8     # seconds between batches
DEFAULT_CONF_THRESHOLD = 0.5   # minimum confidence to raise alert
MAX_HISTORY          = 500     # cap on in-memory alert history
LIME_SAMPLES         = 3000    # neighbourhood samples for LIME
LIME_TOP_FEATURES    = 10      # features to show in LIME explanation
SHAP_BACKGROUND      = 100     # background samples for DeepExplainer

# ---------------------------------------------------------------------------
# UI colours (kept in sync with app.py CSS vars)
# ---------------------------------------------------------------------------
ATTACK_COLORS: dict[str, str] = {
    "BENIGN":              "#39d353",
    "DDoS":                "#f85149",
    "PortScan":            "#58a6ff",
    "Bot":                 "#bc8cff",
    "Infiltration":        "#ffa657",
    "Web Attack - Brute Force": "#d29922",
    "Web Attack - SQLi":   "#ff7b72",
    "Web Attack - XSS":    "#e3b341",
    "DoS Hulk":            "#f0883e",
    "DoS GoldenEye":       "#e3b341",
    "DoS Slowloris":       "#3fb950",
    "DoS Slowhttptest":    "#56d364",
    "SSH-Patator":         "#79c0ff",
    "FTP-Patator":         "#a5d6ff",
}
DEFAULT_COLOR = "#8b949e"

# ---------------------------------------------------------------------------
# LSTM loader helper
# ---------------------------------------------------------------------------
def load_lstm_model():
    """Extract lstm_model.tar.gz and return a loaded Keras model."""
    import tarfile
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise ImportError("TensorFlow is required to load the LSTM model.") from e

    if not LSTM_EXTRACT_DIR.exists() or not any(LSTM_EXTRACT_DIR.iterdir()):
        LSTM_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(LSTM_MODEL_PATH, "r:gz") as tar:
            tar.extractall(LSTM_EXTRACT_DIR)
    return tf.keras.models.load_model(str(LSTM_EXTRACT_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_level_str = os.environ.get("XAI_NIDS_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, _log_level_str, logging.INFO)
