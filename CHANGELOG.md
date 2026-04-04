# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- Actual trained model weights committed after full CICIDS-2017 run
- Populated `metrics_summary.json` with real benchmark numbers
- `paper/` — research writeup with results and comparison tables
- Notebook cell outputs saved for GitHub rendering

---

## [0.3.0] — 2026-04-04

### Added
- `docs/architecture/` — complete architecture documentation
  - `system_architecture.svg` — full 5-layer SVG system diagram
  - `dataflow.md` — 5 Mermaid diagrams (system, preprocessing, inference sequence, SHAP, Docker)
  - `components.md` — detailed spec for every module
  - `decisions.md` — 7 Architecture Decision Records (ADRs)
  - `README.md` — folder index and quick-reference
- `models/` artifact store
  - `README.md` — artifact guide with per-model specs and loading snippets
  - `model_registry.yaml` — full YAML registry with hyperparameters, XAI methods, checksum slots
  - `metrics_summary.json` — three-model comparison schema (stub, populated after training)
  - `rf_metrics.json`, `xgb_metrics.json`, `lstm_metrics.json` — per-model metric stubs
  - `feature_importance_rf.json` — Gini importance stub
- `data/README.md` — full dataset guide: CICIDS-2017 schema, attack class table, preprocessing steps, feature descriptions
- `data/processed/.gitkeep` + `data/samples/.gitkeep` — ensure dirs exist in fresh clones
- `.github/workflows/ci.yml` — GitHub Actions CI: lint (flake8) + pytest + sample-data smoke test on Python 3.10 and 3.11
- `scripts/compute_checksums.py` — SHA-256 checksum writer for model artifacts → `model_registry.yaml`
- `CHANGELOG.md` — this file
- `SECURITY.md` — responsible disclosure policy
- `paper/draft.md` — research paper draft with full methodology, results tables, and SHAP analysis

---

## [0.2.0] — 2026-03-20

### Added
- `src/models/` — three full model implementations
  - `random_forest.py` — RandomForestClassifier (200 trees, balanced class weight)
  - `xgboost_model.py` — XGBClassifier (300 trees, early stopping, hist method)
  - `lstm_model.py` — two-layer LSTM with sliding-window sequences (time_steps=5)
- `src/preprocessing/` — `cleaner.py`, `scaler.py`, `smote_balancer.py`
- `src/explainability/` — `shap_explainer.py`, `summary_plot.py`, `waterfall.py`
- `src/utils/` — `metrics.py`, `pcap_converter.py`, `report_generator.py`
- `dashboard/app.py` — Streamlit dashboard with model comparison and SHAP visualisation
- `scripts/build_processed.py` — full preprocessing pipeline script
- `scripts/generate_samples.py` — synthetic sample CSV generator (14 attack classes)
- `notebooks/` — 4 Jupyter notebooks: EDA, preprocessing, model training, XAI SHAP
- `tests/` — pytest suite: `test_models.py`, `test_preprocessing.py`, `test_explainability.py`

---

## [0.1.0] — 2026-03-01

### Added
- Initial repository scaffold
- `README.md` — project overview, architecture diagram, quickstart
- `CONTRIBUTING.md` — contribution guidelines
- `Dockerfile` + `docker-compose.yml` — containerised environment
- `requirements.txt` — pinned Python dependencies
- `.gitignore` — excludes raw data, model artifacts, virtual envs
- `LICENSE` — MIT
