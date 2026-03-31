# Contributing to XAI-Based Network Intrusion Detection System

Thank you for your interest in contributing. This guide covers how to improve
ML models, add new attack detectors, enhance the SHAP explainability layer,
or expand the Streamlit dashboard.

---

## Getting Started

```bash
git clone https://github.com/ChandraVerse/xai-network-intrusion-detection.git
cd xai-network-intrusion-detection
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## How to Contribute

### Adding or Improving an ML Model

1. **Create your model file** in `src/models/` following the naming convention:
   `<algorithm_name>_model.py`
2. **Implement the standard interface**:
   - `train(X_train, y_train)` — fit the model
   - `evaluate(X_test, y_test)` — return classification report + confusion matrix
   - `save(path)` — serialize with joblib or Keras `.h5`
3. **Integrate SHAP explainability** — every new model must include a compatible
   SHAP explainer in `src/explainability/shap_explainer.py`
4. **Add to the model comparison notebook** — update `notebooks/03_model_training.ipynb`
   with your model's ROC curve and F1 score
5. **Update the README performance table** in `README.md`

### Model Requirements Checklist

Every new model contribution MUST include:
- [ ] Model training script with `argparse` CLI and `--data` flag
- [ ] Classification report (precision, recall, F1 per class)
- [ ] Confusion matrix heatmap (Seaborn or Plotly)
- [ ] ROC curve with AUC score
- [ ] SHAP explainer initialization and at least one waterfall chart
- [ ] Serialized model artifact saved to `models/`
- [ ] Unit tests in `tests/test_<model_name>.py`
- [ ] Docstrings on all functions and classes

### Improving the SHAP Explainability Layer

- Add new explanation types (e.g., SHAP interaction values, dependence plots) in
  `src/explainability/`
- Ensure all new visualizations are renderable inside the Streamlit dashboard
- Test with both tree-based (RF/XGBoost) and deep learning (LSTM) explainers

### Contributing to the Streamlit Dashboard

- New features go in `dashboard/pages/` as separate Streamlit page modules
- Follow the existing tab naming convention: emoji + descriptive label
- All new pages must be registered in `dashboard/app.py`
- Do not introduce new heavyweight dependencies without updating `requirements.txt`

### Adding Dataset Support

- New dataset loaders go in `src/preprocessing/`
- Must support the same 78-feature output schema as CICIDS-2017 OR include a
  feature mapping layer
- Document the dataset source, license, and download instructions in `README.md`

---

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/new-model-isolation-forest`
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(models):` — new ML model
   - `feat(explainability):` — new SHAP visualization
   - `feat(dashboard):` — new dashboard feature
   - `fix(preprocessing):` — bug fix in pipeline
   - `docs:` — documentation only
   - `refactor:` — code restructure without behavior change
4. Push and open a PR with:
   - Description of what model or feature was added
   - Performance metrics achieved on CICIDS-2017
   - Screenshot of SHAP output or dashboard change (if applicable)
   - Confirmation that all existing tests still pass

---

## Code Style

```bash
# Format Python
black src/ dashboard/

# Lint
flake8 src/ dashboard/ --max-line-length=100

# Run tests
pytest tests/ -v

# Check notebook outputs are cleared before committing
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb
```

---

## Reporting Issues

Open a GitHub Issue with one of these labels:

| Label | Use Case |
|-------|----------|
| `bug` | Model produces incorrect predictions or pipeline crashes |
| `false-positive` | Model flags benign traffic incorrectly |
| `enhancement` | New feature or model suggestion |
| `documentation` | README or notebook improvement |
| `question` | General usage or methodology questions |

For bugs, always include:
- Python version and OS
- Relevant error traceback
- Dataset file name and size used
- Command or notebook cell that triggered the issue

---

## Contact

**Author**: Chandra Sekhar Chakraborty  
📧 chakrabortychandrasekhar185@gmail.com  
🔗 [GitHub](https://github.com/ChandraVerse) | [Twitter / X](https://twitter.com/CS_Chakraborty)
