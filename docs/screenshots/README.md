# Screenshots

This directory contains dashboard and model output screenshots generated when running the full pipeline.

## How to Generate

Run the full pipeline end-to-end and screenshots will be saved here automatically:

```bash
# 1. Download CICIDS-2017 dataset (see data/README.md)
# 2. Run notebooks in order:
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_explainability.ipynb

# 3. Launch dashboard (screenshots captured from the running app)
cd dashboard && streamlit run app.py
```

## Expected Screenshots

| File | Description |
|------|-------------|
| `screenshot1_streamlit_dashboard.png` | Streamlit dashboard — live prediction with SHAP waterfall |
| `screenshot2_model_comparison.png` | RF vs XGBoost vs LSTM ROC curves and confusion matrices |
| `screenshot3_shap_summary.png` | SHAP beeswarm global feature importance plot |
| `screenshot4_dataset_distribution.png` | CICIDS-2017 class distribution bar chart from EDA |

> **Note:** Screenshots are not committed to keep repository size small.  
> The trained model artifacts (`.pkl`, `.h5`) are similarly excluded via `.gitignore`.  
> See [Releases](../../releases) for pre-built model artifacts once the full pipeline is executed.
