# docs/screenshots/

This directory contains dashboard and model output screenshots generated after running the full pipeline on CICIDS-2017 data.

## How to generate screenshots

1. **Download CICIDS-2017** from the [UNB dataset page](https://www.unb.ca/cic/datasets/ids-2017.html) (~1 GB)
2. Place the CSVs in `data/raw/`
3. Run the notebooks in order:
   ```
   notebooks/01_eda.ipynb
   notebooks/02_preprocessing.ipynb
   notebooks/03_model_training.ipynb
   notebooks/04_xai_shap.ipynb
   ```
4. Launch the dashboard:
   ```bash
   docker-compose up
   # or: streamlit run dashboard/app.py
   ```
5. Take screenshots and save them here with the names below

## Expected files

| File | Contents |
|---|---|
| `screenshot1_streamlit_dashboard.png` | Streamlit — live prediction + SHAP waterfall panel |
| `screenshot2_model_comparison.png` | Notebook 03 — RF vs XGBoost vs LSTM ROC curves + confusion matrix |
| `screenshot3_shap_summary.png` | Notebook 04 — SHAP beeswarm summary plot |
| `screenshot4_dataset_distribution.png` | Notebook 01 — CICIDS-2017 class distribution bar chart |

> **Note:** Screenshots are not committed to keep repo size manageable. Run the pipeline locally on CICIDS-2017 data to generate them. The README references these paths — once generated and committed, they will render correctly on GitHub.
