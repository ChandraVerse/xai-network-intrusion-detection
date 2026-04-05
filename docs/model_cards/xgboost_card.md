# Model Card: XGBoost Classifier

## Model Details

- **Architecture**: `xgboost.XGBClassifier`
- **Hyperparameters**: `n_estimators=300`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method=hist`, `early_stopping_rounds=20`
- **Training script**: `scripts/bootstrap_artifacts.py`
- **Artifact**: `models/xgboost_model.pkl`

## Intended Use

Same as Random Forest: multi-class NIDS on CICIDS-2017 features. XGBoost typically achieves marginally higher F1 at the cost of slightly longer training time.

## Dataset

Identical split to Random Forest (see `random_forest_card.md`).

## Performance (Test Set)

| Metric | Value |
|---|---|
| Accuracy | See `models/xgb_metrics.json` |
| Macro F1 | See `models/xgb_metrics.json` |
| Mean FPR | See `models/xgb_metrics.json` |
| Inference | < 1 ms / flow |

## Explainability

- **Method**: SHAP `TreeExplainer` (same interface as RF)
- **LIME support**: `src/explainability/lime_explainer.py`

## Limitations

Same as Random Forest. Additionally: XGBoost `.pkl` artifacts serialised via `joblib` — requires compatible xgboost version on load (`pip install xgboost>=2.0`).
