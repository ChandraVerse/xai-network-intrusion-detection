# Architecture Diagrams

## System Architecture Overview

![XAI-NIDS Architecture](./architecture_diagram.png)

The diagram above shows the full 6-layer pipeline:

| Layer | Component | Purpose |
|-------|-----------|--------|
| **[1] Input** | PCAP / CICFlowMeter / CSV | Raw network traffic ingestion |
| **[2] Preprocessing** | `cleaner.py` → `scaler.py` → `smote_balancer.py` | Feature cleaning, scaling, class balancing |
| **[3] Detection** | Random Forest, XGBoost, LSTM + Ensemble | Multi-class attack classification |
| **[4] XAI** | TreeExplainer, DeepExplainer, LIME | Per-alert Shapley value explanations |
| **[5] Dashboard** | Streamlit (`dashboard/app.py`) | Live detection, model comparison, SHAP viz |
| **[6] CI/CD** | GitHub Actions + Docker | Automated testing and deployment |

## Regenerate

```bash
pip install matplotlib
python docs/architecture/generate_diagram.py
```

## Export Formats

- `architecture_diagram.png` — 2730×3330 px @ 150 DPI (high-res, commit this)
- For PDF export: `matplotlib.pyplot.savefig('diagram.pdf')` in `generate_diagram.py`
- For SVG export: change extension to `.svg` (vector, infinite resolution)
