# End-to-End Data Flow

This document describes the complete data flow through the XAI-NIDS system — from raw network traffic to analyst-facing SHAP explanation — using Mermaid diagrams.

---

## 1. System-Level Data Flow

```mermaid
flowchart TD
    A(["🌐 Raw Network Traffic\n(PCAP / Live Capture)"]) --> B["CICFlowMeter\n78-feature extraction"]
    B --> C[("feature_vectors.csv")]
    Z(["📁 Direct CSV Upload\n(pre-extracted 78 features)"]) --> C
    C --> D["Preprocessing Pipeline"]
    D --> E[("train_balanced.csv\ntest.csv")]
    E --> F["Random Forest"]
    E --> G["XGBoost"]
    E --> H["LSTM\n(reshaped 3D: samples×5×78)"]
    F & G & H --> I["Ensemble Voting\n(Majority / Stacking)"]
    I --> J(["Attack Label + Confidence Score"])
    J --> K["SHAP TreeExplainer\n(RF + XGBoost)"]
    J --> L["SHAP DeepExplainer\n(LSTM)"]
    K & L --> M(["Explanation Object\n(SHAP values per feature)"])
    M --> N["Streamlit Dashboard"]
    N --> O(["SOC Analyst\nAlert + Waterfall Chart"])
```

---

## 2. Preprocessing Sub-Flow

```mermaid
flowchart LR
    A(["Raw CSVs\ndata/raw/"]) --> B["Drop Inf / NaN\ncleaner.py"]
    B --> C["Drop zero-variance features"]
    C --> D["LabelEncoder\n14 classes → integers"]
    D --> E["Stratified 80/20 split"]
    E --> F["MinMaxScaler\nfit on TRAIN only"]
    F --> G["SMOTE\napplied to TRAIN only"]
    G --> H[("train_balanced.csv")]
    E --> I[("test.csv\nunmodified distribution")]
    F --> J[("minmax_scaler.pkl")]
    D --> K[("label_encoder.pkl")]
```

> **Data leakage guard:** The scaler and SMOTE are fitted exclusively on the training split. The test set reflects true real-world class distribution, including extreme imbalance (Infiltration: 36 samples).

---

## 3. Inference Flow (Single Network Flow)

```mermaid
sequenceDiagram
    participant Analyst as SOC Analyst
    participant UI as Streamlit UI
    participant Pre as Preprocessor
    participant RF as Random Forest
    participant XGB as XGBoost
    participant LSTM as LSTM
    participant SHAP as SHAP Explainer

    Analyst->>UI: Upload network_flows.csv
    UI->>Pre: scale(flow_features)
    Pre-->>UI: scaled_features [1 × 78]
    UI->>RF: predict(scaled_features)
    UI->>XGB: predict(scaled_features)
    UI->>LSTM: predict(reshaped [1 × 5 × 78])
    RF-->>UI: DDoS (0.973)
    XGB-->>UI: DDoS (0.961)
    LSTM-->>UI: DDoS (0.944)
    UI->>SHAP: explain(RF, scaled_features)
    SHAP-->>UI: shap_values [78]
    UI->>Analyst: 🔴 DDoS · 97.3% confidence\nWaterfall: Flow Duration +0.43, Fwd Packets +0.29 …
```

---

## 4. SHAP Explanation Types

```mermaid
flowchart LR
    A(["Prediction\n+ SHAP Values"]) --> B["Waterfall Chart\nPer-alert top-10 features"]
    A --> C["Force Plot\nInteractive HTML export"]
    A --> D["Beeswarm Summary\nGlobal feature ranking"]
    A --> E["Dependence Plot\nFeature X vs SHAP(X) colored by Y"]
    B --> F(["SOC Triage:\nWhy this alert?"])
    C --> G(["Export: PDF Report"])
    D --> H(["Model Validation:\nWhat did the model learn?"])
    E --> I(["Feature Interaction\nAnalysis"])
```

---

## 5. Deployment Architecture

```mermaid
flowchart TB
    subgraph Docker Container
        A["Streamlit App\ndashboard/app.py"] --> B["src/ modules"]
        B --> C["Pre-trained Models\nmodels/*.pkl / *.h5"]
        B --> D["Sample Data\ndata/samples/"]
    end
    E(["Host: 0.0.0.0:8501"]) <--> A
    F(["Volume Mount:\n/your/data → /app/data/raw"]) --> D
```

**Container entry point:** `streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0`

---

## Data Artefacts Summary

| Artefact | Location | Produced By | Consumed By |
|----------|----------|-------------|-------------|
| `train_balanced.csv` | `data/processed/` | `02_preprocessing.ipynb` | `03_model_training.ipynb` |
| `test.csv` | `data/processed/` | `02_preprocessing.ipynb` | `03_model_training.ipynb`, `04_xai_shap.ipynb` |
| `minmax_scaler.pkl` | `data/processed/` | `02_preprocessing.ipynb` | Dashboard, CLI scripts |
| `label_encoder.pkl` | `data/processed/` | `02_preprocessing.ipynb` | Dashboard, CLI scripts |
| `random_forest.pkl` | `models/` | `03_model_training.ipynb` | `04_xai_shap.ipynb`, Dashboard |
| `xgboost_model.pkl` | `models/` | `03_model_training.ipynb` | `04_xai_shap.ipynb`, Dashboard |
| `lstm_model.h5` | `models/` | `03_model_training.ipynb` | `04_xai_shap.ipynb`, Dashboard |
| `feature_cols.json` | `data/processed/` | `02_preprocessing.ipynb` | All downstream modules |
| `label_map.json` | `data/processed/` | `02_preprocessing.ipynb` | Dashboard, SHAP visualisations |
