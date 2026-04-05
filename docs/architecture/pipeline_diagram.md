# Pipeline Diagram

This file contains the Mermaid source for the full XAI-NIDS pipeline.
GitHub renders Mermaid diagrams natively in Markdown — open this file on GitHub to see the interactive diagram.

---

## Full Pipeline (Data → Models → XAI → Dashboard)

```mermaid
flowchart TD
    A(["🌐 Raw Data<br/>CICIDS-2017 CSV / PCAP"]) --> B

    subgraph PRE ["📦 Stage 1 — Preprocessing"]
        B["notebooks/01_eda.ipynb<br/>Exploratory Analysis"]
        B --> C["notebooks/02_preprocessing.ipynb<br/>Clean · Scale · Encode"]
        C --> D[("data/processed/<br/>X_train.npz · X_test.npz<br/>y_train.npy · y_test.npy<br/>scaler.pkl · label_encoder.pkl")]
    end

    subgraph TRAIN ["🤖 Stage 2 — Model Training"]
        D --> E1["Random Forest<br/>src/models/random_forest.py"]
        D --> E2["XGBoost<br/>src/models/xgboost_model.py"]
        D --> E3["LSTM<br/>src/models/lstm_model.py"]
        E1 --> F1[("models/random_forest.pkl")]
        E2 --> F2[("models/xgboost_model.pkl")]
        E3 --> F3[("models/lstm_model.tar.gz")]
    end

    subgraph XAI ["🔍 Stage 3 — Explainability (XAI)"]
        F1 & F2 --> G1["SHAP TreeExplainer<br/>src/explainability/shap_explainer.py"]
        F3      --> G2["SHAP DeepExplainer<br/>src/explainability/shap_explainer.py"]
        F1 & F2 & F3 --> G3["LIME TabularExplainer<br/>src/explainability/lime_explainer.py"]
        G1 --> H1["summary_plot.py<br/>Beeswarm · Bar"]
        G1 --> H2["waterfall.py<br/>Per-alert waterfall"]
        G3 --> H3["lime_explainer.py<br/>plot_weights()"]
    end

    subgraph EVAL ["📊 Stage 4 — Evaluation"]
        F1 & F2 & F3 --> I["notebooks/05_model_comparison.ipynb"]
        I --> J1["Confusion Matrices"]
        I --> J2["ROC-AUC Curves"]
        I --> J3["SHAP vs LIME Agreement"]
        I --> J4["Inference Latency"]
        I --> J5[("reports/model_comparison.json")]
    end

    subgraph DASH ["🖥️ Stage 5 — SOC Dashboard"]
        F1 & F2 & G1 & G3 --> K["dashboard/app.py<br/>Streamlit"]
        K --> L1["Real-time flow classification"]
        K --> L2["SHAP waterfall per alert"]
        K --> L3["LIME rule explanation"]
        K --> L4["Model comparison panel"]
    end

    subgraph CI ["⚙️ CI/CD — GitHub Actions"]
        M1["Lint (flake8)"]
        M2["Unit + Integration Tests<br/>pytest --cov"]
        M3["Security Scan (Bandit)"]
        M4["Data Schema Validation (pandera)"]
        M5["Docker Build"]
    end

    style PRE   fill:#e8f4f8,stroke:#2196f3
    style TRAIN fill:#e8f8e8,stroke:#4caf50
    style XAI   fill:#fff3e0,stroke:#ff9800
    style EVAL  fill:#fce4ec,stroke:#e91e63
    style DASH  fill:#ede7f6,stroke:#9c27b0
    style CI    fill:#f5f5f5,stroke:#9e9e9e
```

---

## Module Dependency Graph

```mermaid
graph LR
    subgraph utils
        U1[logger.py]
        U2[metrics.py]
        U3[report_generator.py]
        U4[pcap_converter.py]
    end

    subgraph preprocessing
        P1[cleaner.py]
        P2[encoder.py]
        P3[scaler.py]
    end

    subgraph models
        M1[random_forest.py]
        M2[xgboost_model.py]
        M3[lstm_model.py]
    end

    subgraph explainability
        E1[shap_explainer.py]
        E2[lime_explainer.py]
        E3[summary_plot.py]
        E4[waterfall.py]
    end

    P1 --> U1
    P2 --> U1
    P3 --> U1
    M1 --> U1
    M2 --> U1
    M3 --> U1
    E1 --> U1
    E2 --> U1
    E1 --> U2
    E2 --> U2
    U3 --> U2
    E3 --> E1
    E4 --> E1

    style utils         fill:#e3f2fd
    style preprocessing fill:#e8f5e9
    style models        fill:#fff9c4
    style explainability fill:#fce4ec
```

---

## Data Schema

```mermaid
erDiagram
    RAW_CSV {
        string Label
        float  Flow_Duration
        float  Total_Fwd_Packets
        float  Total_Backward_Packets
        float  Flow_Bytes_per_s
        float  Flow_IAT_Mean
        string "... 73 more features"
    }

    PROCESSED {
        npz    X_train
        npz    X_test
        npy    y_train
        npy    y_test
        pkl    scaler
        pkl    label_encoder
        json   feature_names
    }

    MODELS {
        pkl    random_forest
        pkl    xgboost_model
        targz  lstm_model
        json   label_map
        json   metrics_summary
    }

    REPORTS {
        json   model_comparison
        png    confusion_matrices
        png    roc_curves
        png    shap_vs_lime
        png    latency_benchmark
    }

    RAW_CSV ||--o{ PROCESSED : "preprocessing"
    PROCESSED ||--|{ MODELS : "training"
    MODELS ||--|{ REPORTS : "evaluation"
```
