# Architecture Decision Records (ADRs)

This document captures the key design decisions made during the development of the XAI-NIDS system — including the reasoning behind each choice and the alternatives that were considered.

---

## ADR-001 — Dataset: CICIDS-2017

**Decision:** Use CICIDS-2017 as the benchmark dataset.

**Rationale:**
- Most widely cited IDS benchmark (200+ papers) — results are directly comparable to published literature
- Contains 14 labelled attack classes covering the full range of MITRE ATT&CK network techniques
- 2.8 million flows — large enough for robust ML training without trivial accuracy
- Free for academic use with no licensing restrictions

**Alternatives considered:**
- *NSL-KDD:* Older, fewer attack types, widely criticised for unrealistic distribution
- *UNSW-NB15:* Valid alternative but fewer citation comparisons available
- *Live capture:* Would require a dedicated lab network; CICIDS-2017 was generated in a controlled but realistic environment

---

## ADR-002 — Three Models (RF + XGBoost + LSTM)

**Decision:** Train three distinct model architectures rather than a single best model.

**Rationale:**
- RF and XGBoost are both `TreeExplainer`-compatible, enabling fast exact SHAP — critical for sub-second SOC triage
- LSTM captures temporal patterns that tree models cannot (slow-rate DoS attacks)
- Each model has a different precision/recall trade-off: XGBoost minimises false positives, LSTM maximises slow-attack recall
- Three models enable an ensemble vote, improving robustness
- Portfolio value: demonstrates knowledge across tree ensembles and recurrent architectures

**Alternatives considered:**
- *Single best model:* Simpler, but sacrifices temporal detection and model comparison value
- *Transformer / Attention-based:* Higher complexity, longer training time, harder SHAP integration, not necessary to beat CICIDS-2017 baselines

---

## ADR-003 — SHAP as the XAI Method

**Decision:** Use SHAP (SHapley Additive exPlanations) for all model explanations.

**Rationale:**
- Game-theory grounded — Shapley values are the only explanation method that satisfies all four desirable axioms (efficiency, symmetry, dummy, additivity)
- Works for all three model types: `TreeExplainer` for RF/XGBoost, `DeepExplainer` for LSTM
- `TreeExplainer` runs in milliseconds per flow — compatible with real-time SOC latency requirements
- Widely adopted in both industry and academic security research
- Produces actionable output: ranked per-feature contribution that an analyst can verify in raw packet data

**Alternatives considered:**
- *LIME:* Local approximations, less consistent, no global counterpart
- *Integrated Gradients:* Neural networks only — not compatible with RF/XGBoost
- *Feature Importance (Gini):* Global only, not per-alert, provides no triage value

---

## ADR-004 — SMOTE Applied to Training Set Only

**Decision:** Apply SMOTE after the train/test split, exclusively on the training set.

**Rationale:**
- Applying SMOTE before splitting would leak synthetic minority class information into the test set, inflating performance metrics
- The test set must reflect the true real-world class distribution (including the severe imbalance) to produce valid generalisation metrics
- Infiltration class has only 36 real samples — without SMOTE on the training set, the model would be blind to this attack type

**Implementation detail:** The SMOTE `sampling_strategy='not majority'` setting raises all minority classes to match the majority class count without reducing majority class samples.

---

## ADR-005 — Streamlit as the Dashboard Framework

**Decision:** Use Streamlit for the analyst-facing UI.

**Rationale:**
- Fastest path from trained model to interactive web UI — no web development knowledge required
- Native support for Matplotlib/Plotly/SHAP charts with `st.pyplot()` and `st.plotly_chart()`
- File uploader widget provides the exact SOC workflow: upload CSV → get results instantly
- Deployable as a single Python process inside Docker without a separate frontend build step

**Alternatives considered:**
- *Dash (Plotly):* More customisable but requires more boilerplate and a Flask backend
- *Gradio:* Well-suited for single-model demos but limited for multi-tab, multi-model workflows
- *Full React frontend:* Maximum flexibility but requires a separate API layer and significant development overhead inappropriate for a research prototype

---

## ADR-006 — Docker Containerisation

**Decision:** Package the entire system (models, dashboard, preprocessing) as a single Docker image.

**Rationale:**
- Eliminates the "works on my machine" problem — the system runs identically on any OS
- Pre-trained model artifacts are bundled in the image so SOC analysts can run the dashboard with zero ML setup
- Volume mount support allows retraining on a custom dataset without rebuilding the image
- A single `docker-compose up` command is the only barrier between a new user and a running NIDS

**What is excluded from the image:**
- Raw CICIDS-2017 data (~1 GB) — too large and requires separate academic-use download
- Notebook execution environment — notebooks are for development/research, not production inference

---

## ADR-007 — MinMaxScaler over StandardScaler

**Decision:** Use `MinMaxScaler` for feature normalisation rather than `StandardScaler`.

**Rationale:**
- Network flow features (packet counts, byte rates, durations) are strictly non-negative — scaling to [0, 1] is semantically natural
- `StandardScaler` produces negative values for below-mean features, which complicates SHAP value interpretation (a "negative" byte count is meaningless)
- LSTM input benefits from bounded [0, 1] range — prevents vanishing/exploding gradients
- Tree models (RF, XGBoost) are scale-invariant, so MinMax vs Standard makes no difference to their accuracy
