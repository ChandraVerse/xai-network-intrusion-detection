#!/usr/bin/env python3
"""
docs/architecture/generate_diagram.py

Regenerates the architecture_diagram.png used in the README.

Usage:
    python docs/architecture/generate_diagram.py
    # Output: docs/architecture/architecture_diagram.png
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), 'architecture_diagram.png')

C = {
    'bg': '#0d1117', 'panel': '#161b22', 'border': '#30363d',
    'teal': '#01696f', 'teal_lt': '#4f98a3',
    'orange': '#da7101',
    'green': '#437a22', 'green_lt': '#6daa45',
    'purple': '#7a39bb', 'purple_lt': '#a86fdf',
    'red': '#a12c7b', 'red_lt': '#d163a7',
    'text': '#e6edf3', 'muted': '#8b949e', 'arrow': '#58a6ff',
}


def box(ax, x, y, w, h, label, sublabel=None, color=None, fontsize=11):
    color = color or C['teal']
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
        linewidth=1.5, edgecolor=color, facecolor=C['panel']))
    ax.add_patch(FancyBboxPatch((x+0.04, y+h-0.13), w-0.08, 0.09,
        boxstyle='round,pad=0.01', linewidth=0, edgecolor='none',
        facecolor=color, alpha=0.85))
    cy = y + h/2 + (0.13 if sublabel else 0)
    ax.text(x+w/2, cy, label, ha='center', va='center',
        fontsize=fontsize, fontweight='bold', color=C['text'])
    if sublabel:
        ax.text(x+w/2, y+h/2-0.22, sublabel, ha='center', va='center',
            fontsize=8.2, color=C['muted'], fontstyle='italic')


def arr_d(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2.0))


def arr_r(ax, x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.6))


def arr_c(ax, xf, yf, xt, yt):
    ax.annotate('', xy=(xt, yt), xytext=(xf, yf),
        arrowprops=dict(arrowstyle='->', color=C['purple_lt'], lw=1.4))


def sec(ax, x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
        linewidth=2, edgecolor=color, facecolor=color, alpha=0.07))
    ax.text(x+0.3, y+h-0.3, title, fontsize=9.5, fontweight='bold',
        color=color, va='top')


def build():
    fig, ax = plt.subplots(figsize=(18, 22))
    ax.set_xlim(0, 18); ax.set_ylim(0, 22); ax.axis('off')
    fig.patch.set_facecolor(C['bg']); ax.set_facecolor(C['bg'])

    ax.text(9, 21.45, 'XAI-NIDS  --  System Architecture',
        ha='center', fontsize=17, fontweight='bold', color=C['teal_lt'])
    ax.text(9, 21.05, 'Explainable AI-Based Network Intrusion Detection System',
        ha='center', fontsize=10, color=C['muted'], fontstyle='italic')
    ax.plot([0.5, 17.5], [20.75, 20.75], color=C['border'], lw=1.2)

    sec(ax, 0.4, 19.15, 17.2, 1.4, '[1] INPUT LAYER', C['teal'])
    box(ax, 0.7,  19.3, 4.8, 0.9, 'Raw Network Traffic', 'PCAP / Live Capture', C['teal'])
    box(ax, 6.6,  19.3, 4.8, 0.9, 'CICFlowMeter',        '78-feature extraction', C['teal'])
    box(ax, 12.5, 19.3, 4.8, 0.9, 'CSV Upload',          'Pre-extracted flows', C['teal'])
    arr_r(ax, 5.5, 6.6, 19.75); arr_r(ax, 11.4, 12.5, 19.75)

    sec(ax, 0.4, 16.55, 17.2, 2.4, '[2] PREPROCESSING PIPELINE', C['orange'])
    arr_d(ax, 9, 19.3, 18.95)
    box(ax, 0.7,  16.75, 4.8, 0.9, 'cleaner.py',        'Drop NaN / Inf', C['orange'])
    box(ax, 6.6,  16.75, 4.8, 0.9, 'scaler.py',         'MinMaxScaler (train-fit only)', C['orange'])
    box(ax, 12.5, 16.75, 4.8, 0.9, 'smote_balancer.py', 'SMOTE minority classes', C['orange'])
    arr_r(ax, 5.5, 6.6, 17.2); arr_r(ax, 11.4, 12.5, 17.2)
    ax.text(9, 16.63,
        'Artifacts: train_balanced.csv  |  test.csv  |  minmax_scaler.pkl  |  label_encoder.pkl  |  feature_cols.json',
        ha='center', fontsize=7.8, color=C['muted'], fontstyle='italic')

    sec(ax, 0.4, 13.05, 17.2, 3.3, '[3] ML DETECTION ENGINE', C['purple'])
    arr_d(ax, 9, 16.75, 16.38)
    box(ax, 0.7,  13.8, 5.0, 1.65, 'Random Forest',
        'n_estimators=200  |  F1: 0.997  |  FPR: 0.28%', C['purple'], fontsize=10)
    box(ax, 6.5,  13.8, 5.0, 1.65, 'XGBoost',
        'n_estimators=300  |  F1: 0.994  |  FPR: 0.21%', C['purple'], fontsize=10)
    box(ax, 12.3, 13.8, 5.0, 1.65, 'LSTM  (2-layer)',
        'input shape (5, 78)  |  F1: 0.981  |  FPR: 0.51%', C['purple'], fontsize=10)
    box(ax, 4.5, 13.15, 9.0, 0.5, 'Ensemble Voting  --  Majority Vote / Soft Probability',
        color=C['purple_lt'], fontsize=9)
    arr_c(ax, 5.7, 13.8, 7.2, 13.65)
    arr_c(ax, 9.0, 13.8, 9.0, 13.65)
    arr_c(ax, 12.3, 13.8, 10.8, 13.65)

    sec(ax, 0.4, 9.95, 17.2, 2.9, '[4] XAI EXPLAINABILITY LAYER  (SHAP)', C['red'])
    arr_d(ax, 9, 13.15, 12.85)
    box(ax, 0.7,  10.65, 5.0, 1.55, 'TreeExplainer',
        'RF + XGBoost  |  Exact Shapley values', C['red'], fontsize=10)
    box(ax, 6.5,  10.65, 5.0, 1.55, 'DeepExplainer',
        'LSTM  |  Gradient-based approx.', C['red'], fontsize=10)
    box(ax, 12.3, 10.65, 5.0, 1.55, 'LIME Explainer',
        'Model-agnostic  |  Perturbation', C['red'], fontsize=10)
    ax.text(9, 10.5,
        'Output per alert:  Waterfall chart  |  Beeswarm summary  |  Dependence plots  |  Force plot HTML',
        ha='center', fontsize=8, color=C['muted'], fontstyle='italic')
    ax.text(9, 10.15,
        'Top features:  Flow Duration  |  Bwd Pkt Length Max  |  Flow Bytes/s  |  Fwd IAT Total  |  Dst Port',
        ha='center', fontsize=8, color=C['red_lt'])

    sec(ax, 0.4, 6.75, 17.2, 2.95, '[5] STREAMLIT DASHBOARD  --  dashboard/app.py', C['green'])
    arr_d(ax, 9, 9.95, 9.7)
    box(ax, 0.7,  7.5, 3.8, 1.55, 'Live Detection',
        'Upload CSV -> predict\nSHAP waterfall per alert', C['green'], fontsize=9)
    box(ax, 4.9,  7.5, 3.8, 1.55, 'Model Comparison',
        'ROC curves  |  F1\nConfusion matrices', C['green'], fontsize=9)
    box(ax, 9.1,  7.5, 3.8, 1.55, 'Global SHAP',
        'Beeswarm summary\nDependence plots', C['green'], fontsize=9)
    box(ax, 13.3, 7.5, 3.8, 1.55, 'Export Report',
        'PDF alert report\nvia ReportLab', C['green'], fontsize=9)
    ax.text(9, 7.37,
        'http://localhost:8501    |    docker-compose up --build    |    streamlit run dashboard/app.py',
        ha='center', fontsize=8.5, color=C['green_lt'], fontweight='bold')

    sec(ax, 0.4, 4.15, 17.2, 2.38, '[6] CI/CD + DEPLOYMENT', C['teal_lt'])
    arr_d(ax, 9, 6.75, 6.53)
    box(ax, 0.7,  4.38, 5.0, 1.55, 'GitHub Actions CI',
        'ci.yml  |  Python 3.10 + 3.11\npytest  |  flake8  |  coverage', C['teal_lt'], fontsize=9)
    box(ax, 6.5,  4.38, 5.0, 1.55, 'Docker',
        'Dockerfile + docker-compose\nOne-command deployment', C['teal_lt'], fontsize=9)
    box(ax, 12.3, 4.38, 5.0, 1.55, 'Model Artifacts',
        'random_forest.pkl  |  xgboost.pkl\nlstm_model.tar.gz', C['teal_lt'], fontsize=9)

    ax.plot([0.5, 17.5], [4.0, 4.0], color=C['border'], lw=1)
    ax.text(9, 3.62,
        'Dataset:  CICIDS-2017  --  2.8M labeled flows  |  78 CICFlowMeter features  |  14 attack classes',
        ha='center', fontsize=9, color=C['text'], fontweight='bold')

    stack = [
        ('scikit-learn', C['purple_lt']), ('XGBoost', C['orange']),
        ('TensorFlow/Keras', C['red_lt']), ('SHAP', C['teal_lt']),
        ('Streamlit', C['green_lt']), ('Docker', C['arrow']),
        ('pandas / NumPy', C['muted']),
    ]
    xp = 1.0
    for name, col in stack:
        ax.text(xp, 3.18, f'[+] {name}', fontsize=8, color=col, va='center')
        xp += len(name) * 0.145 + 0.65

    ax.plot([0.5, 17.5], [2.88, 2.88], color=C['border'], lw=1)
    ax.text(9, 2.55, 'Attack Classes  --  MITRE ATT&CK Mapped',
        ha='center', fontsize=9, fontweight='bold', color=C['text'])
    ax.text(9, 2.22,
        'DDoS (T1498)  |  DoS Hulk/GoldenEye/Slowloris (T1499)  |  FTP-Patator/SSH-Patator (T1110)',
        ha='center', fontsize=7.8, color=C['muted'])
    ax.text(9, 1.92,
        'PortScan (T1046)  |  Web Attack XSS/SQLi/BruteForce (T1059/T1190)  |  Infiltration (T1078)  |  Bot (T1071)',
        ha='center', fontsize=7.8, color=C['muted'])
    ax.text(9, 1.48,
        'github.com/ChandraVerse/xai-network-intrusion-detection  |  MIT License  |  April 2026',
        ha='center', fontsize=8, color=C['border'], fontstyle='italic')

    ax.add_patch(FancyBboxPatch((0.1, 0.1), 17.8, 21.8,
        boxstyle='round,pad=0.05', linewidth=2,
        edgecolor=C['teal'], facecolor='none', alpha=0.45))

    plt.tight_layout(pad=0)
    plt.savefig(OUT, dpi=150, bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    plt.close()
    print(f'[OK] Saved: {OUT}')


if __name__ == '__main__':
    build()
