# 🎗️ Breast Cancer Diagnosis Prediction

A machine learning system that classifies breast tumors as malignant or benign using the Breast Cancer Wisconsin Dataset. Includes a full data analysis pipeline and an interactive Streamlit dashboard.

---

## Overview

This project has three main components:

- **`breast_cancer1.py`** — Exploratory data analysis (EDA) and model training script
- **`app.py`** — Interactive Streamlit web dashboard
- **`breast-cancer.csv`** — The dataset (569 patient samples, 30 features)

The model uses a Random Forest Classifier trained on 30 tumor measurements to predict diagnosis.

---

## Dataset

**Source:** Breast Cancer Wisconsin Dataset (UCI Machine Learning Repository)

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 numeric measurements |
| Target | `M` = Malignant, `B` = Benign |
| Class split | 357 benign (62.7%), 212 malignant (37.3%) |

Each tumor has 30 measurements across 10 characteristics (mean, standard error, and worst value for each): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

---

## Project Structure

```
breast_cancer_analysis/
├── README.md
├── app.py                 # Streamlit dashboard (887 lines)
├── breast_cancer1.py      # EDA + ML script (176 lines)
├── breast-cancer.csv      # Dataset
├── config.toml            # Streamlit config
└── cina.png               # App icon
```

---

## Installation

**Requirements:** Python 3.8+, 4GB RAM

```bash
# 1. Clone the repo
git clone https://github.com/ankhanhcute/breast_cancer_analysis.git
cd breast_cancer_analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn pillow
```

Verify the install worked:
```bash
python -c "import streamlit, pandas, sklearn; print('All good!')"
```

---

## Usage

### Run the analysis script

Performs EDA and trains the model. Output goes to your terminal.

```bash
python breast_cancer1.py
```

This will load the dataset, show summary stats, generate charts, train a Random Forest on 80% of the data, and print accuracy + classification metrics.

### Launch the dashboard

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser. The app has five sections navigable from the sidebar:

| Section | What it shows |
|---|---|
| **Overview** | Dataset summary and class distribution |
| **EDA** | Interactive histograms, scatter plots, and correlation heatmap |
| **Hypothesis Testing** | T-tests and Levene's variance tests comparing tumor types |
| **ML Model** | Accuracy, confusion matrix, feature importance ranking |
| **Predict Tumor** | Input 30 measurements and get a real-time prediction |

Press `Ctrl+C` in the terminal to stop the app.

---

## Key Findings

- **Malignant tumors are larger** — mean radius ~17.5mm vs ~12.2mm for benign
- **Compactness and concavity** are among the strongest predictors of malignancy
- **Radius, perimeter, and area** are highly correlated (r > 0.95) — expected since they all measure size
- T-tests confirm statistically significant differences between tumor types across most features (p < 0.05)
- Random Forest achieves high accuracy on the 20% held-out test set

---

## Tech Stack

| Package | Purpose |
|---|---|
| `scikit-learn` | Random Forest classifier, metrics, train/test split |
| `pandas` / `numpy` | Data manipulation and numerical ops |
| `matplotlib` / `seaborn` | Visualization |
| `scipy` | Statistical tests (t-test, Levene's, Shapiro) |
| `streamlit` | Interactive web dashboard |
| `pillow` | Image handling in the app |

---

## License

Educational/research use. Do whatever you want with it.

---

*Made by [Ellie Phan](https://github.com/ankhanhcute)*
