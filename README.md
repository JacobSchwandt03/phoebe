# Predicting Secondary School Performance Patterns by means of Regularised Regression

**D200 Coursework** - MPhil Economics and Data Science - University of Cambridge - 2025/26

## Research Question

Do demographic and lifestyle features predict the grade trajectory of a student (G3 - G1) conditional on starting performance (G1). This distinguishes between features which correlate with where a student begins and those that predict how much they improve over a school year. 

## Data

Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

Dataset covers Portuguese secondary school students across two subjects: Mathematics (395 observations) and Portuguese (649) observations. The data set contains 30 demographic and lifestyle features including parental education, health status and study time. 

**Aside on pre-processing** 38 Mathematics student recorded G3 = 0, this indicates mid-year or end-of-year dropouts as supposed to failure. We simply exclude these findings from all models. 

Data file are not tracked by git. Simply place `student-mat.csv` and `student-pot.csv` in `data/raw/`.

## Structure

```
phoebe/
├── data/
│   └── raw/               # student-mat.csv, student-por.csv (not tracked)
├── figures/               # generated plots (not tracked)
├── src/
│   └── models.py          # feature preparation, Lasso/Ridge fitting, evaluation
├── analysis.ipynb         # main analysis notebook
├── pyproject.toml         # project metadata and dependencies
├── environment.yml        # conda environment specification
└── README.md
```

## Setup

**1. Create and activate the conda environment**

```bash
conda env create -f environment.yml
conda activate phoebe
```

**2. Install the project in editable mode**

```bash
pip install -e .
```

**3. Launch Jupyter**

```bash
jupyter notebook analysis.ipynb
```

Select `phoebe` kernel in the top right corner of the notebook.

---

## Reproducing the Analysis

All results are produced by running `analysis.ipynb` top to bottom. The notebook is structured in four sections:

**1. Data Overview**
Summary statistics, missing value check, grade distributions (KDE), quantitative correlation heatmap, categorical feature distributions, and exploratory relationships: parental education cross-tabulation (Medu × Fedu), absences vs final grade, romantic relationships vs grade distribution, and mother's occupation vs grades.

**2. Grade Persistence and Trajectory Motivation**
Grade band transition heatmaps (G1→G2, G1→G3) showing within-year persistence. G1 vs G3 scatter with OLS fit (slope = 0.89) establishing mild mean convergence. Paired violin plots comparing feature associations with starting grade (G1) versus trajectory (G3 − G1). Lasso regularisation paths (G1 baseline vs G3 − G1) with entry-alpha comparison chart.

**3. Regression Models**
Lasso, Ridge, and Elastic Net predicting G3 − G1 using numeric-only and one-hot encoded features. G1 baseline Lasso as dataset validation (R² ≈ 0.13). All trajectory models yield R² ≈ 0.03. Robustness checks on the Portuguese dataset across all six specifications confirm the null result. Hyperparameter selection via 5-fold `GridSearchCV`. Features standardised using `StandardScaler` fitted on training data only to prevent leakage.

**4. Classification Models**
L1-penalised logistic regression and Random Forest predicting whether a student's trajectory improves or holds steady versus declines (G3 − G1 ≥ 0). Evaluated against a dummy classifier baseline using accuracy, binary cross entropy, and ROC-AUC. SHAP values for feature attribution on the Random Forest. CV MSE path for optimal alpha selection.

---

## Reference

Cortez, P. and Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira (Eds.), *Proceedings of 5th Annual Future Business Technology Conference*, Porto, Portugal, pp. 5–12.
