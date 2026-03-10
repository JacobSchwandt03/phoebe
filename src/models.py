"""Model fitting and evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

MODELS = {
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
}


def prepare_features(df):
    """Prepare feature matrix and outcome arrays from the student dataset.

    Drops grade columns (G1, G2, G3) and any non-numeric columns from the
    dataframe to produce the feature matrix X, then computes two outcome
    arrays as grade differences.

    Parameters
    ----------
    df : pd.DataFrame
        Raw student dataset containing at least G1, G2, and G3 columns.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with grade and non-numeric columns removed.
    y_g3_g1 : pd.Series
        Outcome array of G3 minus G1 (end-of-year gain from period 1).
    y_g2_g1 : pd.Series
        Outcome array of G2 minus G1 (mid-year gain from period 1).
    """

    y_g3_g1 = df["G3"] - df["G1"]
    y_g2_g1 = df["G2"] - df["G1"]
    X = df.drop(columns=["G1", "G2", "G3"]).select_dtypes(include=[np.number])
    return X, y_g3_g1, y_g2_g1


def encode_features(df):
    """Extract features (including categorical) and outcome arrays.

    Drops grade columns (G1, G2, G3) but retains categorical columns.
    Call apply_dummies() on train/test splits separately after splitting
    to avoid leaking test-set category information into the feature space.

    Parameters
    ----------
    df : pd.DataFrame
        Raw student dataset containing at least G1, G2, and G3 columns.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with grade columns removed; categoricals are untouched.
    y_g3_g1 : pd.Series
        Outcome array of G3 minus G1.
    y_g2_g1 : pd.Series
        Outcome array of G2 minus G1.
    """
    y_g3_g1 = df["G3"] - df["G1"]
    y_g2_g1 = df["G2"] - df["G1"]
    X = df.drop(columns=["G1", "G2", "G3"])
    return X, y_g3_g1, y_g2_g1


def apply_dummies(X_train, X_test):
    """One-hot encode categoricals fit on train only, then align test columns.

    Encodes using only categories present in the training split, then
    reindexes the test split to the same columns (filling unseen categories
    with 0). This prevents test-set information leaking into the feature space.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix with raw categorical columns.
    X_test : pd.DataFrame
        Test feature matrix with raw categorical columns.

    Returns
    -------
    X_train_enc : pd.DataFrame
        One-hot encoded training features.
    X_test_enc : pd.DataFrame
        One-hot encoded test features, aligned to training columns.
    """
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, drop_first=True).reindex(
        columns=X_train_enc.columns, fill_value=0
    )
    return X_train_enc, X_test_enc


def fit_lasso(X_train, y_train):
    """Fit a Lasso regression model using GridSearchCV.

    Searches over a range of alpha regularization values using 5-fold
    cross-validation and returns the best-fitted model.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training outcome array.

    Returns
    -------
    model : GridSearchCV
        Fitted GridSearchCV object wrapping a Lasso estimator, with
        best_estimator_ set to the model selected by cross-validation.
    """
    alphas = np.logspace(-3, 1, 50)
    param_grid = {"alpha": alphas}
    lasso = Lasso(max_iter=10000, random_state=42)
    grid_search = GridSearchCV(
        estimator=lasso, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def fit_ridge(X_train, y_train):
    """Fit a Ridge regression model using GridSearchCV.

    Searches over a range of alpha regularization values using 5-fold
    cross-validation and returns the best-fitted model.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training outcome array.

    Returns
    -------
    model : GridSearchCV
        Fitted GridSearchCV object wrapping a Ridge estimator, with
        best_estimator_ set to the model selected by cross-validation.
    """
    alphas = np.logspace(-3, 1, 50)
    param_grid = {"alpha": alphas}
    ridge = Ridge(max_iter=10000)
    grid_search = GridSearchCV(
        estimator=ridge, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def fit_elastic_net(X_train, y_train):
    """
    Fit ElasticNet via grid search over alpha and l1_ratio.
    l1_ratio=1 -> pure Lasso, l1_ratio=0 -> pure Ridge.
    Uses 5-fold CV optimising neg MSE.
    """
    alphas = np.logspace(-3, 1, 30)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

    param_grid = {
        "alpha": alphas,
        "l1_ratio": l1_ratios,
    }

    enet = ElasticNet(max_iter=10000)
    grid_search = GridSearchCV(
        estimator=enet, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def fit_logistic(X_train, y_train):
    """
    Fit L1-penalised logistic regression via grid search over C.
    C is the inverse of regularisation strength (smaller C = stronger penalty).
    Uses saga solver with l1_ratio=1 (pure L1 / Lasso penalty).
    Loss minimised is binary cross entropy (log loss).
    Uses 5-fold CV optimising balanced accuracy.
    """
    C_grid = np.logspace(-3, 2, 50)
    param_grid = {"C": C_grid}

    logit = LogisticRegression(
        l1_ratio=1,
        solver="saga",
        max_iter=10000,
        random_state=42,
        class_weight="balanced",
    )
    grid_search = GridSearchCV(
        estimator=logit, param_grid=param_grid, cv=5, scoring="balanced_accuracy"
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def fit_random_forest(X_train, y_train):
    """
    Fit a Random Forest classifier via grid search over
    n_estimators and max_depth.
    Uses 5-fold CV optimising accuracy.
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 10, None],
        "min_samples_leaf": [1, 5, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5, scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a fitted classifier (GridSearchCV or estimator).
    Returns accuracy, log loss (cross entropy), ROC-AUC,
    and a classification report.
    """
    if isinstance(model, GridSearchCV):
        best_model = model.best_estimator_
    else:
        best_model = model

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    cross_entropy = log_loss(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(
        y_test, y_pred, target_names=["Fail", "Pass"], zero_division=0
    )

    return {
        "accuracy": accuracy,
        "cross_entropy": cross_entropy,
        "roc_auc": roc_auc,
        "report": report,
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted model on test data.

    Computes prediction error metrics and extracts coefficients for
    features with non-zero weights.

    Parameters
    ----------
    model : fitted estimator
        A fitted model with a predict method. If wrapped in GridSearchCV,
        the best_estimator_ is used to access coefficients.
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_samples,)
        True outcome values for the test set.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'mse': float, mean squared error on the test set.
        - 'r2': float, R-squared score on the test set.
        - 'nonzero_coefs': dict mapping feature name to coefficient value
          for all features with a non-zero Lasso coefficient.
    """

    if isinstance(model, GridSearchCV):
        best_model = model.best_estimator_
    else:
        best_model = model

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    nonzero_coefs = {
        feature: coef
        for feature, coef in zip(X_test.columns, best_model.coef_)
        if abs(coef) > 1e-6
    }

    results = {"mse": mse, "r2": r2, "nonzero_coefs": nonzero_coefs}
    return results
