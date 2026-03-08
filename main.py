"""Full modelling pipeline — run end to end with `python main.py`."""

import json
from pathlib import Path

from src.data import load_math, load_portuguese, preprocess
from src.models import fit_and_evaluate
from src.plots import feature_importance

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run(subject: str, loader):
    print(f"\n=== {subject} ===")
    df = loader()
    X, y = preprocess(df)

    results = {}
    for model_name in ("linear", "ridge", "random_forest"):
        out = fit_and_evaluate(X, y, model_name)
        results[model_name] = {
            "mae": out["mae"],
            "r2": out["r2"],
            "cv_r2": out["cv_r2"],
        }
        print(
            f"{model_name:>15}  MAE={out['mae']:.3f}  R2={out['r2']:.3f}"
            f"  CV-R2={out['cv_r2']:.3f}"
        )

        if hasattr(out["model"], "feature_importances_"):
            feature_importance(out["model"], list(X.columns), subject)

    (RESULTS_DIR / f"{subject.lower()}_results.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    from src.data import load_math, load_portuguese

    run("Math", load_math)
    run("Portuguese", load_portuguese)
