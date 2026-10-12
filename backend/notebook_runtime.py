"""
notebook_runtime.py
--------------------
Executes the *actual* project notebook's code cells (in order) inside a
sandboxed namespace, and hands back the pandas DataFrame at the one point
in the pipeline that matters for retraining: right after the deficiency
target columns have been engineered and the data has been cleaned, but
*before* the notebook label-encodes the categorical columns in place.

Why not literally reuse the notebook's own trained model / encoder / scaler
objects? Because this notebook reuses a single `le` / `scaler` variable
name across all three targets (each re-fit call overwrites the previous
one), so by the time the notebook finishes running there is no way to
recover separate, inference-safe encoders per column. Instead, we take the
notebook's own dataframe (so any edit you make to the data-generation,
target rules, or cleaning cells is honored) and apply our own robust,
per-column encoding + scaling + training on top of it - the same
consistent path every time, regardless of how the notebook happens to be
structured.

This makes the notebook itself the single source of truth: edit a rule,
save the .ipynb, and the backend watcher (see main.py) re-executes it and
hot-swaps in a freshly trained model.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless - never try to open a GUI window
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype

TARGET_COLUMNS = [
    "Iron_Deficiency_Risk",
    "Calcium_Deficiency_Risk",
    "Vitamin_D_Deficiency_Risk",
    "Vitamin_B12_Deficiency_Risk",
]
CATEGORICAL_COLUMNS = ["Gender", "Diet_Type", "Physical_Activity"]


def _load_code_cells(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]


def extract_clean_dataframe(nb_path: Path, timeout_seconds: float = 300.0) -> pd.DataFrame:
    """
    Run the notebook's code cells top-to-bottom in a shared namespace and
    return the dataframe snapshot from the last moment before categorical
    encoding happens. Cells that error (e.g. rely on Jupyter-only display
    quirks) are logged and skipped rather than aborting the whole run.
    """
    cells = _load_code_cells(nb_path)

    namespace = {
        "__name__": "__notebook_runtime__",
        "display": lambda *a, **k: None,   # Jupyter's display(), no-op here
    }
    plt.show = lambda *a, **k: None  # never blocks / opens a window

    snapshot = None
    start = time.time()

    for i, code in enumerate(cells):
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Notebook execution exceeded {timeout_seconds:.0f}s while retraining."
            )
        try:
            exec(compile(code, f"<notebook cell {i}>", "exec"), namespace)
        except Exception as exc:
            print(f"[notebook-runtime] cell {i} raised {type(exc).__name__}: {exc} — skipping")
            continue

        df = namespace.get("df")
        if isinstance(df, pd.DataFrame) and all(t in df.columns for t in TARGET_COLUMNS):
            # "Still categorical" = not yet numerically encoded. Checking
            # dtype == object breaks on pandas >= 2.x/3.x, which may use a
            # dedicated string dtype instead of plain `object` - checking
            # is_numeric_dtype instead works across all pandas versions.
            still_categorical = all(
                col not in df.columns or not is_numeric_dtype(df[col])
                for col in CATEGORICAL_COLUMNS
            )
            if still_categorical:
                snapshot = df.copy()
            else:
                # Encoding has just happened - the snapshot we already have
                # (from the previous matching cell) is what we want. No
                # need to run the rest of the notebook (EDA plots, model
                # training cells, etc).
                break

    if snapshot is None:
        raise RuntimeError(
            "Could not find a clean dataframe containing "
            f"{TARGET_COLUMNS} with un-encoded categorical columns "
            f"{CATEGORICAL_COLUMNS}. Check that the notebook still creates "
            "these target/feature columns with these exact names."
        )

    return snapshot
