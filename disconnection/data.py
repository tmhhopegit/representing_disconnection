"""Reading and arranging the data.

load_chaco        regional disconnection (ChaCo) scores saved by the ChaCo/NeMo
                  tool as <folder>/Lesion_<n>/Chaco116_MNI.mat (GetDisconn.m and
                  GetDisconMat.m, which were identical)
select_cohort     the patient selections of MakeRegMatrices.m ("small" and "big")
long_format       one row per behavioural assessment with the matching imaging
                  predictors, for mixed-effects models (MakeMixedEffectsDataset.m)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_chaco(folder, numbers, filename="Chaco116_MNI.mat", subfolder="Lesion_{n}"):
    """Stack the 'Regions' vector of each patient's ChaCo result into a matrix.

    folder:   the directory holding Lesion_<n> subfolders (the original used the
              hard-coded Windows path 'LesionImages\\Lesion_<n>\\Chaco116_MNI.mat')
    numbers:  the <n> of each patient, in the order wanted
    Missing files give a row of NaN (with a warning) instead of stopping."""
    from scipy.io import loadmat
    import warnings
    rows = []
    for n in numbers:
        path = Path(folder) / subfolder.format(n=n) / filename
        if not path.exists():
            warnings.warn(f"missing {path}")
            rows.append(None)
            continue
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        key = [k for k in m if not k.startswith("__")][0]
        obj = m[key]
        obj = obj[0] if isinstance(obj, np.ndarray) and obj.dtype == object else obj
        rows.append(np.asarray(obj.Regions, float).ravel())
    width = max((len(r) for r in rows if r is not None), default=0)
    return np.vstack([r if r is not None else np.full(width, np.nan) for r in rows])


def select_cohort(df: pd.DataFrame, which: str = "small") -> np.ndarray:
    """MakeRegMatrices.m's samples, from a table with columns order, age,
    months_post_stroke, lesion_volume, right_volume, handedness_left (0/1) and
    language_native (1 = native speaker).
    "big":   first assessment of each patient (order == 1)
    "small": also left-hemisphere stroke only (lesion volume > 0, right-hemisphere
             volume 0), right-handed, native speaker, age < 82, > 6 months post
             stroke."""
    s = df["order"].to_numpy() == 1
    if which == "small":
        s &= (df["handedness_left"].to_numpy() == 0) & (df["language_native"].to_numpy() > 0)
        s &= (df["lesion_volume"].to_numpy() > 0) & (df["right_volume"].to_numpy() == 0)
        s &= (df["age"].to_numpy() < 82) & (df["months_post_stroke"].to_numpy() > 6)
    elif which != "big":
        raise ValueError("which must be 'small' or 'big'")
    return s


def long_format(assessments: pd.DataFrame, imaging: pd.DataFrame, id_col="id") -> pd.DataFrame:
    """Attach each patient's imaging predictors to every one of their behavioural
    assessments (one row per assessment). assessments: id, time, scores...;
    imaging: id plus predictor columns (the first row per id is used, as the
    original did). Assessments without imaging are dropped (the original left
    them as empty rows)."""
    img = imaging.drop_duplicates(id_col, keep="first")
    return assessments.merge(img, on=id_col, how="inner")
