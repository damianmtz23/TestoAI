from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

def _find_col(candidates: List[str], cols: List[str]) -> Optional[str]:
    """
    Find the first column in `cols` that matches any of the case-insensitive
    `candidates`. This lets us handle datasets with slightly different names,
    e.g. 'Protein_g' vs 'protein'.
    """
    lower = {c.lower(): c for c in cols}
    for want in candidates:
        if want.lower() in lower:
            return lower[want.lower()]
    return None


def _require_cols(df: pd.DataFrame, needed: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Map logical names ('protein','fat','carbs') -> actual df column names.
    Raises a clear error if a required column is missing.
    """
    resolved = {}
    cols = list(df.columns)
    for logical, choices in needed.items():
        col = _find_col(choices, cols)
        if col is None:
            raise KeyError(
                f"Missing a '{logical}' column; looked for any of: {choices}. "
                "Check your dataset or adjust synonyms in recommend.py."
            )
        resolved[logical] = col
    return resolved


def recommend(targets: Dict[str, float], top_k: int = 20) -> pd.DataFrame:
    

    path = Path("data/emb_df.parquet")
    if not path.exists():
        raise FileNotFoundError(
            "Couldn't find data/emb_df.parquet. Run `python train.py` first "
            "to generate artifacts."
        )

    df = pd.read_parquet(path)

    colmap = _require_cols(
        df,
        needed={
            "protein": ["Protein_g", "protein_g", "protein"],
            "fat": ["Fat_g", "fat_g", "fat"],
            "carbs": ["Carb_g", "carbs_g", "Carb", "carbs"],
        },
    )
    p = df[colmap["protein"]].astype(float)
    f = df[colmap["fat"]].astype(float)
    c = df[colmap["carbs"]].astype(float)

    def pick(d: Dict[str, float], *keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in d:
                return float(d[k])
            if k.lower() in d:
                return float(d[k.lower()])
        return default

    t_pro = pick(targets, "protein", "protein_g")
    t_fat = pick(targets, "fat", "fat_g")
    t_carb = pick(targets, "carbs", "carb", "carb_g", "carbs_g")

    X = np.column_stack([p.values, f.values, c.values]).astype(float)
    W = np.diag([1.6, 0.7, 0.5])
    Xw = X @ W

    t = np.array([t_pro, t_fat, t_carb], dtype=float)
    tw = t @ W

    Xw_norm = Xw / (np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-8)
    tw_norm = tw / (np.linalg.norm(tw) + 1e-8)

    score = Xw_norm @ tw_norm


    out_cols = []
    for nice in ["Description", "description", "Food_Description", "name", "long_desc"]:
        if nice in df.columns:
            out_cols.append(nice)
            break
    for grp in ["FoodGroup", "food_group", "Group", "group"]:
        if grp in df.columns:
            out_cols.append(grp)
            break

    out_cols += [colmap["protein"], colmap["fat"], colmap["carbs"]]
    result = df.assign(score=score)[out_cols + ["score"]].sort_values("score", ascending=False).head(top_k)
    
    return result.reset_index(drop=True)
