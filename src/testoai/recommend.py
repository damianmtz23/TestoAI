# src/testoai/recommend.py
# Canonical recommendation engine, shared by demo.py and cli.py.
# Algorithm ported as-is from the original demo.py (embeddings + food-group rules).

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data") / "emb_df.parquet"

# Default testosterone-supportive seeds used to build the similarity prototype.
SEED_FOODS = [
    "Mollusks, oyster, eastern, wild, raw",
    "Beef, variety meats and by-products, liver, raw",
    "Egg, yolk, raw, fresh",
]

# Group weights kept in sync with train.py's group_score column.
GROUP_WEIGHTS = {2: 1.0, 1: 0.3, 0: 0.0}

IDEAL_GROUPS = ["Beef Products", "Dairy and Egg Products", "Fruits and Fruit Juices"]
OKAY_GROUPS = [
    "Lamb, Veal, and Game Products",
    "Finfish and Shellfish Products",
    "Nut and Seed Products",
    "Pork Products",
    "Poultry Products",
]

_BAD_TERMS = (
    "with fruit|yogurt|milkshake|smoothie|pudding|dessert|custard|ice cream|imitation|"
    "rendered fat|subcutaneous fat|separable fat|fat only|suet|tallow|lard|corned|patties|"
    "cured|bacon|skin only|retail cuts"
)

_FRUIT_EXCLUDE = (
    "juice|smoothie|dried|puree|syrup|powder|fruit punch|cocktail|v8|babyfood|canned|frozen"
)

_BEEF_EXCLUDE = (
    "subcutaneous fat|seam fat|intermuscular fat|retail cuts, separable fat|suet|tallow|lard|"
    "marrow|canned|imitation|patties|corned|processed"
)

_ACTIVITY_ALIASES = {"moderate": "medium", "normal": "medium"}

# Columns the recommender actually reads from data/emb_df.parquet. Kept in
# sync with train.py's artifact generation.
REQUIRED_COLUMNS = {"Descrip", "FoodGroup", "Energy_kcal", "Fat_g", "Sugar_g", "group_score"}

_cache: Dict[str, Tuple[pd.DataFrame, List[str]]] = {}


# --- small helpers -----------------------------------------------------

def _normalize_activity(activity: Optional[str]) -> str:
    level = (activity or "medium").strip().lower()
    level = _ACTIVITY_ALIASES.get(level, level)
    return level if level in ("low", "medium", "high") else "medium"


def is_organ(desc: str) -> bool:
    d = desc.lower()
    return any(k in d for k in ["liver", "kidney", "heart", "pancreas", "brain", "sweetbread", "spleen", "tongue"])


def format_food_description(desc: str) -> str:
    blacklist = [
        "separable lean", "separable fat", "retail cuts", "trimmed to", "all grades",
        "composite of trimmed", "choice", "select", "prime", "raw", "cooked", "imported",
        "grades", "composite", "domestic", "manufacturing", "boneless", "lean only", "fat only",
        "broiled", "grilled", "braised", "roasted", "moist heat", "dry heat",
    ]
    parts = [p.strip() for p in desc.lower().split(",") if p.strip()]
    clean = [p for p in parts if not any(b in p for b in blacklist)]
    cook = [p for p in parts if any(x in p for x in ["cooked", "raw", "broiled", "grilled", "braised", "roasted", "moist heat", "dry heat"])]
    main = " ".join(clean[:3]).title() if clean else " ".join(parts[:2]).title()
    return f"{main} ({', '.join(set(cook)).title()})" if cook else main


def _classify_fruit_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    """Adds fruit_type column based on Sugar_g distance to min/mean/max within fruits."""
    fruit_pool = df[df["FoodGroup"] == "Fruits and Fruit Juices"].copy()
    fruit_pool = fruit_pool[~fruit_pool["Descrip"].str.lower().str.contains(_FRUIT_EXCLUDE)]

    if fruit_pool.empty or fruit_pool["Sugar_g"].isna().all():
        df["fruit_type"] = "unknown"
        return df

    sugar_min = fruit_pool["Sugar_g"].min()
    sugar_max = fruit_pool["Sugar_g"].max()
    sugar_mean = fruit_pool["Sugar_g"].mean()

    def bucket(s):
        if pd.isna(s):
            return "unknown"
        dmin, dmax, dmean = abs(s - sugar_min), abs(s - sugar_max), abs(s - sugar_mean)
        if dmin < dmean and dmin < dmax:
            return "low_sugar"
        if dmax < dmean and dmax < dmin:
            return "high_sugar"
        return "moderate_sugar"

    df["fruit_type"] = df["Sugar_g"].apply(bucket)
    return df


def _activity_filter(row, level: str) -> bool:
    kcal = row.get("Energy_kcal", None)
    fat = row.get("Fat_g", None)
    if kcal is None or fat is None:
        return True
    if level == "low":
        return kcal <= 250 and fat <= 12
    if level == "medium":
        return kcal <= 400
    return True  # high


def _build_prototype(filtered: pd.DataFrame, emb_cols: List[str], seeds: Iterable[str]):
    valid = filtered[filtered["Descrip"].isin(list(seeds))][emb_cols]
    if valid.empty:
        raise ValueError("Seed foods missing from filtered set; try different activity/group.")
    return valid.mean(axis=0).values.reshape(1, -1)


def _beef_block(df_beef: pd.DataFrame, level: str) -> List[dict]:
    sub = df_beef.copy()
    sub = sub[~sub["Descrip"].str.lower().str.contains(_BEEF_EXCLUDE)]

    if level == "low":
        sub["adjusted_sim"] = sub["sim_to_ideal"] - 0.02 * sub["Fat_g"].fillna(0)
    elif level == "medium":
        sub["adjusted_sim"] = sub["sim_to_ideal"]
    else:
        sub["adjusted_sim"] = sub["sim_to_ideal"] + 0.02 * sub["Fat_g"].fillna(0)

    sub = sub.sort_values("adjusted_sim", ascending=False)
    organs = sub[sub["Descrip"].apply(is_organ)]
    others = sub[~sub["Descrip"].apply(is_organ)]

    picks, seen = [], set()

    def base_key(d):
        return ", ".join(d.split(",")[:2]).strip().lower()

    if not organs.empty:
        r = organs.iloc[0]
        picks.append(r.to_dict())
        seen.add(base_key(r["Descrip"]))

    for _, r in others.iterrows():
        k = base_key(r["Descrip"])
        if k in seen:
            continue
        picks.append(r.to_dict())
        seen.add(k)
        if len(picks) >= 4:
            break
    return picks


# --- artifact loading ----------------------------------------------------

def load_embeddings(path: Optional[Union[str, Path]] = None, refresh: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Load the trained embeddings artifact (data/emb_df.parquet), with fruit
    classification applied. Cached per path so repeated calls (e.g. from an
    interactive loop) don't re-read the parquet file each time."""
    path = Path(path) if path else DATA_PATH
    key = str(path)
    if not refresh and key in _cache:
        return _cache[key]

    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python train.py")

    df = pd.read_parquet(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}. "
            "The training artifact may be stale — regenerate it with `python train.py`."
        )

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found. Re-run training.")

    df = _classify_fruit_dynamic(df)
    _cache[key] = (df, emb_cols)
    return df, emb_cols


# --- canonical recommender -------------------------------------------------

def recommend(
    food_groups: Optional[Union[str, Iterable[str]]] = None,
    activity: str = "moderate",
    top_k: Optional[int] = None,
    targets: Optional[Dict[str, float]] = None,
    seeds: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Rank foods using the learned nutrient embeddings, activity-level
    adjustments, and the food-group nutrition rules originally in demo.py.

    food_groups: a single FoodGroup name, a list of names, or None for all groups.
    activity: "low", "medium"/"moderate", or "high".
    top_k: cap on the number of picks returned; None (default) returns the full
        curated set, matching the original demo.py behavior.
    targets: accepted for interface compatibility with earlier macro-based
        callers, but not used by this embedding-based engine (there's no
        macro -> embedding mapping without changing the model). Use `seeds`
        to influence the ranking instead.
    seeds: food descriptions used to build the similarity prototype; defaults
        to SEED_FOODS.
    """
    level = _normalize_activity(activity)
    df, emb_cols = load_embeddings()

    if isinstance(food_groups, str):
        requested_groups = [food_groups] if food_groups else []
    else:
        requested_groups = list(food_groups) if food_groups else []

    known_groups = set(df["FoodGroup"].unique())
    for g in requested_groups:
        if g not in known_groups:
            raise ValueError(f"Unknown FoodGroup: {g!r}")

    filtered = df[df.apply(lambda r: _activity_filter(r, level), axis=1)].copy()
    filtered = filtered[~filtered["Descrip"].str.lower().str.contains(_BAD_TERMS)]

    proto = _build_prototype(filtered, emb_cols, seeds or SEED_FOODS)

    filtered["sim_to_ideal"] = cosine_similarity(filtered[emb_cols], proto).flatten()
    filtered["sim_to_ideal"] *= filtered["group_score"].map(GROUP_WEIGHTS).fillna(0.0)

    if requested_groups:
        filtered = filtered[filtered["FoodGroup"].isin(requested_groups)]

    # Fruit pick based on activity level.
    fruit_pool = filtered[filtered["FoodGroup"] == "Fruits and Fruit Juices"].copy()
    fruit_pool = fruit_pool[~fruit_pool["Descrip"].str.lower().str.contains(_FRUIT_EXCLUDE)]
    fruit_bucket = {"low": "low_sugar", "medium": "moderate_sugar", "high": "high_sugar"}[level]
    fruit_choice = fruit_pool[fruit_pool["fruit_type"] == fruit_bucket]
    top_fruit = fruit_choice.sort_values("sim_to_ideal", ascending=False).head(1)

    final: List[dict] = []

    for g in IDEAL_GROUPS:
        sub = filtered[filtered["FoodGroup"] == g].copy()
        if g == "Beef Products":
            final.extend(_beef_block(sub, level))
        elif g == "Dairy and Egg Products":
            eggs = sub[sub["Descrip"].str.lower().str.contains("egg")]
            others = sub[~sub["Descrip"].str.lower().str.contains("egg")].sort_values("sim_to_ideal", ascending=False)
            picks = []
            if not eggs.empty:
                picks.append(eggs.iloc[0].to_dict())
                top_n = 3
            else:
                top_n = 4
            picks.extend(others.head(top_n).to_dict("records"))
            final.extend(picks)
        elif g == "Fruits and Fruit Juices":
            if not top_fruit.empty:
                final.extend(top_fruit.to_dict("records"))

    for g in OKAY_GROUPS:
        sub = filtered[(filtered["FoodGroup"] == g) & (~filtered["Descrip"].apply(is_organ))].copy()
        if level == "low":
            sub["adjusted_sim"] = sub["sim_to_ideal"] - 0.03 * sub["Fat_g"].fillna(0)
        elif level == "medium":
            sub["adjusted_sim"] = sub["sim_to_ideal"]
        else:
            sub["adjusted_sim"] = sub["sim_to_ideal"] + 0.015 * sub["Fat_g"].fillna(0)
        if not sub.empty:
            final.append(sub.sort_values("adjusted_sim", ascending=False).iloc[0].to_dict())

    if not final:
        return pd.DataFrame(columns=list(df.columns) + ["final_score"])

    out = pd.DataFrame(final).copy()
    out["final_score"] = out["adjusted_sim"] if "adjusted_sim" in out else out["sim_to_ideal"]
    out["final_score"] = out["final_score"].fillna(out["sim_to_ideal"])
    out = out.sort_values("final_score", ascending=False)
    if top_k:
        out = out.head(top_k)
    return out.reset_index(drop=True)
