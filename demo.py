# demo.py
# Loads precomputed embeddings (emb_df.parquet) and prints instant recommendations.

import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join("data", "emb_df.parquet")

# Helpers
def classify_fruit_dynamic(df):
    """Adds fruit_type column based on Sugar_g distance to min/mean/max within fruits."""
    fruit_pool = df[df["FoodGroup"] == "Fruits and Fruit Juices"].copy()
    fruit_pool = fruit_pool[~fruit_pool["Descrip"].str.lower().str.contains(
        "juice|smoothie|dried|puree|syrup|powder|fruit punch|cocktail|v8|babyfood|canned|frozen"
    )]

    if fruit_pool.empty or fruit_pool["Sugar_g"].isna().all():
        df["fruit_type"] = "unknown"
        return df

    sugar_min = fruit_pool["Sugar_g"].min()
    sugar_max = fruit_pool["Sugar_g"].max()
    sugar_mean = fruit_pool["Sugar_g"].mean()

    def bucket(s):
        if pd.isna(s): return "unknown"
        dmin, dmax, dmean = abs(s - sugar_min), abs(s - sugar_max), abs(s - sugar_mean)
        if dmin < dmean and dmin < dmax: return "low_sugar"
        if dmax < dmean and dmax < dmin: return "high_sugar"
        return "moderate_sugar"

    df["fruit_type"] = df["Sugar_g"].apply(bucket)
    return df

def is_organ(desc: str) -> bool:
    d = desc.lower()
    return any(k in d for k in ["liver","kidney","heart","pancreas","brain","sweetbread","spleen","tongue"])

def activity_filter(row, level):
    kcal = row.get("Energy_kcal", None)
    fat  = row.get("Fat_g", None)
    if kcal is None or fat is None:
        return True
    if level == "low":
        return kcal <= 250 and fat <= 12
    if level == "medium":
        return kcal <= 400
    return True  # high

def format_food_description(desc: str) -> str:
    blacklist = [
        "separable lean","separable fat","retail cuts","trimmed to","all grades",
        "composite of trimmed","choice","select","prime","raw","cooked","imported",
        "grades","composite","domestic","manufacturing","boneless","lean only","fat only",
        "broiled","grilled","braised","roasted","moist heat","dry heat"
    ]
    parts = [p.strip() for p in desc.lower().split(",") if p.strip()]
    clean = [p for p in parts if not any(b in p for b in blacklist)]
    cook  = [p for p in parts if any(x in p for x in ["cooked","raw","broiled","grilled","braised","roasted","moist heat","dry heat"])]
    main = " ".join(clean[:3]).title() if clean else " ".join(parts[:2]).title()
    return f"{main} ({', '.join(set(cook)).title()})" if cook else main

# Load embeddings
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("data/emb_df.parquet not found. Run: python train.py")

emb_df = pd.read_parquet(DATA_PATH)

# Identify embedding columns
emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
if not emb_cols:
    raise ValueError("No embedding columns found. Re-run training.")

# Fruit classification for later
emb_df = classify_fruit_dynamic(emb_df)

# Group weights (kept in sync with training)
group_weights = {2: 1.0, 1: 0.3, 0: 0.0}

# Default testosterone-supportive seeds
seeds = [
    "Mollusks, oyster, eastern, wild, raw",
    "Beef, variety meats and by-products, liver, raw",
    "Egg, yolk, raw, fresh"
]

print("=== Testosterone-Boosting Food Recommender (Demo) ===")
print("Press Enter to consider all FoodGroups, or type an exact FoodGroup name.")
print("Type 'exit' to quit.\n")

while True:
    grp = input("FoodGroup> ").strip()
    if grp.lower() in ("exit","quit"):
        break
    if grp and grp not in emb_df["FoodGroup"].unique():
        print("  Not found. Try again or leave blank to use all groups.\n")
        continue

    print("\nSelect your activity level:")
    print("  1) Low    (0–2 workouts/week)")
    print("  2) Medium (3–4 workouts/week)")
    print("  3) High   (5+ workouts/week)")
    lv = input("Activity Level (1/2/3)> ").strip()
    level = "low" if lv == "1" else "high" if lv == "3" else "medium"

    # Activity filter
    filtered = emb_df[emb_df.apply(lambda r: activity_filter(r, level), axis=1)].copy()

    # Exclude processed/combined items
    bad_terms = "with fruit|yogurt|milkshake|smoothie|pudding|dessert|custard|ice cream|imitation"
    filtered = filtered[~filtered["Descrip"].str.lower().str.contains(bad_terms)]

    # Build prototype from seeds
    valid = filtered[filtered["Descrip"].isin(seeds)][emb_cols]
    if valid.empty:
        print("Seed foods missing from filtered set; try again with different activity/group.\n")
        continue
    proto = valid.mean(axis=0).values.reshape(1, -1)

    # Base similarity
    filtered["sim_to_ideal"] = cosine_similarity(filtered[emb_cols], proto).flatten()
    filtered["sim_to_ideal"] *= filtered["group_score"].map(group_weights).fillna(0.0)

    # Optional group restriction
    if grp:
        filtered = filtered[filtered["FoodGroup"] == grp]

    # Fruit pick based on activity
    fruit_pool = filtered[filtered["FoodGroup"] == "Fruits and Fruit Juices"].copy()
    if level == "low":
        fruit_choice = fruit_pool[fruit_pool["fruit_type"] == "low_sugar"]
    elif level == "medium":
        fruit_choice = fruit_pool[fruit_pool["fruit_type"] == "moderate_sugar"]
    else:
        fruit_choice = fruit_pool[fruit_pool["fruit_type"] == "high_sugar"]
    top_fruit = fruit_choice.sort_values("sim_to_ideal", ascending=False).head(1)

    # Beef group logic (1 organ max, 3 muscle meats)
    final = []

    def beef_block(df_beef):
        sub = df_beef.copy()
        sub = sub[~sub["Descrip"].str.lower().str.contains(
            "subcutaneous fat|seam fat|intermuscular fat|retail cuts, separable fat|suet|tallow|lard|marrow|canned|imitation|patties|corned|processed"
        )]
        if level == "low":
            sub["adjusted_sim"] = sub["sim_to_ideal"] * (1 - sub["Fat_g"].fillna(0))
        elif level == "medium":
            sub["adjusted_sim"] = sub["sim_to_ideal"]
        else:
            sub["adjusted_sim"] = sub["sim_to_ideal"] * (1 + sub["Fat_g"].fillna(0))
        sub = sub.sort_values("adjusted_sim", ascending=False)
        organs = sub[sub["Descrip"].apply(is_organ)]
        others = sub[~sub["Descrip"].apply(is_organ)]

        picks, seen = [], set()
        def base_key(d): return ", ".join(d.split(",")[:2]).strip().lower()
        if not organs.empty:
            r = organs.iloc[0]; picks.append(r); seen.add(base_key(r["Descrip"]))
        for _, r in others.iterrows():
            k = base_key(r["Descrip"])
            if k in seen: continue
            picks.append(r); seen.add(k)
            if len(picks) >= 4: break
        return picks

    # Collect ideal-group picks
    ideal_groups = ["Beef Products", "Dairy and Egg Products", "Fruits and Fruit Juices"]
    for g in ideal_groups:
        sub = filtered[filtered["FoodGroup"] == g].copy()
        if g == "Beef Products":
            final.extend(beef_block(sub))
        elif g == "Dairy and Egg Products":
            eggs = sub[sub["Descrip"].str.lower().str.contains("egg")]
            others = sub[~sub["Descrip"].str.lower().str.contains("egg")]
            picks = []
            if not eggs.empty: picks.append(eggs.iloc[0])
            picks.extend(others.sort_values("sim_to_ideal", ascending=False).head(3 if not eggs.empty else 4))
            final.extend(picks)
        elif g == "Fruits and Fruit Juices":
            if not top_fruit.empty:
                final.extend(top_fruit.to_dict("records"))

    # Add one from each okay group (no organs)
    okay_groups = ["Lamb, Veal, and Game Products", "Finfish and Shellfish Products",
                   "Nut and Seed Products", "Pork Products", "Poultry Products"]
    for g in okay_groups:
        sub = filtered[(filtered["FoodGroup"] == g) & (~filtered["Descrip"].apply(is_organ))].copy()
        if level == "low":
            sub["adjusted_sim"] = sub["sim_to_ideal"] - 0.03 * sub["Fat_g"].fillna(0)
        elif level == "medium":
            sub["adjusted_sim"] = sub["sim_to_ideal"]
        else:
            sub["adjusted_sim"] = sub["sim_to_ideal"] + 0.015 * sub["Fat_g"].fillna(0)
        if not sub.empty:
            final.append(sub.sort_values("adjusted_sim", ascending=False).iloc[0])

    if not final:
        print("No recommendations found under current filters.\n")
        continue

    import pandas as pd
    out = pd.DataFrame(final).copy()
    out["final_score"] = out.get("adjusted_sim", out["sim_to_ideal"])
    out = out.sort_values("final_score", ascending=False)

    print("\n🧠 Final Testosterone-Boosting Picks:\n")
    for group in out["FoodGroup"].unique():
        print(f"🍽️  {group}")
        for _, row in out[out["FoodGroup"] == group].iterrows():
            print(f"  - {format_food_description(row['Descrip'])} — Score: {round(row['final_score'], 2)}")
        print()
