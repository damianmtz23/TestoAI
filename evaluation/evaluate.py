# evaluation/evaluate.py
# Final ML evaluation: does the learned autoencoder embedding help compared
# with simpler nutrient-based baselines? This is a descriptive, portfolio-
# level comparison, not an accuracy benchmark -- there's no labeled ground
# truth for "correct" food recommendations.
#
# Methods 1-3 use the exact same seed foods, candidate set, cosine-similarity
# approach, and TOP_N; only the feature representation changes, so any
# difference between them is attributable to the representation. Method 4
# calls the real recommend() function and is a system-level comparison that
# also includes activity/food-group rules, not a pure representation test.

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from testoai.recommend import SEED_FOODS, recommend  # noqa: E402

TOP_N = 10

DATA_PATH = Path("data") / "emb_df.parquet"
SCALER_PATH = Path("model") / "scaler.pkl"
META_PATH = Path("model") / "meta.json"
RESULTS_PATH = Path(__file__).parent / "results.csv"

MACRO_COLS = ["Protein_g", "Fat_g", "Carb_g"]


def load_artifacts():
    missing = [p for p in (DATA_PATH, SCALER_PATH, META_PATH) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        sys.exit(f"Missing training artifact(s): {names}. Run: python train.py")

    df = pd.read_parquet(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = json.load(open(META_PATH))

    nutr_cols = meta["nutr_cols"]
    emb_cols = meta["emb_cols"]
    missing_cols = set(nutr_cols + emb_cols) - set(df.columns)
    if missing_cols:
        sys.exit(
            f"{DATA_PATH} is missing expected columns {sorted(missing_cols)}. "
            "Re-run: python train.py"
        )
    return df, scaler, nutr_cols, emb_cols


def build_prototype(matrix, seed_mask):
    return matrix[seed_mask].mean(axis=0).reshape(1, -1)


def score_candidates(df, matrix, seed_mask, score_col):
    """Cosine similarity of every row to the seed prototype, scored on the
    full candidate set (seeds included) so full-ranking correlations are
    comparable across methods."""
    proto = build_prototype(matrix, seed_mask)
    scores = cosine_similarity(matrix, proto).flatten()
    out = df.copy()
    out[score_col] = scores
    return out


def top_n_excluding_seeds(scored_df, score_col, n=TOP_N):
    non_seed = scored_df[~scored_df["Descrip"].isin(SEED_FOODS)]
    return non_seed.sort_values(score_col, ascending=False).head(n)


def summarize(top_df):
    return {
        "avg_protein_g": top_df["Protein_g"].mean(),
        "avg_sugar_g": top_df["Sugar_g"].mean(),
        "avg_fiber_g": top_df["Fiber_g"].mean(),
        "avg_energy_kcal": top_df["Energy_kcal"].mean(),
        "unique_food_groups": top_df["FoodGroup"].nunique(),
    }


def overlap(top_a, top_b):
    a, b = set(top_a["Descrip"]), set(top_b["Descrip"])
    count = len(a & b)
    pct = 100 * count / min(len(a), len(b)) if a and b else 0.0
    return count, pct


def rank_correlation(scored_a, scored_b, score_col_a, score_col_b):
    """Pearson correlation of rank positions == Spearman correlation, computed
    without an extra dependency. Both inputs must share the same row order/set."""
    merged = scored_a[["Descrip", score_col_a]].merge(
        scored_b[["Descrip", score_col_b]], on="Descrip"
    )
    rank_a = merged[score_col_a].rank()
    rank_b = merged[score_col_b].rank()
    return rank_a.corr(rank_b)


def print_top10(label, top_df, score_col):
    print(f"\n--- {label}: top {len(top_df)} ---")
    for _, row in top_df.iterrows():
        print(f"  {row['Descrip']:<55} | {row['FoodGroup']:<30} | score={row[score_col]:.4f}")


def main():
    df, scaler, nutr_cols, emb_cols = load_artifacts()
    print(f"Loaded artifact with {len(df)} rows from {DATA_PATH}.")
    if len(df) < 500:
        print(
            "WARNING: this looks like the small bundled sample fixture, not the "
            "full USDA/Kaggle dataset. Results below are a smoke test only."
        )

    seed_mask = df["Descrip"].isin(SEED_FOODS).to_numpy()

    # Normalize the full nutrient matrix with the fitted scaler, then select
    # subsets for methods 1 and 2.
    norm_matrix = scaler.transform(df[nutr_cols])
    norm_df = pd.DataFrame(norm_matrix, columns=nutr_cols)
    macro_matrix = norm_df[MACRO_COLS].to_numpy()
    raw_matrix = norm_df[nutr_cols].to_numpy()
    emb_matrix = df[emb_cols].to_numpy()

    scored_macro = score_candidates(df, macro_matrix, seed_mask, "macro_score")
    scored_raw = score_candidates(df, raw_matrix, seed_mask, "raw_score")
    scored_emb = score_candidates(df, emb_matrix, seed_mask, "emb_score")

    top_macro = top_n_excluding_seeds(scored_macro, "macro_score")
    top_raw = top_n_excluding_seeds(scored_raw, "raw_score")
    top_emb = top_n_excluding_seeds(scored_emb, "emb_score")

    # Method 4: canonical recommender, system-level comparison (includes rules).
    final_df = recommend(activity="medium", top_k=None, per_group_limit=50)
    final_df = final_df[~final_df["Descrip"].isin(SEED_FOODS)]
    top_final = final_df.head(TOP_N)

    methods = {
        "macro_baseline": (top_macro, "macro_score"),
        "raw_nutrient_baseline": (top_raw, "raw_score"),
        "autoencoder_embedding": (top_emb, "emb_score"),
        "final_testoai": (top_final, "final_score"),
    }

    # --- summary metrics ---
    print("\n=== Summary metrics (top-{}) ===".format(TOP_N))
    rows = []
    for name, (top_df, score_col) in methods.items():
        stats = summarize(top_df)
        rows.append({"method": name, "top_n": len(top_df), **stats})
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # --- overlap ---
    print("\n=== Top-N overlap (count / pct) ===")
    overlap_pairs = [
        ("macro_baseline", "raw_nutrient_baseline", top_macro, top_raw),
        ("macro_baseline", "autoencoder_embedding", top_macro, top_emb),
        ("raw_nutrient_baseline", "autoencoder_embedding", top_raw, top_emb),
        ("autoencoder_embedding", "final_testoai", top_emb, top_final),
    ]
    overlap_lookup = {}
    for name_a, name_b, ta, tb in overlap_pairs:
        count, pct = overlap(ta, tb)
        overlap_lookup[(name_a, name_b)] = (count, pct)
        print(f"  {name_a} vs {name_b}: {count}/{TOP_N} ({pct:.0f}%)")

    # overlap of each method's top-N with the embedding method's top-N, for the CSV.
    embedding_overlap = {}
    for name, (top_df, _) in methods.items():
        count, pct = overlap(top_df, top_emb)
        embedding_overlap[name] = (count, pct)

    # --- rank correlation across full candidate rankings (methods 1-3 only) ---
    print("\n=== Rank correlation (full candidate ranking, methods 1-3) ===")
    corr_macro_raw = rank_correlation(scored_macro, scored_raw, "macro_score", "raw_score")
    corr_macro_emb = rank_correlation(scored_macro, scored_emb, "macro_score", "emb_score")
    corr_raw_emb = rank_correlation(scored_raw, scored_emb, "raw_score", "emb_score")
    print(f"  macro vs raw nutrient:        {corr_macro_raw:.3f}")
    print(f"  macro vs embedding:           {corr_macro_emb:.3f}")
    print(f"  raw nutrient vs embedding:    {corr_raw_emb:.3f}")

    rank_corr_with_embedding = {
        "macro_baseline": corr_macro_emb,
        "raw_nutrient_baseline": corr_raw_emb,
        "autoencoder_embedding": 1.0,
        "final_testoai": None,
    }

    # --- representative recommendations ---
    print("\n=== Representative top-{} recommendations ===".format(TOP_N))
    for name, (top_df, score_col) in methods.items():
        print_top10(name, top_df, score_col)

    # --- write results.csv ---
    for row in rows:
        name = row["method"]
        count, pct = embedding_overlap[name]
        row["overlap_with_embedding_count"] = count
        row["overlap_with_embedding_pct"] = pct
        row["rank_corr_with_embedding"] = rank_corr_with_embedding[name]
        row["dataset_rows"] = len(df)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
