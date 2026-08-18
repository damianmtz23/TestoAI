# app.py
# Minimal Streamlit UI over the canonical recommender in src/testoai/recommend.py.
# No recommendation/scoring logic lives here — it only collects input and displays output.

import base64
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from testoai.recommend import (  # noqa: E402
    IDEAL_GROUPS,
    OKAY_GROUPS,
    format_food_description,
    load_embeddings,
    recommend,
)

ALL_GROUPS_LABEL = "All food groups"
BACKGROUND_IMAGE_PATH = Path(__file__).parent / "assets" / "primal_bg.png"

# Dark overlay keeps content readable over the background image; used as a
# plain color fallback too if the image can't be loaded.
_FALLBACK_STYLE = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #14100c;
}
</style>
"""


def apply_background(image_path: Path) -> None:
    """Apply a fixed, non-tiling background image with a dark readability
    overlay. Falls back to a plain dark background if the asset is missing
    or unreadable, so the app never crashes over a styling concern."""
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode()
    except OSError:
        st.markdown(_FALLBACK_STYLE, unsafe_allow_html=True)
        return

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(rgba(10, 8, 6, 0.72), rgba(10, 8, 6, 0.72)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(0, 0, 0, 0);
        }}
        div[data-testid="stForm"] {{
            background-color: rgba(20, 16, 12, 0.55);
            border-radius: 0.5rem;
            padding: 1.25rem 1.25rem 0.5rem 1.25rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="TestoAI", page_icon="🥩")
apply_background(BACKGROUND_IMAGE_PATH)

st.title("TestoAI")
st.write(
    "A nutrient-aware food recommendation system. It suggests foods that may "
    "support healthy testosterone levels, based on learned nutrient embeddings."
)

try:
    emb_df, _ = load_embeddings()
except FileNotFoundError:
    st.error("Training artifact not found. Run `python train.py` to generate it, then reload this app.")
    st.stop()
except ValueError as e:
    st.error(f"Training artifact is invalid or stale: {e}")
    st.stop()

# Only groups the recommender actually scores (see IDEAL_GROUPS/OKAY_GROUPS in
# recommend.py) — the raw artifact has many more FoodGroup values that would
# always come back empty (e.g. "Vegetables and Vegetable Products").
supported_groups = set(IDEAL_GROUPS) | set(OKAY_GROUPS)
known_groups = set(emb_df["FoodGroup"].unique())
food_group_options = [ALL_GROUPS_LABEL] + sorted(supported_groups & known_groups)

with st.form("recommend_form"):
    activity = st.selectbox("Activity level", ["Low", "Medium", "High"], index=1)
    food_group = st.selectbox("Food group", food_group_options, index=0)
    top_k = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    selected_groups = None if food_group == ALL_GROUPS_LABEL else food_group
    try:
        results = recommend(
            food_groups=selected_groups,
            activity=activity.lower(),
            top_k=top_k,
            per_group_limit=top_k,
        )
    except ValueError as e:
        st.error(str(e))
        results = None

    if results is not None:
        if results.empty:
            st.warning(
                "No recommendations found under the current filters. "
                "Try a different food group or activity level."
            )
        else:
            display_df = results.copy()
            display_df["Food"] = display_df["Descrip"].apply(format_food_description)
            display_df["Score"] = display_df["final_score"].round(3)
            display_df = display_df.rename(columns={"FoodGroup": "Food Group"})
            st.dataframe(
                display_df[["Food", "Food Group", "Score"]],
                hide_index=True,
                width="stretch",
            )

st.divider()
st.subheader("How it works")
st.write(
    "Nutrient data for each food is encoded into learned embeddings, which are then "
    "compared against nutrient-profile prototypes for testosterone-supportive foods "
    "using cosine similarity. Practical rules based on activity level and food group "
    "further adjust the ranking."
)
st.caption("These recommendations are informational only and are not medical advice.")
