# tests/test_recommend.py

import pandas as pd
import pytest

from testoai.recommend import load_embeddings, recommend


def test_recommend_returns_results_from_sample_artifact(artifact_path):
    out = recommend()
    assert isinstance(out, pd.DataFrame)
    assert not out.empty


def test_recommend_output_has_expected_columns(artifact_path):
    out = recommend()
    for col in ("FoodGroup", "Descrip", "final_score"):
        assert col in out.columns


@pytest.mark.parametrize("top_k", [1, 2])
def test_top_k_limits_number_of_recommendations(artifact_path, top_k):
    out = recommend(top_k=top_k)
    assert len(out) <= top_k


def test_food_group_filter_restricts_results_to_requested_group(artifact_path):
    out = recommend(food_groups="Beef Products")
    assert not out.empty
    assert set(out["FoodGroup"].unique()) == {"Beef Products"}


@pytest.mark.parametrize("activity", ["low", "medium", "moderate", "high"])
def test_activity_levels_run_without_crashing(artifact_path, activity):
    out = recommend(activity=activity)
    assert isinstance(out, pd.DataFrame)


@pytest.mark.parametrize(
    "activity,expected_fruit",
    [
        ("low", "Strawberries, raw"),
        ("medium", "Apples, raw, with skin"),
        ("high", "Bananas, raw"),
    ],
)
def test_fruit_group_pick_matches_activity_sugar_bucket(artifact_path, activity, expected_fruit):
    out = recommend(food_groups="Fruits and Fruit Juices", activity=activity)
    assert set(out["FoodGroup"].unique()) == {"Fruits and Fruit Juices"}
    assert list(out["Descrip"]) == [expected_fruit]


def test_unrecommendable_group_returns_empty_result(artifact_path):
    out = recommend(food_groups="Baked Products")
    assert out.empty


def test_unknown_food_group_raises_value_error(artifact_path):
    with pytest.raises(ValueError):
        recommend(food_groups="Not A Real FoodGroup")


def test_missing_required_columns_raises_clear_error(tmp_path, monkeypatch):
    import testoai.recommend as recommend_module

    bad_df = pd.DataFrame({"Descrip": ["x"], "FoodGroup": ["Beef Products"], "emb_0": [0.1]})
    path = tmp_path / "bad_emb_df.parquet"
    bad_df.to_parquet(path, index=False)
    monkeypatch.setattr(recommend_module, "DATA_PATH", path)
    recommend_module._cache.clear()

    with pytest.raises(ValueError, match="missing required columns"):
        load_embeddings()

    recommend_module._cache.clear()


def test_missing_artifact_file_raises_clear_error(tmp_path, monkeypatch):
    import testoai.recommend as recommend_module

    monkeypatch.setattr(recommend_module, "DATA_PATH", tmp_path / "does_not_exist.parquet")
    recommend_module._cache.clear()

    with pytest.raises(FileNotFoundError):
        load_embeddings()
