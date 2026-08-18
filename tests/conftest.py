# tests/conftest.py
# Shared fixtures for a small, self-contained embeddings artifact so tests
# never need the real training pipeline, Kaggle, or network access.

import pandas as pd
import pytest

from testoai import recommend as recommend_module

ROWS = [
    # Beef Products (ideal group, group_score=2)
    dict(Descrip="Beef, variety meats and by-products, liver, raw", FoodGroup="Beef Products",
         Energy_kcal=135, Fat_g=3.8, Sugar_g=0.0, group_score=2, emb_0=0.90, emb_1=0.10),
    dict(Descrip="Beef, ground, 85% lean meat / 15% fat, raw", FoodGroup="Beef Products",
         Energy_kcal=215, Fat_g=15.0, Sugar_g=0.0, group_score=2, emb_0=0.85, emb_1=0.15),
    dict(Descrip="Beef, tenderloin, raw", FoodGroup="Beef Products",
         Energy_kcal=186, Fat_g=10.8, Sugar_g=0.0, group_score=2, emb_0=0.80, emb_1=0.20),
    # Dairy and Egg Products (ideal group)
    dict(Descrip="Egg, yolk, raw, fresh", FoodGroup="Dairy and Egg Products",
         Energy_kcal=322, Fat_g=26.5, Sugar_g=0.6, group_score=2, emb_0=0.70, emb_1=0.30),
    dict(Descrip="Milk, whole, 3.25% milkfat", FoodGroup="Dairy and Egg Products",
         Energy_kcal=61, Fat_g=3.3, Sugar_g=5.1, group_score=2, emb_0=0.50, emb_1=0.10),
    dict(Descrip="Cheese, cheddar", FoodGroup="Dairy and Egg Products",
         Energy_kcal=403, Fat_g=33.1, Sugar_g=0.5, group_score=2, emb_0=0.40, emb_1=0.20),
    # Fruits and Fruit Juices (ideal group) - sugar spread chosen so low/medium/high
    # activity levels each resolve to a different fruit_type bucket.
    dict(Descrip="Apples, raw, with skin", FoodGroup="Fruits and Fruit Juices",
         Energy_kcal=52, Fat_g=0.2, Sugar_g=10.4, group_score=2, emb_0=0.20, emb_1=0.90),
    dict(Descrip="Bananas, raw", FoodGroup="Fruits and Fruit Juices",
         Energy_kcal=89, Fat_g=0.3, Sugar_g=12.2, group_score=2, emb_0=0.25, emb_1=0.85),
    dict(Descrip="Strawberries, raw", FoodGroup="Fruits and Fruit Juices",
         Energy_kcal=32, Fat_g=0.3, Sugar_g=4.9, group_score=2, emb_0=0.15, emb_1=0.80),
    # Finfish and Shellfish Products (okay group) - includes a default seed food.
    dict(Descrip="Mollusks, oyster, eastern, wild, raw", FoodGroup="Finfish and Shellfish Products",
         Energy_kcal=81, Fat_g=1.1, Sugar_g=0.0, group_score=1, emb_0=0.60, emb_1=0.40),
    # Poultry Products (okay group)
    dict(Descrip="Chicken, broilers or fryers, breast, meat only, raw", FoodGroup="Poultry Products",
         Energy_kcal=120, Fat_g=2.6, Sugar_g=0.0, group_score=1, emb_0=0.55, emb_1=0.35),
    # Baked Products - not an ideal/okay group (group_score=0), used to test the
    # empty-result path when filtering to a group the recommender never picks from.
    dict(Descrip="Bread, white, commercially prepared", FoodGroup="Baked Products",
         Energy_kcal=266, Fat_g=3.3, Sugar_g=5.0, group_score=0, emb_0=0.10, emb_1=0.10),
]


@pytest.fixture
def artifact_df():
    return pd.DataFrame(ROWS)


@pytest.fixture
def artifact_path(tmp_path, monkeypatch, artifact_df):
    """Write a small embeddings artifact and point testoai.recommend at it."""
    path = tmp_path / "emb_df.parquet"
    artifact_df.to_parquet(path, index=False)
    monkeypatch.setattr(recommend_module, "DATA_PATH", path)
    recommend_module._cache.clear()
    yield path
    recommend_module._cache.clear()
