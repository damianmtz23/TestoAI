# src/testoai/cli.py

import argparse

import pandas as pd

from testoai.recommend import format_food_description, recommend  # canonical engine


def parse_food_groups(s: str):
    """Turn 'Beef Products,Poultry Products' into ['Beef Products', 'Poultry Products']."""
    if not s:
        return None
    groups = [g.strip() for g in s.split(",") if g.strip()]
    return groups or None


def main(argv=None):
    # 1) define the command and its flags
    p = argparse.ArgumentParser(
        prog="testoai",
        description="TestoAI – testosterone-supportive food recommender."
    )
    p.add_argument(
        "--food-groups",
        default=None,
        help="Comma-separated FoodGroup names to restrict results to (default: all groups)"
    )
    p.add_argument(
        "--activity",
        choices=["low", "moderate", "medium", "high"],
        default="moderate"
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Cap the number of recommendations returned (default: full curated set)"
    )

    # 2) read the user's inputs from the command line
    args = p.parse_args(argv)     # if argv is None, argparse uses sys.argv
    food_groups = parse_food_groups(args.food_groups)

    # 3) call the shared recommendation engine (same one demo.py uses)
    try:
        df = recommend(food_groups=food_groups, activity=args.activity, top_k=args.k)
    except FileNotFoundError as e:
        print(str(e))             # friendly message if artifacts are missing
        return
    except ValueError as e:
        print(str(e))             # friendly message if group is unknown/seeds missing
        return

    if df.empty:
        print("No recommendations found under current filters.")
        return

    # 4) print a neat table
    df = df.copy()
    df["Description"] = df["Descrip"].apply(format_food_description)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df[["FoodGroup", "Description", "final_score"]])
