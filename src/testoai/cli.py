# src/testoai/cli.py

import argparse
import pandas as pd                   # <-- needed for option_context
from testoai.recommend import recommend  # calls your ranking function


def parse_targets(s: str):
    """
    Turn 'protein=194,fat=65,carbs=247' into {'protein': 194.0, 'fat': 65.0, 'carbs': 247.0}
    """
    out = {}
    for kv in s.split(","):       # split on commas → ['protein=194', 'fat=65', 'carbs=247']
        if not kv.strip():        # skip empty pieces (just in case)
            continue
        k, v = kv.split("=")      # split 'protein=194' into k='protein', v='194'
        out[k.strip()] = float(v) # store as numbers (floats)
    return out


def main(argv=None):
    # 1) define the command and its flags
    p = argparse.ArgumentParser(
        prog="testoai",
        description="TestoAI – nutrient-aware food recommender."
    )
    p.add_argument(
        "--targets",
        default="protein=194,fat=65,carbs=247",
        help="Comma-separated targets like 'protein=194,fat=65,carbs=247'"
    )
    p.add_argument(
        "--activity",
        choices=["low", "moderate", "high"],
        default="moderate"
    )
    p.add_argument(
        "--k",
        type=int,
        default=20,
        help="Top-K recommendations to return"
    )

    # 2) read the user’s inputs from the command line
    args = p.parse_args(argv)     # if argv is None, argparse uses sys.argv
    targets = parse_targets(args.targets)

    # 3) call the library logic
    try:
        df = recommend(targets=targets, top_k=args.k)
    except FileNotFoundError as e:
        print(str(e))             # friendly message if artifacts are missing
        return
    except KeyError as e:
        print("Column mapping error:", e)  # friendly message if column names don’t match
        return

    # 4) print a neat table
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df)
