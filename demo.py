# demo.py
# Interactive shell that collects user input and displays results from the
# canonical recommender in src/testoai/recommend.py.

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from testoai.recommend import format_food_description, load_embeddings, recommend

print("=== Testosterone-Boosting Food Recommender (Demo) ===")
print("Press Enter to consider all FoodGroups, or type an exact FoodGroup name.")
print("Type 'exit' to quit.\n")

try:
    emb_df, _ = load_embeddings()
except FileNotFoundError as e:
    print(str(e))
    sys.exit(1)

while True:
    try:
        grp = input("FoodGroup> ").strip()
    except EOFError:
        print()
        break
    if grp.lower() in ("exit", "quit"):
        break
    if grp and grp not in emb_df["FoodGroup"].unique():
        print("  Not found. Try again or leave blank to use all groups.\n")
        continue

    print("\nSelect your activity level:")
    print("  1) Low    (0–2 workouts/week)")
    print("  2) Medium (3–4 workouts/week)")
    print("  3) High   (5+ workouts/week)")
    try:
        lv = input("Activity Level (1/2/3)> ").strip()
    except EOFError:
        print()
        break
    level = "low" if lv == "1" else "high" if lv == "3" else "medium"

    try:
        out = recommend(food_groups=grp or None, activity=level)
    except ValueError as e:
        print(f"{e}\n")
        continue

    if out.empty:
        print("No recommendations found under current filters.\n")
        continue

    print("\nFinal Testosterone-Boosting Picks:\n")
    for group in out["FoodGroup"].unique():
        print(group)
        for _, row in out[out["FoodGroup"] == group].iterrows():
            print(f"  - {format_food_description(row['Descrip'])} — Score: {round(row['final_score'], 2)}")
        print()
