# Macro & Micronutrient Food Recommender

Recommend nutrient‑dense foods that support testosterone production by learning nutrient embeddings with a tiny autoencoder and matching to a testosterone‑supporting nutrient profile — filtered by activity level and food group rules.

---

## Overview

This project learns a compact representation of foods from the **USDA National Nutrient Database** and recommends items that best match a nutrient “prototype” designed to promote testosterone production. The prototype is built from seed foods known for their high nutrient density and key micronutrients (e.g., zinc, magnesium, selenium, B vitamins) — such as oysters, egg yolk, and beef liver. It prioritizes protein‑dense animal foods while still considering fruit and vegetable options based on their sugar/fiber profiles and mineral content.

**Core ideas**

* **Embeddings via Autoencoder** (PyTorch): 13→64→16 latent →64→13; MSE reconstruction loss.
* **Prototype matching**: Mean embedding of testosterone‑boosting seed foods → cosine similarity to every item.
* **Rules & weights**: Favor testosterone‑supportive food groups; cap organ meats; dedupe near‑duplicates; exclude processed items.
* **Activity‑aware**: Adjust scoring and calorie/fat filters for low/medium/high activity.
* **Dynamic fruit sugar buckets**: “low/moderate/high sugar” based on dataset distribution.

---

## Quickstart

### Requirements

* Python 3.10+
* `pandas`, `scikit-learn`, `torch`, `matplotlib`, `kagglehub`

```bash
pip install -r requirements.txt
```

### Run (CLI)

```bash
python ml_macro_project.py
```

Example session:

```
=== Testosterone‑Boosting Food Recommender ===
FoodGroup>  
Select your activity level:
   1. Low   (0–2 workouts/week)
   2. Medium (3–4 workouts/week)
   3. High  (5+ workouts/week)
Activity Level (1/2/3)> 2

Final Testosterone-Boosting Picks:

Beef Products
  - Beef Liver (Raw) — Score: 0.91
  - Beef Ribeye (Grilled) — Score: 0.86

Dairy and Egg Products
  - Egg (Whole) — Score: 0.83
  - Cottage Cheese — Score: 0.78

Fruits and Fruit Juices
  - Banana — Score: 0.72
  - Kiwi — Score: 0.68
```

---

## Data

* **Source:** USDA National Nutrient Database (via `kagglehub`)
* **Columns used:** `Energy_kcal`, `Protein_g`, `Fat_g`, `Carb_g`, `Zinc_mg`, `Magnesium_mg`, `VitB6_mg`, `VitB12_mcg`, `Selenium_mcg`, `VitA_mcg`, `Iron_mg`, `Sugar_g`, `Fiber_g`.
* **Preprocessing:** drop NAs; MinMax scaling to \[0,1].

---

## Model & Method

* **Autoencoder**: Linear → ReLU → Linear (latent=16) → Linear → ReLU → Linear; trained \~10 epochs with Adam (1e‑3) and MSE.
* **Embeddings**: Use encoder output as the nutrient embedding for each food.
* **Prototype vector**: Mean of seed foods rich in testosterone‑supportive micronutrients.
* **Similarity**: Cosine similarity to prototype.
* **Group weighting**: Ideal groups (Beef, Dairy/Egg, Fruits, select Vegetables) weighted above Okay groups (Lamb/Veal/Game, Fish, Nuts/Seeds, Pork, Poultry).
* **Activity filter**: Calorie/fat thresholds vary by low/medium/high activity.
* **Fruit logic**: Classify fruits into low/moderate/high sugar; pick based on activity.
* **Safety/quality rules**: Exclude processed/combined items; at most one organ meat per set; dedupe by base description.

---

## Project Structure (current)

```
ml_macro_project/
├── ml_macro_project.py
├── README.md
└── (dataset auto‑downloaded by kagglehub on first run)
```

## Project Structure (planned)

```
ml_macro_project/
├── data/
├── model/
├── train.py
├── demo.py
├── ml_macro_project.py
├── requirements.txt
└── README.md
```

---

## Roadmap

* [ ] Step 1: Polish docs (this README) and add function/class docstrings.
* [ ] Step 2: Split training & demo; save/load pretrained artifacts.
* [ ] Step 3: Minimal API (FastAPI) for later UI.
* [ ] Step 4 (optional): Simple Streamlit front‑end.

---

## Notes

* Results depend on chosen seeds and preprocessing choices.
* The scorer intentionally biases toward nutrient‑dense, testosterone‑supportive foods while allowing balanced variety.

## License

MIT recommended for simplicity.

## Acknowledgments

* USDA National Nutrient Database
* PyTorch, scikit‑learn
