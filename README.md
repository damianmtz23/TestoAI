# Macro & Micronutrient Food Recommender

Recommend nutrient-dense foods that support testosterone production by learning nutrient embeddings with a tiny autoencoder and matching to a testosterone-supporting nutrient profile — filtered by activity level and food group rules.

---

## Overview

This project learns a compact representation of foods from the **USDA National Nutrient Database** and recommends items that best match a nutrient “prototype” designed to promote testosterone production. The prototype is built from seed foods known for their high nutrient density and key micronutrients (e.g., zinc, magnesium, selenium, B vitamins) — such as oysters, egg yolk, and beef liver. It prioritizes protein-dense animal foods while still considering fruit and vegetable options based on their sugar/fiber profiles and mineral content.

**Core ideas**

* **Embeddings via Autoencoder** (PyTorch): 13→64→16 latent →64→13; MSE reconstruction loss.
* **Prototype matching**: Mean embedding of testosterone-boosting seed foods → cosine similarity to every item.
* **Rules & weights**: Favor testosterone-supportive food groups; cap organ meats; dedupe near-duplicates; exclude processed items.
* **Activity-aware**: Adjust scoring and calorie/fat filters for low/medium/high activity.
* **Dynamic fruit sugar buckets**: “low/moderate/high sugar” based on dataset distribution.

---

## Quickstart

### Install

```bash
pip install -r requirements.txt
```

### (Optional) Train to regenerate artifacts

```bash
python train.py
```

This will download the dataset via `kagglehub`, train the autoencoder, and save:

* `data/emb_df.parquet`
* `model/encoder.pt`
* `model/scaler.pkl`
* `model/meta.json`

### Run the demo (instant)

```bash
python demo.py
```

Then follow the prompts (choose activity level, optional FoodGroup) to see recommendations.

---


## Sample output
See `sample_output.txt` for a full example run.

Example (truncated):

## Data

* **Source:** USDA National Nutrient Database (via `kagglehub`)
* **Columns used:** `Energy_kcal`, `Protein_g`, `Fat_g`, `Carb_g`, `Zinc_mg`, `Magnesium_mg`, `VitB6_mg`, `VitB12_mcg`, `Selenium_mcg`, `VitA_mcg`, `Iron_mg`, `Sugar_g`, `Fiber_g`.
* **Preprocessing:** drop NAs; MinMax scaling to \[0,1].

---

## Model & Method

* **Autoencoder**: Linear → ReLU → Linear (latent=16) → Linear → ReLU → Linear; trained \~10 epochs with Adam (1e-3) and MSE.
* **Embeddings**: Use encoder output as the nutrient embedding for each food.
* **Prototype vector**: Mean of seed foods rich in testosterone-supportive micronutrients.
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
├── requirements.txt
├── README.md
└── (dataset auto-downloaded by kagglehub on first run)
```

## Project Structure (planned)

```
ml_macro_project/
├── data/
├── model/
├── train.py          # trains model & saves artifacts (see Quickstart)
├── demo.py           # loads artifacts & prints recommendations
├── ml_macro_project.py
├── requirements.txt
└── README.md
```

---

## Roadmap

### Short‑term

* [ ] Split training and demo; save artifacts (`data/emb_df.parquet`, `model/encoder.pt`, `model/scaler.pkl`, `model/meta.json`).
* [ ] Add sample output file (`sample_output.txt`) from a demo run.
* [ ] Add `.gitignore` (ignore `data/`, `model/`, `__pycache__/`).
* [ ] Add concise docstrings and inline comments.

### Future enhancements

* [ ] Pin package versions or add a lock file.
* [ ] Add CLI flags (e.g., `--activity`, `--group`, `--seeds path.json`).
* [ ] Create an evaluation notebook for protein density & micronutrient coverage (zinc, magnesium, selenium, B6, B12).
* [ ] Provide a minimal API with FastAPI (`/recommend`).
* [ ] Optional UI (Streamlit) using the API.
* [ ] Unit tests for filters & scoring (pytest).
* [ ] Expand vegetables and micronutrient weighting.
* [ ] Dockerfile for one‑command setup.

---

## Notes

* Results depend on chosen seeds and preprocessing choices.
* The scorer intentionally biases toward nutrient-dense, testosterone-supportive foods while allowing balanced variety.

## License

MIT recommended for simplicity.

## Acknowledgments

* USDA National Nutrient Database
* PyTorch, scikit-learn
