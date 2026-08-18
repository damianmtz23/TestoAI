# Macro & Micronutrient Food Recommender (TestoAI)

> **What this is:** A small ML project that learns **nutrient embeddings** from the **USDA/Kaggle** dataset and recommends foods that best match a testosterone‑supportive nutrient profile.

---

## Overview (high level)

This project trains a tiny **autoencoder** (PyTorch) on hand‑picked **macro** and **micro** features, then compares each food to a **prototype** built from seed items (e.g., oysters, egg yolk, beef liver) known for key micronutrients. The recommender is **protein‑forward**, with light rules (activity awareness, food‑group preferences, and basic filters) to keep results practical.

* Dataset: USDA National Nutrient Database (via Kaggle).
* Goal: Hit macro targets and favor micro‑dense foods (e.g., **zinc**, **magnesium**, **selenium**, **B‑vitamins**) while avoiding ultra‑processed items.
* Artifacts: `data/emb_df.parquet`, `model/encoder.pt`, `model/scaler.pkl`, `model/meta.json`.

---

## How foods are scored (the important part)

**Features used** (columns, units):

* **Macros**: Protein\_g, Fat\_g, Carb\_g, Energy\_kcal
* **Micros**: Zinc\_mg, Magnesium\_mg, Selenium\_mcg, VitB6\_mg, VitB12\_mcg, VitA\_mcg, Iron\_mg, Sugar\_g, Fiber\_g
* **Scaling**: per‑nutrient **MinMax** to \[0,1] (keeps mg vs g comparable).

**Embedding learning** (train time):

* Autoencoder: `13 → 64 → 16 (latent) → 64 → 13`, ReLU, **MSE** loss, Adam (1e‑3), \~10 epochs (tunable).
* The encoder output is the **food embedding**.

**Prototype & similarity** (inference time):

* Build a **prototype vector** = mean embedding of seed foods (oysters, egg yolk, liver, etc.).
* Score each food by **cosine similarity** to the prototype (**higher = better**).

**Rules & weights** (post‑scoring tweaks):

* **Protein‑forward** emphasis (boost items with strong protein signal).
* **Activity‑aware** fat handling (slightly more lenient at high activity, stricter at low).
* **Food‑group preferences** (e.g., Beef / Dairy‑Egg → Ideal; keep variety with Fruits/Veg based on sugar/fiber/minerals).
* **Fruit sugar buckets** (low / moderate / high sugar by distribution percentile).
* **Quality filters**: drop ultra‑processed/combined items; cap organ meats; de‑dupe near‑duplicates.

**Final score (conceptual)**:
`score = α · cosine(embedding, prototype)  +  β · macro‑alignment  −  penalties(rules)`  (defaults are simple; weights are easy to tune.)

---

## Quickstart (fresh clone)

```bash
git clone <repo>
cd TestoAI

# 1) Install the package (editable, pulls in requirements.txt deps)
python -m pip install -e .

# 2) Train: builds embeddings + saves artifacts to data/ and model/ (both gitignored)
python train.py

# 3) Try the interactive demo
python demo.py

# ...or the CLI
testoai --activity high --k 5
testoai --food-groups "Fruits and Fruit Juices" --activity high --k 10
```

`data/` and `model/` are intentionally not committed — `train.py` regenerates
them from scratch every time, so nothing from the original developer's
machine is required.

**About the dataset:** `train.py` tries to pull the full USDA National
Nutrient Database via `kagglehub` first (this needs a Kaggle account and API
token — see [kagglehub docs](https://github.com/Kaggle/kagglehub) for setup).
If that's unavailable (no credentials, offline, etc.), it automatically falls
back to a small bundled sample dataset at `fixtures/sample_nutrients.csv` (~25
foods covering every FoodGroup used by the recommender), so `python train.py`
always produces working artifacts — just with less variety in the results
than the full dataset.

---

## Example output (truncated)

```
                  FoodGroup                          Description  final_score
0            Beef Products         Beef Liver (Raw)                  0.967060
1  Dairy and Egg Products                  Egg Yolk (Raw)          0.928033
2          Poultry Products      Chicken Breast (Raw)                0.926318
```

---

## Project structure

```
TestoAI/
├─ src/testoai/
│  ├─ __init__.py
│  ├─ recommend.py        # ranking/scoring logic (cosine + rules)
│  └─ cli.py              # parses flags, calls recommend(), prints a table
├─ train.py               # regenerates artifacts from full (or sample) dataset
├─ demo.py                # interactive demo using saved artifacts
├─ data/                  # (generated, gitignored) embeddings & tables
├─ model/                 # (generated, gitignored) encoder weights, scaler, meta
├─ fixtures/
│  └─ sample_nutrients.csv  # tiny tracked dataset used when kagglehub is unavailable
├─ requirements.txt
├─ pyproject.toml         # makes package installable; creates `testoai` command
├─ README.md
└─ .gitignore
```

---

---

## Roadmap

**Near-term (polish)**

* Provide small **pretrained AE weights** and attach to a GitHub Release.
* Tiny **evaluation notebook** comparing cosine vs macro-distance with top-K examples.
* **Unit tests** (pytest): column mapping, protein-forward monotonicity, fruit sugar buckets.
* **GitHub Actions**: ruff/black + pytest (Py 3.10/3.11).
* Sample **config files**: `examples/targets.yaml`, `examples/presets.yaml`.

**Next (nice-to-have)**

* Minimal **FastAPI** endpoint (`/recommend`) + 1-page **Streamlit** UI.
* **Dockerfile** for one-command runs.
* Expanded **data docs** (features, seed foods, assumptions/filters) linked from README.

---

## Notes & Disclaimer

Educational project; **not medical advice**. Results depend on dataset quality, scaling, chosen seeds/rules.

## License

MIT (see `LICENSE`).
