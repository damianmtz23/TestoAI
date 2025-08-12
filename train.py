# train.py
# Trains the nutrient autoencoder, builds embeddings, and saves artifacts for the demo.

import os
import json
import joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Config / paths
DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LATENT_DIM = 16
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
RANDOM_STATE = 42

# Load USDA dataset via kagglehub
import kagglehub
path = kagglehub.dataset_download("haithemhermessi/usda-national-nutrient-database")
df = pd.read_csv(os.path.join(path, "train.csv"))

# Columns to keep/normalize
nutr_cols = [
    'Energy_kcal', 'Protein_g', 'Fat_g', 'Carb_g',
    'Zinc_mg', 'Magnesium_mg', 'VitB6_mg', 'VitB12_mcg',
    'Selenium_mcg', 'VitA_mcg', 'Iron_mg', 'Sugar_g', 'Fiber_g'
]

meta_cols = ['Descrip', 'FoodGroup']  # kept for IDs and grouping

df_clean = df[meta_cols + nutr_cols].dropna().reset_index(drop=True)

# Normalize
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(df_clean[nutr_cols])
features = pd.DataFrame(X_norm, columns=nutr_cols)

# Torch tensors
X = torch.tensor(features.values, dtype=torch.float32)

# Dataloaders
class FoodDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

X_train, X_val = train_test_split(X, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
train_loader = DataLoader(FoodDataset(X_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(FoodDataset(X_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder(input_dim=X.shape[1], latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Train
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.size(0)
    train_loss /= len(train_loader.dataset)

    # quick val
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            val_loss += criterion(recon, batch).item() * batch.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")


# Build embeddings DataFrame and save
model.eval()
with torch.no_grad():
    _, Z = model(X.to(device))
    Z = Z.cpu().numpy()

emb_cols = [f"emb_{i}" for i in range(Z.shape[1])]
emb_df = pd.DataFrame(Z, columns=emb_cols)
emb_df['Descrip'] = df_clean['Descrip'].values
emb_df = emb_df.merge(df_clean[meta_cols + ['Energy_kcal','Fat_g','Sugar_g','Fiber_g']], on='Descrip', how='left')

# Add group score for later weighting
ideal_groups = ["Beef Products", "Dairy and Egg Products", "Fruits and Fruit Juices"]
okay_groups  = ["Lamb, Veal, and Game Products", "Finfish and Shellfish Products",
                "Nut and Seed Products", "Pork Products", "Poultry Products"]

def group_filter(group):
    if group in ideal_groups: return 2
    if group in okay_groups:  return 1
    return 0

emb_df["group_score"] = emb_df["FoodGroup"].apply(group_filter)

# Save artifacts
emb_path = os.path.join(DATA_DIR, "emb_df.parquet")
emb_df.to_parquet(emb_path, index=False)

torch.save(model.encoder.state_dict(), os.path.join(MODEL_DIR, "encoder.pt"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Save a small metadata file
meta = {
    "nutr_cols": nutr_cols,
    "emb_cols": emb_cols,
    "ideal_groups": ideal_groups,
    "okay_groups": okay_groups,
    "latent_dim": LATENT_DIM
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved: {emb_path}, model/encoder.pt, model/scaler.pkl, model/meta.json")
print("Training complete.")
