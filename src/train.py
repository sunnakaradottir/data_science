# train.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import yaml
import wandb
from dotenv import load_dotenv
from src.model import AutoEncoder, DEVICE
from pathlib import Path

# path to .env file (one level up)
dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

# ...existing code...

def flatten_config(config):
    """Extracts flat config from sweep-style YAML for normal runs."""
    params = config.get("parameters", {})
    flat = {}
    for k, v in params.items():
        if "value" in v:
            flat[k] = v["value"]
        elif "values" in v:
            flat[k] = v["values"][0]  # pick first for normal run
        elif "min" in v and "max" in v:
            flat[k] = v.get("min")    # pick min for normal run
        else:
            flat[k] = v
    return flat

with open("./config.yaml") as file:
    CONFIG = yaml.safe_load(file)

def _train_cluster(X_cluster: np.ndarray, in_dim: int, cfg, cid: int) -> tuple[AutoEncoder, float]:
    """
    Train an autoencoder on a specific cluster's data.
    Parameters
    ----------
    X_cluster : np.ndarray
    """
    X_train, X_val = train_test_split(X_cluster, test_size=cfg["val_split"], random_state=cfg["seed"])
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train_t), batch_size=cfg["batch_size"], shuffle=True)

    net = AutoEncoder(in_dim=in_dim, hidden_units=cfg["hidden_dim"], latent_features=cfg["latent"], num_layers=cfg["num_layers"]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    crit = nn.MSELoss()

    print("training started...")
    print(f"Cluster ID: {cid}, Cluster Size: {X_cluster.shape[0]}")
    best_val_mse = float('inf')

    for epoch in range(1, cfg["epochs"] + 1):
        net.train()
        total = 0.0
        for i, (xb,) in enumerate(loader):
            xb = xb.to(DEVICE)
            loss = crit(net(xb)["x_hat"], xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"  Epoch {epoch}/{cfg['epochs']}, Loss: {loss.item():.6f}")

        train_mse = total / len(loader.dataset)

        net.eval()
        with torch.no_grad():
            xv = X_val_t.to(DEVICE)
            val_mse = F.mse_loss(net(xv)["x_hat"], xv, reduction="mean").item()

        best_val_mse = min(best_val_mse, val_mse)

        wandb.log({
            "epoch": epoch,
            "cluster": cid,
            f"train_mse": train_mse,
            f"val_mse": val_mse,
        })

    return net, best_val_mse

def run(X_scaled: np.ndarray, labels: np.ndarray, sweep=False):
    """
    Start ONE wandb run and train an AE for a single cluster.
    If sweep=True, use sweep config; else, flatten config and pick first values.
    """
    if sweep:
        wandb.init(project="card-fraud-ae", entity="card-fraud-gang")
        cfg = dict(wandb.config)
    else:
        flat_cfg = flatten_config(CONFIG)
        wandb.init(project="card-fraud-ae", entity="card-fraud-gang", config=flat_cfg)
        cfg = flat_cfg
    
    # OVERRIDE cluster_id from config if present
    cluster_id = cfg["cluster_id"]

    # set seeds and create output dir
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    in_dim = X_scaled.shape[1]

    # Choose rows from the specified cluster
    Xc = X_scaled[labels == cluster_id]
    print(f"Training cluster {cluster_id} (n={len(Xc)})")

    # Nickname the run
    if wandb.run:
        import datetime
        # timestamp readable 
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.run.name = f"ae_cluster_{cluster_id}_" + time_str
    
    # info logs
    wandb.log ({
        "cluster_id": cluster_id,
        "cluster_size": len(Xc),
    })

    # Train the autoencoder for this cluster
    model, best_val = _train_cluster(Xc, in_dim, cfg, cluster_id)

    # Save the trained model
    model_path = os.path.join(cfg["out_dir"], f"ae_cluster_{cluster_id}_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    wandb.save(model_path)

    # sweep summary (single scalar per run)
    wandb.log({
        "best_val_mse": best_val,
    })
    wandb.run.summary["best_val_mse"] = best_val
    
    print("✅ Training complete.")
