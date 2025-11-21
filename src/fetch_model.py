import os
import wandb

PROJECT = "data_science"
ENTITY = "card-fraud-gang"
BEST_DIR = "models"

# Ensure the folder exists
os.makedirs(BEST_DIR, exist_ok=True)

def download_best_model_for_sweep(sweep_id: str, cluster_id: int):
    """
    Given a W&B sweep ID and cluster ID:
    - Fetch all runs in the sweep
    - Select run with lowest best_val_mse
    - Download its model checkpoint
    """
    api = wandb.Api()

    # Get the sweep object
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")

    best_run = None
    best_mse = float("inf")

    # Iterate over all runs in the sweep
    for run in sweep.runs:
        mse = run.summary.get("best_val_mse")
        try:
            mse = float(mse)
        except (TypeError, ValueError):
            mse = None
        if mse is not None and mse < best_mse:
            best_mse = mse
            best_run = run

    if best_run is None:
        print(f"[ERROR] No valid runs found in sweep {sweep_id}")
        return

    print(f"Cluster {cluster_id} – Best run: {best_run.id}, MSE={best_mse}")

    # Find the model file in the run's artifacts/files
    files = best_run.files()
    model_files = [f for f in files if f.name.endswith(".pt")]

    if not model_files:
        print(f"[ERROR] No .pt model file found for run {best_run.id}")
        return
    
    SAVE_PATH = BEST_DIR + f"/ae_cluster_{cluster_id}"

    model_file = model_files[0]

    # Get the config from the run and save it
    config = best_run.config
    print(f"[INFO] Config for best run: {config}")
    config_path = os.path.join(SAVE_PATH, f"ae_cluster_{cluster_id}_config.txt")
    

    # Get run_name
    run_name = best_run.name

    out_path = os.path.join(SAVE_PATH, f"ae_cluster_{cluster_id}.pt")
    model_file.download(root=SAVE_PATH, replace=True)

    # Rename to consistent filename
    os.rename(os.path.join(SAVE_PATH, model_file.name), out_path)

    with open(config_path, "w") as f:
        f.write(f"run_name: {run_name}\n")
        f.write(f"best_val_mse: {best_mse}\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        

    print(f"[OK] Saved best model for cluster {cluster_id} to {out_path}")


if __name__ == "__main__":
    # Fill these with your sweep IDs (one per cluster)
    SWEEP_IDS = {
        0: "qj1xrm4c",  # cluster 0 sweep id
        1: "aue80fvv",  # cluster 1 sweep id
        2: "yu2uc1y5",  # cluster 2 sweep id
        3: "l8dpnh7i",  # cluster 3 sweep id
    }

    for cid, sweep_id in SWEEP_IDS.items():
        download_best_model_for_sweep(sweep_id, cluster_id=cid)
