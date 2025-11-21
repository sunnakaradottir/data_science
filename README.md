# Credit Card Fraud Anomaly Detection via Clustering and AutoEncoders 💳

Lightweight pipeline for clustering normal credit card transactions and training a per-cluster autoencoder to spot anomalies. Data comes from the Kaggle credit card fraud dataset tracked with Git LFS.

## Quick start
- Install Git LFS and pull the dataset: `git lfs install && git lfs pull`
- Create an env and install deps: `conda create --name <my-env> python=3.12.8 && conda activate <my-env> && pip install -r requirements.txt`
- Set your W&B key (needed for logging): create a .env file and set `WANDB_API_KEY=...`

## Common workflows
### Prepare splits, scaler, and KMeans once:
  ```bash
  python - <<'PY'
  from main import main_setup
  main_setup()
  PY
  ```
### Train one cluster autoencoder 
To train an autoencoder for a specific cluster (e.g., cluster 2), run:
```bash
python - <<'PY'
from main import main_single_cluster
main_single_cluster()
PY
```
This uses the config similar to `ae_cluster_2_config.yaml`. Adjust `cluster_id` in that file to change the cluster being trained.

### WandB Sweeps
Model training was mostly done via W&B sweeps using `config.yaml` to find optimal hyperparameters.

Process is exactly the same as above, just ensure `sweep=True` in `run()` inside `main_single_cluster()`.

Also ensure that you are using a config file similar to `config.yaml` with parameter ranges defined.

## Evaluation and Thresholding
Evaluation and thresholding code is TBD. The idea is to use the `tune_data.csv` which contains both fraud and sampled normal transactions to find per-cluster reconstruction error thresholds that balance TPR and FPR.

## Project layout (high value files)
```text
.
├── main.py                   — entry point; wire up prep + training
├── ae_cluster_2_config.yaml  — Training config for a single cluster and single run
├── config.yaml               — Sweep config with hyperparameter ranges
├── requirements.txt          — dependencies
├── job.sh                    — job script for HPC runs
├── src/
│   ├── data.py               — split base/tune, fit scaler, load helpers
│   ├── cluster.py            — KMeans helpers
|   ├── kmeans.py             - Our own KMeans implementation used in pipeline
│   ├── train.py              — per-cluster autoencoder training + W&B logging
│   ├── model.py              — AutoEncoder definition
│   ├── eval.py               — TBD
│   └── fetch_model.py        — Download best model for given sweeps
├── data/
│   ├── creditcard.csv        — raw dataset (Git LFS)
│   ├── base_data.csv         — normals for clustering/training (created)
│   └── tune_data.csv         — fraud + sampled normals for thresholding
├── models/                   — Best models fetched via fetch_model.py
└── notebooks/                - Final notebook and experiments
```

## Notes
- To change which cluster trains, update `cluster_id` in config file used.
- Generated artifacts (scaler, KMeans, cluster distribution) land in `data/`. W&B runs are named with cluster IDs and timestamps for quick filtering.
