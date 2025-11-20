#!/bin/bash

#BSUB -q gpuv100
#BSUB -W 18:00
#BSUB -J ae_gpu_cluster_2
#BSUB -o ae_gpu_job_cluster_2_%J.out
#BSUB -e ae_gpu_job_cluster_2_%J.err

# Send an email when the job starts
#BSUB -B

# Send an email when the job ends
#BSUB -N

# Specify the email address to send notifications
#BSUB -u s241645@dtu.dk

# Request number of CPU cores
#BSUB -n 4

# Request single node
#BSUB -R "span[hosts=1]"

# Request memory
#BSUB -R "rusage[mem=8GB]"

# Request GPU
#BSUB -gpu "num=1:mode=exclusive_process"


# Initalize conda environment
source ~/load_data_science.sh

# Get the cpu model name
lscpu | grep "Model name"

# Get the GPU model name
echo "GPU Model(s):"
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq

# Wandb agents
# Cluster 0 = wandb agent --count 50 card-fraud-gang/data_science/qj1xrm4c
# Cluster 1 = wandb agent --count 50 card-fraud-gang/data_science/aue80fvv
# Cluster 2 = wandb agent --count 50 card-fraud-gang/data_science/yu2uc1y5
# Cluster 3 = wandb agent --count 50 card-fraud-gang/data_science/awuytuzf
echo "Starting wandb agent for cluster 2"
wandb agent --count 50 card-fraud-gang/data_science/yu2uc1y5