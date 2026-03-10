#!/bin/bash -l

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL
#SBATCH --job-name=natsl
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%A_%a-out.txt
#SBATCH --mem=384G
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:h100:4

#SBATCH --array=0-0

conda activate NAtSL

NGPUS=4

MODEL_CONFIG=natsl_340M  # or change to the model that you want to train on
base=$(dirname "$PWD")

export MASTER_PORT=63333

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

working_dir=${base}/experiments

export export OMP_NUM_THREADS=32
cd $working_dir


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE  --rdzv_backend c10d --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" --local-ranks-filter 0 --role rank --rdzv_id=$SLURM_JOB_ID \
        train.py --job.config_file flame/flame/models/fla.toml --job.dump_folder nats_exp_small/${MODEL_CONFIG}-15B/batch32.seqlen4096.warmup1024.update1.steps20480.lr3e-4 \
        --training.batch_size 8 \
        --training.seq_len 4096 \
        --training.gradient_accumulation_steps 4 \
        --training.max_norm 1.0 \
        --training.prefetch_factor 2 \
        --training.steps 20480 \
        --optimizer.name AdamW \
        --optimizer.lr 3e-4 \
        --training.skip_nan_inf \
        --training.compile \
        --lr_scheduler.decay_type cosine \
        --training.num_workers 8 \
        --training.dataset_split train \
        --model.config configs/${MODEL_CONFIG}.json  \
        --checkpoint.load_step -1 \
        --checkpoint.keep_latest_k 2 \
        --training.skip_nan_inf

# convert model to HF model
python -u flame/flame/utils/convert_dcp_to_hf.py --path  nats_exp_small/${MODEL_CONFIG}-15B/batch32.seqlen4096.warmup1024.update1.steps20480.lr3e-4 --step 20480 --config  configs/${MODEL_CONFIG}.json --tokenizer fla-hub/transformer-1.3B-100B
