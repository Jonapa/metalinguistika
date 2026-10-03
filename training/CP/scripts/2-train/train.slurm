#!/bin/bash
#SBATCH --account=ehu26
#SBATCH --qos=acc_resa
#SBATCH --job-name=train
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --time=06:00:00

CUDA_VER=13.3

module load cuda/${CUDA_VER} || { echo "ERROR: module cuda/${CUDA_VER} not found" >&2; exit 1; }

export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
echo "Using CUDA_HOME=$CUDA_HOME"

source /gpfs/scratch/ehu26/ehu129750/build-env/bin/activate

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Removed 9.0a: ValueError: invalid literal for int() with base 10: '0a'
export ARCH_LIST="9.0"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DeepSpeed Op Builder: Installed CUDA version 13.3 does not match the version torch was compiled with 13.0, unable to compile cuda/cpp extensions without a matching cuda version.
export DS_SKIP_CUDA_CHECK=1

export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export CUDAHOSTCXX=$CXX

export TRITON_CACHE_DIR=/gpfs/scratch/ehu26/ehu129750/.cache/.triton
export TORCHINDUCTOR_CACHE_DIR=/gpfs/scratch/ehu26/ehu129750/.cache/.inductor
export CUDA_CACHE_PATH=/gpfs/scratch/ehu26/ehu129750/.cache/.nv
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DeepSpeed
export TORCH_EXTENSIONS_DIR=/gpfs/scratch/ehu26/ehu129750/.cache/torch_extensions

projectPath="/gpfs/scratch/ehu26/ehu129750/metalinguistika/"

export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export CPATH=$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:${CPATH}}
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}

############################################################################################################################################

# https://docs.axolotl.ai/docs/telemetry.html
export AXOLOTL_DO_NOT_TRACK=1

export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

export HF_HUB_OFFLINE=1 # Offline mode for Hugging Face
export WANDB_MODE=offline

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 # 1 hour, prevent HEARTBEAT_TIMEOUT error

# Enable asynchronous error handling for NCCL operations
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Enable the memory allocator to allocate memory in a way that is more likely to be compatible with NCCL
export PYTORCH_ALLOC_CONF=expandable_segments:True

# https://github.com/huggingface/accelerate/issues/314
# https://docs.axolotl.ai/docs/nccl.html
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL

MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

############################################################################################################################################

# 80 cores per node shared by 4 ranks -> 20 threads each
export OMP_NUM_THREADS=20

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export PYTHONPATH=${projectPath}/training/CP/scripts/0-train_configs/common:$PYTHONPATH

# copy the prepared dataset to local NVMe on EVERY node (one task per node),
# not just on the node running this batch script
SHARED_CACHE=${projectPath}/training/CP/cache/dataset_cache
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DATASET_HASH=a67107cfe3b51650984e2c908275eb20
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
LOCAL_CACHE=${TMPDIR:?TMPDIR is not set}/dataset_cache   # node-local NVMe

srun --ntasks-per-node=1 bash -c "mkdir -p '$LOCAL_CACHE' && cp -r '$SHARED_CACHE/$DATASET_HASH' '$LOCAL_CACHE/'" \
    || { echo "ERROR: staging dataset to local disk failed" >&2; exit 1; }

# Note: \$SLURM_NODEID is expanded here on the first node, so every node gets 0.
# That's fine with --rdzv_backend c10d (it assigns node ranks itself).
srun accelerate launch \
--multi_gpu \
--num_processes 64 \
--num_machines 16 \
--mixed_precision bf16 \
--dynamo_backend "no" \
--rdzv_backend c10d \
--main_process_ip $MAIN_PROCESS_IP \
--main_process_port $MASTER_PORT \
--machine_rank $SLURM_NODEID \
-m axolotl.cli.train "${projectPath}/training/CP/scripts/0-train_configs/train_config.yaml" \
    --dataset_prepared_path="$LOCAL_CACHE" \
    --dataloader_num_workers=4 \
    --dataloader_prefetch_factor=2

source /gpfs/scratch/ehu26/ehu129750/build-env/bin/deactivate