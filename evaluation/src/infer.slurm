#!/bin/bash
#SBATCH --account=ehu26
#SBATCH --qos=acc_resa
#SBATCH --job-name=infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=08:00:00

module load cuda/13.3

source /gpfs/scratch/ehu26/ehu129750/build-env/bin/activate

export ARCH_LIST="9.0a"
export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export CUDAHOSTCXX=$CXX
export MAX_JOBS=$(( $(nproc) / 2 ))
export CMAKE_PREFIX_PATH=$(dirname $(which python))

export TRITON_CACHE_DIR=/gpfs/scratch/ehu26/ehu129750/.cache/.triton
export TORCHINDUCTOR_CACHE_DIR=/gpfs/scratch/ehu26/ehu129750/.cache/.inductor
export CUDA_CACHE_PATH=/gpfs/scratch/ehu26/ehu129750/.cache/.nv
export TORCH_EXTENSIONS_DIR=/gpfs/scratch/ehu26/ehu129750/.cache/torch_extensions

export CUDA_LAUNCH_BLOCKING=1

DATASET="08-11-26"
FILENAME="${DATASET}.preprocessed"

INFER_MODELS=(
  "/gpfs/scratch/ehu26/ehu129750/metalinguistika/training/CP/cache/model/CP"
  "/gpfs/scratch/ehu26/ehu129750/metalinguistika/training/CP/cache/model/CP_2"
  "/gpfs/scratch/ehu26/ehu129750/metalinguistika/training/CP/cache/model/CP_X4"
  "/gpfs/scratch/ehu26/ehu129750/metalinguistika/training/CP/cache/model/CP_X4_2"
)

for INFER_MODEL in "${INFER_MODELS[@]}"; do
    python /gpfs/scratch/ehu26/ehu129750/metalinguistika/evaluation/src/01_infer_vllm.py \
        --input-path "/gpfs/scratch/ehu26/ehu129750/metalinguistika/evaluation/input/${DATASET}/${FILENAME}.json" \
        --output-path "/gpfs/scratch/ehu26/ehu129750/metalinguistika/evaluation/output/infer/${FILENAME}__${INFER_MODEL//[\/:]/_}.json" \
        --model "${INFER_MODEL}" \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.8
done

source /gpfs/scratch/ehu26/ehu129750/build-env/bin/deactivate