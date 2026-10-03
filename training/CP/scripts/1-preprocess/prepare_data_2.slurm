#!/bin/bash
#SBATCH --account=ehu26
#SBATCH --qos=acc_resa
#SBATCH --job-name=prepare_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

source /gpfs/scratch/ehu26/ehu129750/build-env/bin/activate

projectPath="/gpfs/scratch/ehu26/ehu129750/metalinguistika/"

export WANDB_MODE=offline
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export HF_HUB_OFFLINE=1

mkdir -p "${projectPath}/training/CP/cache/dataset_cache"
mkdir -p "${projectPath}/training/CP/cache/model"
mkdir -p "${projectPath}/training/CP/cache/hf"
export TMPDIR="${projectPath}/training/CP/cache"
export HF_HOME="${projectPath}/training/CP/cache/hf"

# https://docs.axolotl.ai/docs/telemetry.html
export AXOLOTL_DO_NOT_TRACK=1


# match cpu number
export OMP_NUM_THREADS=16
export PYTHONPATH=${projectPath}/training/CP/scripts/0-train_configs/common:$PYTHONPATH

python -m axolotl.cli.preprocess "${projectPath}/training/CP/scripts/0-train_configs/train_config_2.yaml"

source /gpfs/scratch/ehu26/ehu129750/build-env/bin/deactivate