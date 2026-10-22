#!/usr/bin/env bash

module load cuda/13.3

source /gpfs/scratch/ehu26/ehu129750/build-env-swift/bin/activate

export ARCH_LIST="9.0a"
export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
export NVTE_CUDA_ARCHS="90"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export CUDAHOSTCXX=$CXX
export MAX_JOBS=$(( $(nproc) / 2 ))
export CMAKE_PREFIX_PATH=$(dirname $(which python))

# TransformerEngine was built against the pip cuDNN/NCCL wheels, not system ones
export NVTE_FRAMEWORK=pytorch
export CUDNN_PATH=$(python -c "import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])")
export CUDNN_HOME=$CUDNN_PATH
export NCCL_HOME=$(python -c "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])")
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for i in {1..100}; do echo -n '#'; done; echo
python -c "import swift; print(f'ms-swift: {swift.__version__}')"
python -c "import megatron.core; print(f'Megatron-Core: {megatron.core.__version__}')"
python -c "import transformer_engine.pytorch as te; import transformer_engine; print(f'TransformerEngine: {transformer_engine.__version__}')"
python -c "import apex; print(apex.__file__)"
python -c "import lm_eval; print(f'LM Evaluation Harness: {lm_eval.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python -c "import importlib.metadata as m; print('Liger Kernel:', m.version('liger-kernel'))"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import torchao; print(f'TorchAO: {torchao.__version__}')"
python -c "import flash_attn; print('FlashAttention 2: OK')"
python -c "import flash_attn_3; print('FlashAttention 3: OK')"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4: OK')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import platform; print(f'Python: {platform.python_version()}')"
for i in {1..100}; do echo -n '#'; done; echo

for i in {1..100}; do echo -n '#'; done; echo
DS_ACCELERATOR=cuda ds_report
for i in {1..100}; do echo -n '#'; done; echo

python -m pip check

bash /gpfs/scratch/ehu26/ehu129750/metalinguistika/environment/mn5/gpu_env_check-2.sh

source /gpfs/scratch/ehu26/ehu129750/build-env-swift/bin/deactivate