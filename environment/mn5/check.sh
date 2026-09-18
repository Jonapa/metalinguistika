source /gpfs/scratch/ehpc678/jonapa/build-env/bin/activate

module load cuda/13.3

export ARCH_LIST="9.0a"
export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export CUDAHOSTCXX=$CXX
export MAX_JOBS=$(( $(nproc) / 2 ))
export CMAKE_PREFIX_PATH=$(dirname $(which python))

export TRITON_CACHE_DIR=/gpfs/scratch/ehpc678/jonapa/.cache/.triton
export TORCHINDUCTOR_CACHE_DIR=/gpfs/scratch/ehpc678/jonapa/.cache/.inductor
export CUDA_CACHE_PATH=/gpfs/scratch/ehpc678/jonapa/.cache/.nv

for i in {1..100}; do echo -n '#'; done; echo
axolotl --help
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
# python -c "import flash_attn_3; print('FlashAttention 3: OK')"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4: OK')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import platform; print(f'Python: {platform.python_version()}')"
for i in {1..100}; do echo -n '#'; done; echo

for i in {1..100}; do echo -n '#'; done; echo
ds_report
for i in {1..100}; do echo -n '#'; done; echo

python -m pip check

sh /home/ehu/ehu129750/jonapa/gpu_env_check.sh

source /gpfs/scratch/ehpc678/jonapa/build-env/bin/deactivate