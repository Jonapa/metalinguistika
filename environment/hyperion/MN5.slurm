#!/bin/bash
#SBATCH --job-name=MN5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --time=48:00:00

module load Miniforge3
module load CUDA/13.0.0

echo "PATH: $PATH" 
nvcc --version # verify that nvcc is in your PATH
echo "CUDA_HOME: $CUDA_HOME"
${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOME

# With compilers
    # CUDA 12.8/12.9 -> Up to GCC 14.x is supported (with 14 fails, falling back to 13)
    # https://docs.nvidia.com/cuda/archive/12.9.0/cuda-installation-guide-linux/index.html#system-requirements
    # https://docs.nvidia.com/cuda/archive/12.9.0/cuda-installation-guide-linux/index.html#host-compiler-support-policy
# setuptools: 
    # vllm: https://github.com/vllm-project/vllm/blob/main/requirements/common.txt#L39
conda create -y -n build-env -c conda-forge \
    python=3.12 \
    sysroot_linux-64=2.28 \
    "gcc_linux-64<14" \
    "gxx_linux-64<14" \
    "setuptools<81.0.0" \
    rust \
    setuptools-rust \
    conda-pack






##############################################################
PY="/home/jonapa/.conda/envs/build-env/bin/python"
UNPIN="/scratch/jonapa/unpin_installed.py"
CU="cu130" # CUDA 13.0
# change also flash attention 2
export ARCH_LIST="9.0a" # Common arch numbers: V100 7.0, T4 7.5, A100 8.0, A6000/3090 8.6, L40S/4090 8.9, H100/H200 9.0a, B200 10.0a, RTX 5090 12.0a
conda activate /home/jonapa/.conda/envs/build-env
##############################################################





export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
# https://discuss.pytorch.org/t/building-from-source-without-root-access/205320
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# https://cmake.org/cmake/help/latest/envvar/CUDAHOSTCXX.html
export CUDAHOSTCXX=$CXX

#######################################################################################################################

# https://github.com/pytorch/pytorch#from-source
export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
export USE_CUDA=1
export USE_CUDNN=0 # No cuDNN 12.9/13.0 version in Hyperion, so disable it to avoid build errors
export MAX_JOBS=$(( $(nproc) / 2 ))
export NVCC_THREADS=$(( $(nproc) / MAX_JOBS ))

git clone https://github.com/pytorch/pytorch.git
cd pytorch/
# https://github.com/pytorch/pytorch/releases
git checkout v2.13.0 
# 2.14.0 is the latest, flash attention: https://github.com/Dao-AILab/flash-attention/issues/2863
git submodule sync
git submodule update --init --recursive

# pip does not recognise cmake and ninja installed via conda, so install them via pip
"$PY" -m pip install cmake ninja

# https://github.com/scientific-python/spin/blob/main/pyproject.toml
# only for installation purposes, so no need to fix this 
# spin (from torch)
    # click!=8.3.0,<8.4,>=8
# huggingface-hub (from transformers)
    # click<9.0.0,>=8.4.2


# vllm
    # numpy==2.3.5 (inferred version from vllm pip install)
        # axolotl >=2.1,<3.0
    # protobuf==6.33.6 (protobuf!=6.30.*,!=6.31.*,!=6.32.*,!=6.33.0.*,!=6.33.1.*,!=6.33.2.*,!=6.33.3.*,!=6.33.4.*,>=5.29.6, inferred version from vllm pip install, vllm most restrictive package)
    # apache-tvm-ffi==0.1.11 (fixed in requirements/cuda.txt)
        # flash-attn-4 apache-tvm-ffi<0.2,>=0.1.12 (reduced to beta19 https://github.com/Dao-AILab/flash-attention/commit/00469573f86e5bc9bfdf83d5073649c8a6638409)
    # nvidia-cutlass-dsl[cu13]==4.6.2 (fixed in requirements/cuda.txt)
# axolotl
    # tomlkit 0.13.3 (inferred version from vllm pip install)
    # pillow 11.3.0 (inferred version from vllm pip install)
        # pip install done in torchvision
    # fsspec-2025.10.0 (inferred version from vllm pip install)
    # starlette<1.0,>=0.40.0 (from gradio) -> force-unpin in unpin_installed.py, not important just for web interfaces
        # vllm starlette<2.0.0,>=1.0.1
    # pandas 2.3.3 (inferred version from vllm pip install)

# Avoid uninstall
"$PY" -m pip install numpy==2.3.5 apache-tvm-ffi==0.1.11 "nvidia-cutlass-dsl[cu13]==4.6.2" protobuf==6.33.6 tomlkit==0.13.3 "fsspec==2025.10.0" "starlette<2.0.0,>=1.0.1" pandas==2.3.3

"$PY" -m pip install -r requirements.txt
"$PY" -m pip install mkl-static mkl-include
make triton
"$PY" -m pip install --no-build-isolation .

for i in {1..100}; do echo -n '#'; done; echo
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import platform; print(f'Python: {platform.python_version()}')"
for i in {1..100}; do echo -n '#'; done; echo

#######################################################################################################################

# https://github.com/dao-ailab/flash-attention#installation-and-features
cd ../
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
# https://github.com/Dao-AILab/flash-attention/releases
# currently beta30 release, but beta19 has vllm dependency "apache-tvm-ffi>=0.1.5,<0.2"
git checkout fa4-v4.0.0.beta19

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTN_CUDA_ARCHS="90"

# FlashAttention 3: (works with CUDA 12.9, not with CUDA 13.0)
# cd hopper/
# "$PY" setup.py install
# cd ../

# FlashAttention 2 
"$PY" -m pip install --no-build-isolation .

# FlashAttention 4: (Only CUDA 13.0)
"$PY" -m pip install --no-build-isolation "flash_attn/cute[cu13]"

for i in {1..100}; do echo -n '#'; done; echo
python -c "import flash_attn; print('FlashAttention 2: OK')"
# python -c "import flash_attn_3; print('FlashAttention 3: OK')"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4: OK')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import platform; print(f'Python: {platform.python_version()}')"
for i in {1..100}; do echo -n '#'; done; echo

#######################################################################################################################

# Check out Torchao and Torch version compatibility matrix
# https://github.com/pytorch/ao/issues/2919
# https://github.com/pytorch/ao/releases/tag/v0.17.0
cd ../
git clone https://github.com/pytorch/ao.git
cd ao/
# https://github.com/pytorch/ao/releases
git checkout v0.18.0

"$PY" -m pip install --no-build-isolation .

for i in {1..100}; do echo -n '#'; done; echo
python -c "import torchao; print(f'TorchAO: {torchao.__version__}')"
python -c "import flash_attn; print('FlashAttention 2: OK')"
# python -c "import flash_attn_3; print('FlashAttention 3: OK')"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4: OK')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import platform; print(f'Python: {platform.python_version()}')"
for i in {1..100}; do echo -n '#'; done; echo

#######################################################################################################################

# Check out Torchvision and Torch version compatibility matrix
# https://github.com/pytorch/vision#installation
export FORCE_CUDA=1
cd ../
git clone https://github.com/pytorch/vision.git
cd vision/
# https://github.com/pytorch/vision/releases
# 0.29.0 is the latest, same compilation error as pytorch
git checkout v0.28.0

# Python Imaging Library
"$PY" -m pip install pillow==11.3.0
"$PY" -m pip install --no-build-isolation .

sleep 5
for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://docs.pytorch.org/audio/main/build.html
cd ../
git clone https://github.com/pytorch/audio
cd audio/
# https://github.com/pytorch/audio/releases
git checkout v2.11.0

"$PY" -m pip install --no-build-isolation .

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://github.com/linkedin/Liger-Kernel#installation
cd ../
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel/
# https://github.com/linkedin/Liger-Kernel/releases
git checkout v0.8.3

touch src/liger_kernel/ops/experimental/__init__.py
"$PY" -m pip install --no-build-isolation .

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://github.com/deepspeedai/DeepSpeed/blob/master/environment.yml
cd ../
git clone https://github.com/deepspeedai/DeepSpeed.git
cd DeepSpeed/
# https://github.com/deepspeedai/DeepSpeed/releases
git checkout v0.19.7

# https://www.deepspeed.ai/tutorials/advanced-install/
# https://pydevtools.com/handbook/how-to/how-to-install-deepspeed/#pre-compile-ops-at-install-time
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1
export DS_BUILD_FUSED_ADAM=1
export DS_BUILD_PIN_MEMORY=1

"$PY" -m pip install -r requirements/requirements.txt
"$PY" -m pip install --no-build-isolation .

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# https://github.com/huggingface/transformers#installation
cd ../
git clone https://github.com/huggingface/transformers.git
cd transformers/
# https://github.com/huggingface/transformers/releases
git checkout v5.17.0

"$PY" -m pip install --no-build-isolation .

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://github.com/huggingface/accelerate
cd ../
git clone https://github.com/huggingface/accelerate.git
cd accelerate/
# https://github.com/huggingface/accelerate/releases
git checkout v1.15.0 

"$PY" -m pip install --no-build-isolation .

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://github.com/huggingface/peft#quickstart
cd ../
git clone https://github.com/huggingface/peft.git
cd peft/
# https://github.com/huggingface/peft/releases
git checkout v0.21.0

"$PY" -m pip install --no-build-isolation .

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#use-an-existing-pytorch-installation
# https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#build-wheel-from-source
cd ../
git clone https://github.com/vllm-project/vllm.git
cd vllm/
# https://github.com/vllm-project/vllm/releases
git checkout v0.29.0

# Pin the version: a dirty tree makes setuptools-scm stamp the build date into it, and a build that crosses midnight then fails pip's filename check.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.29.0

# For building pyproject.toml
"$PY" -m pip install maturin pdm-backend flit_core meson-python hatchling hatch-fancy-pypi-readme hatch-vcs poetry-core poetry-dynamic-versioning setuptools-scm
"$PY" use_existing_torch.py
"$PY" -m pip install -r requirements/build/cuda.txt
"$PY" -m pip install --no-build-isolation . --extra-index-url https://download.pytorch.org/whl/"$CU"

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://github.com/EleutherAI/lm-evaluation-harness/#install
# https://github.com/EleutherAI/lm-evaluation-harness/releases
cd ../
git clone --depth 1 --branch v0.4.13 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness/

# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/pyproject.toml
# dependencies:
    # hf: torch, transformers, accelerate, peft
    # vllm

"$PY" -m pip install --no-build-isolation .

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

cd ../
git clone https://github.com/axolotl-ai-cloud/axolotl.git

cd axolotl/
# https://github.com/axolotl-ai-cloud/axolotl/releases
# https://github.com/axolotl-ai-cloud/axolotl/commit/ebcc96c904d6474799a85d18f50dd6a4e84af3ac
git checkout ebcc96c

# Change library versions in pyproject.toml to match the versions installed above
# Only print
# python unpin_installed.py pyproject.toml --sections main --dry-run --force-unpin gradio
# Modify file
# python unpin_installed.py pyproject.toml --sections main --force-unpin gradio
"$PY" "$UNPIN" pyproject.toml --sections main --force-unpin gradio

"$PY" -m pip install --no-build-isolation .

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

conda deactivate

# module load Miniforge3
# conda activate build-env
# conda pack -n build-env -o build-env.tar.gz


# scp /scratch/jonapa/build-env.tar.gz  ehu_p518_10_1@dt.scayle.es:/home/ehu_p518_10/ehu_p518_10_1
#### cpubash
# mkdir build-env
# tar -xzf build-env.tar.gz -C build-env
# module load CUDA/13.0.0
# source build-env/bin/activate
# conda-unpack
# source build-env/bin/deactivate

# scp /scratch/jonapa/build-env.tar.gz  ehu129750@transfer1.bsc.es:/gpfs/scratch/ehpc678/jonapa
#### cpubash
# cd /gpfs/scratch/ehpc678/jonapa
# mkdir build-env
# tar -xzf build-env.tar.gz -C build-env
# source build-env/bin/activate
# conda-unpack
# source build-env/bin/deactivate
# bash check.sh > output.out