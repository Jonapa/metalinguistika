#!/bin/bash
#SBATCH --job-name=MN5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --qos=long
#SBATCH --time=48:00:00



#--account=hitz-exclusive
#--partition=hitz-exclusive

module load Miniforge3
module load CUDA/13.0.0

echo "PATH: $PATH" 
nvcc --version # verify that nvcc is in your PATH
echo "CUDA_HOME: $CUDA_HOME"
${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOME

# With compilers
    # CUDA 12.8/12.9 -> Up to GCC 14.x is supported (with 14 fails, falling back to 13 (<14))
    # https://docs.nvidia.com/cuda/archive/12.9.0/cuda-installation-guide-linux/index.html#system-requirements
    # https://docs.nvidia.com/cuda/archive/12.9.0/cuda-installation-guide-linux/index.html#host-compiler-support-policy
# setuptools: 
    # vllm: https://github.com/vllm-project/vllm/blob/main/requirements/common.txt#L39
conda create -y -n build-env -c conda-forge \
    python=3.12 \
    sysroot_linux-64=2.28 \
    "gcc_linux-64<15" \
    "gxx_linux-64<15" \
    "setuptools<81.0.0" \
    rust \
    setuptools-rust \
    conda-pack \
    elfutils # vllm: fatal error: elfutils/libdwfl.h: No such file or directory 






##############################################################
PY="/home/jonapa/.conda/envs/build-env/bin/python"
UNPIN="/scratch/jonapa/unpin_installed.py"
CU="cu130" # CUDA 13.0
export ARCH_LIST="9.0a" # Common arch numbers: V100 7.0, T4 7.5, A100 8.0, A6000/3090 8.6, L40S/4090 8.9, H100/H200 9.0a, B200 10.0a, RTX 5090 12.0a
export FLASH_ATTN_CUDA_ARCHS="90"
export FLASH_ATTENTION_DISABLE_SM80=TRUE
# DeepSpeed's op_builder can't parse arch suffixes like "9.0a"
DS_ARCH_LIST="${ARCH_LIST//a/}"   # "9.0a" -> "9.0"
export NVTE_CUDA_ARCHS="90"
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
git checkout v2.14.1 
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


# torchvision
    # pillow
# vllm
    # numpy==2.3.5 (inferred version from vllm pip install)
    # protobuf==6.33.6 (protobuf!=6.30.*,!=6.31.*,!=6.32.*,!=6.33.0.*,!=6.33.1.*,!=6.33.2.*,!=6.33.3.*,!=6.33.4.*,>=5.29.6, inferred version from vllm pip install, vllm most restrictive package)
    # apache-tvm-ffi==0.1.11
        # flash-attn-4 (fa4-v4.0.0.beta34 -- Allow apache-tvm-ffi 0.1.11: https://github.com/Dao-AILab/flash-attention/pull/2948)
    # nvidia-cutlass-dsl[cu13]==4.8.0 (fixed in requirements/cuda.txt)
    ### transformers 5.19.0
# ms-swift
    # datasets>=4.0,<4.8.5
    # fsspec<=2026.2.0,>=2023.1.0
    ### peft 0.20.0

# Avoid uninstall
"$PY" -m pip install pillow numpy==2.3.5 protobuf==6.33.6 apache-tvm-ffi==0.1.11 "nvidia-cutlass-dsl[cu13]==4.8.0" "datasets>=4.0,<4.8.5" "fsspec<=2026.2.0,>=2023.1.0"

"$PY" -m pip install -r requirements.txt
"$PY" -m pip install mkl-static mkl-include
make triton
"$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import platform; print(f'Python: {platform.python_version()}')"
for i in {1..100}; do echo -n '#'; done; echo

#######################################################################################################################

# https://github.com/dao-ailab/flash-attention#installation-and-features
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
# https://github.com/Dao-AILab/flash-attention/releases
git checkout fa4-v4.0.0.beta35

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

# FlashAttention 3
cd hopper/
"$PY" setup.py install
cd ../

# FlashAttention 2 
"$PY" -m pip install --no-build-isolation .

# FlashAttention 4: (Only CUDA 13.0)
"$PY" -m pip install --no-build-isolation "flash_attn/cute[cu13]"

cd ../

for i in {1..100}; do echo -n '#'; done; echo
python -c "import flash_attn; print('FlashAttention 2: OK')"
python -c "import flash_attn_3; print('FlashAttention 3: OK')"
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
git clone https://github.com/pytorch/ao.git
cd ao/
# https://github.com/pytorch/ao/releases
git checkout v0.18.0

"$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
python -c "import torchao; print(f'TorchAO: {torchao.__version__}')"
python -c "import flash_attn; print('FlashAttention 2: OK')"
python -c "import flash_attn_3; print('FlashAttention 3: OK')"
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
git clone https://github.com/pytorch/vision.git
cd vision/
# https://github.com/pytorch/vision/releases
git checkout v0.29.1

"$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://docs.pytorch.org/audio/main/build.html
git clone https://github.com/pytorch/audio
cd audio/
# https://github.com/pytorch/audio/releases
git checkout v2.11.0

"$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://github.com/linkedin/Liger-Kernel#installation
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel/
# https://github.com/linkedin/Liger-Kernel/releases
git checkout v0.8.5

# create empty __init__.py file in the experimental directory to avoid import errors
touch src/liger_kernel/ops/experimental/__init__.py
"$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://github.com/deepspeedai/DeepSpeed/blob/master/environment.yml
git clone https://github.com/deepspeedai/DeepSpeed.git
cd DeepSpeed/
# https://github.com/deepspeedai/DeepSpeed/releases
git checkout v0.19.8

# https://github.com/deepspeedai/DeepSpeed/blob/master/accelerator/real_accelerator.py
# See variable: SUPPORTED_ACCELERATOR_LIST
export DS_ACCELERATOR=cuda

# https://www.deepspeed.ai/tutorials/advanced-install/
# https://pydevtools.com/handbook/how-to/how-to-install-deepspeed/#pre-compile-ops-at-install-time
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1
export DS_BUILD_FUSED_ADAM=1
export DS_BUILD_PIN_MEMORY=1

"$PY" -m pip install -r requirements/requirements.txt
TORCH_CUDA_ARCH_LIST="$DS_ARCH_LIST" "$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# https://github.com/huggingface/transformers#installation
git clone https://github.com/huggingface/transformers.git
cd transformers/
# https://github.com/huggingface/transformers/releases
# TODO: 5.20.2, but see vllm compatibility
git checkout v5.19.0

"$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://github.com/huggingface/accelerate
git clone https://github.com/huggingface/accelerate.git
cd accelerate/
# https://github.com/huggingface/accelerate/releases
git checkout v1.15.0 

"$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

# https://github.com/huggingface/peft#quickstart
git clone https://github.com/huggingface/peft.git
cd peft/
# https://github.com/huggingface/peft/releases
# TODO: 0.21.2, but see ms-swift compatibility
git checkout v0.20.0

"$PY" -m pip install --no-build-isolation .

cd ../

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
python -c "import flash_attn_3; print('FlashAttention 3: OK')"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4: OK')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import platform; print(f'Python: {platform.python_version()}')"
for i in {1..100}; do echo -n '#'; done; echo

#######################################################################################################################

# https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#use-an-existing-pytorch-installation
# https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#build-wheel-from-source
git clone https://github.com/vllm-project/vllm.git
cd vllm/
# https://github.com/vllm-project/vllm/releases
# TODO: wait for release, which bumps transformers
git checkout v0.32.0
# Pin the version: a dirty tree makes setuptools-scm stamp the build date into it, and a build that crosses midnight then fails pip's filename check.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.32.0

# vllm: fatal error: elfutils/libdwfl.h: No such file or directory 
ls $CONDA_PREFIX/include/elfutils/libdwfl.h
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# For building pyproject.toml
"$PY" -m pip install maturin pdm-backend flit_core meson-python hatchling hatch-fancy-pypi-readme hatch-vcs poetry-core poetry-dynamic-versioning setuptools-scm
"$PY" tools/use_existing_torch.py
"$PY" -m pip install -r requirements/build/cuda.txt
# https://vllm.ai/ 
# See quickstart command
"$PY" -m pip install --no-build-isolation . --extra-index-url https://download.pytorch.org/whl/"$CU"

cd ../

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
python -c "import flash_attn_3; print('FlashAttention 3: OK')"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4: OK')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import platform; print(f'Python: {platform.python_version()}')"
for i in {1..100}; do echo -n '#'; done; echo

#######################################################################################################################

# https://github.com/EleutherAI/lm-evaluation-harness/#install
# https://github.com/EleutherAI/lm-evaluation-harness/releases
git clone --depth 1 --branch v0.4.13 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness/

# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/pyproject.toml
"$PY" -m pip install --no-build-isolation ".[hf, vllm]"

cd ../

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
python -c "import flash_attn_3; print('FlashAttention 3: OK')"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4: OK')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import platform; print(f'Python: {platform.python_version()}')"
for i in {1..100}; do echo -n '#'; done; echo

#######################################################################################################################

git clone https://github.com/NVIDIA/apex.git
cd apex/
# https://github.com/NVIDIA/apex/tags
git checkout "26.09"
# https://github.com/nvidia/apex#from-source
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 "$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
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

#######################################################################################################################

git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine/
# https://github.com/NVIDIA/TransformerEngine/releases
git checkout v2.20.2

git submodule update --init --recursive

# https://docs.nvidia.com/deeplearning/transformer-engine/installation.html
export NVTE_FRAMEWORK=pytorch
"$PY" -m pip install nvidia-cudnn-cu13

export CUDNN_PATH=$("$PY" -c "import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])")
export CUDNN_HOME=$CUDNN_PATH
export LD_LIBRARY_PATH="$CUDNN_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PY" -m pip install nvidia-nccl-cu13

export NCCL_HOME=$("$PY" -c "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])")
# The wheel ships only libnccl.so.2, but the NCCL EP Makefile links with -lnccl,
# which needs the unversioned libnccl.so.
ln -sf libnccl.so.2 "$NCCL_HOME/lib/libnccl.so"
export LD_LIBRARY_PATH="$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# CMake's find_library ignores NCCL_HOME and LD_LIBRARY_PATH; point it at the wheel's lib dir.
export CMAKE_LIBRARY_PATH="$NCCL_HOME/lib${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
# Make nvcc's default host compiler conda's g++
export NVCC_CCBIN="$CXX"
# The PyTorch extension build ignores CUDNN_PATH/NCCL_HOME for headers; gcc honors CPATH.
export CPATH="$CUDNN_PATH/include:$NCCL_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDNN_PATH/lib:$NCCL_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

"$PY" -m pip install "pybind11[global]"
# Unset conda's LDFLAGS for this build only: NCCL EP's Makefile passes its gcc-style -Wl,... flags to nvcc, which rejects them (conda gcc is still used via CXX).
env -u LDFLAGS "$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
python -c "import transformer_engine.pytorch as te; import transformer_engine; print(transformer_engine.__version__)"
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

#######################################################################################################################

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/
# https://github.com/NVIDIA/Megatron-LM/releases
git checkout core_v0.19.2

"$PY" -m pip install --no-build-isolation ".[te]"

cd ../

for i in {1..100}; do echo -n '#'; done; echo
python -c "import megatron.core; print(f'Megatron-Core: {megatron.core.__version__}')"
python -c "import transformer_engine.pytorch as te; import transformer_engine; print(transformer_engine.__version__)"
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

#######################################################################################################################

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift/
# https://github.com/modelscope/ms-swift/releases
git checkout v4.6.0 

# TODO: wait for release, which bumps peft
"$PY" -m pip install -r requirements/framework.txt
"$PY" -m pip install -r requirements/megatron.txt
"$PY" -m pip install --no-build-isolation .

cd ../

for i in {1..100}; do echo -n '#'; done; echo
python -c "import swift; print(f'ms-swift: {swift.__version__}')"
python -c "import megatron.core; print(f'Megatron-Core: {megatron.core.__version__}')"
python -c "import transformer_engine.pytorch as te; import transformer_engine; print(transformer_engine.__version__)"
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

# scp /scratch/jonapa/build-env.tar.gz  ehu129750@transfer1.bsc.es:/gpfs/scratch/ehu26/ehu129750
#### cpubash
# cd /gpfs/scratch/ehu26/ehu129750/
# mkdir build-env
# tar -xzf build-env.tar.gz -C build-env
# source build-env/bin/activate
# conda-unpack
# source build-env/bin/deactivate
# bash check.sh > output.out