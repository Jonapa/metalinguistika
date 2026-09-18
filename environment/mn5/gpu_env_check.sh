#!/usr/bin/env bash
# gpu_env_check.sh - verify the ML stack is installed AND actually able to use the GPU.
#
#   bash gpu_env_check.sh             # versions + functional GPU tests
#   RUN_NCCL=1 bash gpu_env_check.sh  # also run a multi-GPU NCCL all_reduce
#   FORCE_COLOR=1 bash ... | tee log  # keep colour when piping
#   NO_COLOR=1 bash ...               # plain text
#
# Exit code is non-zero if any required check fails. Warnings never fail the run.

set -u

# ---------------------------------------------------------------- presentation
W=$(tput cols 2>/dev/null || echo 100)
[ "$W" -gt 100 ] && W=100
[ "$W" -lt 64 ] && W=64
LBLW=24                                   # label column width

if [ -n "${NO_COLOR:-}" ]; then
  USE_COLOR=0
elif [ -n "${FORCE_COLOR:-}" ] || [ -t 1 ]; then
  USE_COLOR=1
else
  USE_COLOR=0
fi

if [ "$USE_COLOR" = 1 ]; then
  G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; C=$'\033[36m'
  D=$'\033[2m'; B=$'\033[1m'; N=$'\033[0m'
else
  G=''; R=''; Y=''; C=''; D=''; B=''; N=''
fi

# Box/tick glyphs only if the locale can render them.
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
  *UTF-8*|*utf8*|*UTF8*|*utf-8*) LINE='─'; DLINE='━'; ELL='…'; ASCII=0 ;;
  *)                             LINE='-'; DLINE='='; ELL='~'; ASCII=1 ;;
esac

pass=0; fail=0; warn=0; skip=0
FAIL_L=(); FAIL_D=(); WARN_L=(); WARN_D=(); SKIP_L=()

repeat() { local n=$1 ch=$2; [ "$n" -gt 0 ] || return 0; local i s=''; for ((i=0;i<n;i++)); do s+="$ch"; done; printf '%s' "$s"; }

banner() {
  printf '%s%s%s\n' "$B" "$(repeat "$W" "$DLINE")" "$N"
  printf '%s  %s%s\n' "$B" "$1" "$N"
  printf '%s  %s%s\n' "$D" "$2" "$N"
  printf '%s%s%s\n' "$B" "$(repeat "$W" "$DLINE")" "$N"
}

section() {                               # section "DRIVER & HARDWARE" [number]
  local title n sep='·'
  [ "$ASCII" = 1 ] && sep='|'
  if [ $# -gt 1 ]; then title="  $2 $sep $1  "; else title="  $1  "; fi
  n=$(( W - ${#title} - 2 ))
  [ "$n" -lt 0 ] && n=0
  printf '\n%s%s%s%s%s\n' "$C" "$(repeat 2 "$LINE")" "$title" "$(repeat "$n" "$LINE")" "$N"
}

tidy() {                                  # collapse python tracebacks to the useful lines
  awk '
    /^Traceback \(most recent call last\):[[:space:]]*$/ { next }
    /^  File "/ { frame = $0; next }
    /^[[:space:]]{4,}/ { next }
    /^[[:space:]]*$/ { next }
    { if (frame != "") { sub(/^[[:space:]]+/, "", frame); print frame; frame = "" } print }
  '
}

row() {                                   # row KIND LABEL VALUE
  local kind=$1 label=$2 value=$3 badge col dots n avail
  case $kind in
    ok)   badge="${G}[ OK ]${N}"; col='' ;;
    fail) badge="${R}${B}[FAIL]${N}"; col=$D ;;
    warn) badge="${Y}[WARN]${N}"; col=$D ;;
    skip) badge="${D}[SKIP]${N}"; col=$D ;;
    *)    badge="${D}[ -- ]${N}"; col=$D ;;
  esac
  n=$(( LBLW - ${#label} )); [ "$n" -lt 1 ] && n=1
  dots=$(printf '%*s' "$n" '' | tr ' ' '.')
  avail=$(( W - 10 - LBLW ))
  value=$(printf '%s' "$value" | tr '\n\t\r' '   ' | tr -s ' ')
  [ ${#value} -gt "$avail" ] && value="${value:0:$((avail-1))}${ELL}"
  printf '  %s %s%s%s %s%s%s\n' "$badge" "$D" "$label" "$N$dots" "$col" "$value" "$N"
}

_record() {                               # _record KIND LABEL DETAIL
  case $1 in
    fail) FAIL_L+=("$2"); FAIL_D+=("$3"); fail=$((fail+1)) ;;
    warn) WARN_L+=("$2"); WARN_D+=("$3"); warn=$((warn+1)) ;;
  esac
}

_run() {                                  # _run KIND_ON_FAIL LABEL cmd...
  local onfail=$1 label=$2; shift 2
  local out rc last
  out=$("$@" 2>&1); rc=$?
  last=$(printf '%s' "$out" | grep -v '^[[:space:]]*$' | tail -n1)
  if [ "$rc" -eq 0 ]; then
    row ok "$label" "${last:-done}"; pass=$((pass+1))
  else
    [ -n "$last" ] || { last="command failed, exit status $rc"; out="$last"; }
    row "$onfail" "$label" "$last"
    _record "$onfail" "$label" "$out"
  fi
}

check() { _run fail "$@"; }               # required: failure is an error
opt()   { _run warn "$@"; }               # non-package probe: failure is a warning

# opt_pkg LABEL MODULE cmd...
#   Optional COMPONENT, not optional CORRECTNESS. If MODULE is not installed the
#   check is skipped; if it IS installed it must work, and failure is an error.
#   (A broken-but-present package is a real defect, not an absent feature.)
opt_pkg() {
  local label=$1 mod=$2; shift 2
  if ! python -c "import importlib.util as u, sys
try: sys.exit(0 if u.find_spec('$mod') else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
    row skip "$label" "not installed"; skip=$((skip+1)); SKIP_L+=("$label"); return 0
  fi
  _run fail "$label" "$@"
}

banner "GPU ENVIRONMENT CHECK" \
       "$(hostname 2>/dev/null || echo host?) · $(date '+%Y-%m-%d %H:%M:%S %Z')"

# ------------------------------------------------------------------- section 1
section "DRIVER & HARDWARE" 1
check "nvidia-smi"           bash -c "set -o pipefail; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | paste -sd' / '"
opt   "CUDA_VISIBLE_DEVICES" bash -c "echo \"${CUDA_VISIBLE_DEVICES:-unset (all GPUs visible)}\""
opt   "nvcc"                 bash -c "set -o pipefail; nvcc --version | tail -n2 | head -n1"

# ------------------------------------------------------------------- section 2
section "PACKAGES  (import + version)" 2
check "Python"           python -c "import platform; print(platform.python_version())"
check "Torch"            python -c "import torch; print(torch.__version__)"
check "TorchVision"      python -c "import torchvision; print(torchvision.__version__)"
check "TorchAudio"       python -c "import torchaudio; print(torchaudio.__version__)"
check "Triton"           python -c "import triton; print(triton.__version__)"
check "Transformers"     python -c "import transformers; print(transformers.__version__)"
check "Accelerate"       python -c "import accelerate; print(accelerate.__version__)"
check "PEFT"             python -c "import peft; print(peft.__version__)"
check "DeepSpeed"        python -c "import deepspeed; print(deepspeed.__version__)"
# Deep import, not just the top-level package: liger_kernel/__init__.py is thin,
# so `import liger_kernel` succeeds even when the installed tree is incomplete.
opt_pkg "Liger Kernel"     liger_kernel         python -c "import liger_kernel.transformers, importlib.metadata as m; print(m.version('liger-kernel'))"
opt_pkg "TorchAO"          torchao              python -c "import torchao; print(torchao.__version__)"
opt_pkg "bitsandbytes"     bitsandbytes         python -c "import bitsandbytes as bnb; print(bnb.__version__)"
check   "FlashAttention 2"                      python -c "import flash_attn; print(flash_attn.__version__)"
opt_pkg "FlashAttention 3" flash_attn_interface python -c "import flash_attn_interface; print('flash_attn_interface importable')"
opt_pkg "FlashAttention 4" flash_attn.cute      python -c "from flash_attn.cute import flash_attn_func; print('flash_attn.cute importable')"
check "axolotl"          bash -c "axolotl --help >/dev/null 2>&1 && python -c \"import importlib.metadata as m; print(m.version('axolotl'))\""

# ------------------------------------------------------------------- section 3
section "GPU FUNCTIONAL TESTS" 3
res=$(mktemp)
# Triton's @jit reads its own source off disk (inspect.getsourcefile), so this
# block cannot be piped through stdin - write it to a real .py file and run that.
pyf=$(mktemp /tmp/gpuchk_XXXXXX.py)
cat > "$pyf" <<'PY'
import os, re, sys, time

RES = os.environ.get("GPUCHK_RESULT_FILE")
W = int(os.environ.get("GPUCHK_W", "100"))
LBLW = int(os.environ.get("GPUCHK_LBLW", "24"))
ELL = os.environ.get("GPUCHK_ELL", "...")
if os.environ.get("GPUCHK_COLOR") == "1":
    G, R, Y, D, B, N = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
else:
    G = R = Y = D = B = N = ""

BADGE = {"ok": f"{G}[ OK ]{N}", "fail": f"{R}{B}[FAIL]{N}",
         "warn": f"{Y}[WARN]{N}", "skip": f"{D}[SKIP]{N}"}
c = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
detail = []            # (kind, label, full message)

def row(kind, label, value, dur=None):
    tail = f"  {dur:.1f}s" if (dur and dur >= 1.0) else ""
    dots = "." * max(1, LBLW - len(label))
    avail = W - 10 - LBLW - len(tail)
    value = " ".join(str(value).split())
    if len(value) > avail:
        value = value[:max(0, avail - 1)] + ELL
    col = D if kind in ("fail", "warn") else ""
    print(f"  {BADGE[kind]} {D}{label}{N}{dots} {col}{value}{N}{D}{tail}{N}", flush=True)

def installed(mod):
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False

def run(label, fn, requires=None):
    # requires=MODULE -> skip when absent, but FAIL when present and broken.
    if requires and not installed(requires):
        row("skip", label, f"{requires} not installed")
        detail.append(("skip", label, ""))
        c["skip"] += 1
        return
    t0 = time.perf_counter()
    try:
        msg = fn() or ""
        row("ok", label, msg, time.perf_counter() - t0)
        c["pass"] += 1
    except Exception as e:
        full = f"{type(e).__name__}: {e}"
        row("fail", label, full, time.perf_counter() - t0)
        detail.append(("fail", label, full))
        c["fail"] += 1

def finish(code=None):
    if RES:
        try:
            with open(RES, "w") as f:
                f.write(f"COUNTS\t{c['pass']}\t{c['fail']}\t{c['warn']}\t{c['skip']}\n")
                for kind, label, msg in detail:
                    f.write(f"{kind}\t{label}\t" + " ".join(msg.split()) + "\n")
        except OSError:
            pass
    sys.exit(code if code is not None else (1 if c["fail"] else 0))

try:
    import torch
except Exception as e:
    row("fail", "import torch", f"{type(e).__name__}: {e}")
    detail.append(("fail", "import torch", f"{type(e).__name__}: {e}"))
    c["fail"] += 1
    finish()

# --- the check that actually matters -----------------------------------------
def t_available():
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False - CPU-only wheel, no driver, or no visible device")
    n = torch.cuda.device_count()
    assert n > 0, "zero CUDA devices visible"
    cudnn = torch.backends.cudnn.version()
    if not cudnn:
        cudnn = "NOT LINKED" if not torch.backends.cudnn.is_available() else "unknown"
    return f"{n} device(s) | CUDA {torch.version.cuda} | cuDNN {cudnn}"

def t_devices():
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        out.append(f"[{i}] {p.name} sm_{p.major}{p.minor} {free/1024**3:.0f}/{total/1024**3:.0f}GiB free")
    return " | ".join(out)

def t_arch():
    archs = torch.cuda.get_arch_list()
    missing = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        tag = f"sm_{p.major}{p.minor}"
        if not any(a.startswith(tag) for a in archs):
            missing.append(tag)
    assert not missing, f"torch has no kernels for {missing}; built for {' '.join(archs)}"
    return f"torch built for {' '.join(archs)}"

def t_matmul():
    torch.manual_seed(0)
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    got = (a @ b).float().cpu()
    torch.cuda.synchronize()
    assert torch.isfinite(got).all(), "non-finite output"
    ref = a.float().cpu() @ b.float().cpu()
    err = (got - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-2, f"GPU result disagrees with CPU (rel err {err:.3g})"
    return f"fp16 2048^3 matmul correct, rel err {err:.1e}"

def t_bf16():
    assert torch.cuda.is_bf16_supported(), "bf16 unsupported on this device"
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    return f"bf16 matmul ok | matmul tf32={torch.backends.cuda.matmul.allow_tf32}"

def t_build_tags():
    import torchvision, torchaudio
    v = {"torch": torch.__version__, "torchvision": torchvision.__version__,
         "torchaudio": torchaudio.__version__}
    tags = {k: (s.split("+", 1)[1] if "+" in s else "") for k, s in v.items()}
    cu = {k: t for k, t in tags.items() if re.fullmatch(r"cu\d+", t)}
    if len(cu) == 3:                       # wheels from the pytorch index
        assert len(set(cu.values())) == 1, f"mixed CUDA builds: {cu}"
        return f"all wheels built +{next(iter(cu.values()))}"
    if cu:
        raise AssertionError(f"some wheels CUDA-tagged, some not: {tags}")
    # Source/container builds carry a git hash instead of a cuXXX tag, so there
    # is nothing to compare - the extension-load tests below are the real proof.
    return "source builds (" + ", ".join(
        f"{k}={t or 'untagged'}" for k, t in tags.items()) + ")"

def t_torchvision_ext():
    from torchvision.ops import nms
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]], device="cuda")
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    keep = nms(boxes, scores, 0.5)
    torch.cuda.synchronize()
    return f"CUDA extension loaded, nms kept {keep.numel()} boxes"

def t_sdpa():
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    assert torch.isfinite(o).all()
    return "torch SDPA flash backend runs"

def t_flash2():
    import flash_attn
    from flash_attn import flash_attn_func
    q = torch.randn(2, 512, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    o = flash_attn_func(q, q, q, causal=True)
    o.float().sum().backward()             # backward kernels are compiled separately
    torch.cuda.synchronize()
    assert torch.isfinite(o).all() and torch.isfinite(q.grad).all()
    return f"v{flash_attn.__version__} forward + backward ok"

def t_flash3():
    from flash_attn_interface import flash_attn_func as fa3
    q = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
    o = fa3(q, q, q, causal=True)
    o = o[0] if isinstance(o, tuple) else o
    torch.cuda.synchronize()
    assert torch.isfinite(o).all()
    return "flash_attn_interface forward ok"

def t_flash4():
    from flash_attn.cute import flash_attn_func as fa4
    q = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
    o = fa4(q, q, q, causal=True)
    o = o[0] if isinstance(o, tuple) else o
    torch.cuda.synchronize()
    assert torch.isfinite(o).all()
    return "flash_attn.cute forward ok"

def t_triton():
    import triton, triton.language as tl

    @triton.jit
    def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

    n = 8192
    x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
    o = torch.empty_like(x)
    _add[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
    torch.cuda.synchronize()
    assert torch.allclose(o, x + y)
    return f"v{triton.__version__} JIT compile + launch ok"

def t_deepspeed():
    from deepspeed.accelerator import get_accelerator
    acc = get_accelerator()
    assert acc.is_available(), "DeepSpeed accelerator not available"
    from deepspeed.ops.adam import FusedAdam  # noqa: F401
    built = ""
    try:
        from deepspeed.git_version_info import installed_ops
        names = [k for k, v in installed_ops.items() if v]
        built = " | prebuilt: " + (", ".join(names) if names else "none, JIT at runtime")
    except Exception:
        pass
    return f"accelerator={acc.device_name()}{built}"

def t_liger():
    # Public API per README > Low-level APIs > Model Kernels.
    # apply_liger_kernel_to_llama is the entrypoint trainer integrations call.
    import inspect
    from liger_kernel.transformers import LigerRMSNorm
    from liger_kernel.transformers import apply_liger_kernel_to_llama  # noqa: F401
    params = inspect.signature(LigerRMSNorm.__init__).parameters
    m = (LigerRMSNorm(hidden_size=128) if "hidden_size" in params
         else LigerRMSNorm(128)).cuda().to(torch.bfloat16)
    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    m(x).float().sum().backward()
    torch.cuda.synchronize()
    return "patching API importable, RMSNorm fwd + bwd ok"

def t_nccl_p2p():
    n = torch.cuda.device_count()
    ver = ".".join(map(str, torch.cuda.nccl.version()))
    if n < 2:
        return f"NCCL {ver} | single GPU, P2P n/a"
    peers = sum(torch.cuda.can_device_access_peer(i, j)
                for i in range(n) for j in range(n) if i != j)
    return f"NCCL {ver} | {peers}/{n*(n-1)} P2P pairs direct"

run("cuda available",       t_available)
if not torch.cuda.is_available():
    finish()
run("devices",              t_devices)
run("kernel arch support",  t_arch)
run("real matmul",          t_matmul)
run("bf16",                 t_bf16)
run("cuda build tags",      t_build_tags)
run("torchvision cuda ops", t_torchvision_ext)
run("torch SDPA flash",     t_sdpa)
run("flash-attn 2 fwd+bwd", t_flash2)
run("flash-attn 3 fwd",     t_flash3, requires="flash_attn_interface")
run("flash-attn 4 fwd",     t_flash4, requires="flash_attn.cute")
run("triton kernel",        t_triton)
run("deepspeed",            t_deepspeed)
run("liger kernel",         t_liger, requires="liger_kernel")
run("nccl / p2p",           t_nccl_p2p)
finish(0)
PY

GPUCHK_RESULT_FILE="$res" GPUCHK_W="$W" GPUCHK_LBLW="$LBLW" \
GPUCHK_COLOR="$USE_COLOR" GPUCHK_ELL="$ELL" python "$pyf"

if [ -s "$res" ]; then
  while IFS=$'\t' read -r f1 f2 f3 f4 f5; do
    case "$f1" in
      COUNTS) pass=$((pass+f2)); fail=$((fail+f3)); warn=$((warn+f4)); skip=$((skip+f5)) ;;
      fail)   FAIL_L+=("$f2"); FAIL_D+=("$f3") ;;
      warn)   WARN_L+=("$f2"); WARN_D+=("$f3") ;;
      skip)   SKIP_L+=("$f2") ;;
    esac
  done < "$res"
else
  row fail "gpu functional tests" "python process died (segfault or import crash)"
  _record fail "gpu functional tests" "The functional test process exited without writing results - usually a segfault from an ABI mismatch. Re-run: python $pyf"
fi
rm -f "$res" "$pyf"

# ------------------------------------------------------------------- section 4
section "INFERENCE ENGINE" 4
opt_pkg "vLLM" vllm python -c "import vllm; from vllm.platforms import current_platform; print(f'v{vllm.__version__} on {current_platform.get_device_name()}')"

# ------------------------------------------------------------------- section 5
if [ "${RUN_NCCL:-0}" = "1" ]; then
  section "MULTI-GPU" 5
  ngpu=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
  if [ "$ngpu" -lt 2 ]; then
    row warn "nccl all_reduce" "needs >=2 GPUs, found $ngpu"
    _record warn "nccl all_reduce" "Skipped: needs at least 2 visible GPUs, found $ngpu."
  else
    cat > /tmp/_nccl_check.py <<'PY'
import torch, torch.distributed as dist
dist.init_process_group("nccl")
r, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
t = torch.full((1 << 22,), float(r + 1), device="cuda")
dist.all_reduce(t)
torch.cuda.synchronize()
expected = ws * (ws + 1) / 2
assert abs(t[0].item() - expected) < 1e-3, f"got {t[0].item()}, expected {expected}"
if r == 0:
    print(f"all_reduce correct across {ws} GPUs")
dist.destroy_process_group()
PY
    check "nccl all_reduce" torchrun --nproc_per_node="$ngpu" /tmp/_nccl_check.py
    rm -f /tmp/_nccl_check.py
  fi
fi

# --------------------------------------------------------------------- summary
detail_block() {                          # detail_block HEADER COLOR labels[] details[]
  local header=$1 color=$2; shift 2
  local -n L=$1; local -n Dd=$2
  [ "${#L[@]}" -eq 0 ] && return 0
  printf '\n  %s%s%s\n' "$color$B" "$header" "$N"
  local i
  for i in "${!L[@]}"; do
    printf '    %s%s%s\n' "$color" "${L[$i]}" "$N"
    printf '%s\n' "${Dd[$i]}" | tidy | fold -s -w $((W-8)) | while IFS= read -r ln; do
      printf '      %s%s%s\n' "$D" "$ln" "$N"
    done
  done
}

section "SUMMARY"
printf '  %s%d passed%s   %s%d failed%s   %s%d warnings%s   %s%d skipped%s\n' \
  "$G" "$pass" "$N" "$R" "$fail" "$N" "$Y" "$warn" "$N" "$D" "$skip" "$N"
if [ "${#SKIP_L[@]}" -gt 0 ]; then
  printf '  %snot installed: %s%s\n' "$D" "$(IFS=', '; echo "${SKIP_L[*]}")" "$N"
fi

detail_block "NEEDS ATTENTION" "$R" FAIL_L FAIL_D
detail_block "WARNINGS (optional components)" "$Y" WARN_L WARN_D

printf '\n%s%s%s\n' "$B" "$(repeat "$W" "$DLINE")" "$N"
if [ "$fail" -eq 0 ]; then
  printf '  %sENVIRONMENT READY%s' "$G$B" "$N"
  [ "$skip" -gt 0 ] && printf ' %s(%d optional component(s) not installed)%s' "$D" "$skip" "$N"
  printf '\n'
else
  printf '  %s%d REQUIRED CHECK(S) FAILED%s - see NEEDS ATTENTION above\n' "$R$B" "$fail" "$N"
fi
printf '%s%s%s\n' "$B" "$(repeat "$W" "$DLINE")" "$N"

[ "$fail" -eq 0 ]