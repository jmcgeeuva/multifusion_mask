#!/usr/bin/env python3
"""
Diagnostic script: emulates test.py startup up to load_checkpoint and reports
exactly where the chain breaks.

Run with (from multifusion_mask root):
    conda activate ssiai_adv
    python scripts/debug_checkpoint_load.py

Exit codes:
    0  everything passed
    1  at least one FAIL detected
"""

import sys
import os
import traceback
import subprocess
import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "IS-Fusion"))

CKPT_CONFIG  = "./IS-Fusion/ckpt/IS-Fusion_epoch_10.pth"
CKPT_ACTUAL  = "./pretrained_models/IS-Fusion_epoch_10.pth"
CONFIG_FILE  = "config/isfusion_0075voxel.py"
CUDA_TIMEOUT = 20  # seconds before declaring model.cuda() hung

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

_failures = []


def p(tag, msg):
    print(f"{tag} {msg}", flush=True)
    if tag == FAIL:
        _failures.append(msg)


def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# 1. Python / environment
# ---------------------------------------------------------------------------
section("1. Python & Conda Environment")
p(INFO, f"Python : {sys.version.split()[0]}  ({sys.prefix})")
p(INFO, f"CWD    : {os.getcwd()}")


# ---------------------------------------------------------------------------
# 2. CUDA / GPU compatibility
# ---------------------------------------------------------------------------
section("2. CUDA & GPU Compatibility")
try:
    import torch
    p(PASS, f"torch {torch.__version__} imported")
    p(INFO, f"PyTorch CUDA runtime : {torch.version.cuda}")

    arch_list = torch.cuda.get_arch_list()
    p(INFO, f"Compiled arch list   : {arch_list}")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        p(PASS, f"CUDA is available — {n} device(s)")
        all_compat = True
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            cap  = torch.cuda.get_device_capability(i)
            sm   = f"sm_{cap[0]}{cap[1]}"
            compute = f"compute_{cap[0]}{cap[1]}"
            compat  = sm in arch_list or compute in arch_list
            if not compat:
                all_compat = False
            tag  = PASS if compat else FAIL
            note = "" if compat else f"  <-- {sm} NOT in PyTorch arch list; CUDA ops will hang/fail"
            p(tag, f"GPU {i}: {name}  capability=sm_{cap[0]}{cap[1]}{note}")
        if not all_compat:
            p(FAIL, "GPU architecture incompatibility detected — model.cuda() WILL hang")
            p(INFO, "Remedy: upgrade PyTorch to a version that supports your GPU's SM version")
    else:
        p(FAIL, "CUDA is NOT available (no GPU or driver mismatch)")
except Exception as e:
    p(FAIL, f"torch import failed: {e}")
    traceback.print_exc()
    sys.exit(1)


# ---------------------------------------------------------------------------
# 3. Core mm* library imports
# ---------------------------------------------------------------------------
section("3. Core Library Imports")
for label, mod in [
    ("mmcv",             "mmcv"),
    ("mmdet",            "mmdet"),
    ("mmseg",            "mmseg"),
    ("mmdet3d",          "mmdet3d"),
    ("spconv",           "spconv"),
]:
    try:
        m   = __import__(mod)
        ver = getattr(m, "__version__", "?")
        src = getattr(m, "__file__", "?")
        p(PASS, f"{label} {ver}  ({os.path.dirname(src)})")
    except Exception as e:
        p(FAIL, f"{label}: {e}")


# ---------------------------------------------------------------------------
# 4. mmcv.runner.load_checkpoint
# ---------------------------------------------------------------------------
section("4. mmcv.runner.load_checkpoint import")
try:
    from mmcv.runner import load_checkpoint
    p(PASS, "load_checkpoint imported from mmcv.runner")
except Exception as e:
    p(FAIL, f"{e}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 5. Checkpoint file on disk
# ---------------------------------------------------------------------------
section("5. Checkpoint File on Disk")
ckpt_path = None
for candidate in [CKPT_CONFIG, CKPT_ACTUAL]:
    if os.path.exists(candidate):
        ckpt_path = candidate
        break

if ckpt_path is None:
    p(FAIL, f"Checkpoint not found at config path : {CKPT_CONFIG}")
    p(FAIL, f"Checkpoint not found at actual path : {CKPT_ACTUAL}")
else:
    size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    p(PASS, f"Found: {ckpt_path}  ({size_mb:.0f} MB)")
    if ckpt_path != CKPT_CONFIG:
        p(WARN, f"PATH MISMATCH — config.checkpoint = '{CKPT_CONFIG}'")
        p(WARN, f"  but the real file is at          '{CKPT_ACTUAL}'")
        p(WARN,  "  test.py line ~519 will fail with 'not a checkpoint file'")
        p(INFO,  "  Fix A (symlink): mkdir -p IS-Fusion/ckpt && "
                 "ln -s ../../pretrained_models/IS-Fusion_epoch_10.pth IS-Fusion/ckpt/")
        p(INFO,  "  Fix B (config):  change checkpoint= in the config to './pretrained_models/IS-Fusion_epoch_10.pth'")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        p(PASS, f"torch.load succeeded — top-level keys: {list(ckpt.keys())}")
        if "meta" in ckpt:
            meta = ckpt["meta"]
            p(INFO, "Checkpoint was built with:")
            for k in ["mmcv_version","mmdet_version","mmseg_version","mmdet3d_version","epoch"]:
                p(INFO, f"    {k}: {meta.get(k,'N/A')}")
            for line in meta.get("env_info", "").splitlines():
                if any(x in line for x in ["Python:","PyTorch:","CUDA_HOME","GPU 0","CUDA Runtime"]):
                    p(INFO, f"    {line.strip()}")
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            p(INFO, f"state_dict: {len(sd)} entries, first key: {next(iter(sd))}")
    except Exception as e:
        p(FAIL, f"torch.load failed: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 6. Config parsing
# ---------------------------------------------------------------------------
section("6. Config Parsing")
cfg = None
try:
    from mmcv import Config
    cfg = Config.fromfile(CONFIG_FILE)
    p(PASS, f"Config loaded: {CONFIG_FILE}")
    p(INFO, f"cfg.checkpoint = '{cfg.checkpoint}'")
    if not os.path.exists(cfg.checkpoint):
        p(FAIL, f"cfg.checkpoint path does not exist on disk — load_checkpoint will fail")
        p(INFO, f"  Actual file is at: {CKPT_ACTUAL}")
except Exception as e:
    p(FAIL, f"Config.fromfile failed: {e}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 7. Model build (CPU)
# ---------------------------------------------------------------------------
section("7. Model Build (CPU)")
model = None
if cfg is not None:
    try:
        from mmdet3d.models import build_model
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        p(PASS, f"build_model: {type(model).__name__}")
    except Exception as e:
        p(FAIL, f"build_model failed: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 8. load_checkpoint — exact call from test.py line ~519
# ---------------------------------------------------------------------------
section("8. load_checkpoint (mirrors test.py ~line 519)")
checkpoint_loaded = False
if model is not None:
    # Test A: with cfg.checkpoint (the path test.py actually uses)
    p(INFO, f"Test A — using cfg.checkpoint: '{cfg.checkpoint}'")
    try:
        from mmcv.runner import load_checkpoint as _lc
        _ = _lc(model, cfg.checkpoint, map_location="cpu")
        p(PASS, "Test A passed")
        checkpoint_loaded = True
    except Exception as e:
        p(FAIL, f"Test A failed: {e}")

    if not checkpoint_loaded and ckpt_path is not None and ckpt_path != cfg.checkpoint:
        p(INFO, f"Test B — using corrected path: '{ckpt_path}'")
        try:
            from mmcv.runner import load_checkpoint as _lc
            checkpoint = _lc(model, ckpt_path, map_location="cpu")
            p(PASS, "Test B passed (works when path is corrected)")
            checkpoint_loaded = True
        except Exception as e:
            p(FAIL, f"Test B also failed: {e}")
            traceback.print_exc()
else:
    p(WARN, "Skipped — model not built (see step 7)")


# ---------------------------------------------------------------------------
# 9. model.cuda() with timeout — the GPU-arch hang point
#
# SIGALRM cannot interrupt a hang inside the CUDA C driver (the GPU call
# blocks in uninterruptible kernel code).  We run model.cuda() in a child
# process and kill the whole process if it exceeds the timeout.
# ---------------------------------------------------------------------------
section("9. model.cuda() — GPU Architecture Hang Test (subprocess)")

_CUDA_TEST_SCRIPT = """\
import sys, os
sys.path.insert(0, '{isfusion_path}')
os.chdir('{root}')
import torch
from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
cfg = Config.fromfile('{config}')
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, '{ckpt}', map_location='cpu')
print('BEFORE_CUDA', flush=True)
model.cuda()
print('AFTER_CUDA', flush=True)
"""

if model is not None and cfg is not None and ckpt_path is not None:
    resolved_ckpt = ckpt_path if os.path.exists(ckpt_path) else cfg.checkpoint
    script_src = _CUDA_TEST_SCRIPT.format(
        isfusion_path=os.path.join(ROOT, "IS-Fusion"),
        root=ROOT,
        config=CONFIG_FILE,
        ckpt=resolved_ckpt,
    )
    p(INFO, f"Launching child process for model.cuda() with {CUDA_TIMEOUT}s hard kill ...")
    import subprocess
    python_exe = sys.executable  # same interpreter as this script
    proc = subprocess.Popen(
        [python_exe, "-c", script_src],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=CUDA_TIMEOUT)
        if "AFTER_CUDA" in stdout:
            p(PASS, "model.cuda() returned — GPU is compatible")
        elif "BEFORE_CUDA" in stdout:
            p(FAIL, "Child printed BEFORE_CUDA but never AFTER_CUDA — model.cuda() hung "
                    f"and was killed after {CUDA_TIMEOUT}s")
            p(FAIL, "This is the sm_120 Blackwell / PyTorch 1.10.1 arch mismatch hang.")
        else:
            p(FAIL, f"Child exited (rc={proc.returncode}) without reaching model.cuda().")
            p(INFO, f"Output: {stdout.strip()[:300]}")
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        p(FAIL, f"model.cuda() HUNG — child process killed after {CUDA_TIMEOUT}s")
        p(FAIL, "SIGALRM cannot interrupt CUDA C-level driver calls; subprocess kill required.")
        p(FAIL, "Confirmed: sm_120 Blackwell GPU is incompatible with PyTorch 1.10.1 (max sm_86).")
        p(INFO, "The hang occurs inside cuDeviceGet / cuMemAlloc in the CUDA driver, "
                "not in Python — no Python signal can unblock it.")
        p(INFO, "Remedy: rebuild env with PyTorch >= 2.6 which adds Blackwell (sm_120) support.")
else:
    p(WARN, "Skipped — model not built or checkpoint missing")


# ---------------------------------------------------------------------------
# 10. spconv CUDA forward-pass smoke test (subprocess — also hangs on Blackwell)
# ---------------------------------------------------------------------------
section("10. spconv Custom CUDA Op Smoke Test (subprocess)")
_SPCONV_SCRIPT = """\
import torch, spconv.pytorch as sp
dummy_f   = torch.randn(10, 4).cuda()
dummy_idx = torch.zeros(10, 4, dtype=torch.int32).cuda()
sp_t  = sp.SparseConvTensor(dummy_f, dummy_idx, [10, 10, 10], 1)
conv  = sp.SubMConv3d(4, 8, 3, bias=False).cuda()
_ = conv(sp_t)
print('SPCONV_OK', flush=True)
"""
try:
    import spconv
    p(PASS, f"spconv {spconv.__version__} imported (CPU import OK)")
    import subprocess
    proc2 = subprocess.Popen(
        [sys.executable, "-c", _SPCONV_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        out2, _ = proc2.communicate(timeout=CUDA_TIMEOUT)
        if "SPCONV_OK" in out2:
            p(PASS, "spconv CUDA forward pass succeeded")
        else:
            p(FAIL, f"spconv CUDA child exited (rc={proc2.returncode}) without SPCONV_OK")
            p(INFO, out2.strip()[:300])
    except subprocess.TimeoutExpired:
        proc2.kill(); proc2.communicate()
        p(FAIL, f"spconv CUDA forward pass HUNG — child killed after {CUDA_TIMEOUT}s")
        p(INFO, "Confirms all CUDA ops (not just model params) hang on sm_120 Blackwell.")
except ImportError as e:
    p(FAIL, f"spconv not installed: {e}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("Summary")
if not _failures:
    print(f"\n{PASS} All checks passed — environment looks healthy.\n", flush=True)
    sys.exit(0)
else:
    print(f"\n{FAIL} {len(_failures)} failure(s) detected:\n", flush=True)
    for i, f in enumerate(_failures, 1):
        print(f"  {i}. {f}", flush=True)

    print("""
Root causes identified so far:

  [A] CHECKPOINT PATH MISMATCH
      cfg.checkpoint = './IS-Fusion/ckpt/IS-Fusion_epoch_10.pth'  (does not exist)
      Actual file   = './pretrained_models/IS-Fusion_epoch_10.pth'
      Quick fix:
        mkdir -p IS-Fusion/ckpt
        ln -s ../../pretrained_models/IS-Fusion_epoch_10.pth IS-Fusion/ckpt/

  [B] GPU ARCHITECTURE INCOMPATIBILITY (HPC UPDATE)
      Old HPC GPUs : Tesla V100   (sm_70)  — supported by PyTorch 1.10.1
      New HPC GPUs : RTX PRO 6000 (sm_120) — NOT supported by PyTorch 1.10.1
      model.cuda() hangs because CUDA kernels cannot be dispatched.
      Required: rebuild the full environment for CUDA 12+ / Blackwell support.

      Minimum version requirements for Blackwell (sm_120):
        PyTorch  >= 2.6.0
        CUDA     >= 12.8  (driver >= 570)
        spconv   >= 2.3   (rebuild from source for cu128 / sm_120)
        mmcv-full: must be rebuilt from source for the new PyTorch/CUDA versions
                   (no prebuilt mmcv-full wheels for PyTorch 2.x yet —
                    use mmcv >= 2.x which supports PyTorch 2.x)

      Migration path:
        1. conda create -n isfusion_bw python=3.10 -y
        2. pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
        3. pip install mmcv==2.x.x  (check openmmlab for latest)
        4. Re-install mmdet, mmdet3d, spconv for new versions
        5. Verify custom IS-Fusion ops are compatible with mmdet3d 1.x API
           (IS-Fusion was written for mmdet3d 0.16 which has API differences)
""", flush=True)
    sys.exit(1)
