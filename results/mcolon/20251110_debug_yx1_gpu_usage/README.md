# GPU Usage Diagnostics for build01B_compile_yx1_images_torch.py

## Problem
The GPU may not be used during build01B execution even when available on GPU nodes, causing very slow processing.

## Root Cause
The wrapper `run_build01.py` does not pass the `device` parameter to `build_ff_from_yx1()`, so the function uses its default parameter evaluation which may evaluate to "cpu" at module import time.

## Diagnostic Setup

### Files Modified
1. **`src/build/build01B_compile_yx1_images_torch.py`**
   - Added GPU diagnostics logging at function entry
   - Added per-stack logging of device and memory usage
   - Added PyTorch profiler traces (first 3 stacks only)
   - All controlled by `MSEQ_YX1_DEBUG=1` env variable

2. **`src/build/export_utils.py`**
   - Added profiler imports

### Files Created
1. **`debug_build01B_20251020.sh`** - Runner script for interactive GPU node testing

## How to Run

### Terminal 1: Monitor GPU Usage
```bash
# Start GPU monitoring before running the build
nvidia-smi dmon -s pucvmt -d 5 -i ${CUDA_VISIBLE_DEVICES:-0} | tee gpu_monitor.log

# Or use watch for simpler output
watch -n 2 nvidia-smi
```

### Terminal 2: Run Debug Build
```bash
# Run the debug script (optionally override experiment via positional arg)
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251110_debug_yx1_gpu_usage
MSEQ_YX1_DEBUG=1 bash debug_build01B_20251020.sh [20251020] 2>&1 | tee build_debug.log

# You can also override microscope/overwrite behavior via env vars:
#   MICROSCOPE=YX1 OVERWRITE=1 MSEQ_YX1_DEBUG=1 bash debug_build01B_20251020.sh 20251020
```

The script now calls `run_build01()` directly (the module exposes no CLI), so you should see the usual
validation + build logs in the console. Profiler traces continue to land in `$REPO_ROOT`.

## What to Look For

### 1. In the Build Log (`build_debug.log`)

Look for the GPU diagnostics section at the start:
```
============================================================
GPU DIAGNOSTICS - build_ff_from_yx1
Torch CUDA available: True/False
CUDA device count: N
CUDA version: X.X
Selected device: cuda/cpu
GPU 0: [GPU name]
n_workers: 1
============================================================
```

**Expected:** `CUDA available: True`, `Selected device: cuda`
**Bug confirmed if:** `CUDA available: False` or `Selected device: cpu` on GPU node

### 2. During Focus Stacking

Look for repeated log entries like:
```
_focus_stack: tensor shape=torch.Size([Z, Y, X]) device=cuda:0 cuda_available=True
_focus_stack: cuda mem allocated XXX.X MB
```

**Expected:** `device=cuda:0` and increasing memory allocation
**Bug confirmed if:** `device=cpu`

### 3. In GPU Monitor Log (`gpu_monitor.log`)

Look for non-zero values in:
- **sm** column: GPU utilization % (should be 50-100% during focus stacking)
- **mem** column: GPU memory usage (should increase when processing)
- **enc/dec** columns: May show activity depending on operations

**Expected:** High sm% and increasing mem during build
**Bug confirmed if:** All zeros or no change throughout build

### 4. Profiler Traces

Three trace files will be generated: `profiler_trace_stack_000.json`, `profiler_trace_stack_001.json`, `profiler_trace_stack_002.json`

**To view:**
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select a trace file
4. Look for operations:
   - **CUDA operations:** Should see `cudaLaunchKernel`, `cudaMemcpyAsync`, GPU streams
   - **CPU operations:** Will only see host-side operations

**Expected:** Many CUDA kernel launches (conv2d operations on GPU)
**Bug confirmed if:** Only CPU operations visible, no CUDA activity

## Common Issues

### Issue 1: CUDA Not Available
```
Torch CUDA available: False
Selected device: cpu
```

**Possible causes:**
- Not running on GPU node
- CUDA_VISIBLE_DEVICES not set or set to empty string
- PyTorch not compiled with CUDA support
- CUDA driver/library mismatch

**Check:**
```bash
nvidia-smi  # Should show GPU
echo $CUDA_VISIBLE_DEVICES  # Should show device ID(s)
python -c "import torch; print(torch.version.cuda)"  # Should show CUDA version
```

### Issue 2: Device Set to CPU Despite CUDA Available
```
Torch CUDA available: True
Selected device: cpu
```

**Cause:** The wrapper is not passing device parameter, and default evaluation chose CPU

**Solution:** Needs code fix to pass device parameter through the call chain

### Issue 3: Multiprocessing Interference
```
n_workers: 4 (GPU works best with n_workers=1)
```

**Cause:** Multiprocessing can cause CUDA context issues
**Solution:** Use `n_workers=1` for GPU builds (set in config or via CLI)

## Next Steps Based on Results

### If GPU is being used correctly:
- GPU utilization high (>50%)
- CUDA ops in profiler traces
- Log shows `device=cuda:0`

→ **GPU is working!** The issue may be elsewhere (data loading, I/O, etc.)

### If GPU is NOT being used:
- GPU utilization zero
- Only CPU ops in traces
- Log shows `device=cpu`

→ **Bug confirmed.** Need to:
1. Fix `run_build01.py` to pass device parameter
2. Ensure CUDA environment is properly initialized before module imports
3. Add explicit GPU device validation at entry points

## Files to Review for Fix

If bug is confirmed, these files need changes:
1. **`src/run_morphseq_pipeline/steps/run_build01.py`** - Add device parameter passing
2. **`src/run_morphseq_pipeline/run_experiment_manager_qsub.sh`** - Add CUDA environment setup
3. **`src/build/build01B_compile_yx1_images_torch.py`** - Keep diagnostic logging for future debugging
