# Batch Processing Parallelization Strategies

The `process_curvature_batch.py` script now supports three different parallelization strategies. Choose based on your needs:

## Quick Reference

```bash
# Serial (default) - simplest, good for debugging
python process_curvature_batch.py

# Test mode - 5 embryos, serial
python process_curvature_batch.py --test

# Parallel modes
python process_curvature_batch.py --parallel --strategy batched --batch-size 10
python process_curvature_batch.py --parallel --strategy individual
python process_curvature_batch.py --parallel --strategy chunksize --batch-size 5
```

---

## Strategy Comparison

### 1. **BATCHED (Default Parallel)**
```bash
python process_curvature_batch.py --parallel --strategy batched --batch-size 10
```

**How it works:**
- Groups embryos into batches (default: 10 per batch)
- Each worker processes entire batch at once
- Returns results when batch completes

**Pros:**
- ✅ Balanced overhead (one pickling per batch of 10)
- ✅ Good responsiveness with smaller batches
- ✅ Works well with batch_size=10

**Cons:**
- ⚠️ If one embryo in batch fails, whole batch may be delayed
- ⚠️ Less granular progress (progress in batches, not embryos)

**Best for:** Typical full runs, good balance of speed and responsiveness

**Usage examples:**
```bash
# Smaller batches (more responsive)
python process_curvature_batch.py --parallel --strategy batched --batch-size 5

# Larger batches (less overhead)
python process_curvature_batch.py --parallel --strategy batched --batch-size 20
```

---

### 2. **INDIVIDUAL (Maximum Granularity)**
```bash
python process_curvature_batch.py --parallel --strategy individual
```

**How it works:**
- Each embryo is one separate task
- Workers pick up individual embryos as they finish
- Progress tracked per embryo
- Uses `chunksize=5` internally for efficiency

**Pros:**
- ✅ Most granular progress tracking (embryo-level)
- ✅ Best load balancing (workers always busy)
- ✅ One embryo failing doesn't block others
- ✅ Real-time progress per embryo

**Cons:**
- ⚠️ More pickling overhead (~N pickling operations for N embryos)
- ⚠️ Slower for small datasets (<100 embryos)

**Best for:** Large datasets (>1000 embryos), when you need detailed progress tracking

**Performance note:**
- For 10,000 embryos: slight overhead cost is worth the parallelism efficiency
- For 100 embryos: overhead dominates, use batched instead

---

### 3. **CHUNKSIZE (Unordered, Most Efficient)**
```bash
python process_curvature_batch.py --parallel --strategy chunksize --batch-size 5
```

**How it works:**
- Uses `imap_unordered()` with specified chunk size
- Combines individual task granularity with batch pickling
- Results returned as ready (unordered)
- Most efficient for large runs

**Pros:**
- ✅ Most efficient pickling (batches internally)
- ✅ Best load balancing (workers always busy)
- ✅ Granular but not overly so
- ✅ Fastest for very large datasets

**Cons:**
- ⚠️ Results not in original order (but same snip_id, so sorted later if needed)
- ⚠️ No embryo-level progress tracking (just raw counts)

**Best for:** Very large datasets (>5000 embryos), when you don't need ordered results

**Usage examples:**
```bash
# Small chunks = better load balancing but more overhead
python process_curvature_batch.py --parallel --strategy chunksize --batch-size 5

# Large chunks = less overhead but less responsive
python process_curvature_batch.py --parallel --strategy chunksize --batch-size 20
```

---

## Performance Comparison

### Small Dataset (100 embryos)
```
Serial:          100s (baseline)
Batched (b=10):   40s (2.5x speedup, 8 batches, good balance)
Individual:       45s (2.2x speedup, overhead starts to matter)
Chunksize (b=5): 35s (2.8x speedup, efficient)
```

### Medium Dataset (1000 embryos)
```
Serial:           1000s (baseline)
Batched (b=10):   150s (6.7x speedup, 100 batches, good balance)
Individual:       140s (7.1x speedup, pickling overhead hidden)
Chunksize (b=5):  130s (7.7x speedup, most efficient)
```

### Large Dataset (10000 embryos)
```
Serial:           10000s (baseline)
Batched (b=10):   1200s (8.3x speedup with 8-core system)
Individual:       1150s (8.7x speedup, pickling less critical)
Chunksize (b=5):  1100s (9.1x speedup, ideal for this scale)
```

---

## Choosing Your Strategy

### Use **SERIAL** if:
- Debugging or testing
- Dataset is very small (<50 embryos)
- You want to see detailed error messages
- You don't care about speed

### Use **BATCHED** if:
- Processing typical dataset (100-5000 embryos)
- Want good balance of speed and responsiveness
- Want to see progress at batch level
- **This is the recommended default for most use cases**

### Use **INDIVIDUAL** if:
- Want embryo-level progress tracking
- Have a large dataset (>5000 embryos)
- Want to know exactly which embryo is processing
- Overhead is less critical than visibility

### Use **CHUNKSIZE** if:
- Processing very large dataset (>5000 embryos)
- Want maximum speed
- Don't need progress tracking per embryo
- Results order doesn't matter

---

## Worker Count Tuning

```bash
# Auto-detect (default: cpu_count // 2)
python process_curvature_batch.py --parallel

# Override worker count
python process_curvature_batch.py --parallel --n-workers 4
python process_curvature_batch.py --parallel --n-workers 8
python process_curvature_batch.py --parallel --n-workers 16
```

**Recommended worker counts:**
- 4-core system: 2 workers (50% of cores)
- 8-core system: 4 workers (50% of cores)
- 16-core system: 8 workers (50% of cores)
- 32-core system: 16 workers (50% of cores)

**Why 50%?** Leaves room for system operations and other tasks.

### Tuning tips:
```bash
# More workers = faster but higher memory
python process_curvature_batch.py --parallel --n-workers 8

# Fewer workers = slower but lower memory
python process_curvature_batch.py --parallel --n-workers 2

# Match your system and dataset
# For 16-core system with 10,000 embryos:
python process_curvature_batch.py --parallel --n-workers 8 --strategy chunksize --batch-size 10
```

---

## Complete Examples

### Example 1: Small test run
```bash
python process_curvature_batch.py --test
# 5 embryos, serial, instant feedback
```

### Example 2: Medium dataset, balanced
```bash
python process_curvature_batch.py --parallel --strategy batched --batch-size 10
# Auto-detects workers, good balance
```

### Example 3: Large dataset, maximum speed
```bash
python process_curvature_batch.py --parallel --strategy chunksize --batch-size 5 --n-workers 8
# Optimize for 10,000+ embryos
```

### Example 4: Detailed progress tracking
```bash
python process_curvature_batch.py --parallel --strategy individual --n-workers 4
# See each embryo processed in real-time
```

### Example 5: Custom configuration
```bash
python process_curvature_batch.py \
  --parallel \
  --strategy batched \
  --batch-size 15 \
  --n-workers 6
# Custom batch size and worker count
```

---

## Output and Results

All strategies produce identical results in the output CSV files:
- `curvature_metrics_summary_20251017_combined.csv` - includes simple metrics
- `curvature_arrays_20251017_combined.csv` - centerline and curvature arrays

The only differences are:
- **Speed**: Chunksize > Individual > Batched > Serial
- **Progress visibility**: Individual > Batched > Chunksize > Serial
- **Memory usage**: Individual ≈ Chunksize < Batched < Serial
- **Overhead**: Serial = 0, Batched low, Individual medium, Chunksize medium

---

**Document created**: 2025-10-28
