# Group Prior Estimation (Step 8) Optimization Plan

## Overview

The `estimate_group_priors()` EM loop in `core/group_priors.py` has significant
optimization opportunities: redundant expensive computations, and Python loops
over subjects/sessions/clusters that can be replaced with vectorized einsum
operations.

## Files modified

| File | Changes |
|------|---------|
| `src/pymshbm/core/group_priors.py` | All optimizations below |
| `src/pymshbm/math/vmf.py` | Added optional `log_c` param to `vmf_log_probability` |
| `tests/test_core/test_group_priors.py` | Added numerical regression test |

## Optimizations

### 1. Cache vmf_log_probability from E-step -> reuse in cost [DONE]

**Problem:** Each inner EM iteration calls `_e_step` then `_compute_em_cost`.
Both compute `vmf_log_probability(X, nu, kappa)` for every (subject, session)
slice -- an `(81924, 642) @ (642, 17)` matrix multiply repeated S*T times,
done twice.

**Fix:** `_e_step` returns list of per-subject `log_vmf_total` arrays.
`_compute_em_cost` accepts optional `log_vmf_cache` to skip recomputation.

**Estimated savings:** ~25% of total inner-loop compute.

- [x] Modify `_e_step` to return `log_vmf_cache: list[np.ndarray]`
- [x] Modify `_compute_em_cost` to accept optional precomputed log_vmf_cache
- [x] Update `vmf_clustering_subject_session` to pass cached values

### 2. Vectorize `_m_step` kappa update [DONE]

**Problem:** Python loop `for s in S: for t in T:` computing `X @ nu` and
accumulating scalar sums.

**Fix:** Single einsum + broadcast:
```python
dots = np.einsum("ndst,dlts->nlst", data, params.s_t_nu)
kappa_num = np.nansum(params.s_lambda[:, :, :, np.newaxis] * dots)
kappa_den = np.nansum(params.s_lambda)
```

- [x] Replace double loop with einsum

### 3. Vectorize `_m_step` s_t_nu update [DONE]

**Problem:** Python loop `for s in S: for t in T:` computing `X.T @ sl` per
slice.

**Fix:** Batch einsum + broadcast + vectorized convergence:
```python
weighted_data = np.einsum("ndst,nls->dlst", data, params.s_lambda)
lambda_X = kappa * weighted_data.transpose(0,1,3,2) + sigma * s_psi[:,:,None,:]
```

- [x] Replace double loop with einsum + broadcast
- [x] Vectorize convergence check

### 4. Vectorize `intra_subject_var` s_psi accumulation [DONE]

**Problem:** Loop over S with inner loop over T summing `sigma * s_t_nu`.

**Fix:**
```python
accum = sigma[None, :, None] * s_t_nu.sum(axis=2) + epsil[None, :, None] * mu[:, :, None]
```

- [x] Replace S*T loop with vectorized sum + broadcast

### 5. Vectorize `intra_subject_var` sigma update [DONE]

**Problem:** Triple Python loop L*S*T computing scalar inner products.

**Fix:**
```python
rbar_all = np.einsum('dls,dlts->l', s_psi, s_t_nu) / (S * T)
```

- [x] Replace triple loop with einsum

### 6. Vectorize `inter_subject_var` epsil update [DONE]

**Problem:** Double loop L*S computing inner products.

**Fix:**
```python
rbar_all = np.einsum('dls,dl->l', s_psi, mu) / S
```

- [x] Replace double loop with einsum

### 7. Precompute cdln once per inner iteration [DONE]

**Problem:** `kappa` is uniform (`np.full(L, val)`), so `cdln` computes L
identical Bessel evaluations. Also recomputed for each (subject, session) call.

**Fix:** Added optional `log_c` parameter to `vmf_log_probability`. Precompute
once per E-step and pass through. Also precomputed in `_compute_initial_s_lambda`
and `_compute_em_cost` fallback path.

- [x] Add optional `log_c` parameter to `vmf_log_probability`
- [x] Precompute in `_e_step`, `_compute_initial_s_lambda`, `_compute_em_cost`

## Round 1 Verification

All 157 tests pass. Numerical regression test confirms identical output
to pre-optimization code (atol=1e-6 on mu, atol=1e-4 on sigma, atol=0.01
on cost record).

---

## Round 2: Memory-bandwidth and BLAS optimizations

Real-world dimensions: N=81924, D=1284, S=5, T=12, L=15.
Data tensor: (81924, 1284, 5, 12) float32 = **25.2 GB** (was 50.5 GB in float64).

### 8. Hoist `weighted_data` out of M-step inner loop [DONE]

**Problem:** `weighted_data = einsum("ndst,nls->dlst", data, s_lambda)` is
recomputed every M-step inner iteration (~10 iterations), but `s_lambda` is
constant within the M-step. Each call reads the full data tensor.
9 of 10 calls are redundant.

**Fix:** Added `_compute_weighted_data()` helper. Called once per inner EM
iteration before the M-step loop, passed as parameter.

- [x] Move `weighted_data` computation before M-step inner loop
- [x] Pass `weighted_data` as parameter to inner loop body

### 9. Accumulate `kappa_num` without (N,L,S,T) intermediate [DONE]

**Problem:** `dots = einsum("ndst,dlts->nlst", data, s_t_nu)` materializes
a 590 MB intermediate (81924 × 15 × 5 × 12), then sums it to a scalar.

**Fix:** Loop over (s,t) pairs with explicit `data[:,:,s,t] @ s_t_nu[:,:,t,s]`
BLAS GEMM, accumulate kappa_num directly. No large intermediate.

- [x] Replace dots einsum with S×T matmul accumulation loop

### 10. Replace remaining 4D einsums with explicit BLAS matmuls [DONE]

**Problem:** 4D einsum patterns may not dispatch to optimized BLAS.

**Fix:** `_compute_weighted_data()` uses explicit `data[:,:,s,t].T @ sl`
per (s,t) slice. E-step already used per-slice matmuls.

- [x] Replace `weighted_data` einsum with S×T matmul loop
- [x] E-step already uses explicit matmuls per-slice

### 11. Use float32 for data tensor [DONE]

**Problem:** Data tensor was 50.5 GB in float64.

**Fix:** Changed Zarr store dtype to float32 in `_open_profiles_zarr`.
All downstream operations (normalized tensor cache, group priors) get
float32 data. Parameters stay float64; numpy auto-upcasts in matmuls.

- [x] Changed Zarr dtype from float64 to float32
- [x] Matmuls auto-upcast (numpy handles mixed dtypes)
- [x] Added regression test for float32 group priors

### 12. Vectorize `inv_ad` with Halley/Newton-bisection [DONE]

**Problem:** `inv_ad` uses scalar `brentq` root-finding, called L×50 times
per outer iteration.

**Fix:** Added `inv_ad_batch(d, rbar_array)` using damped Newton-Raphson
with bisection fallback. Derivative: `A_d'(k) = 1 - A_d(k)^2 - (d-1)/k * A_d(k)`.
Actually more accurate than scalar `inv_ad` (which falls back to approximation
when brentq fails for large d).

- [x] Add `inv_ad_batch` to `math/vmf.py`
- [x] Add tests (roundtrip, high dimension, recovers rbar)
- [x] Replace L-loops in `intra_subject_var` and `inter_subject_var`

### 13. Hoist constants in `_compute_em_cost` [DONE]

**Problem:** `log_theta` and `log_c` computed inside subject loop despite
being constant.

**Fix:** Hoisted both before the loop. `_compute_initial_s_lambda` already
uses per-slice matmuls, no further change needed.

- [x] Hoist `log_theta` and `log_c` before subject loop
- [x] `_compute_initial_s_lambda` already uses per-slice matmuls

## Round 2 Verification

All 162 tests pass. Numerical regression test confirms identical output
(float64 path unchanged). float32 regression test confirms close results.

```bash
uv run python -m pytest tests/ -v  # 162 passed
```

---

## Round 3: Medial wall exclusion

### 14. Exclude medial wall vertices at step 6 [DONE]

**Problem:** ~30% of vertices (medial wall) have all-zero profiles.
These are carried through steps 7-8 as dead weight in every matmul.
With N=81924, this wastes ~30% of all computation.

**Approach:**
1. `_detect_valid_vertices(data)` → boolean mask (N,), True for non-zero rows
2. Filter data for steps 7 (k-means) and 8 (group priors): `data_reduced = data[mask]`
3. After step 8, expand `theta` and `s_lambda` back to full surface
4. Step 9 uses full data + expanded group priors (it handles medial wall internally)
5. CIFTI output unchanged (full surface labels)

**Estimated savings:** ~30% reduction in all step 7-8 matmul costs.

- [x] Add `_detect_valid_vertices()` helper
- [x] Filter data before steps 7-8 in `run_wrapper`
- [x] Expand theta/s_lambda back to full surface before step 9
- [x] Add 3 tests for mask detection (all valid, excludes zeros, partial zero kept)
- [x] `_compute_initial_centroids` accepts `valid_mask` parameter
- [x] All 165 tests pass

## Round 3 Verification

```bash
uv run python -m pytest tests/ -v  # 165 passed
```

---

## Round 4: Memory mapping and multiprocessing

### 15. Memory-map the data tensor [DONE]

**Problem:** `_load_profiles_tensor` loads the full 25.2 GB tensor into memory
via `zarr.open_array(...)[:]`. On machines with limited RAM this causes heavy
swapping, and the full tensor must be resident even though the EM loops only
access one (subject, session) slice at a time.

**Fix:** Cache the normalized tensor as `.npy` instead of Zarr. Return
`np.load(path, mmap_mode='r')` so the OS pages in data on demand. Also cache
the medial-wall-filtered reduced tensor as a separate `.npy` memmap.

- [x] Change `_load_profiles_tensor` cache from Zarr to `.npy`
- [x] Return `np.memmap` via `np.load(..., mmap_mode='r')`
- [x] Cache reduced tensor as `.npy` memmap in `run_wrapper`
- [x] Update existing tests for new cache format

### 16. Multiprocessing over subjects [DONE]

**Problem:** The S×T matmul loops in `_e_step`, `_compute_weighted_data`,
`_m_step` (kappa accumulation), `_compute_em_cost`, and
`_compute_initial_s_lambda` are embarrassingly parallel across subjects.
With S=5 subjects on a multi-core machine, only one core is utilized.

**Fix:** Use `concurrent.futures.ProcessPoolExecutor` with a persistent pool.
Workers open their own memory-mapped view of the data tensor via
`_init_worker(data_path)`. Each subject is processed by a separate worker.
Falls back to sequential processing when data is not memory-mapped (e.g.
small test arrays).

- [x] Add module-level worker state and `_init_worker` initializer
- [x] Add per-subject worker functions for each parallelizable operation
- [x] `_create_pool` auto-detects memmap and creates pool
- [x] Pool created once in `estimate_group_priors`, passed through to inner functions
- [x] Graceful fallback: non-memmap data uses sequential path (tests unaffected)
- [x] Add test verifying parallel path matches sequential results

## Round 4 Verification

```bash
uv run python -m pytest tests/ -v
```
