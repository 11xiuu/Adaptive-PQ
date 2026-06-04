# AdaptivePQ

AdaptivePQ is a C++17 experimental ANN benchmark for SIFT-style float vectors.
It currently implements:

- binary `.tensor` dataset loading
- IVF coarse quantization
- per-list PQ training
- uniform PQ baselines
- per-IVF-list adaptive kcenter PQ configs
- PQ-estimate filtering with exact-distance refinement
- JSON benchmark output and Python plotting

## Build

```bash
cmake -S . -B build
cmake --build build
```

CMake defaults to `Release` and adds:

```text
-O3 -march=native
```

## Dataset Format

The loader expects binary tensor files:

```text
int64 n
int64 d
float data[n * d]
```

The current hard-coded dataset directory is:

```text
/home/wls/4TB/dataset/Sift10M
```

Expected files:

```text
data.tensor
query.tensor
groundtruth.tensor
```

`groundtruth.tensor` is not loaded by the benchmark. Exact ground truth is computed by brute force against the loaded base vectors.

## Run

Run from the build directory:

```bash
cd build
./adaptivepq
```

The current benchmark settings are in `main.cpp`, including:

```cpp
const std::int64_t max_base_n = 1000000;
const Dataset query((dataset_dir / "query.tensor").string(), 100);
const std::int64_t topk = 10;
const std::int64_t ivf_list_size = std::sqrt(index_base.n);
const float adaptive_radius_scale = 0.03f;
```

## IVF Cache

IVF coarse kmeans and assignment are cached so different PQ configs do not rebuild IVF every time.

Cache directory:

```text
/home/wls/4TB/dataset/index_buffer/PQ/
```

Cache filename pattern:

```text
ivf_n{n}_d{d}_lists{ivf}.bin
```

The cache stores:

- coarse centroids
- IVF list ids

PQ codebooks and PQ codes are rebuilt for each config.

## PQ Configs

The benchmark currently compares:

```text
8x8bit
16x4bit
kcenter8
kcenter16
```

### Uniform Baselines

`8x8bit`:

- 8 uniform subspaces
- each codebook size = 256

`16x4bit`:

- 16 uniform subspaces
- each codebook size = 16

Uniform configs are global and reused for every IVF list.

### Adaptive Kcenter Configs

`kcenter8` and `kcenter16` are generated separately inside each IVF list.

For each IVF list:

1. Start with every dimension as its own subspace.
2. Compute kcenter center count for each subspace.
3. Pick the subspace with the smallest center count.
4. Try merging it with every other active subspace.
5. Choose the merge with the smallest center-count growth.
6. Repeat until the target subspace count is reached.

`kcenter8` merges to 8 subspaces.

`kcenter16` merges to 16 subspaces.

Each adaptive subspace uses:

```cpp
codebook_size = kcenter_center_count
```

The kcenter radius is global, not per-list:

```cpp
radius_sq = adaptive_radius_scale * global_variance_sum(base)
```

This global radius is computed once in `Index::build()`.

## Index Build

For each config:

1. Load or build IVF.
2. For each IVF list:
   - choose config
   - train subspace PQ codebooks
   - encode list vectors
   - record quantization error for each original vector

Quantization error is:

```text
sum over subspaces ||sub_vector - assigned_codeword||^2
```

The code storage is currently:

```cpp
std::vector<std::int16_t> codes;
```

Layout:

```cpp
codes[row * subspace_count + subspace_id]
```

Codebook size is clamped to `int16_t` range during training.

## Query Logic

For each query:

1. Find nearest IVF coarse lists using `nprobe`.
2. For each probed list:
   - use that list's own config
   - copy query dimensions into contiguous subspace buffers
   - compute distance tables against the list's PQ codebooks
   - scan PQ codes and aggregate estimated distance
3. Maintain a top-k max heap of exact distances.

The refine filter uses:

```cpp
lower_bound = pq_dist - alpha * quantization_error[id]
```

If:

```cpp
lower_bound > current_worst_topk_exact_distance
```

the vector is skipped.

Otherwise, exact L2 distance to the original vector is computed and inserted into the top-k heap if useful.

Current alpha settings:

```text
8x8bit:   0.6
16x4bit:  0.6
kcenter8: 1.0, 0.8, 0.6
kcenter16: 1.0, 0.8, 0.6
```

## Benchmark Output

Results are written to:

```text
result/benchmark_results.json
```

The benchmark prints and stores:

- build time
- average bits per vector
- average query time
- recall@10
- exact distance computations per query
- PQ skipped vectors per query
- exact ratio
- skipped ratio

Example row:

```text
alpha,nprobe,time_ms,recall@10,avg_bits,exact_per_query,skipped_per_query,exact_ratio,skipped_ratio
```

## Config Dumps

After each index build, a few non-empty IVF list configs are exported:

```text
result/list_configs_8x8bit.json
result/list_configs_16x4bit.json
result/list_configs_kcenter8.json
result/list_configs_kcenter16.json
```

Each dump includes:

- list id
- list size
- subspace dims
- requested codebook size
- actual trained k

## Plotting

Generate plots after running the benchmark:

```bash
python3 scripts/plot_benchmark.py
```

The plot is saved to:

```text
result/benchmark_results.png
```

The plot shows:

- time-recall curve
- exact/refine ratio curve

The plotting script clips curves to a shared useful range:

- recall must be at least the best recall at the smallest `nprobe`
- time must be no larger than the best time at the largest `nprobe`

## Notes

- PCA code was removed from the current benchmark path.
- IVF is cached, PQ is rebuilt for every config.
- Adaptive kcenter config is per-IVF-list, not global.
- Query subspaces are copied into contiguous buffers before distance-table computation.
- Distance tables are still stored as vector-of-vector buffers, reused across probed lists.
- Codes are `int16_t`, not bit-packed.
