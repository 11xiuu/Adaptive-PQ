# AdaptivePQ

Experimental ANN benchmark comparing three PQ-based search pipelines on SIFT vectors.

## Build

```bash
cmake -S . -B build
cmake --build build
```

Produces three executables:

| Target | File | Description |
|---|---|---|
| `adaptivepq` | main.cpp | **Local PQ** — per-list PQ on original vectors |
| `adaptivepq_residual` | main_residual.cpp | **Residual PQ** — per-list PQ on IVF residuals |
| `adaptivepq_faiss` | main_faiss.cpp | **FAISS IVFPQ** baseline (global PQ + residual) |

## Run

```bash
cd build
./adaptivepq              # Local PQ
./adaptivepq_residual     # Residual PQ
./adaptivepq_faiss        # FAISS IVFPQ
```

Each reads from `/home/zhaixiue/data/sift/sift_base.fvecs` and writes results to `result/{dataset}_{size}/`.

## Architecture

### 1. Local PQ (`adaptivepq`)

- IVF coarse quantization → assign vectors to `nlist = sqrt(N)` lists
- For each list, train per-list PQ codebooks:
  - **Uniform config**: fixed subspace split (e.g. 8×8 = 8 subspaces × 256 centroids)
  - **Kcenter config**: adaptive subspace merging via greedy kcenter covering
- Query: find nearest nprobe lists → PQ ADC distance tables → alpha-pruned refinement

### 2. Residual PQ (`adaptivepq_residual`)

Same as Local PQ, but:

- **Build**: each vector is reduced to its residual `x - coarse_centroid[list]` before PQ training
- **Query**: distance table uses `(query - coarse_centroid)` instead of raw query
- Kcenter radius is computed from residual variance (not original data variance)

### 3. FAISS IVFPQ (`adaptivepq_faiss`)

FAISS `IndexIVFPQ` with:
- `nprobe` sweep (same values as the custom pipelines)
- Rerank top-500 candidates with exact L2
- Two configs: M=8 nbits=8, M=16 nbits=4

## Configs

All configs use the same nprobe list: `{1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50}`

| Config | Subspaces | Codebook | Alpha | Algorithm |
|---|---|---|---|---|
| `8x8bit{-local,-residual}` | 8 uniform | 256 | 0.8 | Fixed partition |
| `16x4bit{-local,-residual}` | 16 uniform | 16 | 0.8 | Fixed partition |
| `kcenter8{-residual}` | 8 adaptive | kcenter count | 1.0, 0.8 | Greedy merge |
| `kcenter16{-residual}` | 16 adaptive | kcenter count | 1.0, 0.8 | Greedy merge |
| `faiss-IVFPQ 8*8` | 8 | 256 | — | FAISS |
| `faiss-IVFPQ 16*4` | 16 | 16 | — | FAISS |

## Pruning

Adaptive pruning uses the **L2-norm triangle inequality** (not squared-distance approximation):

```
bound = √pq_dist - α · √qe
if (bound² > heap_front.exact_sq) → skip vector
```

- α = 1.0: strict lower bound (no false negatives)
- α = 0.8: tighter pruning (may trade recall for speed)

## Results

Each run produces per-config JSON files in `result/{dataset}_{size}/`:

```
result/sift_100000/
├── 8x8bit-local_a0.8.json
├── 16x4bit-local_a0.8.json
├── kcenter8_a1.0.json
├── kcenter8_a0.8.json
├── kcenter16_a1.0.json
├── kcenter16_a0.8.json
├── faiss-IVFPQ_8*8.json
├── faiss-IVFPQ_16*4.json
└── time_recall_curve.png
```

## Plotting

```bash
python3 scripts/plot_benchmark.py
```

Scans all dataset subdirectories in `result/` and generates a time-recall curve per dataset. Curves clip at recall < 0.7.

## IVF Cache

IVF centroids and list assignments are cached to speed up repeated runs with different PQ configs. Cache is in `./data/index_buffer/PQ/`.
