# Deployment benchmark report

## Data source

All raw measurements are stored in `benchmark.csv`. Each row is uniquely keyed by
`run_id, framework, backend, precision, mode`; derived speedups are intentionally
not stored because they can be calculated directly from the raw FPS values.

The legacy `pytorch.csv`, `tensorrt.csv`, `tensorrt_optimized.csv`,
`speed_comparison.csv`, and `optimization_comparison.csv` files were consolidated
into this table and removed.

## Protocol

- GPU: NVIDIA GeForce RTX 4060
- Input: batch 1, 640 x 640
- Warmup: 100 iterations
- Timed: 1000 iterations with CUDA synchronization
- TensorRT: FP16 deployment graph including the deploy postprocessor
- `model`: fixed device input; excludes image I/O, preprocessing, H2D, and D2H
- `end-to-end`: image read/decode, resize, H2D, TensorRT graph, and result D2H

Benchmarks abort before writing results when another non-parent compute process is
using the GPU.

## Historical model-scope results

The model-scope rows dated 2026-07-13 were preserved during consolidation. They
will be replaced, and matching end-to-end rows added, by the next idle-GPU run.

| Model | Dataset | TensorRT FP16 FPS | Mean latency (ms) |
|---|---|---:|---:|
| DEIM | UA-DETRAC | 334.84 | 2.986 |
| D-FINE | UA-DETRAC | 343.85 | 2.908 |
| CaS 4x MoE base03 | DAIR-V2X | 55.28 | 18.088 |
| CaS 4x MoE base05 | UA-DETRAC | 56.09 | 17.829 |
| CaS 0.5x MoE base05 | DAIR-V2X | 210.38 | 4.753 |
| CaS 0.5x dense-expert deploy | DAIR-V2X | 314.53 | 3.179 |

## Reproduction

```bash
/root/autodl-tmp/cas_trt_env/bin/python \
  experiments/CaS-DETR/tools/benchmark/benchmark_artifact_engines.py \
  --warmup 100 --iterations 1000
```

The command benchmarks all six archived engines and upserts both `model` and
`end-to-end` rows into `benchmark.csv` without duplicating historical PyTorch rows.
