# Tile extraction benchmark: slideflow vs. slideforge

Benchmarks AutoMIL's current tile extraction (`sf.Dataset.extract_tiles`) against the experimental
[slideforge](https://github.com/WaibelJonas/slideforge) backend (`automil/tiling.py`).

Results and a detailed comparison of the two backends: [RESULTS.md](RESULTS.md).

## Setup

slideforge is built from source and needs a Rust toolchain:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
pip install -e ".[slideforge]"
```

The slideflow backend additionally needs a working slide reader (cucim, or libvips + pyvips).

## Running

```sh
# 1. Speed / resources: writes <out>/results.jsonl, one line per run
python benchmarks/tiling/bench.py --slides /data/svs --out /data/bench_tiling \
    --resolution Low --resolution High --threads 1 --threads 4 --threads 8 --repeats 3 [--drop-caches]

# 2. Equivalence: tile overlap + pixel similarity between two runs kept with --keep-tfrecords
#    (tfrecords_dir from results.jsonl)
python benchmarks/tiling/compare.py <slideflow_tfrecords_dir> <slideforge_tfrecords_dir> \
    --tile-um 500 --mpp 0.5 --out comparison.csv

# 3. Calibration: sweep slideforge's min_tissue_fraction against slideflow's Otsu QC tile selection
python benchmarks/tiling/calibrate.py --slides /data/svs --out /data/calibration --resolution High --resolution Low

# 4. Figures: tile selection maps and matched tile pairs from two runs kept with --keep-tfrecords
python benchmarks/tiling/figures.py <slideflow_tfrecords_dir> <slideforge_tfrecords_dir> --slides /data/svs \
    --tile-um 149 --mpp 0.499 --out benchmarks/tiling/figures --prefix high
```

Each run happens in a fresh subprocess with its own slideflow project. Extracted tfrecords are deleted after each run
unless `--keep-tfrecords last|all` is passed (needed for `compare.py`), and pending writes are flushed (`sync`) before each
run. Every result records kernel I/O stall time (`io_full_pct`); runs above `--max-io-stall` (5%) get a warning.
Runs both with and without tissue detection by default; `--no-tissue` isolates raw read/resize/encode throughput.

`--threads` sets rayon threads for slideforge and worker processes (`num_threads`) for slideflow.

## Known differences (not a like-for-like comparison yet)

| | slideflow | slideforge |
|---|---|---|
| Tissue detection | Otsu on the overview's saturation + per-tile grayspace filter | Otsu on the overview's grayscale + `min_tissue_fraction`, rechecked per tile |
| JPEG quality | 95 (cucim backend) | 95 (`jpeg_quality`, set by `SlideforgeBackend`) |
| Stain normalization | Reinhard, slideflow target | Reinhard, own target |
| Slide formats | Everything cucim / libvips reads | SVS only |
| Edge tiles | Dropped (`use_edge_tiles=False`) | Dropped in grid mode |
| `.index` files | Written | Created afterwards by `SlideforgeBackend` |

The `min_tissue_fraction=0.4` default was calibrated against slideflow's tissue-detected tiles on the three OpenSlide
CMU slides: it gives the highest tile overlap (Jaccard 0.94 at High, 0.96 at Low) and won every leave-one-slide-out
split within 0.01 (`calibrate.py`, rerun it for other slide sets). The two never match exactly, since they threshold different signals (grayscale vs. saturation).
