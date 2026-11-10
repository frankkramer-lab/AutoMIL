# ==============================================================================#
#  AutoMIL - Automated Machine Learning for Image Classification in            #
#  Whole-Slide Imaging with Multiple Instance Learning                         #
#                                                                              #
#  Copyright (C) 2026 Jonas Waibel                                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.      #
# ==============================================================================#
"""
Compares the tfrecords of two tile extraction runs (e.g. slideflow vs. slideforge).

Tiles are matched per slide by their stored center location (``loc_x``/``loc_y``, level-0 pixels).
For every slide present in both runs this reports the tile overlap and, for a sample of matched
tiles, pixel-level similarity (mean absolute difference and SSIM).

Usage:
    python benchmarks/tiling/compare.py <reference_tfrecords_dir> <candidate_tfrecords_dir> \\
        --tile-um 500 --mpp 0.5 --out comparison.csv

The tfrecords directories are listed as ``tfrecords_dir`` in ``results.jsonl``.
"""

from __future__ import annotations

import io
from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import KDTree
from skimage.metrics import structural_similarity
from slideflow.io import get_locations_from_tfrecord, get_tfrecord_by_index


def _decode(record: dict) -> np.ndarray:
    image_raw = record["image_raw"]
    if not isinstance(image_raw, bytes):
        image_raw = image_raw.numpy()
    return np.asarray(Image.open(io.BytesIO(image_raw)).convert("RGB"))


def compare_slide(
    reference: Path,
    candidate: Path,
    tolerance_px: float,
    max_pairs: int,
    rng: np.random.Generator,
) -> dict:
    """Matches tiles of two tfrecords for the same slide and computes similarity metrics"""
    ref_locs = np.asarray(get_locations_from_tfrecord(str(reference)), dtype=float).reshape(-1, 2)
    cand_locs = np.asarray(get_locations_from_tfrecord(str(candidate)), dtype=float).reshape(-1, 2)

    row = {"slide": reference.stem, "n_reference": len(ref_locs), "n_candidate": len(cand_locs)}
    if len(ref_locs) == 0 or len(cand_locs) == 0:
        return row | {"n_matched": 0, "jaccard": 0.0}

    # Nearest-neighbour match of tile centers within tolerance
    distances, cand_idx = KDTree(cand_locs).query(ref_locs, distance_upper_bound=tolerance_px)
    distances, cand_idx = np.ravel(distances), np.ravel(cand_idx)  # typed as possibly scalar
    matched = np.flatnonzero(np.isfinite(distances))
    n_matched = len(np.unique(cand_idx[matched]))
    row |= {
        "n_matched": n_matched,
        "jaccard": n_matched / (len(ref_locs) + len(cand_locs) - n_matched),
        "mean_offset_px": float(distances[matched].mean()) if len(matched) else None,
    }

    # Pixel similarity on a sample of matched pairs
    sample = rng.choice(matched, size=min(max_pairs, len(matched)), replace=False) if len(matched) else []
    mad, ssim = [], []
    for ref_i in sample:
        ref_img = _decode(get_tfrecord_by_index(str(reference), int(ref_i)))
        cand_img = _decode(get_tfrecord_by_index(str(candidate), int(cand_idx[ref_i])))
        if ref_img.shape != cand_img.shape:
            cand_img = np.asarray(Image.fromarray(cand_img).resize(ref_img.shape[1::-1]))
        mad.append(np.abs(ref_img.astype(np.int16) - cand_img.astype(np.int16)).mean())
        ssim.append(structural_similarity(ref_img, cand_img, channel_axis=-1))

    return row | {
        "n_compared": len(sample),
        "mean_abs_diff": float(np.mean(mad)) if mad else None,
        "mean_ssim": float(np.mean(ssim)) if ssim else None,
    }


@click.command()
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--tile-um", type=float, required=True, help="Tile size in microns (tile_um in results.jsonl)")
@click.option("--mpp", type=float, required=True, help="Level-0 microns per pixel of the slides")
@click.option(
    "--tolerance",
    type=float,
    default=0.25,
    show_default=True,
    help="Max center offset for a match, as a fraction of the tile size",
)
@click.option("--max-pairs", type=int, default=50, show_default=True, help="Matched tiles per slide to compare")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write per-slide CSV")
def main(
    reference_dir: Path,
    candidate_dir: Path,
    tile_um: float,
    mpp: float,
    tolerance: float,
    max_pairs: int,
    seed: int,
    out: Path | None,
) -> None:
    rng = np.random.default_rng(seed)
    tolerance_px = tolerance * tile_um / mpp

    reference = {p.stem: p for p in reference_dir.glob("*.tfrecords")}
    candidate = {p.stem: p for p in candidate_dir.glob("*.tfrecords")}
    shared = sorted(reference.keys() & candidate.keys())

    if missing := sorted(reference.keys() ^ candidate.keys()):
        click.echo(f"Slides only present in one run ({len(missing)}): {missing}")

    df = pd.DataFrame([compare_slide(reference[s], candidate[s], tolerance_px, max_pairs, rng) for s in shared])
    click.echo(df.to_string(index=False))
    click.echo("\nOverall:")
    click.echo(df[["n_reference", "n_candidate", "n_matched"]].sum().to_string())
    click.echo(df[["jaccard", "mean_abs_diff", "mean_ssim"]].mean().to_string())

    if out is not None:
        df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
