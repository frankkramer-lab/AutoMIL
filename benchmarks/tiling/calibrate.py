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
Calibrates slideforge's ``min_tissue_fraction`` against slideflow's tissue detection.

For each resolution preset, a slideflow run with Otsu QC serves as the reference tile set. ``SlideforgeBackend``
is then run for a sweep of ``min_tissue_fraction`` values, and its kept tiles are matched to the reference by
their stored center location (both backends tile on the same grid). The fraction with the highest overall
Jaccard overlap is reported, together with a leave-one-slide-out check of how well it generalizes.

Usage:
    python benchmarks/tiling/calibrate.py --slides /data/svs --out /data/calibration --resolution High --resolution Low
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd

BENCH = Path(__file__).parent / "bench.py"


def _locations(tfrecords_dir: Path) -> dict[str, set[tuple[int, int]]]:
    """Tile center locations per slide"""
    from slideflow.io import get_locations_from_tfrecord

    return {p.stem: set(map(tuple, get_locations_from_tfrecord(str(p)))) for p in tfrecords_dir.glob("*.tfrecords")}


def _jaccard(df: pd.DataFrame) -> pd.Series:
    return df.both / (df.slideflow + df.slideforge - df.both)


def _reference(slides: Path, run_dir: Path, resolution: str, threads: int) -> dict:
    """Runs (or reuses) a slideflow reference extraction with Otsu QC via ``bench.py run``"""
    if not (run_dir / "run.json").exists():
        config = {
            "backend": "slideflow",
            "resolution": resolution,
            "threads": threads,
            "tissue": True,
            "stain": False,
            "min_tissue_fraction": 0.0,  # unused by slideflow
        }
        subprocess.run(
            [
                sys.executable,
                str(BENCH),
                "run",
                "--slides",
                str(slides),
                "--run-dir",
                str(run_dir),
                "--config",
                json.dumps(config),
            ],
            check=True,
        )
    return json.loads((run_dir / "run.json").read_text())


@click.command()
@click.option("--slides", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option(
    "--resolution",
    "resolutions",
    multiple=True,
    default=["High", "Low"],
    show_default=True,
    help="AutoMIL resolution preset name(s)",
)
@click.option("--start", type=float, default=0.05, show_default=True)
@click.option("--stop", type=float, default=0.95, show_default=True)
@click.option("--step", type=float, default=0.05, show_default=True)
@click.option("--threads", type=int, default=8, show_default=True)
def main(
    slides: Path,
    out: Path,
    resolutions: tuple[str, ...],
    start: float,
    stop: float,
    step: float,
    threads: int,
) -> None:
    """Sweeps min_tissue_fraction and writes <out>/calibration.csv"""
    from automil.tiling import SlideforgeBackend

    out.mkdir(parents=True, exist_ok=True)
    fractions = [round(f, 4) for f in np.arange(start, stop + step / 2, step)]
    slide_paths = sorted(str(p) for p in slides.iterdir() if p.is_file())
    rows = []

    for resolution in resolutions:
        run = _reference(slides, out / f"ref_{resolution}", resolution, threads)
        reference = _locations(Path(run["tfrecords_dir"]))
        click.echo(
            f"== {resolution}: tile_px={run['tile_px']} tile_um={run['tile_um']} slideflow tiles={run['n_tiles']}"
        )

        for fraction in fractions:
            tmp = out / "tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            # Minimal stand-in for sf.Dataset, so the real SlideforgeBackend code path is calibrated
            dataset = SimpleNamespace(
                tile_px=run["tile_px"],
                tile_um=run["tile_um"],
                slide_paths=lambda: slide_paths,
                tfrecords_folders=lambda: [str(tmp)],
                update_manifest=lambda force_update: None,
            )
            # JPEG quality doesn't affect tile selection, so keep the files small
            SlideforgeBackend(min_tissue_fraction=fraction, jpeg_quality=50, threads=threads, overwrite=True).extract(
                dataset, tissue_detection="otsu", stain_normalizer=None
            )
            candidate = _locations(tmp)
            for slide, ref_locs in sorted(reference.items()):
                cand_locs = candidate.get(slide, set())
                rows.append(
                    {
                        "resolution": resolution,
                        "fraction": fraction,
                        "slide": slide,
                        "slideflow": len(ref_locs),
                        "slideforge": len(cand_locs),
                        "both": len(ref_locs & cand_locs),
                    }
                )
        shutil.rmtree(out / "tmp", ignore_errors=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration.csv", index=False)

    for resolution, group in df.groupby("resolution", sort=False):
        totals = group.groupby("fraction")[["slideflow", "slideforge", "both"]].sum()
        jaccard = _jaccard(totals)
        best = jaccard.idxmax()
        click.echo(
            f"\n== {resolution}: best fraction {best} (jaccard {jaccard[best]:.3f}, "
            f"slideforge/slideflow tiles {totals.slideforge[best]}/{totals.slideflow[best]})"
        )
        click.echo(jaccard.round(3).to_string())

        # Leave-one-slide-out: pick the fraction on the other slides, evaluate on the held-out one
        if group.slide.nunique() > 1:
            for held_out in sorted(group.slide.unique()):
                train = group[group.slide != held_out].groupby("fraction")[["slideflow", "slideforge", "both"]].sum()
                chosen = _jaccard(train).idxmax()
                test = group[group.slide == held_out].set_index("fraction")
                click.echo(
                    f"  leave out {held_out}: chosen {chosen} -> held-out jaccard "
                    f"{_jaccard(test)[chosen]:.3f} (best possible {_jaccard(test).max():.3f})"
                )


if __name__ == "__main__":
    main()
