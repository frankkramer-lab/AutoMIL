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
Benchmarks tile extraction backends (slideflow vs. slideforge).

Every configuration runs in a fresh subprocess with its own slideflow project, so runs neither share
extracted tfrecords nor memory. The parent samples the subprocess tree for peak memory, records kernel
pressure stall time (PSI) per run and appends one JSON line per run to ``<out>/results.jsonl``.

Extracted tfrecords are deleted after each run by default, so the benchmark doesn't grow the disk (on WSL,
growing the virtual disk caused write stalls). Pass ``--keep-tfrecords last`` to keep one run per config and
use ``compare.py`` on those to check how equivalent the outputs are.

Usage:
    python benchmarks/tiling/bench.py --slides /data/svs --out /data/bench_tiling \\
        --resolution Low --threads 1 --threads 4 --threads 8 --repeats 3
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import click
import psutil


def _sample_peak_memory(process: psutil.Process, stop: threading.Event, peak: list[int]) -> None:
    """Samples summed peak memory of `process` and its child processes until `stop` is set.

    Args:
        process (psutil.Process): Parent process
        stop (threading.Event): Stop event to listen for
        peak (list[int]): Memory peak list.
    """
    # Runs in background until `stop`
    while not stop.is_set():
        total = 0
        for proc in [process, *process.children(recursive=True)]:
            try:
                total += proc.memory_full_info().pss
            except psutil.Error:
                pass
        peak[0] = max(peak[0], total)
        time.sleep(0.2)


def _pressure_totals() -> dict[str, int] | None:
    """Returns cumulative kernel pressure stall time in microseconds (Linux PSI), or None if unavailable

    Returns:
        dict[str, int] | None: A dict containing kernel pressure stall times for cpu, io and memory, or None
        if unavailable.
    """
    totals = {}
    try:
        for kind in ("cpu", "io", "memory"):
            for line in Path(f"/proc/pressure/{kind}").read_text().splitlines():
                part, *fields = line.split()
                totals[f"{kind}_{part}"] = int(dict(f.split("=") for f in fields)["total"])
    except (OSError, ValueError, KeyError):
        return None
    return totals


def _drop_page_cache() -> bool:
    """Drops the OS page cache (needs passwordless sudo). Returns whether it succeeded."""
    result = subprocess.run(
        ["sudo", "-n", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
        capture_output=True,
    )
    return result.returncode == 0


def _environment() -> dict:
    """Returns a summary of the current environment

    Returns:
        dict: Summary dict containing environment details such as:
        python version, cpu, memory (in Gb), slideflow and slideforge versions
    """
    from importlib.metadata import PackageNotFoundError, version

    import slideflow

    try:
        slideforge_version = version("slideforge")
    except PackageNotFoundError:
        slideforge_version = None

    return {
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "mem_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "slideflow": slideflow.__version__,
        "slideflow_backend": os.environ.get("SF_SLIDE_BACKEND", "cucim"),
        "slideforge": slideforge_version,
    }


@click.group(invoke_without_command=True)
@click.option(
    "--slides", type=click.Path(exists=True, file_okay=False, path_type=Path), help="Directory containing slides"
)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), help="Output directory")
@click.option("--backend", "backends", multiple=True, default=["slideflow", "slideforge"], show_default=True)
@click.option(
    "--resolution",
    "resolutions",
    multiple=True,
    default=["Low"],
    show_default=True,
    help="AutoMIL resolution preset name(s)",
)
@click.option(
    "--threads",
    "threads_list",
    multiple=True,
    type=int,
    default=[8],
    show_default=True,
    help="Cores per run: rayon threads (slideforge) / worker processes (slideflow)",
)
@click.option(
    "--tissue/--no-tissue",
    "tissue_modes",
    default=None,
    help="Restrict to runs with/without tissue detection (default: both)",
)
@click.option("--stain/--no-stain", default=False, show_default=True, help="Enable Reinhard stain normalization")
@click.option("--min-tissue-fraction", type=float, default=0.4, show_default=True)
@click.option("--repeats", type=int, default=3, show_default=True)
@click.option("--warmup/--no-warmup", default=True, show_default=True, help="Discard one untimed run per config")
@click.option("--drop-caches", is_flag=True, help="Drop the page cache before each run (needs passwordless sudo)")
@click.option(
    "--sync/--no-sync",
    default=True,
    show_default=True,
    help="Flush pending writes (sync) before each run, so a run doesn't pay for the previous one's output",
)
@click.option(
    "--keep-tfrecords",
    type=click.Choice(["none", "last", "all"]),
    default="none",
    show_default=True,
    help="Which runs keep their extracted tfrecords (last: the last timed repeat of each config)",
)
@click.option(
    "--max-io-stall",
    type=float,
    default=5.0,
    show_default=True,
    help="Warn when all tasks were stalled on I/O for more than this percent of a run",
)
@click.pass_context
def main(
    ctx: click.Context,
    slides: Path | None,
    out: Path | None,
    backends: tuple[str, ...],
    resolutions: tuple[str, ...],
    threads_list: tuple[int, ...],
    tissue_modes: bool | None,
    stain: bool,
    min_tissue_fraction: float,
    repeats: int,
    warmup: bool,
    drop_caches: bool,
    sync: bool,
    keep_tfrecords: str,
    max_io_stall: float,
) -> None:
    """Runs the full benchmark matrix"""
    if ctx.invoked_subcommand is not None:
        return
    if slides is None or out is None:
        raise click.UsageError("--slides and --out are required")

    out.mkdir(parents=True, exist_ok=True)
    results_file = out / "results.jsonl"
    environment = _environment()
    tissue_values = [True, False] if tissue_modes is None else [tissue_modes]

    configs = [
        {
            "backend": backend,
            "resolution": resolution,
            "threads": threads,
            "tissue": tissue,
            "stain": stain,
            "min_tissue_fraction": min_tissue_fraction,
        }
        for backend, resolution, threads, tissue in itertools.product(
            backends, resolutions, threads_list, tissue_values
        )
    ]

    for config in configs:
        run_ids = (["warmup"] if warmup else []) + list(range(repeats))
        for repeat in run_ids:
            name = "_".join(f"{k}-{v}" for k, v in config.items() if k != "min_tissue_fraction") + f"_r{repeat}"
            run_dir = out / "runs" / name
            if run_dir.exists():
                subprocess.run(["rm", "-rf", str(run_dir)], check=True)

            cache_dropped = _drop_page_cache() if drop_caches else False
            if sync:
                os.sync()
            click.echo(f"[{time.strftime('%H:%M:%S')}] {name}")

            proc_start = time.perf_counter()

            proc = psutil.Popen(
                [
                    sys.executable,
                    __file__,
                    "run",
                    "--slides",
                    str(slides),
                    "--run-dir",
                    str(run_dir),
                    "--config",
                    json.dumps(config),
                ]
            )
            stop, peak = threading.Event(), [0]
            sampler = threading.Thread(target=_sample_peak_memory, args=(proc, stop, peak), daemon=True)
            sampler.start()
            returncode = proc.wait()
            proc_s = time.perf_counter() - proc_start
            stop.set()
            sampler.join()

            keep = keep_tfrecords == "all" or (keep_tfrecords == "last" and repeat == repeats - 1)
            if not keep:
                subprocess.run(["rm", "-rf", str(run_dir / "project")], check=True)
            if repeat == "warmup":
                continue

            run_result_file = run_dir / "run.json"
            run_result = json.loads(run_result_file.read_text()) if run_result_file.exists() else {}
            record = {
                **config,
                "repeat": repeat,
                "returncode": returncode,
                "peak_pss_mb": round(peak[0] / 1e6, 1),
                "cache_dropped": cache_dropped,
                "synced": sync,
                "proc_s": round(proc_s, 3),
                **run_result,
                "tfrecords_kept": keep,
                "env": environment,
            }
            with results_file.open("a") as f:
                f.write(json.dumps(record) + "\n")

            # slideflow logs per-slide errors without failing, so check what was actually extracted
            n_empty = sum(n == 0 for n in run_result.get("tiles_per_slide", {}).values())
            if returncode != 0 or run_result.get("n_slides_extracted") != run_result.get("n_slides") or n_empty:
                click.secho(
                    f"  WARNING: returncode={returncode}, extracted {run_result.get('n_slides_extracted')}"
                    f"/{run_result.get('n_slides')} slides, {n_empty} without tiles",
                    fg="yellow",
                )
            io_full_pct = run_result.get("io_full_pct")
            if io_full_pct is not None and io_full_pct > max_io_stall:
                click.secho(
                    f"  WARNING: all tasks were stalled on I/O for {io_full_pct}% of the run, timings are unreliable",
                    fg="yellow",
                )

    click.echo(f"Results written to {results_file}")


@main.command()
@click.option("--slides", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--run-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--config", "config_json", required=True)
def run(slides: Path, run_dir: Path, config_json: str) -> None:
    """Runs a single configuration (invoked as a subprocess by the matrix)"""
    import pandas as pd
    import slideflow as sf
    from slideflow.io import get_tfrecord_length
    from slideflow.util import is_slide, path_to_name

    from automil.dataset import Dataset
    from automil.tiling import get_tile_backend
    from automil.util import RESOLUTION_PRESETS, TileBackendType

    config = json.loads(config_json)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal project: one annotation row per slide, no labels needed
    slide_names = sorted({path_to_name(str(p)) for p in slides.iterdir() if is_slide(str(p))})
    annotations_file = run_dir / "annotations.csv"
    pd.DataFrame({"patient": slide_names, "slide": slide_names}).to_csv(annotations_file, index=False)
    project = sf.create_project(
        name="AutoMIL",
        root=str(run_dir / "project"),
        slides=str(slides),
        annotations=str(annotations_file),
    )

    # Reuse AutoMIL's tile size / MPP logic so both backends get the same geometry
    automil_dataset = Dataset(
        project,
        RESOLUTION_PRESETS[config["resolution"]],
        label_map=[],
        slide_dir=slides,
        verbose=False,
    )
    dataset = project.dataset(tile_px=automil_dataset.tile_px, tile_um=automil_dataset.tile_um)

    match config["backend"]:
        case "slideflow":
            # Without tissue detection, also disable slideflow's default grayspace filter (0.6) so both backends
            # keep every grid tile and the run measures pure read/resize/encode throughput
            filters = {} if config["tissue"] else {"grayspace_fraction": 1}
            backend = get_tile_backend(
                TileBackendType.Slideflow, report=False, num_threads=config["threads"], **filters
            )
        case "slideforge":
            backend = get_tile_backend(
                TileBackendType.Slideforge,
                threads=config["threads"],
                min_tissue_fraction=config["min_tissue_fraction"],
            )
        case other:
            raise click.BadParameter(f"Unknown backend '{other}'")

    pressure_start = _pressure_totals()
    start_wall, start_cpu = time.perf_counter(), _cpu_seconds()
    backend.extract(
        dataset,
        tissue_detection="otsu" if config["tissue"] else None,
        stain_normalizer="reinhard" if config["stain"] else None,
    )
    wall_s, cpu_s = time.perf_counter() - start_wall, _cpu_seconds() - start_cpu
    pressure_end = _pressure_totals()

    # System-wide stall time during the timed extraction, in seconds (Linux PSI)
    pressure = (
        {f"{k}_stall_s": round((pressure_end[k] - pressure_start[k]) / 1e6, 3) for k in pressure_start}
        if pressure_start and pressure_end
        else {}
    )

    tfrecords_dir = Path(dataset.tfrecords_folders()[0])
    tfrecords = sorted(tfrecords_dir.glob("*.tfrecords"))
    tiles_per_slide = {p.stem: get_tfrecord_length(str(p)) for p in tfrecords}
    n_tiles = sum(tiles_per_slide.values())

    result = {
        "tile_px": automil_dataset.tile_px,
        "tile_um": automil_dataset.tile_um,
        "mpp": automil_dataset.mpp,
        "n_slides": len(slide_names),
        "n_slides_extracted": len(tfrecords),
        "n_tiles": n_tiles,
        "wall_s": round(wall_s, 3),
        "cpu_s": round(cpu_s, 3),
        "tiles_per_s": round(n_tiles / wall_s, 1) if wall_s > 0 else None,
        "io_full_pct": round(100 * pressure["io_full_stall_s"] / wall_s, 1) if pressure and wall_s > 0 else None,
        **pressure,
        "disk_mb": round(sum(p.stat().st_size for p in tfrecords) / 1e6, 1),
        "tfrecords_dir": str(tfrecords_dir),
        "tiles_per_slide": tiles_per_slide,
    }
    (run_dir / "run.json").write_text(json.dumps(result, indent=2))


def _cpu_seconds() -> float:
    """User + system CPU time of this process and its (finished) children"""
    import resource

    usage = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(u.ru_utime + u.ru_stime for u in usage)


if __name__ == "__main__":
    main()
