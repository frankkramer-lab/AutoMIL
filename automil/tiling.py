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
EXPERIMENTAL | Tile extraction backends.

Two interchangeable tiling backends:

- `slideflow`: The default, wraps around `slideflow.Dataset.extract_tiles`
- `slideforge`: Experimental, uses the Rust library `slideforge`
(binding available under `pip install automil[slideforge]`, requires a Rust toolchain to build)

Note:
    The two backends are *not* pixel-identical. slideforge thresholds tissue
    per tile (Otsu on each tile + `min_tissue_fraction`) instead of using a
    slide-level QC mask, uses its own Reinhard target and only reads SVS slides.
"""

from __future__ import annotations

import mmap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import slideflow as sf
from slideflow.util import path_to_name
from slideflow.util.tfrecord2idx import create_index, save_index

from .util import TileBackendType


# === Helpers === #
def _read_protobuf_varint(buffer: mmap.mmap, position: int) -> tuple[int, int]:
    """Decodes a protobuf base-128 varint starting at `position`.

    Args:
        buffer (mmap.mmap): Memory-mapped file buffer to read from
        position (int): Position the varint starts at

    Returns:
        tuple[int, int]: The decoded value and the position after the varint
    """
    value, shift = 0, 0
    while True:
        byte = buffer[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7


def _read_length_delimited_field(buffer: mmap.mmap, position: int) -> tuple[int, int, int]:
    """Reads the header (tag and length) of a length-delimited protobuf field starting at `position`.

    Args:
        buffer (mmap.mmap): Memory-mapped file buffer to read from
        position (int): Position the field starts at

    Raises:
        ValueError: If the field's wire type is not that of a length-delimited field

    Returns:
        tuple[int, int, int]: The field's tag and the start and end positions of its payload
    """
    tag, position = _read_protobuf_varint(buffer, position)
    if (wire_type := tag & 0x7) != 2:
        raise ValueError(f"unexpected wire type {wire_type} at position {position - 1}")
    length, position = _read_protobuf_varint(buffer, position)
    return tag, position, position + length


def _read_int64_feature(buffer: mmap.mmap, start: int, end: int) -> int | None:
    """First value of a serialized `tf.train.Feature` holding an `Int64List`.

    Args:
        buffer (mmap.mmap): Memory-mapped file buffer to read from
        start (int): Position the serialized `tf.train.Feature` starts at
        end (int): End of the enclosing map entry's value

    Raises:
        ValueError: If the `tf.train.Feature` exceeds its map entry or the `Int64List` has an unexpected
            field tag

    Returns:
        int | None: The first value, or None if the feature is not an `Int64List` or is empty
    """
    tag, position, list_end = _read_length_delimited_field(buffer, start)
    if list_end > end:
        raise ValueError("tf.train.Feature exceeds its map entry")
    # Not an Int64List
    if tag != 0x1A or position >= list_end:
        return None
    tag, position = _read_protobuf_varint(buffer, position)
    # Values are either packed (0x0A) or unpacked (0x08)
    if tag not in (0x0A, 0x08):
        raise ValueError(f"unexpected Int64List field tag {tag}")
    # Packed values are prefixed by their byte length
    if tag == 0x0A:
        _, position = _read_protobuf_varint(buffer, position)
    value, _ = _read_protobuf_varint(buffer, position)
    return value - (1 << 64) if value >= 1 << 63 else value


def _record_location(buffer: mmap.mmap, start: int, end: int) -> tuple[int, int] | None:
    """Reads `(loc_x, loc_y)` from a serialized `tf.train.Example` in `buffer[start:end]`.

    Walks the protobuf by field lengths, so the encoded image is skipped without being copied.

    Args:
        buffer (mmap.mmap): Memory-mapped file buffer to read from
        start (int): Position the serialized `tf.train.Example` starts at
        end (int): Position the serialized `tf.train.Example` ends at

    Raises:
        ValueError: If the record is not a `tf.train.Example`

    Returns:
        tuple[int, int] | None: The tile location, or None if the record has no location features
    """
    tag, position, features_end = _read_length_delimited_field(buffer, start)
    # Example.features
    if tag != 0x0A or features_end > end:
        raise ValueError("record is not a tf.train.Example")

    location_keys = (b"loc_x", b"loc_y")
    locations: dict[bytes, int | None] = {}
    while position < features_end:
        # Features.feature map entry
        _, position, entry_end = _read_length_delimited_field(buffer, position)
        key, value_range = None, None
        while position < entry_end:
            field, position, field_end = _read_length_delimited_field(buffer, position)
            if field == 0x0A:
                key = buffer[position:field_end]
            elif field == 0x12:
                value_range = (position, field_end)
            position = field_end
        if key in location_keys and value_range is not None:
            locations[key] = _read_int64_feature(buffer, *value_range)

    x, y = (locations.get(k) for k in location_keys)
    return None if x is None or y is None else (x, y)


def _scan_tfrecord(path: Path) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Reads the byte offset/length and tile location of every record in a tfrecord.

    Args:
        path (Path): Path to the tfrecord

    Raises:
        ValueError: If a record is truncated or not a `tf.train.Example`, or only some records have locations

    Returns:
        tuple[list[list[int]], list[tuple[int, int]]]: `[offset, length]` per record, and the tile locations
            (empty if the records have none)
    """
    offsets: list[list[int]] = []
    locations: list[tuple[int, int]] = []
    read_access = mmap.ACCESS_READ

    with path.open("rb") as file_stream:
        size = path.stat().st_size
        if size == 0:
            return offsets, locations
        with mmap.mmap(file_stream.fileno(), 0, access=read_access) as buffer:
            position = 0
            while position < size:
                # u64 length, u32 length crc, payload, u32 payload crc
                length = int.from_bytes(buffer[position : position + 8], "little")
                end = position + 12 + length + 4
                if end > size:
                    raise ValueError(f"truncated record at byte {position}")
                offsets.append([position, end - position])
                payload_start = position + 12
                if (location := _record_location(buffer, payload_start, payload_start + length)) is not None:
                    locations.append(location)
                position = end
    if locations and len(locations) != len(offsets):
        raise ValueError("only some records have tile locations")
    return offsets, locations


def write_index(tfrecord_path: Path) -> None:
    """Writes the slideflow index of a tfrecord, with record offsets and tile locations.

    Produces the same index as `slideflow.util.tfrecord2idx.create_index`, which decodes every record
    in Python and takes seconds per slide. Here records are walked by their length fields and only the
    location features are decoded. Falls back to `create_index` if a record can't be parsed.

    Args:
        tfrecord_path (Path): Path to .tfrecord file
    """
    index_file = tfrecord_path.with_suffix(".index")
    # slideflow prefers .index over .index.npz over .index.npy, so a stale index would shadow the new one
    for suffix in ("", ".npz", ".npy"):
        Path(f"{index_file}{suffix}").unlink(missing_ok=True)

    try:
        offsets, locations = _scan_tfrecord(tfrecord_path)
    # Fallback: slideflow's create_index
    except (ValueError, IndexError):
        create_index(str(tfrecord_path))
        return
    save_index(np.array(offsets), str(index_file), locations=np.array(locations))


# === Backends === #
class SlideflowBackend:
    """Current tile extraction via `sf.Dataset.extract_tiles`"""

    def __init__(self, report: bool = True, **extract_kwargs) -> None:
        """
        Initializes a `SlideflowBackend` instance.
        Args:
            report (bool, optional): Whether to generate PDF extraction report.
            **extract_kwargs: Passed through to `sf.Dataset.extract_tiles` (e.g. `num_threads`).
        """
        self.report = report
        self.extract_kwargs = extract_kwargs

    def extract(
        self,
        dataset: sf.Dataset,
        tissue_detection: str | None,
        stain_normalizer: str | None,
    ) -> None:
        """Extracts tiles from the given dataset using `sf.Dataset.extract_tiles`.

        Args:
            dataset (sf.Dataset): a sf.Dataset unstance for which to extract tiles
            tissue_detection (str | None): Tissue detection method to use
            stain_normalizer (str | None): Stain normalization method to use
        """
        dataset.extract_tiles(
            qc=tissue_detection,
            normalizer=stain_normalizer,
            report=self.report,
            **self.extract_kwargs,
        )


class SlideforgeBackend:
    """Experimental tile extraction via the `slideforge` binding.

    Tiles are extracted in grid mode (`tile_px`/`tile_um`, non-overlapping) to match
    slideflow's tiling convention. Since slideforge does not write `.index` files,
    these are created afterwards with `write_index`.
    """

    def __init__(
        self,
        min_tissue_fraction: float = 0.4,
        jpeg_quality: int = 95,
        threads: int | None = None,
        slide_workers: int = 1,
        overwrite: bool = False,
    ) -> None:
        """
        Args:
            min_tissue_fraction (float, optional): Minimum per-tile tissue fraction to keep a tile.
                Only applied if tissue detection is enabled. The default best reproduces slideflow's Otsu QC tile
                selection (calibrated on CMU-1/2/3 at the High and Low presets).
            jpeg_quality (int, optional): JPEG quality of the written tiles. Defaults to slideflow's quality.
            threads (int | None, optional): Threads per slide (rayon). `None` uses all cores.
            slide_workers (int, optional): Number of slides extracted concurrently.
            overwrite (bool, optional): Re-extract slides whose tfrecords already exist.
        """
        try:
            import slideforge
        except ImportError as e:
            raise ImportError(
                "The slideforge tiler requires the slideforge binding: pip install 'automil[slideforge]'"
            ) from e

        # The compiled binding ships no type stubs
        self._slideforge: Any = slideforge
        self.min_tissue_fraction = min_tissue_fraction
        self.jpeg_quality = jpeg_quality
        self.threads = threads
        self.slide_workers = slide_workers
        self.overwrite = overwrite

    def extract(
        self,
        dataset: sf.Dataset,
        tissue_detection: str | None,
        stain_normalizer: str | None,
    ) -> None:
        """Extract tiles from the given `sf.Dataset` using slideforge.

        Args:
            dataset (sf.Dataset): a sf.Dataset unstance for which to extract tiles.
            tissue_detection (str | None): Tissue detection method to use.
            stain_normalizer (str | None): Stain normalization method to use.

        Raises:
            ValueError: If the selected stain_normalizer is not `reinhard`
            ValueError: If a magnification string is passed instead of integer tile_px and tile_um
        """
        if stain_normalizer is not None and stain_normalizer != "reinhard":
            raise ValueError(f"slideforge only supports 'reinhard' stain normalization, got '{stain_normalizer}'")
        stain_normalize = False if stain_normalizer is None else True
        min_tissue_fraction = self.min_tissue_fraction if tissue_detection else None

        # slideflow also allows magnification strings (e.g. "20x") as tile_um, slideforge needs microns
        tile_px, tile_um = dataset.tile_px, dataset.tile_um
        if not isinstance(tile_px, int) or not isinstance(tile_um, int):
            raise ValueError(f"slideforge requires integer tile_px and tile_um, got {tile_px!r} and {tile_um!r}")

        tfrecords_dir = Path(dataset.tfrecords_folders()[0])
        tfrecords_dir.mkdir(parents=True, exist_ok=True)

        def _extract_slide(slide_path: str) -> None:
            tfrecord_path = tfrecords_dir / f"{path_to_name(slide_path)}.tfrecords"
            if tfrecord_path.exists() and not self.overwrite:
                return
            self._slideforge.extract(
                slide_path,
                tfrecord=str(tfrecord_path),
                target_mpp=tile_um / tile_px,
                tile_px=tile_px,
                tile_um=float(tile_um),
                min_tissue_fraction=min_tissue_fraction,
                stain_normalize=stain_normalize,
                jpeg_quality=self.jpeg_quality,
                threads=self.threads,
            )
            write_index(tfrecord_path)

        # TODO | This should unlock the GIL, but check anyway
        with ThreadPoolExecutor(max_workers=self.slide_workers) as pool:
            list(pool.map(_extract_slide, dataset.slide_paths()))

        # Force update of .index files
        dataset.update_manifest(force_update=True)


def get_tile_backend(tiler: TileBackendType, **kwargs) -> SlideflowBackend | SlideforgeBackend:
    """Returns an instance of the tile extraction backend selected by `tiler`.

    Args:
        tiler (TileBackendType): Backend to create, `TileBackendType.Slideflow` or `TileBackendType.Slideforge`
        **kwargs: Passed to the backend's constructor, e.g. `num_threads` for `SlideflowBackend` or
            `min_tissue_fraction` and `threads` for `SlideforgeBackend`

    Raises:
        ImportError: If `TileBackendType.Slideforge` is requested but the `slideforge` binding is not installed
        ValueError: If `tiler` is not a `TileBackendType` member

    Returns:
        SlideflowBackend | SlideforgeBackend: The requested tile extraction backend.

    Example:
        ```python
        backend = get_tile_backend(TileBackendType.Slideforge, threads=8)
        backend.extract(dataset, tissue_detection="otsu", stain_normalizer="reinhard")
        ```
    """
    match tiler:
        case TileBackendType.Slideflow:
            return SlideflowBackend(**kwargs)
        case TileBackendType.Slideforge:
            return SlideforgeBackend(**kwargs)
        case _:
            raise ValueError(f"Unsupported tile backend: {tiler}")
