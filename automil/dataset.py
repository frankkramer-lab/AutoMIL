#==============================================================================#
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
#==============================================================================#
"""
Dataset management utilities for AutoMIL.

This module provides the :class:`automil.dataset.Dataset` class, which is responsible for preparing a
slideflow-compatible dataset source that can be passed to downstream pipeline stages.
It supports both raw and pre-tiled whole-slide image datasets, and handles
tiling, the conversion from .png to .tiff, label filtering, and feature extraction.

The resulting datasets are stored as TFRecords and feature bags within respective folders in the
project directory.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import pandas as pd
import slideflow as sf
import torch
from slideflow.slide import qc

from .util import (COMMON_MPP_VALUES, FEATURE_EXTRACTOR, INFO_CLR,
                   RESOLUTION_PRESETS, SUCCESS_CLR, LogLevel, get_vlog)
from .util.logging import render_kv_table
from .util.pretiled import pretiled_to_tfrecords
from .util.slide import calculate_average_mpp, get_mpp_from_slide
from .util.tiff_conversion import batch_conversion_concurrent, batch_generator


# === Helpers === #
def get_unique_labels(annotations_file: Path, label_column: str) -> list[str]:
    """Extract list of unique labels from the specified column in the annotations file.

    Args:
        annotations_file: Path to the annotations CSV file
        label_column: Name of the column containing labels

    Returns:
        List of unique labels found in the specified column.
    """
    annotations = pd.read_csv(annotations_file)
    return [str(label) for label in annotations[label_column].dropna().unique()]

class Dataset():
    """Prepares and manages dataset sources for downstream pipeline stages.

    This class handles and executes all dataset-related preprocessing steps, including:

    - Computing microns-per-pixel (MPP) or selecting a sensible default
    - Filtering slides by label
    - Extracting tiles or
    - Handling pretiled datasets
    - Generating feature bags for model training

    The dataset lifecycle is tightly coupled to a slideflow ``Project`` instance
    and produces TFRecords and feature bags within the project directory.
    """
    def __init__(
        self,
        project: sf.Project,
        resolution: RESOLUTION_PRESETS,
        label_map: dict | list[str],
        slide_dir: Path | None = None,
        bags_dir:  Path | None = None,
        is_pretiled:  bool = False,
        tiff_conversion: bool = False,
        verbose: bool = True
        ) -> None:
        """Initialize a Dataset manager.

        Args:
            project (sf.Project): Slideflow project associated with this dataset.
            resolution (RESOLUTION_PRESETS): Resolution preset defining tile size and magnification.
            label_map (dict | list[str]): Label mapping used to filter slides.
            slide_dir (Path | None, optional): Directory containing raw or pretiled slides.
            bags_dir (Path | None, optional): Output directory for generated feature bags.
            is_pretiled (bool, optional): Whether the input slides are already tiled.
            tiff_conversion (bool, optional): Whether slides should be converted to TIFF before tiling.
            verbose (bool, optional): Enable verbose logging.
        """
        self.project = project
        self.resolution = resolution

        self.slide_dir = slide_dir
        self.bags_dir  = bags_dir

        self.label_map = label_map
        self.is_pretiled = is_pretiled
        self.tiff_conversion = tiff_conversion

        self.vlog = get_vlog(verbose)

    @cached_property
    def tile_px(self) -> int:
        """Tile size in pixels derived from the resolution preset.

        Returns:
            int: Tile size in pixels.
        """
        return self.resolution.tile_px
    
    @cached_property
    def magnification(self) -> str:
        """Nominal magnification derived from the resolution preset.

        Returns:
            str: Magnification string (e.g., ``"10x"``).
        """

        return self.resolution.magnification
    
    @cached_property
    def mpp(self) -> float:
        """Computed microns-per-pixel (MPP) value.

        The MPP is computed by retrieving MPP values from the slides metadata ifavailable, otherwise
        sensible defaults are used based on magnification.

        Returns:
            float: Microns per pixel.
        """
        return self._compute_mpp(by_average=True)

    @cached_property
    def tile_um(self) -> int:
        """Tile size in micrometers.

        Returns:
            int: Tile size in micrometers.
        """

        return int(self.tile_px * self.mpp)

    @cached_property
    def tfrecords_dir(self) -> Path:
        """Path to directory where tfrecords will be stored"""
        if self.is_pretiled:
            return Path(self.project.root) / "tfrecords" / "pretiled"
        elif self.tiff_conversion:
            return Path(self.project.root) / "tfrecords" / "tiff_buffer"
        else:
            return Path(self.project.root) / "tfrecords"


    def prepare_dataset_source(self) -> sf.Dataset:
        """Prepares a single dataset source for a given resolution preset, or for a set of pretiled slides.

        Performs the following steps:
            1. Compute appropriate MPP for slides
            2. filter slides by label map
            3. Extract tiles (or convert pretiled to tfrecords)
            4. Extract features

        Raises:
            ValueError: If `pretiled` is True but no `slide_dir` is provided

        Returns:
            sf.Dataset: A slideflow dataset
        """
        if self.is_pretiled:
            self.vlog(f"Preparing dataset source from pretiled slides at [{INFO_CLR}]{self.slide_dir}[/]")
        else:
            self.vlog(f"Preparing dataset source at resolution [{INFO_CLR}]{self.resolution.name} "
                f"({self.tile_px}px, {self.tile_um:.2f}um)[/]")

        # Convert pretiled to tfrecords
        if self.is_pretiled:
            if self.slide_dir is None:
                raise ValueError("slide_dir must be provided when pretiled=True")
            dataset = self._convert_pretiled()
            dataset = self._apply_label_filter(dataset)
        else:
            dataset = self.project.dataset(
                sources="AutoMIL",
                tile_px=self.tile_px,
                tile_um=self.tile_um,
            )
            dataset = self._apply_label_filter(dataset)
            self._extract_tiles(dataset)

        self._extract_features(dataset)
        return dataset
    
    def summary(self) -> None:
        rows = [
            ("Resolution Preset", self.resolution.name),
            ("Tile Size (px)", f"{self.tile_px}px"),
            ("Magnification", self.magnification),
            ("Microns-Per-Pixel", f"{self.mpp:.3f}"),
            ("Tile Size (µm)", f"{self.tile_um:.2f}µm"),
            ("Pretiled Input", self.is_pretiled),
            ("TIFF Conversion", self.tiff_conversion),
        ]

        self.vlog("[bold underline]Dataset Summary[/]")
        self.vlog(render_kv_table(rows))

    # === Internals === #
    def _compute_mpp(self, by_average: bool = True) -> float:
        """Computes an appropriate Microns Per Pixel (MPP) for the given slide images.
        If `by_average` is True, the average MPP across all slides is computed.
        Otherwise, the first slide's MPP is used.
        If `slide_dir` is None, MPP is computed based on sensible defaults based on the slide magnification.

        Args:
            by_average (bool, optional): Compute MPP by calculating the average across slides. Defaults to False.

        Returns:
            float: Appropriate MPP value for the given slides 
        """
        global COMMON_MPP_VALUES
        mpp = None

        # Try to compute MPP from slides
        if self.slide_dir is not None and self.slide_dir.exists():
            # Average MPP across slides
            if by_average:
                mpp = calculate_average_mpp(self.slide_dir)
                if mpp is not None:
                    self.vlog(f"Computed average MPP across slides: [{INFO_CLR}]{mpp:.3f}[/]")
            # MPP from first slide
            else:
                first_slide = next(self.slide_dir.glob("*"))
                mpp = get_mpp_from_slide(first_slide)

        # Fallback: Default from common mpp values
        if mpp is None:
            mpp = COMMON_MPP_VALUES.get(self.magnification, 0.5)
            self.vlog(f"Using default MPP for magnification [{INFO_CLR}]{self.magnification}: {mpp:.3f}[/]")

        return mpp
    
    def _apply_label_filter(self, dataset: sf.Dataset) -> sf.Dataset:
        """Apply `label_map` filter to the given dataset

        Args:
            dataset (sf.Dataset): dataset

        Returns:
            sf.Dataset: filtered dataset
        """
        # Extract list of unique labels
        match self.label_map:
            case dict():
                unique_labels = list(self.label_map.values())
            case list():
                unique_labels = self.label_map
            case _:
                unique_labels = []

        if not unique_labels:
            return dataset

        # Retrieve annotation dtypes and cast if necessary
        annotations = dataset.annotations if dataset.annotations is not None else pd.DataFrame()
        if not annotations.empty:
            ann_type = type(annotations["label"].iat[0])
            unique_type = type(unique_labels[0])

            if ann_type != unique_type:
                unique_labels = [ann_type(lbl) for lbl in unique_labels]
        
        self.vlog(f"Filtering for unique labels {unique_labels}")

        return self.project.dataset(
            dataset.tile_px,
            dataset.tile_um,
            filters={"label": unique_labels},
        )
        
    def _convert_pretiled(self) -> sf.Dataset:
        """Converts a pretiled dataset source to tfrecords. Tiling is skipped.

        Raises:
            RuntimeError: If no project annotations file is found
            RuntimeError: If the dataset manifest is empty after conversion

        Returns:
            sf.Dataset: A dataset source with tfrecords
        """
        if not self.project.annotations:
            raise RuntimeError("A project annotations file is required for pretiled datasets.")
        
        elif self.slide_dir is None:
            raise ValueError("slide_dir must be provided when pretiled=True")

        # Prepare TFRecords directory
        tfrecords_dir = self.tfrecords_dir
        tfrecords_dir.mkdir(parents=True, exist_ok=True)

        # Add source if not already present
        if "pretiled" not in self.project.sources:
            self.project.add_source(
                "pretiled",
                tfrecords=str(tfrecords_dir),
                slides=str(self.slide_dir)
            )
        # Change source so slideflow knows where to look for tfrecords
        dataset = self.project.dataset(
            sources=["pretiled"],
            tile_px=self.tile_px,
            tile_um=self.tile_um,
        )

         # Convert pretiled slides to tfrecords
        self.vlog(f"Converting pretiled slides to tfrecords at [{INFO_CLR}]{tfrecords_dir}[/] ...")

        pretiled_to_tfrecords(self.slide_dir, Path(dataset.tfrecords_folders()[0]))

        dataset.rebuild_index()
        dataset.update_manifest(force_update=True)
        
        if len(dataset.manifest()) == 0:
            raise RuntimeError("Pretiled dataset conversion produced an empty manifest.")

        self.vlog(f"Pretiled dataset loaded with [{INFO_CLR}]{len(dataset.manifest())}[/] slides.")

        return dataset
    
    def _extract_tiles(self, dataset: sf.Dataset) -> None:
        """Extracts tiles from a given dataset source. Optionally performs prior tiff conversion.

        Note:
            The tiff conversion process is performed in batches to avoid excessive disk space usage.
            It is recommended to use tiff conversion ONLY when working with slides in formats that are not well-suited for tiling (e.g., .png).

        Args:
            dataset (sf.Dataset): Dataset source for which to extract tiles

        Raises:
            RuntimeError: If the batchwise tiff conversion process fails or encounters a timeout.
        """
        # Default Case: Normal tile extraction
        if not self.tiff_conversion:
            self.vlog(f"Extracting tiles at [{INFO_CLR}]{self.magnification} | tile={self.tile_px}[/]")
            dataset.extract_tiles(
                qc=qc.Otsu(),
                normalizer="reinhard_mask",
                report=True,
            )
            return

        # Optional: batchwise .tiff conversion
        else:
            self.vlog(f"Preparing TIFF conversion pipeline [{INFO_CLR}]({self.tile_px}px @ {self.magnification})[/]")

            # Permanent tiff buffer directory
            tiff_dir = Path(self.project.root) / "tiffs"
            tiff_dir.mkdir(parents=True, exist_ok=True)

            # Need to register a dataset source for the tiff buffer
            if "tiff_buffer" not in self.project.sources:
                self.project.add_source(
                    "tiff_buffer",
                    slides=str(tiff_dir),
                )
            dataset = self.project.dataset(
                sources=["tiff_buffer"],
                tile_px=dataset.tile_px,
                tile_um=dataset.tile_um,
            )

            # Prepare TFRecords directory
            tfrecords_dir = Path(dataset.tfrecords_folders()[0])
            tfrecords_dir.mkdir(parents=True, exist_ok=True)

            # Retieve slide paths and IDs
            slide_list: list[Path] = [path for p in dataset.slide_paths() if (path := Path(p)).exists()]
            slide_ids:  list[str]  = list(set(slide.stem for slide in slide_list)) # Using a set to avoid duplicates

            # Caution: Make sure the tfrecords dont actually exist yet (e.g., from previous runs)
            expected_tfrecords = {sid: tfrecords_dir / f"{sid}.tfrecords" for sid in slide_ids}
            existing = {sid: path for sid, path in expected_tfrecords.items() if path.exists()}
            missing = [sid for sid in slide_ids if sid not in existing.keys()]

            if not missing:
                self.vlog(
                    f"All expected tfrecords already exist in {tfrecords_dir}. Skipping TIFF conversion.",
                    LogLevel.WARNING
                )
                return

            self.vlog(
                f"Found {len(existing)} existing TFRecords — creating {len(missing)} missing ones."
            )

            # Only convert still missing slides
            missing_slides = [slide for slide in slide_list if slide.stem in missing]

            # Size of the tiff buffer batches
            # TODO | Should probably be configurable
            buffer_size = 10

            # Process missing/outdated slides in batches
            for batch_idx, slide_batch in enumerate(batch_generator(missing_slides, buffer_size)):
                self.vlog(f"Converting TIFF batch [{INFO_CLR}]{batch_idx+1}[/] / [{INFO_CLR}]{len(missing_slides)//buffer_size+1}[/]")
                # Batchwise conversion to tiff
                batch_conversion_concurrent(slide_batch, tiff_dir)

                # Extract tiles
                try:
                    dataset.extract_tiles(
                        qc=qc.Otsu(),
                        normalizer="reinhard_mask",
                        mpp_override=self.mpp
                    )
                except Exception as e:
                    raise RuntimeError(f"Error extracting tiles for TIFF batch {batch_idx}: {e}")

            self.vlog(f"[{SUCCESS_CLR}]Finished TIFF conversion[/]")

    def _extract_features(self, dataset: sf.Dataset) -> None:
        """Extracts features from a given (tiled) dataset source and stores them in `bags_dir`

        Args:
            dataset (sf.Dataset): Dataset source for which to generate features
        """
        global FEATURE_EXTRACTOR

        # Prepare bags directory
        bag_dir = Path(self.project.root) / "bags" if self.bags_dir is None else self.bags_dir
        bag_dir.mkdir(exist_ok=True)

        # Build feature extractor model
        extractor = sf.build_feature_extractor(
            name=FEATURE_EXTRACTOR,
            resize=224,
        )
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.vlog(f"Extracting features using [{INFO_CLR}]{num_gpus}[/] GPUs …")

        # Generate feature bags
        dataset.generate_feature_bags(
            model=extractor,
            outdir=str(bag_dir),
            slide_batch_size=32,
            num_gpus=num_gpus,
        )

        self.vlog(f"[{SUCCESS_CLR}]Finished feature extraction.[/]")