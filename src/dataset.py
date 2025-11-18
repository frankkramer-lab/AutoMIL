# dataset.py
"""
Dataset preparation and management for AutoMIL.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import slideflow as sf
import torch
from slideflow.slide import qc

from utils import (COMMON_MPP_VALUES, FEATURE_EXTRACTOR, INFO_CLR,
                   RESOLUTION_PRESETS, SUCCESS_CLR, LogLevel,
                   batch_conversion_concurrent, batch_generator,
                   calculate_average_mpp, get_mpp_from_slide, get_vlog,
                   pretiled_to_tfrecords)


class Dataset():
    """Prepare and manage dataset sources for AutoMIL
    """
    def __init__(
        self,
        project: sf.Project,
        verbose: bool = True
        ) -> None:
        """Prepare and manage dataset sources for AutoMIL

        Args:
            project (sf.Project): Project for which to manage dataset sources
            verbose (bool, optional): Whether to print verbose messages. Defaults to True.
        """
        self.project = project
        self.verbose = verbose
        self.vlog = get_vlog(verbose)

    def prepare_dataset_source(
        self,
        resolution: RESOLUTION_PRESETS,
        label_map: dict | list[str],
        slide_dir: Path | None = None,
        bags_dir:  Path | None = None,
        pretiled:  bool = False,
        tiff_conversion: bool = False,
        ) -> sf.Dataset:
        """Prepares a single dataset source for a given resolution preset.

        Performs the following steps:
            1. Compute appropriate MPP for slides
            2. filter slides by label map
            3. Extract tiles (or convert pretiled to tfrecords)
            4. Extract features

        Args:
            resolution (RESOLUTION_PRESETS): Resolution for which to create dataset source
            label_map (dict | list[str]): label mapping (usually mapping from original categories to floats
            slide_dir (Path | None, optional): Optional directory in which to find slides. Defaults to None.
            pretiled (bool, optional): Whether the input images are already pretiled. If True, tiling is skipped and the input images are directly written to tfrecords. Defaults to False.
            tiff_conversion (bool, optional): Whether to perform tiff conversion. Recommended if input images are in a format unsuited for tiling (e.g .png). Defaults to False.

        Raises:
            ValueError: If `pretiled` is True but no `slide_dir` is provided

        Returns:
            sf.Dataset: A slideflow dataset
        """

        tile_size, magnification = resolution.value

        # Compute appropriate MPP for slides
        mpp = self._compute_mpp(magnification, slide_dir, by_average=True)

        # Compute tile_um
        tile_um = int(tile_size * mpp)

        # Prepare dataset source
        self.vlog(f"Preparing dataset source at resolution [{INFO_CLR}]{resolution.name} ({tile_size}px, {tile_um:.2f}um)[/]")
        dataset = self.project.dataset(tile_px=tile_size, tile_um=tile_um)
        # Filter dataset by with label map
        dataset = self._apply_label_filter(dataset, label_map)

        # Convert pretiled to tfrecords
        if pretiled:
            if slide_dir is None:
                raise ValueError("slide_dir must be provided when pretiled=True")
            dataset = self._convert_pretiled(dataset, slide_dir)
        else:
            self._extract_tiles(dataset, magnification, tile_size, tiff_conversion, mpp)

        self._extract_features(dataset, bags_dir)
        return dataset
    
    # === Internals ===
    def _compute_mpp(
        self,
        magnification: str,
        slide_dir:  Path | None = None,
        by_average: bool = False
        ) -> float:
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
        if slide_dir is not None and slide_dir.exists():
            # Average MPP across slides
            if by_average:
                mpp = calculate_average_mpp(slide_dir)
                if mpp is not None:
                    self.vlog(f"[{INFO_CLR}]Computed average MPP across slides: {mpp:.3f}[/]")
            # MPP from first slide
            else:
                first_slide = next(slide_dir.glob("*"))
                mpp = get_mpp_from_slide(first_slide)

        # Fallback: Default from common mpp values
        if mpp is None:
            mpp = COMMON_MPP_VALUES.get(magnification, 0.5)
            self.vlog(f"[{INFO_CLR}]Using default MPP for magnification {magnification}: {mpp:.3f}[/]")

        return mpp
    
    def _apply_label_filter(
        self,
        dataset: sf.Dataset,
        label_map: dict | list[str],
    ) -> sf.Dataset:

        # Extract list of unique labels
        match label_map:
            case dict():
                unique_labels = list(label_map.values())
            case list():
                unique_labels = label_map
            case _:
                unique_labels = []

        if not unique_labels:
            return dataset

        # Retrieve annotation dtypes and cast if necessary
        annotations = dataset.annotations if dataset.annotations is not None else pd.DataFrame()
        if not annotations.empty:
            ann_type = type(annotations["label"].iloc[0])
            unique_type = type(unique_labels[0])

            if ann_type != unique_type:
                self.vlog(
                    f"Label dtype mismatch ({ann_type} vs {unique_type}). Casting labels.",
                    level=LogLevel.WARNING,
                )
                unique_labels = [ann_type(lbl) for lbl in unique_labels]

        return self.project.dataset(
            dataset.tile_px,
            dataset.tile_um,
            filters={"label": unique_labels},
        )
    

    def _convert_pretiled(
        self,
        dataset: sf.Dataset,
        slide_dir: Path
    ) -> sf.Dataset:
        """Converts a pretiled dataset source to tfrecords. Tiling is skipped.

        Args:
            dataset (sf.Dataset): pretiled dataset source
            slide_dir (Path | None, optional): Slide directory. Defaults to None.

        Raises:
            RuntimeError: If no project annotations file is found
            RuntimeError: If the dataset manifest is empty after conversion

        Returns:
            sf.Dataset: A dataset source with tfrecords
        """
        ann_file = self.project.annotations
        if ann_file is None:
            raise RuntimeError("A project annotations file is required for pretiled datasets.")

        # Prepare TFRecords directory
        tf_dir = Path(dataset.tfrecords_folders()[0])
        tf_dir.mkdir(parents=True, exist_ok=True)

        self.vlog(f"Converting pretiled slides to tfrecords at {tf_dir}")

        pretiled_to_tfrecords(slide_dir, tf_dir)

        dataset.rebuild_index()
        dataset.update_manifest(force_update=True)

        if len(dataset.manifest()) == 0:
            raise RuntimeError("Pretiled dataset conversion produced an empty manifest.")

        self.vlog(f"{SUCCESS_CLR} Pretiled dataset loaded with {len(dataset.manifest())} slides.")

        return dataset
    
    def _extract_tiles(
        self,
        dataset: sf.Dataset,
        magnification: str,
        tile_size: int,
        tiff_conversion: bool,
        mpp_override: float,
    ) -> None:
        """Extracts tiles from a given dataset source. Optionally performs tiff prior tiff conversion.

        Note:
            The tiff conversion process is performed in batches to avoid excessive disk space usage.
            It is recommended to use tiff conversion ONLY when working with slides in formats that are not well-suited for tiling (e.g., .png).

        Args:
            dataset (sf.Dataset): Dataset source for which to extract tiles
            magnification (str): magnification at which to extract tiles
            tile_size (int): tile_size (in pixels e.g 250) at which to extract tiles
            tiff_conversion (bool): Whether to perform tiff conversion prior to tiling. Recommenden with .png slides
            mpp_override (float): Optional value with which to override the Microns-Per-Pixel (MPP) value with which to extract tiles.

        Raises:
            RuntimeError: If the batchwise tiff conversion process fails or encounters a timeout.
        """
        # Default Case: Normal tile extraction
        if not tiff_conversion:
            self.vlog(f"Extracting tiles at {magnification} | tile={tile_size}")
            dataset.extract_tiles(
                qc=qc.Otsu(),
                normalizer="reinhard_mask",
                report=True,
            )
            return

        # Optional: batchwise .tiff conversion
        else:
            self.vlog(f"Preparing TIFF conversion pipeline ({tile_size}px @ {magnification})")

            # Permanent tiff buffer directory
            tiff_dir = Path(self.project.root) / "tiffs"
            tiff_dir.mkdir(parents=True, exist_ok=True)

            # Need to register a dataset source for the tiff buffer
            if "tiff_buffer" not in self.project.sources:
                self.project.add_source("tiff_buffer", slides=str(tiff_dir))
            tiff_dataset = self.project.dataset(
                sources=["tiff_buffer"],
                tile_px=dataset.tile_px,
                tile_um=dataset.tile_um,
            )

            # Prepare TFRecords directory
            tfrecords_dir = Path(tiff_dataset.tfrecords_folders()[0])
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
                    level=LogLevel.WARNING
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
                self.vlog(f"Converting TIFF batch {batch_idx+1} / {len(missing_slides)//buffer_size+1}")
                # Batchwise conversion to tiff
                batch_conversion_concurrent(slide_batch, tiff_dir)

                # Extract tiles
                try:
                    tiff_dataset.extract_tiles(
                        qc=qc.Otsu(),
                        normalizer="reinhard_mask",
                        mpp_override=mpp_override,
                    )
                except Exception as e:
                    raise RuntimeError(f"Error extracting tiles for TIFF batch {batch_idx}: {e}")

            self.vlog(f"[{SUCCESS_CLR}]Finished TIFF conversion (cached)[/]")

    def _extract_features(
        self,
        dataset: sf.Dataset,
        bags_dir: Path | None = None,
    ) -> None:
        """Extracts features from a given (tiled) dataset source and stores them in `bags_dir`

        Args:
            dataset (sf.Dataset): Dataset source for which to generate features
            bags_dir (Path | None, optional): Directory in which to store bags. If `None`, will be set to `project_dir / "bags"`. Defaults to None.
        """
        global FEATURE_EXTRACTOR

        # Prepare bags directory
        bag_dir = Path(self.project.root) / "bags" if bags_dir is None else bags_dir
        bag_dir.mkdir(exist_ok=True)

        # Build feature extractor model
        extractor = sf.build_feature_extractor(
            name=FEATURE_EXTRACTOR,
            resize=224,
        )
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.vlog(f"Extracting features using {num_gpus} GPUs …")

        # Generate feature bags
        dataset.generate_feature_bags(
            model=extractor,
            outdir=str(bag_dir),
            slide_batch_size=32,
            num_gpus=num_gpus,
        )

        self.vlog(f"{SUCCESS_CLR} Finished feature extraction.")