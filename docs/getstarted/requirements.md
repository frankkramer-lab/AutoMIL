# Requirements

This section outlines the software and system requirements needed to run **AutoMIL**.

---

## Python Version

AutoMIL requires **Python ≥ 3.11**.
A 64-bit Python installation is strongly recommended, especially when working with large whole-slide images.

---

## Core Python Dependencies

The following Python packages are required and are installed automatically when installing AutoMIL via `pip install .`:

### Deep Learning and Numerical Computing
- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `scikit-learn`
- `fastai`

### Visualization and Interactive Analysis
- `matplotlib`
- `seaborn`
- `imgui`
- `ipython`

### Whole-Slide Image Processing and MIL Utilities
- `openslide-python`
- `openslide-bin`
- `slideflow`
- `nystrom-attention`
- `cucim`

### General Utilities and Command Line Interface
- `click`
- `pyrfr`
- `swig`

No manual installation of these dependencies is required when installing AutoMIL from source.

---

## Optional Dependencies

### `pyvips`/`libvips`

By default, Slideflow uses the image processing library :material-github: [cuCIM](https://github.com/rapidsai/cucim) for handling WSIs. In certain edge cases, however, cuCIM is not a reliable solution for image processing tasks. To our knowledge, these cases include:

1. **Windows usage**  
   cuCIM is part of the [RAPIDS](https://rapids.ai/) project, which primarily targets Linux. As a result, there is no official PyPI package for Windows.

2. **Working with OME-TIFF files**  
   [OME-TIFF](https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/) is a common WSI format that combines TIFF image data with XML metadata. As of now, OME-TIFF is not supported by cuCIM.

3. **Using AutoMIL’s PNG → TIFF conversion pipeline**  
   For PNG-based datasets (which are generally not recommended for WSIs, but are sometimes used in practice, e.g. in certain public challenges), AutoMIL provides an opt-in preprocessing step to convert PNG images to TIFF. cuCIM is not well suited for processing very large PNG images in this context.

For all three cases, **AutoMIL** provides an optional dependency group that installs the [pyvips](https://github.com/libvips/pyvips) library as an alternative image processing backend. Install it via:

```bash
pip install .[vips]
```

!!! warning "System dependency required for pyvips"

    `pyvips` is only the Python binding for the image processing library
    [libvips](https://www.libvips.org/), which must be installed separately
    on your system.

    === "Windows"

        Download and install the official prebuilt binaries from the libvips website and ensure that the installation directory is added to your `PATH`.
    
    === "Linux (Debian/Ubuntu)"

        ```bash
        sudo apt install libvips
        ```

    === "macOS"

        ```bash
        brew install vips
        ```