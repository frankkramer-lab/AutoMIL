# Requirements

This section outlines the software and system requirements needed to run **AutoMIL**.

---

## Platform / Operating System

AutoMIL builds on top of the :material-microscope: [Slideflow](https://slideflow.dev/overview/) framework for WSI processing, dataset management, and model training. While AutoMIL itself is platform-agnostic, slideflow is primarily developed and tested on **Linux** and depends on system-level libraries such as **[cuCIM](https://docs.rapids.ai/api/cucim/stable/)**. cuCIM is part of the **[RAPIDS](https://rapids.ai/)** collection of GPU-accelerated software solutions for data science, all of which are developed for usage on Linux. Consequently, full functionality, stability, and performance of AutoMIL can only be guaranteed on **Linux**, which is therefore **strongly recommended**.

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

1. **Working with OME-TIFF files**  
   [OME-TIFF](https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/) is a common WSI format that combines TIFF image data with XML metadata. As of now, OME-TIFF is not supported by cuCIM.

2. **Using AutoMIL’s PNG → TIFF conversion pipeline**  
   For PNG-based datasets (which are generally not recommended for WSIs, but are [sometimes used in practice](https://www.kaggle.com/competitions/UBC-OCEAN/)), AutoMIL provides an opt-in preprocessing step to convert PNG images to TIFF. cuCIM is not well suited for processing very large PNG images in this context.

For both cases, **AutoMIL** provides an optional dependency group that installs the [pyvips](https://github.com/libvips/pyvips) library as an alternative image processing backend. Install it via:

```bash
pip install .[vips]
```

!!! warning "System dependency required for pyvips"

   `pyvips` is only the Python binding for the image processing library
    [libvips](https://www.libvips.org/), which must be installed separately
    on your system.
    
    ```bash
    sudo apt install libvips
    ```