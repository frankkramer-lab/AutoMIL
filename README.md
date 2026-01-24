# AutoMIL 
## Automated Machine Learning for Image Classification in Whole-Slide Imaging with Multiple Instance Learning

AutoMIL is a flexible, open-source, end-to-end pipeline for training and evaluating Multiple Instance Learning (MIL) models for image classification on whole-slide images (WSIs).
It provides a modular command-line interface (CLI) that enables straightforward usage and adaptation to diverse WSI datasets.
In addition to the CLI, AutoMIL exposes a Python API for programmatic use, allowing users to build their own custom workflows.


## Features

* A well documented and easy to use Command Line Interface
* A high-level python API for custom development
* Modular project structure for easy adaptation to new datasets
* Support for multiple MIL algorithms and model architectures
* Adaptability to various WSI formats and datasets, including large image sizes and pretiled slides

## Installation

### Requirements
    - Python 3.11+
    - Cuda-compatible GPU
    - cucim or libvips
    - Linux

### Setup

**AutoMIL** can be installed directly from its public [GitHub](https://github.com/your/project) repository. To download the source code, open a terminal, navigate to any directory and run:

```bash
git clone https://github.com/WaibelJonas/AutoMIL.git
```

This will clone the projects source code inside a new directory called `./automil`. Navigate to this directory and install **AutoMIL** in your current python environment:

```bash
pip install .
```

## Quick Start

### Preparing your Dataset

AutoMIL expects your WSI dataset to consist of slide images in one of many supported formats (.tiff, .svs, .tif etc) and a file containing slide-level label information 

A minimal dataset consists of:

- A directory containing slide images
- A .csv metadata file with slide-level annotations

Example directory structure:

```text
dataset/
├── slides/
│   ├── case_001.tiff
│   ├── case_002.tiff
│   └── case_003.tiff
└── annotations.csv
```

With annotations.csv:

```csv
patient,slide,label
001,case_001,0
002,case_002,0
003,case_003,1
```

### Training a Model

To train a basic [Attention_MIL](https://arxiv.org/abs/1802.04712) model on the dataset, run the `automil train` command with default parameters:

```bash
automil train ./dataset/slides ./dataset/annotations.csv results -v
```

Using the verbose flag `-v` will provide you with additional information displayed in stdout, giving you more verbose info and error messages and is recommended

The trained model will be saved in the `results/` directory under `results/models/`.


### Evaluate the trained model

To evaluate the trained model on the same dataset, run the `automil evaluate` command:

```bash
automil evaluate ./results/models/00000_attentionmil_label/ ./dataset/slides ./dataset/annotations.csv -o ./evaluation -v
```

This will create an evaluation report inside the `./evaluation` directory, containing metrics and visualizations of the model performance.
