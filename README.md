# AutoMIL 
## Automated Machine Learning for Image Classification in Whole-Slide Imaging with Multiple Instance Learning

AutoMIL provides a flexible framework for training and evaluating Multiple Instance Learning models for the task of image classification on Whole Slide Image (WSI) datasets.

![Pipeline](data/Dia.png)

## Features

- **Automated Pipeline**: Complete automated workflow encompassing preprocessing, training and evalutuation
- **Image Backend Compatibility**: Support for multiple common image backends (cucim, libvips, openslide) available on both Linux and Windows
- **Memory Optimization**: Automated adjustments of hyperparamters to better adaptat to available VRAM
- **Rich Logging**: Extensive logging and debug functionality with color-coding using [Rich](https://rich.readthedocs.io/en/stable/introduction.html)
- **Configuration**: Support for various slide formats and annotation strucutures

## Installation

### Requirements
- Python 3.11+
- Cuda-compatible GPU
- libvips (Windows) or cucim (Linux)

### Setup

#### pip
```bash
git clone https://github.com/your-repo/AutoMIL.git
cd AutoMIL
pip install -r requirements.txt
```
#### [uv](https://docs.astral.sh/uv/)
```bash
git clone https://github.com/your-repo/AutoMIL.git
cd AutoMIL
uv sync
# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

## Quick Start

### Basic Pipeline
Run the complete AutoMIL training pipeline:

```bash
cd AutoMIL
source .venv/bin/activate
python3 src/cli.py run-pipeline ./slides ./annotations.csv ./project_dir --verbose
```

**Options:**
- `-pc, --patient_column`: Column name for patient IDs (default: "patient")
- `-lc, --label_column`: Column name for labels (default: "label")
- `-sc, --slide_column`: Column name for slide names (optional)
- `-k`: Number of cross-validation folds (default: 3)
- `-t, --transform_labels`: Transform labels to float values
- `-v, --verbose`: Enable detailed logging
- `-c, --cleanup`: Delete project structure after completion

### Batch Size Analysis
Analyze model performance across different batch sizes:

```bash
cd AutoMIL
source .venv/bin/activate
python3 src/cli.py batch-analysis ./slides ./annotations.csv ./project_dir --batch_sizes "2,4,8,16,32" --plot --verbose
```

**Options:**
- All options from `run-pipeline`
- `-bs, --batch_sizes`: Comma-separated batch sizes (default: "2,4,8,16,32")
- `-p, --plot`: Generate plots automatically after analysis