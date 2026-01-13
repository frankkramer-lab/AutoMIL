# Quickstart

This guide walks you through a minimal, end-to-end AutoMIL workflow: from preparing your data to running a first model evaluation. It is intended to provide a high-level overview of the typical AutoMIL pipeline.

---

## (Optional) 1. Activate Your Environment

If you installed AutoMIL in a virtual environment, ensure your venv is activated before running AutoMIL.

```bash
source .venv/bin/activate
```

---

## 2. Prepare the Dataset

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

## 3. Run the basic training pipeline

To train a basic [Attention_MIL](https://arxiv.org/abs/1802.04712) model on the dataset, run the `automil train` command with default parameters:

```bash
automil train ./dataset/slides ./dataset/annotations.csv /results -v
```

Using the verbose flag `-v` will provide you with additional information displayed in stdout, giving you an overview of what is happening in the background:

```text
INFO     Using resolution presets: ['Low']                     
INFO     Using model type: Attention_MIL                             
INFO     Creating project scaffold at results            
INFO     Annotations saved to results/annotations.csv            
INFO     Project scaffold setup complete!                            
INFO     Created project at results                     
INFO     Project Summary                                             
INFO    ╒═══════════════════════╤═══════════════════════════════╕   
        │ Project Directory:    │ results                       │   
        │ Slide Directory:      │ ./dataset/slides              │   
        │ Annotations File:     │ ./dataset/annotations.csv     │   
        │ Patient Column:       │ patient                       │   
        │ Label Column:         │ label                         │   
        │ Slide Column:         │ None (using patient ID)       │   
        │ Transform Labels:     │ False                         │   
        │ Modified Annotations: │ results/annotations.csv       │   
        │ Slideflow Project:    │ Loaded                        │   
        ╘═══════════════════════╧═══════════════════════════════╛   
INFO     Setting up dataset for resolution preset: Low         
INFO     Using default MPP for magnification 10x: 1.000               
INFO     Dataset Summary:                                            
INFO    ╒═══════════════════╤═══════════╕                           
        │ Resolution Preset │ Low       │                           
        │ Tile Size (px)    │ 1000px    │                           
        │ Magnification     │ 10x       │                           
        │ Microns-Per-Pixel │ 1.000     │                           
        │ Tile Size (um)    │ 2000.00um │                           
        │ Pretiled Input    │ False     │                           
        │ TIFF Conversion   │ True      │                           
        ╘═══════════════════╧═══════════╛                           
INFO     Preparing dataset source at resolution Low (1000px,   
        2000.00um)        
```

The trained model will be saved in the `/results` directory under `results/models/`.

## 4. Evaluate the trained model

To evaluate the trained model on the same dataset, run the `automil evaluate` command:

```bash
automil evaluate ./results/models/00000_attentionmil_label/ ./dataset/slides ./dataset/annotations.csv -o ./evaluation -v
```

This will create an evaluation report inside the `./evaluation` directory, containing metrics and visualizations of the model performance.

