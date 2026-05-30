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
Contains help page text for the AutoMIL CLI.
"""

RUN_PIPELINE_HELP = """
    Execute the complete AutoMIL pipeline for whole slide image analysis.
    
    \b
    `run-pipeline` executes the entire AutoMIL workflow, encompassing:
      [1] Project setup and configuration
      [2] Dataset preparation and tile extraction  
      [3] Model training with k-fold cross-validation
      [4] Model evaluation and ensemble creation
      [5] Results comparison and visualization

    \b
    ARGUMENTS:
      SLIDE_DIR        Directory containing whole slide images (.svs, .tiff, etc.)
      ANNOTATION_FILE  .csv file with slide/patient annotations and labels  
      PROJECT_DIR      Output directory for all results and models
    
    \b
    EXAMPLES:
      # Basic usage with default settings
      automil run-pipeline /data/slides /data/annotations.csv ./results
      
      # Multi-resolution training with verbose output
      automil run-pipeline -r "Low,High" -v /data/slides /data/annotations.csv ./results
      
      # Custom model and k-fold settings
      automil run-pipeline -m TransMIL -k 5 /data/slides /data/annotations.csv ./results
      
      # Skip tiling if tiles are pre-extracted
      automil run-pipeline -p /data/slides /data/annotations.csv ./results
      
      # Custom column names in the annotation file
      automil run-pipeline -pc "patient_name" -lc "diagnosis" -sc "slide_name" /data/slides /data/annotations.csv ./results
      
      # Provide a predefined train-test split
      automil run-pipeline --split-file /data/split.json /data/slides /data/annotations.csv ./results

    \b
    ANNOTATION REQUIREMENTS:
      ANNOTATION_FILE must be a CSV file containing at least the following columns:
        - Patient IDs (default column name: "patient")
        - Slide names (default column name: "slide"; optional)
        - Labels (default column name: "label")
      By default, AutoMIL looks for columns named "patient", "slide", and "label".
      By using the options `--patient_column`, `--slide_column`, and `--label_column`,
      users can specify custom column names as needed (see EXAMPLES)
    \b
    MINIMAL ANNOTATION FILE EXAMPLE:
        patient,slide,label
        001,001_1,0
        001,001_2,0
        002,002,1
        003,003,1

    \b
    EXPECTED SLIDE DIRECTORY STRUCTURE:
      SLIDE_DIR should contain whole slide images in supported formats
      such as .svs, .tiff, or .png.
      Example structure:
        /data/slides/
        |-- slide1.svs
        |-- slide2.tiff
        |-- slide3.tiff
      If slides are in PNG, AutoMIL will first convert them to TIFF for easier processing.

    \b
    USING PRETILED DATA:
      If tiles have already been extracted from the slides, use the `--is_pretiled` flag.
      In the case of pretiled data, AutoMIL expects the following directory structure for SLIDE_DIR:
        /data/slides/
        |-- slide1/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
        |-- slide2/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
      Tile names are arbitrary but slide subdirectories must match the slide names in ANNOTATION_FILE.

    \b
    PROVIDING A TRAIN TEST SPLIT:
        Use the `--split-file` option to provide a JSON file defining train-test splits.
        The JSON file should have the following structure:
            {
            "train": ["slide1", "slide2", ...],
            "test":  ["slide3", "slide4", ...]
            }
        or:
            {
            "train": ["slide1", "slide2", ...],
            "validation":  ["slide3", "slide4", ...]
            }

    \b
    OUTPUT STRUCTURE:
      project_dir/
      ├── bags/           # Extracted tile features
      ├── models/         # Trained model checkpoints  
      ├── ensemble/       # Ensemble predictions
      ├── annotations.csv # Processed annotations
      └── results.json    # Performance metrics
    """

TRAIN_HELP = """
    Train a single or multiple MIL models on a given dataset.
    
    \b
    `train` sets up a project and dataset source, then trains MIL models using k-fold cross-validation:
      [1] Project setup and configuration
      [2] Dataset preparation and tile extraction  
      [3] Model training with k-fold cross-validation

    \b
    ARGUMENTS:
      SLIDE_DIR        Directory containing whole slide images (.svs, .tiff, etc.)
      ANNOTATION_FILE  CSV file with slide/patient annotations and labels  
      PROJECT_DIR      Output directory for all results and models
    
    \b
    EXAMPLES:
      # Basic usage with default settings
      automil train /data/slides /data/annotations.csv ./results
      
      # Multi-resolution training with verbose output
      automil train -r "Low,High" -v /data/slides /data/annotations.csv ./results
      
      # Custom model and k-fold settings
      automil train -m TransMIL -k 5 /data/slides /data/annotations.csv ./results
      
      # Skip tiling if tiles are pre-extracted
      automil train -p /data/slides /data/annotations.csv ./results
      
      # Custom column names in the annotation file
      automil train -pc "patient_name" -lc "diagnosis" -sc "slide_name" /data/slides /data/annotations.csv ./results

    \b
    ANNOTATION REQUIREMENTS:
      ANNOTATION_FILE must be a CSV file containing at least the following columns:
        - Patient IDs (default column name: "patient")
        - Slide names (default column name: "slide"; optional)
        - Labels (default column name: "label")
      By default, AutoMIL looks for columns named "patient", "slide", and "label".
      By using the options `--patient_column`, `--slide_column`, and `--label_column`,
      users can specify custom column names as needed (see EXAMPLES)
    \b
    MINIMAL ANNOTATION FILE EXAMPLE:
        patient,slide,label
        001,001_1,0
        001,001_2,0
        002,002,1
        003,003,1

    \b
    EXPECTED SLIDE DIRECTORY STRUCTURE:
      SLIDE_DIR should contain whole slide images in supported formats
      such as .svs, .tiff, or .png.
      Example structure:
        /data/slides/
        |-- slide1.svs
        |-- slide2.tiff
        |-- slide3.tiff
      If slides are in PNG, AutoMIL will first convert them to TIFF for easier processing.

    \b
    USING PRETILED DATA:
      If tiles have already been extracted from the slides, use the `--is_pretiled` flag.
      In the case of pretiled data, AutoMIL expects the following directory structure for SLIDE_DIR:
        /data/slides/
        |-- slide1/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
        |-- slide2/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
      Tile names are arbitrary but slide subdirectories must match the slide names in ANNOTATION_FILE.

    \b
    OUTPUT STRUCTURE:
      project_dir/
      ├── bags/           # Extracted tile features
      ├── models/         # Trained model checkpoints  
      ├── ensemble/       # Ensemble predictions
      ├── annotations.csv # Processed annotations
      └── results.json    # Performance metrics
    """

PREDICT_HELP = """
    Generate predictions using a single or multiple trained MIL models.

    \b
    `predict` loads all available model checkpoints from MODEL_DIR and generates predictions
    on the slides in SLIDE_DIR using the corresponding tile features in BAGS_DIR.
    The results are saved to OUTPUT_FILE.

    \b
    ARGUMENTS:
        SLIDE_DIR     Directory containing whole slide images (.svs, .tiff, ...)
        ANNOTATION_FILE  .csv file with slide/patient annotations and labels
        BAGS_DIR      Directory containing tile feature bags (.pt files)
        MODEL_DIR     Directory containing trained model checkpoints (.pth files)

    \b
    EXAMPLES:
      # Generate predictions with multiple models (generates one output file per model)
      automil predict /data/slides /data/annotations.csv /data/bags /data/models/ -o ./predictions

      # Generate predictions with a single model
      automil predict /data/slides /data/annotations.csv /data/bags /data/models/model_1 -v
    
      # Generate predictions with a single model (override column names)
      automil predict -pc "patient_id" -lc "outcome" -sc "slide_id" /data/slides /data/annotations.csv /data/bags /data/models/model_1 -o ./predictions
    
    \b
    EXPECTED MODEL DIRECTORY STRUCTURE:
        MODEL_DIR can refer to a single model directory (containing one .pth file) or
        a parent directory containing multiple model subdirectories with .pth files.
        EXAMPLE STRUCTURE FOR A SINGLE MODEL:
            /data/models/model_1/
                |-- best_valid.pth
                |....
        EXAMPLE STRUCTURE FOR MULTIPLE MODELS:
          /data/models/
            |-- model_1/
            |    |-- best_valid.pth
            |-- model_2/
            |    |-- best_valid.pth
            |    |...
    
    \b
    ANNOTATION REQUIREMENTS:
      ANNOTATION_FILE must be a CSV file containing at least the following columns:
        - Patient IDs (default column name: "patient")
        - Slide names (default column name: "slide"; optional)
        - Labels (default column name: "label")
      By default, AutoMIL looks for columns named "patient", "slide", and "label".
      By using the options `--patient_column`, `--slide_column`, and `--label_column`,
      users can specify custom column names as needed (see EXAMPLES)
    \b
    MINIMAL ANNOTATION FILE EXAMPLE:
        patient,slide,label
        001,001_1,0
        001,001_2,0
        002,002,1
        003,003,1
    
    \b
    OUTPUT DIRECTORY FORMAT:
        OUTPUT_DIR should be a directory path.
        Predictions will be saved in separate .csv or .parquet files within this directory.
        If multiple models are used, separate output files will be created for each model,
        adding a suffix with the model name to the specified OUTPUT_DIR path.
    """

EVALUATE_HELP = """
    Evaluate a single or multiple trained MIL models.

    \b
    `evaluate` loads all available model checkpoints from MODEL_DIR, generates predictions
    on the slides in SLIDE_DIR using the corresponding tile features in BAGS_DIR and evaluates them.
    The results are saved to OUTPUT_FILE.

    \b
    ARGUMENTS:
        SLIDE_DIR     Directory containing whole slide images (.svs, .tiff, ...)
        ANNOTATION_FILE  .csv file with slide/patient annotations and labels
        BAGS_DIR      Directory containing tile feature bags (.pt files)
        MODEL_DIR     Directory containing trained model checkpoints (.pth files)

    \b
    EXAMPLES:
      # Evaluate a single model
      automil evaluate /data/slides /data/annotations.csv /data/bags /data/models/model_1 -o ./results

      # Evaluate multiple models (generates one output file per model)
      automil evaluate /data/slides /data/annotations.csv /data/bags /data/models/ -v
    
      # Evaluate a single model (override column names)
      automil evaluate -pc "patient_id" -lc "outcome" -sc "slide_id" /data/slides /data/annotations.csv /data/bags /data/models/model_1 -o ./results      
    
    \b
    EXPECTED MODEL DIRECTORY STRUCTURE:
        MODEL_DIR can refer to a single model directory (containing one .pth file) or
        a parent directory containing multiple model subdirectories with .pth files.
        EXAMPLE STRUCTURE FOR A SINGLE MODEL:
            /data/models/model_1/
                |-- best_valid.pth
                |....
        EXAMPLE STRUCTURE FOR MULTIPLE MODELS:
          /data/models/
            |-- model_1/
            |    |-- best_valid.pth
            |-- model_2/
            |    |-- best_valid.pth
            |    |...
    
    \b
    ANNOTATION REQUIREMENTS:
      ANNOTATION_FILE must be a CSV file containing at least the following columns:
        - Patient IDs (default column name: "patient")
        - Slide names (default column name: "slide"; optional)
        - Labels (default column name: "label")
      By default, AutoMIL looks for columns named "patient", "slide", and "label".
      By using the options `--patient_column`, `--slide_column`, and `--label_column`,
      users can specify custom column names as needed (see EXAMPLES)
    \b
    MINIMAL ANNOTATION FILE EXAMPLE:
        patient,slide,label
        001,001_1,0
        001,001_2,0
        002,002,1
        003,003,1

    \b
    OUTPUT DIRECTORY FORMAT:
        OUTPUT_DIR should be a directory path.
        Predictions will be saved in separate .csv or .parquet files within this directory.
        If multiple models are used, separate output files will be created for each model,
        adding a suffix with the model name to the specified OUTPUT_DIR path.
    """

CREATE_SPLIT_HELP = """
    Create a train-test split .json file based on the provided annotations.

    \b
    `create-split` reads the annotation file and generates a train-test split,
    saving it to OUTPUT_FILE in .json format.

    \b
    ARGUMENTS:
      SLIDE_DIR        Directory containing whole slide images (.svs, .tiff, etc.)
      ANNOTATION_FILE  .csv file with slide/patient annotations and labels  

    \b
    EXAMPLES:
      # Basic usage with default settings
      automil create-split /data/slides /data/annotations.csv -o split.json

    \b
    OUTPUT FILE FORMAT:
      The output JSON file will have the following structure:
        {
          "train": ["slide1", "slide2", ...],
          "test":  ["slide3", "slide4", ...]
        }
    """