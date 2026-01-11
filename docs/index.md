# AutoMIL Documentation

AutoMIL is a flexible, open-source, end-to-end pipeline for training and evaluating Multiple Instance Learning (MIL) models for image classification on whole-slide images (WSIs).
It provides a modular command-line interface (CLI) that enables straightforward usage and adaptation to diverse WSI datasets.
In addition to the CLI, AutoMIL exposes a Python API for programmatic use, allowing users to build their own custom workflows.

!!! info "Project Repository"
    The full code is accessible on :material-github: [GitHub](https://github.com/your/project)

??? info "AutoMIL and AutoML"

    **AutoMIL** is deeply rooted in the research field of [AutoML](https://www.automl.org/automl/), focusing on methods and processes to automate the development and deployment of Machine Learning (ML) solutions in order to make them more accessible to the broader public domain without requiring expert knowledge. AutoMIL is the first project of this type for WSI data and Histopathology (to our knowledge)

## Features

* :octicons-terminal-16: - A well documented and easy to use Command Line Interface
* :simple-python: - A high-level python API for custom development
* :open_file_folder: - Modular project structure for easy adaptation to new datasets
* :gear: - Support for multiple MIL algorithms and model architectures
* :octicons-light-bulb-16: Adaptability to various WSI formats and datasets, including large image sizes and pretiled slides

## Quickstart

To get started, go to the [Installation Instructions](getstarted/installation.md). If you already installed AutoMIL and want to quickly try it out, follow the [Quick Start Guide](getstarted/quickstart.md). If you're especially impatient and already have a dataset ready, you can directly jump into training your first model with one of the following commands:

* `automil run-pipeline /dataset annotations.csv /project` - Trains and evaluates a model on the dataset located at `/dataset` with slide-level labels provided in `annotations.csv` and saves all outputs to `/project`.

* `automil train /dataset annotations.csv /project` - Trains a model on the dataset located at `/dataset` with slide-level labels provided in `annotations.csv` and saves the trained model to `/project`.

* `automil predict /model /slides` - Generates predictions on the slides located in `/slides` using the (trained) model located at `/model`.

## Built upon Slideflow

AutoMIL is built on top of the :material-microscope: [Slideflow](https://slideflow.dev/overview/) framework for WSI data handling and preprocessing. Slideflow provides efficient data loading, tiling, and augmentation functionalities specifically designed for Whole Slide Images, making it an ideal foundation for MIL model training. **AutoMIL**s contribution lies in automating the selection of hyperparamaters, model architectures, and training procedures specifically tailored for MIL tasks on WSI data, as well as providing a single entry-point so the user experience is streamlined.

## Project layout

A typical AutoMIL project has the following layout:

    project_dir/
    |-- tfrecords/ # Directory containing .tfrecords generated during preprocessing
    |-- models/ # Directory containing trained model checkpoints
    |-- bags/ # Directory containing generated bags 
    |-- ensemble/ # Directory containing ensemble predictions