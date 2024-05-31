# Project Title: ECG Signal Forecasting Using VQ-VAE, SDE and Integrating Based-Attentions Mechanisms Architectures

## Overview

This project focuses on forecasting ECG signals for patients using a novel approach that combines Vector Quantized Variational Autoencoders (VQ-VAE) and SDE with integratins Transformer architectures. The repository includes various modules and configurations required for the implementation, training, and evaluation of the model.

## Directory Structure

### VQ-VAE Implementation
1. **vqvae.py**
   - Contains the implementation of the VQ-VAE model.

2. **hyper_params.py**
   - Includes the hyperparameters for the VQ-VAE model.

3. **config_vqvae.json**
   - Configuration file for the VQ-VAE model, specifying parameters and settings.

4. **vqvae_train.py**
   - Script to train the VQ-VAE model. It utilizes the configurations specified in `config_vqvae.json` and `hyper_params.py`.

### Data Preparation and Paths
1. **prepare_mitbih.py**
   - Prepares and preprocesses the MIT-BIH dataset before training.

2. **paths.py**
   - Describes the storage locations for datasets, results, checkpoints, and other relevant data paths.
   - Maintains different folder versions and configurations for different experiments.

### Training and Results
1. **training.py**
   - A generic file for training the model. It accepts the model, epochs, and configurations to perform data-parallel training and post-epoch graph/results generation.

2. **train_results.py**
   - Contains classes for handling and displaying training results.

### Transformer Implementation
1. **transformer.py**
   - Includes the implementation of the Transformer architecture.

### SDE Implementation (Out of Scope)
1. **latent_sde.py**
   - Implements the latent SDE (part of a larger project, not within the scope of this work).

2. **hyper_params_sde.py**
   - Contains the hyperparameters for the SNe.

## Setup Instructions

### Prerequisites

Before running the project, ensure you have the following installed:
- Conda: A package and environment management system.

### Environment Setup

1. **Install Conda**
   - Download and install Conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Create Environment**
   - Navigate to the project directory and create the environment using the provided `environment.yml` file:
     ```bash
     conda env create -f environment.yml
     ```

### Dataset Preparation

1. **Prepare the Dataset**
   - Ensure the MIT-BIH dataset is correctly assigned in the `paths.py` file:
     - `train_data`
     - `test_data`
     - `datasets_dir`
   - If not, assign the correct paths in `paths.py` and run:
     ```bash
     python3 prepare_mitbih.py
     ```
   - This script will create the necessary dataset folders and `.pt` files.

### Configuration

1. **Configure Paths**
   - Set the `results_dir` and `checkpoints_dir` in the `paths.py` file.

2. **Configure VQ-VAE**
   - Modify the `config_vqvae.json` file with the desired settings:
     ```json
     {
       "has_attention": true,
       "samples_per_second": 360,
       "input_signal_duration": 3,
       "signals_per_patient": 1,
       "forecast_predictions": 1
     }
     ```

## Running the Training

To start training the VQ-VAE model, run:
```bash
python3 vqvae_train.py
```

## Running the Training

This command will:

- Print initial graphs of results for the reconstructed lead1 and lead2, comparing them to the original signals.
- Save results in a file under the results directory after each epoch, comparing the original signals with the reconstructed signals graphically.
- Save checkpoints for the model weights periodically.
- Save the best model after all epochs are completed. You can change these settings under the fit method in the `vqvae_train.py` file.

### Detailed Training Process

1. **Initial Results**
   - The training process will print the initial graphs of results for the reconstructed lead1 and lead2, comparing them to the original signals.

2. **Epoch Results**
   - Results will be saved in a file under the results directory after each epoch.

3. **Model Checkpoints**
   - Checkpoints for the model weights will be saved every few epochs.

4. **Final Model**
   - The best model will be saved after running all the epochs defined. These settings can be adjusted in the fit method in the `vqvae_train.py` file.
