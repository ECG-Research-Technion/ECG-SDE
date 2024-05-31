# Project Title: ECG Signal Forecasting Using VQ-VAE and Transformer Architectures

## Overview

This project focuses on forecasting ECG signals for patients using a novel approach that combines Vector Quantized Variational Autoencoders (VQ-VAE) and Transformer architectures. The repository includes various modules and configurations required for the implementation, training, and evaluation of the model.

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
   - Contains the hyperparameters for the SDE.

## Usage

To train the VQ-VAE model, run the `vqvae_train.py` script. Ensure that the configuration and hyperparameters files (`config_vqvae.json` and `hyper_params.py`) are set up correctly. Data should be prepared using `prepare_mitbih.py` and paths should be configured in `paths.py`.

### Example Command
```bash
python vqvae_train.py
