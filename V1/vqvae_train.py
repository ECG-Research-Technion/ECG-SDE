import os
import matplotlib as mpl
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Local imports
import sys
from hyper_params import *
from preprocess_data import *
from vqvae import *
from training import *
from paths import *

# Configure GPU settings
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def plt_two_time_series(samples, path, model, epoch):
    X_reconstructed = model(samples)[MODELS_TENSOR_PREDICITONS_KEY]

    for i in range(samples.shape[0]):
        fig, axs = plt.subplots(2, 1, figsize=(20, 10))  # Create 2 subplots for each channel
        
        for channel in range(2):
            orig = samples[i, channel, :].detach().cpu().numpy()  # Original signal for the channel
            pred = X_reconstructed[i, channel, :].detach().cpu().numpy()  # Reconstructed signal
            
            axs[channel].plot(orig, color='orange', label='Original')
            axs[channel].plot(pred, color='grey', label='Reconstructed')
            y_min = -2 if orig.min() > -1 else orig.min()-1
            y_max = 2 if orig.max() < 1 else orig.max()+1
            axs[channel].set_ylim([y_min, y_max])
            axs[channel].set_title(f'Channel {channel+1}: Original vs Reconstructed')
            axs[channel].set_xlabel('Time')
            axs[channel].set_ylabel('ECG')
            axs[channel].legend()
        
        plt.tight_layout()
        res_filename = path / f"before_training.png" if epoch == -1 else path / f"epoch_{epoch+1}_sample_{i}.png"
        plt.savefig(res_filename)
        plt.close()

def post_epoch_fn(verbose, epoch, X_test_data, model):
    if verbose:
        num_samples = 4
        samples = select_samples(X_test_data, num_samples)
        plt_two_time_series(samples, RESULTS_DIR, model, epoch)

def select_samples(data, num_samples):
    random_indices = torch.empty(num_samples)
    forecast_signals = config['samples_per_second']*config['input_signal_duration']*config['forecast_predictions']
    samples = torch.empty(num_samples, 2, forecast_signals)
    while num_samples != 0:
        random_index = torch.randperm(data.size(0) - 1)[0].item()
        # check if the random index exist in the random_indices tensor
        if random_index in random_indices:
            continue
        # add to the random_indices tensor, the random index
        random_indices[num_samples-1] = random_index
        candidate_sample = data[random_index, :, :]
        if is_valid_sample(candidate_sample):
            samples[num_samples-1, :, :] = candidate_sample
            num_samples -= 1
    return samples.to(device)

def is_valid_sample(sample):
    return ((sample[0, :] > 0.75).any() or (sample[0, :] < -0.75).any()) and ((sample[1, :] > 0.75).any() or (sample[1, :] < -0.75).any())

def initialize_data():
    # Load X_train_data and X_test_data

    assert TRAIN_DATA.is_file() and TEST_DATA.is_file(), \
        "TRAIN_DATA and TEST_DATA files do not exist. Please prepare the relevant datasets using the prepare dataset files."

    X_train_data = torch.load(str(TRAIN_DATA))
    X_test_data = torch.load(str(TEST_DATA))

    return X_train_data, X_test_data

def setup_model_and_optim():
    # Setup the VQ-VAE model, optimizer, and scheduler
    vqvae_hp = vqvae_hyperparams()
    vqvae = VQVAE(
        in_channel=vqvae_hp['in_channel'],
        channel=vqvae_hp['channel'],
        n_res_block=vqvae_hp['n_res_block'],
        n_res_channel=vqvae_hp['n_res_channel'],
        embed_dim=vqvae_hp['embed_dim'],
        n_embed=vqvae_hp['n_embed'],
        decay=vqvae_hp['decay'],
        n_dims=vqvae_hp['n_dims'],
        has_transformer=config['has_transformer'],
        has_attention=config['has_attention'],
        n_trans_layers=vqvae_hp['n_trans_layers']
    ).to(device)
    print(f"Number of parameters in the model: {sum(p.numel() for p in vqvae.parameters() if p.requires_grad)}")
    
    optimizer = optim.RAdam(vqvae.parameters(), lr=vqvae_hp['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return vqvae, optimizer, scheduler

def eval_before_training(X_test_data, model):
    mpl.rcParams['figure.max_open_warning'] = 10000
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    num_samples = 4
    samples = select_samples(X_test_data, num_samples)
    plt_two_time_series(samples, RESULTS_DIR, model, epoch=-1)

def run_training(vqvae, optimizer, scheduler, X_train_data, X_test_data):
    # Define training parameters
    criterion = nn.MSELoss()
    trainer = VQVAETrainer(model=vqvae, loss_fn=criterion, optimizer=optimizer, device=device, scheduler=scheduler)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINTS_DIR / 'vqvae_weights'
    checkpoint_file_final = CHECKPOINTS_DIR / '_final'

    # Prepare data loaders
    num_cpu_cores = os.cpu_count()
    dl_train, dl_test = [
        DataLoader(
            X_data,
            num_workers=max(num_cpu_cores - 2, 1),
            batch_size=vqvae_hyperparams()['batch_size'],
            pin_memory=True,
        )
        for X_data in [X_train_data, X_test_data]
    ]

    # Train the model or load from checkpoint
    if not checkpoint_file_final.is_file():
        trainer.fit(dl_train, dl_test,
            num_epochs=2000, early_stopping=6000, print_every=2, save_weights_every=5,
            checkpoints=checkpoint_file,
            post_epoch_fn=post_epoch_fn, X_test_data=X_test_data, model=vqvae)
    else:
        print(f"*** Loading final checkpoint file {checkpoint_file_final} instead of training")
        state_dict = torch.load(f"{checkpoint_file_final}.pt")
        vqvae.load_state_dict(state_dict)

def main():
    # Initialize and load data
    X_train_data, X_test_data = initialize_data()
    # X_train_data = X_train_data[0:32]
    # X_test_data = X_train_data
    print(X_train_data.shape, X_test_data.shape)
    
    # Load hyperparameters and initialize model, optimizer, scheduler
    vqvae, optimizer, scheduler = setup_model_and_optim()

    # Evaluation before training
    eval_before_training(X_test_data, vqvae)

    # Training
    run_training(vqvae, optimizer, scheduler, X_train_data, X_test_data)

if __name__ == '__main__':
    main()
