import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.optim as optim
import torch.nn as nn
import sys
sys.path.append('/home/shadi-omari/SDEProject/Final_Proj/Project')
from hyper_params import *
from preprocess_data import *
from vqvae import *
from training import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from latent_sde import *


def plt_two_time_series(samples, path, vqvae, sde, num_samples=4):    
    with torch.no_grad():
        pred = torch.Tensor([])
        orig = torch.Tensor([])
        for channel in range(2):
            fig, ax = plt.subplots(figsize=(20,5))
            sample = samples[0:1,:,:].to(device)

            quant_b, _, _ = vqvae.encode(sample)
            quant_b = quant_b.view(quant_b.size(0), -1)
            ts_ext = torch.linspace(0, 1, 5)
            zs, kl = sde(ts=ts_ext, y=quant_b, batch_size=quant_b.size(0))
            zs = zs[1:-1]
            zs = zs.reshape(zs.size(0), 8, 5*128)
            X_reconstructed = vqvae.decode(zs)

            X_reconstructed0 = vqvae(sample)
            pred = torch.cat([pred, X_reconstructed0[MODELS_TENSOR_PREDICITONS_KEY][0][channel].detach().cpu()], dim=0)
            orig = torch.cat([orig, sample[0][channel].detach().squeeze(0).cpu()], dim=0)
            
            for i in range(zs.size(0)):
                pred = torch.cat([pred, X_reconstructed[i][channel].detach().cpu()], dim=0)
                orig = torch.cat([orig, samples[i+1][channel].detach().squeeze(0).cpu()], dim=0)

            t = [j for j in range(128*5*num_samples)]
            ax.plot(t, orig, 'b', label="GT")
            ax.plot(t, pred, 'r--', label="Prediction") #
            y_min = -2 if orig.min() > -1 else orig.min()-1
            y_max = 2 if orig.max() < 1 else orig.max()+1
            ax.plot([128, 128], [y_min, y_max], 'k:')
            ax.set_ylim([y_min, y_max])
            ax.set_title('Original vs Reconstructed')
            ax.set_xlabel('time')
            ax.set_ylabel('ECG')
            ax.legend()
            plt.savefig(path[channel])
            plt.close()

            pred = torch.Tensor([])
            orig = torch.Tensor([])

def post_epoch_fn(verbose, epoch, X_test_data, model, wrapper_model):
    # Plot some samples if this is a verbose epoch
    if verbose:
        
        # sample 3 random samples
        num_samples = 3
        samples = torch.empty(num_samples*4,2,128*5)
        while num_samples != 0:
            num_total_samples = X_test_data.size(0) - 3
            random_indices = torch.randperm(num_total_samples)[:1]
            if torch.any(random_indices == samples[0,:,0]):
                continue
            random_indices = random_indices - random_indices%4
            random_indices = torch.arange(random_indices[0], random_indices[0]+4)

            candidate_sample = X_test_data[random_indices, :, :]
            if (torch.any(candidate_sample[:,0,:] > 0.75) or torch.any(candidate_sample[:,0,:] < -0.75)) and (torch.any(candidate_sample[:,1,:] > 0.75) or torch.any(candidate_sample[:,1,:] < -0.75)):
                samples[num_samples*4-4:num_samples*4,:,:] = candidate_sample
                num_samples -= 1

        for i in range(int(samples.shape[0]/4)):
            plt_two_time_series(samples[i*4:i*4+4,:,:], [f'../orig_pred_sde/epoch_{epoch}_channel_0_sample_{i}', f'../orig_pred_sde/epoch_{epoch}_channel_1_sample_{i}'], wrapper_model, model)
            

def main():
    
    ################################
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Timestamp:", timestamp)
    
    #pre-process data
    X_train_filename = '../../../X_train_4.pt' #should be absolute path
    if os.path.exists(X_train_filename):
        #file exists.
        X_train_data = torch.load(X_train_filename).permute(1,0,2)

    else:
        #create a file
        pre_data = PreprocessData('Train')
        X_train_data = pre_data.get_ecgs()
        torch.save(X_train_data, X_train_filename)
        X_train_data = X_train_data.permute(1,0,2)

    X_test_filename = '../../../X_test_4.pt' #should be absolute path
    if os.path.exists(X_test_filename):
        #file exists.
        X_test_data = torch.load(X_test_filename).permute(1,0,2)

    else:
        #create a file
        pre_data = PreprocessData('Test')
        X_test_data = pre_data.get_ecgs()
        torch.save(X_test_data, X_test_filename)
        X_test_data = X_test_data.permute(1,0,2)

#     X_val_filename = '../X_val.pt' #should be absolute path
#     if os.path.exists(X_val_filename):
#         #file exists.
#         X_val_data = torch.load(X_val_filename).to(device).permute(1,0,2)

#     else:
#         #create a file
#         pre_data = PreprocessData('val')
#         X_val_data = pre_data.get_ecgs()
#         torch.save(X_val_data, X_val_filename)
#         X_val_data.to(device).permute(1,0,2)
        
#     num_rows = int(0.85 * X_val_data.shape[0])
#     X_train_data = X_val_data[:num_rows,:,:]
#     X_test_data = X_val_data[num_rows:,:,:]
    
    print(X_train_data.shape)
    print(X_test_data.shape)

    train_dataset = TensorDataset(X_train_data)
    test_dataset = TensorDataset(X_test_data)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Timestamp:", timestamp)   
    
    # Load vqvae hyperparams
    vqvae_hp = vqvae_hyperparams()
    in_channel = vqvae_hp['in_channel']
    channel = vqvae_hp['channel']
    n_res_block = vqvae_hp['n_res_block']
    n_res_channel = vqvae_hp['n_res_channel']
    embed_dim = vqvae_hp['embed_dim']
    n_embed = vqvae_hp['n_embed']
    decay = vqvae_hp['decay']
    n_dims = vqvae_hp['n_dims']
    learning_rate = vqvae_hp['learning_rate']
    
    # Load sde hyperparams
    sde_hp = sde_hyperparams()
    adjoint = sde_hp['adjoint']
    adaptive = sde_hp['adaptive']
    method = sde_hp['method']
    dt = sde_hp['dt']
    rtol = sde_hp['rtol']
    atol = sde_hp['atol']
    batch_size = sde_hp['batch_size']
     
    # Create an instance of the VQVAE class
    vqvae = VQVAE(
        in_channel=in_channel,
        channel=channel,
        n_res_block=n_res_block,
        n_res_channel=n_res_channel,
        embed_dim=embed_dim,
        n_embed=n_embed,
        decay=decay,
        n_dims=n_dims,
    ).to(device)
    vqvae_dp = DataParallel(vqvae).to(device)
    
    vqvae_weights = '../checkpoints/vqvae_weights_final'
    if os.path.isfile(f'{vqvae_weights}.pt'):
        state_dict = torch.load(f'{vqvae_weights}.pt')
        vqvae_dp.load_state_dict(state_dict)
        
    # Create an instance of the Latent SDE class
    latent_sde = LatentSDE().to(device)
    latent_sde_dp = DataParallel(latent_sde).to(device)
    
    # Create data loaders for the train and test sets
    num_cpu_cores = os.cpu_count()
    dl_train, dl_test = [
        DataLoader(
            X_data,
            num_workers=max(num_cpu_cores - 2, 1),
            batch_size=batch_size
        )
        for X_data in [train_dataset, test_dataset]
    ]

    optimizer = optim.Adam(latent_sde.parameters(), lr=1e-7)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    kl_scheduler = LinearScheduler(iters=100)
    
    logpy_metric = EMAMetric()
    kl_metric = EMAMetric()
    loss_metric = EMAMetric()
    
    # Train the model
    trainer = LatentSDETrainer(latent_sde_dp, None, optimizer, device, vqvae_dp, scheduler, kl_scheduler, logpy_metric, kl_metric, loss_metric)
    
    checkpoint_file = '../checkpoints_sde/sde_weights'
    checkpoint_file_final = f'{checkpoint_file}_final'
    if os.path.isfile(f'{checkpoint_file}.pt'):
        os.remove(f'{checkpoint_file}.pt')
        
    print(vqvae)
    print(vqvae_hp)
    print(latent_sde)
    
    mpl.rcParams['figure.max_open_warning'] = 10000
    
    # sample 3 random samples
    num_samples = 3
    samples = torch.empty(num_samples*4,2,128*5)
    while num_samples != 0:
        num_total_samples = X_test_data.size(0) - 3
        random_indices = torch.randperm(num_total_samples)[:1]
        if torch.any(random_indices == samples[0,:,0]):
            continue
        random_indices = random_indices - random_indices%4
        random_indices = torch.arange(random_indices[0], random_indices[0]+4)

        candidate_sample = X_test_data[random_indices, :, :]
        if (torch.any(candidate_sample[:,0,:] > 0.75) or torch.any(candidate_sample[:,0,:] < -0.75)) and (torch.any(candidate_sample[:,1,:] > 0.75) or torch.any(candidate_sample[:,1,:] < -0.75)):
            samples[num_samples*4-4:num_samples*4,:,:] = candidate_sample
            num_samples -= 1
            
        
    print('*** Ecgs before any train:')
    for i in range(int(samples.shape[0]/4)):
        plt_two_time_series(samples[i*4:i*4+4,:,:], [f'../orig_pred_sde/epoch_0_channel_0_sample_{i}', f'../orig_pred_sde/epoch_0_channel_1_sample_{i}'], vqvae, latent_sde)
        
    if os.path.isfile(f'{checkpoint_file_final}.pt'):
        print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        checkpoint_file = checkpoint_file_final
        state_dict = torch.load(f'{checkpoint_file_final}.pt')
        latent_sde_dp.load_state_dict(state_dict)
        
    else:      
        res = trainer.fit(dl_train, dl_test,
                          num_epochs=20, early_stopping=6000, print_every=2, save_weights_every=10,
                          checkpoints=checkpoint_file,
                          post_epoch_fn=post_epoch_fn, X_test_data=X_test_data, model=latent_sde_dp, wrapper_model=vqvae)
        
    for i in range(int(samples.shape[0]/4)):
        plt_two_time_series(samples[i*4:i*4+4,:,:], [f'../orig_pred_sde/after_train_sample_{i*2}', f'../orig_pred_sde/after_train_sample_{i*2+1}'], vqvae, latent_sde)
            
            
if __name__ == '__main__':
    main()