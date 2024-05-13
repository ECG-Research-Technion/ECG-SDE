def vqvae_hyperparams():
    hypers = dict(
        in_channel = 0, enc_out_channel = 0, n_trans_res_block = 0, n_trans_layers = 0, embed_dim = 0, n_embed = 0, decay = 0.0, n_dims = 0,
        batch_size = 0, learning_rate = 0.0, num_epochs = 0
    )
    
    # ========================
    hypers['in_channel'] = 2              # The number of channels in the ECG signal
    
    hypers['channel'] = 8                 # The number of channels in the feature maps produced by the encoder
    
    hypers['n_res_block'] = 16            # The number of residual blocks in the transformer of the VQ-VAE model

    hypers['n_res_channel'] = 64          # The number of channels in the residual blocks of the transformer
        
    hypers['embed_dim'] = 512             # The dimensionality of the vector embeddings produced by the VQ layer
    
    hypers['n_embed'] = 1024              # The number of vectors in the codebook used by the VQ layer
    
    hypers['decay'] = 0.99                # The decay rate used in the exponential moving average update of the codebook 
    
    hypers['n_dims'] = 1                  # The number of dimensions in the input data. For us, this should be the number of samples in the signal
        
    hypers['n_trans_layers'] = 4          # The number of transformer layers in the VQ-VAE model
    
    hypers['batch_size'] = 32             # The number of samples in each batch
    
    hypers['learning_rate'] = 0.001       # The learning rate used by the optimizer

    hypers['num_epochs'] = 2000           # The number of epochs to train the model
    # ========================

    return hypers