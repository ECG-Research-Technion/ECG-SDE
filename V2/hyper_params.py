def vqvae_hyperparams():
    hypers = dict(
        in_channel = 0, enc_out_channel = 0, n_trans_res_block = 0, n_trans_layers = 0, embed_dim = 0, n_embed = 0, decay = 0.0, n_dims = 0,
        batch_size = 0, learning_rate = 0.0, num_epochs = 0
    )
    
    # ========================
    hypers['in_channels'] = 2             # The number of channels in the ECG signal
    
    hypers['enc_channels'] = 64           # The number of channels in the feature maps produced by the encoder
        
    hypers['embed_dim'] = 80              # The dimensionality of the vector embeddings produced by the VQ layer
    
    hypers['n_embed'] = 512               # The number of vectors in the codebook used by the VQ layer
                    
    hypers['batch_size'] = 32             # The number of samples in each batch
    
    hypers['learning_rate'] = 0.001       # The learning rate used by the optimizer
    # ========================

    return hypers