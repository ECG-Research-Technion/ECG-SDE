def sde_hyperparams():
    hypers = dict(
        in_channel = 0, channel = 0, n_res_block = 0, n_res_channel = 0, embed_dim = 0, n_embed = 0, decay = 0.0, n_dims = 0, input_size = 0,
        batch_size = 0, learning_rate = 0.0, num_epochs = 0
    )
    
    hypers['batch_size'] = 4
    
    hypers['likelihood'] = 'laplace' # it may be normal
    
    hypers['scale'] = 0.05
    
    hypers['adjoint'] = True
    
    hypers['adaptive'] = True
    
    hypers['method'] = 'euler' # 'euler', 'milstein', 'srk'
    
    hypers['dt'] = 1e-2
    
    hypers['rtol'] = 1e-3
    
    hypers['atol'] = 1e-3
    
    # ========================
    
    
    return hypers