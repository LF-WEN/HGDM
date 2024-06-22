from configs.default_zinc_config import get_default_configs
def get_config():
    config = get_default_configs()
    config.wandb.project = 'ZINC_ae'
    config.device = 'cuda:0'
    config.model_type = 'ae'

    wandb = config.wandb
    wandb.no_wandb = True

    model = config.model
    model.edge_dim = 1
    model.use_centroid = True
    model.model = 'HGCN'    #help='GCN,HGCN'
    model.use_centroidDec = False
    model.layer_type = 'HGAT'    #help='HGCN,HGAT'
    model.manifold ='PoincareBall'  #help='Euclidean, Lorentz, PoincareBall'
    model.c = 1e-2
    model.learnable_c = False
    model.pred_edge = True

    config.exp_name = f'ae'
    return config


