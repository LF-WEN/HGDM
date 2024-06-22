from configs.default_ego_small_config import get_default_configs
def get_config():
    config = get_default_configs()
    config.wandb.project = 'ego_small_ae'
    config.device = 'cuda:1'
    config.model_type = 'ae'

    wandb = config.wandb
    wandb.no_wandb = True

    model = config.model
    model.edge_dim = 1
    model.use_centroid = True
    model.model = 'HGCN'    #help='GCN,HGCN'
    model.layer_type = 'HGAT'    #help='HGCN,HGAT'
    model.manifold ='PoincareBall'  #help='Euclidean, Lorentz, PoincareBall'
    model.use_centroidDec = True
    model.c = 1e-2
    model.learnable_c = False
    model.pred_edge = True
    config.exp_name = f'ae'
    return config