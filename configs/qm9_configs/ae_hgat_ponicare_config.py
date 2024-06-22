from configs.default_qm9_config import get_default_configs
def get_config():
    config = get_default_configs()
    config.device = 'cuda:1'
    config.model_type = 'ae'

    wandb = config.wandb
    wandb.no_wandb = True

    model = config.model
    model.edge_dim = 1
    model.use_centroid = True
    model.model = 'HGCN'    #help='GCN,HGCN'
    model.layer_type = 'HGAT'    #help='HGCN,HGCNv1,HGAT'
    model.manifold ='PoincareBall'  #help='Euclidean, Lorentz, PoincareBall'
    model.use_centroidDec = True
    model.c = 0.01
    model.learnable_c = False
    model.pred_edge = True
    config.exp_name = f'ae'
    return config


