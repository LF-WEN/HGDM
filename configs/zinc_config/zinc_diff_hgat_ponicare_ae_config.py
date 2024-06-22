from configs.default_zinc_config import get_default_configs
def get_config():
    config = get_default_configs()
    config.device = 'cuda:0'
    config.ae = 'ae'
    config.ae_path = 'checkpoints/ZINC250k/'+config.ae+'/'+config.ae+'.pth'

    wandb = config.wandb
    wandb.no_wandb = True

    model = config.model

    model.GCN_type = 'HGAT'
    model.manifold ='PoincareBall'
    x = config.sde.x
    x.type = 'VP'
    x.beta_min = 0.1
    x.beta_max = 1.

    config.exp_name = f'score_model'
    return config