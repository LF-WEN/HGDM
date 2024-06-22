from configs.default_qm9_config import get_default_configs
def get_config():
    config = get_default_configs()

    config.device = 'cuda:1'
    config.ae = 'ae'
    config.ae_path = 'checkpoints/QM9/'+config.ae+'/'+config.ae+'.pth'

    wandb = config.wandb
    wandb.no_wandb = True
    wandb.online = False

    model = config.model
    model.GCN_type = 'HGAT'  # HGCN HGAT
    model.manifold ='PoincareBall'

    x = config.sde.x
    x.type = 'VP'
    x.beta_min = 0.1
    x.beta_max = 2.


    config.exp_name = f'score_model'

    return config