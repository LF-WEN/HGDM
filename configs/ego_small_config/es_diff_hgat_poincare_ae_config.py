from configs.default_ego_small_config import get_default_configs
def get_config():
    config = get_default_configs()

    config.train.lr = 1e-2
    config.device = 'cuda:1'
    config.ae = 'ae'
    config.ae_path = 'checkpoints/'+config.data.data+'/' + config.ae + '/' + config.ae + '.pth'
    wandb = config.wandb
    wandb.no_wandb = True
    wandb.online = False

    model = config.model
    model.GCN_type = 'HGAT'  # HGCN HGAT
    model.manifold = 'PoincareBall'
    model.c = 1e-2
    x = config.sde.x
    x.type = 'VE'
    x.beta_min = 0.1
    x.beta_max = 4.

    config.exp_name = f'Test_repair_diff_hgat_c={config.model.c}_x.sde={config.sde.x.type}_beta=[{config.sde.x.beta_min},{config.sde.x.beta_max}]_' \
                      f'adj.sde={config.sde.adj.type}_beta=[{config.sde.adj.beta_min},{config.sde.adj.beta_max}]_ae={config.ae}'
    return config