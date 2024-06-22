import ml_collections


def get_default_configs(device,ckpt,snr_x,scale_eps_x,saved_name=None,predictor='Euler',corrector='None'):
    config = ml_collections.ConfigDict()
    config.seed = 1
    config.device = device
    config.ckpt = ckpt
    if saved_name == None:
        config.saved_name = ckpt
    else:
        config.saved_name = saved_name
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.no_wandb = True
    wandb.wandb_usr = 'elma'
    wandb.online = False
    wandb.project = 'QM9Sample'

    # config.model = model = ml_collections.ConfigDict()
    # model.manifold = 'PoincareBall'
    # model.c = 0.1

    config.data = data = ml_collections.ConfigDict()
    data.data = 'QM9'  # help='qm9_config | qm9_second_half (train only on the last 50K samples of the training dataset)'
    data.dir = 'data/'
    data.max_feat_num = 4
    data.max_node_num = 9
    data.batch_size = 1024

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.predictor = predictor  # Reverse Euler
    sampler.corrector = corrector  # None Langevin
    sampler.n_steps = 1
    sampler.snr_x = snr_x
    sampler.scale_eps_x = scale_eps_x
    # sampler.snr_x = 0.1
    # sampler.scale_eps_x = 0.8
    sampler.snr_A = snr_x
    sampler.scale_eps_A = scale_eps_x
    # sampler.snr_A = 0.4
    # sampler.scale_eps_A = 0.8

    config.sample = sample = ml_collections.ConfigDict()
    sample.probability_flow = False
    sample.noise_removal = True
    sample.eps = 1e-5
    sample.use_ema = False
    sample.seed = 1

    # config.exp_name = f'[{config.ckpt}]_snr_x={config.sampler.snr_x}_scale_eps_x={config.sampler.scale_eps_x}'
    config.exp_name = f'[{config.ckpt}]'    # 验证时使用同一个log

    return config

def get_config(device,ckpt,snr_x,scale_eps_x,saved_name=None,predictor='Euler',corrector='None'):
    config = get_default_configs(device,ckpt,float(snr_x),float(scale_eps_x),saved_name,predictor,corrector)
    return config