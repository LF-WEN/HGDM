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
    wandb.online = 'True'
    wandb.project = 'Community_small_Sample'


    config.data = data = ml_collections.ConfigDict()
    data.data = 'community_small'  # help='qm9_config | qm9_second_half (train only on the last 50K samples of the training dataset)'
    data.dir = 'data/'
    data.init = 'deg'
    data.max_feat_num = 10
    data.max_node_num = 20
    data.batch_size = 128
    data.test_split = 0.2

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.predictor = predictor  # Reverse Euler
    sampler.corrector = corrector  # None Langevin
    sampler.n_steps = 1
    sampler.snr_x = snr_x
    sampler.scale_eps_x = scale_eps_x
    # sampler.snr_A = 0.05
    # sampler.scale_eps_A = 0.7
    sampler.snr_A = snr_x
    sampler.scale_eps_A = scale_eps_x

    config.sample = sample = ml_collections.ConfigDict()
    sample.probability_flow= False
    sample.noise_removal= True
    sample.eps=1e-4
    sample.use_ema = False
    sample.seed = 1
    config.exp_name = f'[{config.ckpt}]_snr_x={config.sampler.snr_x}_scale_eps_x={config.sampler.scale_eps_x}'
    return config

def get_config(device,ckpt,snr_x,scale_eps_x,saved_name=None,predictor='Euler',corrector='None'):
    return get_default_configs(device,ckpt,float(snr_x),float(scale_eps_x),saved_name,predictor,corrector)
