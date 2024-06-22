import ml_collections


def get_default_configs():
    config = ml_collections.ConfigDict()
    config.seed = 1

    config.train = train = ml_collections.ConfigDict()
    train.ema = 0.999
    train.weight_decay = 1e-4
    train.num_epochs = 300

    train.lr = 5e-3
    train.lr_schedule = True
    train.lr_decay = 0.999
    train.reduce_mean = True
    train.eps = 1.0e-5
    train.grad_norm = 1.0
    train.print_interval = 1
    train.save_interval = 20
    train.kl_regularization = 1e-2

    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.no_wandb = False
    wandb.wandb_usr = 'elma'
    wandb.online = True
    wandb.project = 'QM9'

    config.model = model = ml_collections.ConfigDict()
    model.normalization_factor = 1
    model.aggregation_method = 'sum'
    model.msg_transform = True
    model.sum_transform = True
    model.use_norm = 'ln'  # 'none,ln'
    model.dropout = 0.
    model.dim = 4
    model.hidden_dim = 16
    model.enc_layers = 3
    model.dec_layers = 3
    model.act = 'LeakyReLU'  # LeakyReLU ReLU
    model.edge_dim = 1
    model.use_centroid = False

    model.x = 'ScoreNetworkX_poincare'
    model.adj = 'ScoreNetworkA_poincare'
    model.conv = 'GCN'
    model.num_heads = 4
    model.depth = 2
    model.adim = 16
    model.nhid = 16
    model.num_layers = 3
    model.num_linears = 3
    model.c_init = 2
    model.c_hid = 8
    model.c_final = 4

    config.sde = sde = ml_collections.ConfigDict()
    sde.x = x = ml_collections.ConfigDict()
    x.type = 'VE'
    x.beta_min = 0.1
    x.beta_max = 1.
    x.num_scales = 1000
    sde.adj = adj = ml_collections.ConfigDict()
    adj.type = 'VE'
    adj.beta_min = 0.1
    adj.beta_max = 1.0
    adj.num_scales = 1000

    config.data = data = ml_collections.ConfigDict()
    data.data = 'QM9'  # help='qm9_config | qm9_second_half (train only on the last 50K samples of the training dataset)'
    data.dir = 'data/'
    data.filter_n_atoms = None
    data.data_augmentation = True
    data.conditioning = []  # nargs='+', help='arguments : homo | lumo | alpha | gap | mu | Cv')
    data.remove_h = False
    data.include_charges = True
    data.init = 'atom'
    data.max_feat_num = 4
    data.max_node_num = 9
    data.batch_size = 1024

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.predictor = 'Reverse'  # Reverse Euler
    sampler.corrector = 'Langevin'  # None Langevin
    sampler.n_steps = 1
    sampler.snr = 0.1
    sampler.scale_eps = 0.7

    config.sample = sample = ml_collections.ConfigDict()
    sample.probability_flow= False
    sample.noise_removal= True
    sample.eps=1e-5
    sample.use_ema = False
    sample.seed = 1
    return config
