import ml_collections


def get_default_configs():
    config = ml_collections.ConfigDict()
    config.seed = 1

    config.train = train = ml_collections.ConfigDict()
    train.ema = 0.999
    train.weight_decay = 1e-4
    train.num_epochs = 5000

    train.lr = 1e-2
    train.lr_schedule = True
    train.lr_decay = 0.999
    train.reduce_mean = True
    train.eps = 1.0e-5
    train.grad_norm = 1.0
    train.print_interval = 100
    train.save_interval = 500
    train.kl_regularization = 1e-5

    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.no_wandb = True
    wandb.wandb_usr = 'elma'
    wandb.online = True
    wandb.project = 'ego_small'

    config.model = model = ml_collections.ConfigDict()
    model.normalization_factor = 1
    model.aggregation_method = 'sum'
    model.msg_transform = True
    model.sum_transform = True
    model.use_norm = 'ln'  # 'none,ln'
    model.dropout = 0.
    model.dim = 17
    model.hidden_dim = 32
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
    model.adim = 32
    model.nhid = 32
    model.num_layers = 5
    model.num_linears = 2
    model.c_init = 2
    model.c_hid = 8
    model.c_final = 4

    config.sde = sde = ml_collections.ConfigDict()
    sde.x = x = ml_collections.ConfigDict()
    x.type = 'VE'
    x.beta_min = 0.01
    x.beta_max = 20.
    x.num_scales = 1000
    sde.adj = adj = ml_collections.ConfigDict()
    adj.type = 'VP'
    adj.beta_min = 0.1
    adj.beta_max = 1.0
    adj.num_scales = 1000
    # config.sde = sde = ml_collections.ConfigDict()
    # sde.x = x = ml_collections.ConfigDict()
    # x.type = 'VP'
    # x.beta_min = 0.1
    # x.beta_max = 20.0
    # x.num_scales = 1000
    # sde.adj = adj = ml_collections.ConfigDict()
    # adj.type = 'VP'
    # adj.beta_min = 0.1
    # adj.beta_max = 20.0
    # adj.num_scales = 1000

    config.data = data = ml_collections.ConfigDict()
    data.data = 'ego_small'  #
    data.dir = 'data/'
    data.init = 'deg'
    data.max_feat_num = 17
    data.max_node_num = 18
    data.batch_size = 128
    data.test_split = 0.2

    return config

def get_config():
    return get_default_configs()
