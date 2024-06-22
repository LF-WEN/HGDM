import geoopt.optim
import ml_collections
import torch
import random
import numpy as np

import synthetic.model
from models.HVAE import HVAE
from models.ScoreNetwork_A import ScoreNetworkA, ScoreNetworkA_poincare, HScoreNetworkA
from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH, ScoreNetworkX_poincare
from sde_graph_lib import VPSDE, VESDE, subVPSDE

from losses import get_sde_loss_fn
from solver import get_pc_sampler, S4_solver
from evaluation.mmd import gaussian, gaussian_emd
from utils.ema import ExponentialMovingAverage


def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def load_device(config):
    # if torch.cuda.is_available():
    #     device = list(range(torch.cuda.device_count()))
    # else:
    #     device = 'cpu'
    device = torch.device(config.device)
    return device


def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX(**params_)
    elif model_type == 'ScoreNetworkX_GMH':
        model = ScoreNetworkX_GMH(**params_)
    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA(**params_)
    elif model_type == 'ScoreNetworkX_poincare':
        model = ScoreNetworkX_poincare(**params_)
    elif model_type == 'ScoreNetworkA_poincare':
        model = ScoreNetworkA_poincare(**params_)
    elif model_type == 'HScoreNetworkA':
        model = HScoreNetworkA(**params_)
    elif model_type == 'ae':
        model = HVAE(ml_collections.ConfigDict(params))
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config, device):
    config_train = config.train
    model = load_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    else:
        model = model.to(device)
    if config.model.manifold == 'Euclidean':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config_train.lr,
                                    weight_decay=config_train.weight_decay)
    else:
        optimizer = geoopt.optim.RiemannianAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=config_train.lr,
                                     weight_decay=config_train.weight_decay)
        # optimizer = geoopt.optim.RiemannianSGD(filter(lambda p: p.requires_grad, model.parameters()),
        #                                         lr=config_train.lr,
        #                                         weight_decay=config_train.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
        #                                         lr=config_train.lr,
        #                                         weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, get_graph_list=False):
    if config.data.data in ['QM9', 'ZINC250k']:
        from utils.data_loader_mol import dataloader
        return dataloader(config, get_graph_list)
    # elif config.data.data == 'synthetic_lobster_data':
    #     import pickle
    #     from torch.utils.data import DataLoader
    #     from synthetic.graph_datasets import SyntheticDataset
    #     from synthetic.collate import collate_fn
    #     with open('data/synthetic_lobster_data.pkl', 'rb') as f:
    #         data = pickle.load(f)
    #     config.data.max_feat_num = data[0]['node_feat'].size(-1)
    #     test_size = int(config.data.test_split * len(data))
    #
    #
    #     train_dataset = SyntheticDataset(data[test_size:])
    #     test_dataset = SyntheticDataset(data[:test_size])
    #
    #     train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size,collate_fn=collate_fn)
    #     test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size,collate_fn=collate_fn)
    #     return train_dataloader,test_dataloader
    else:
        from utils.data_loader import dataloader
        return dataloader(config, get_graph_list)


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    return x_b, adj_b


def load_sde(config_sde,manifold=None):
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == 'VP':
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales,manifold=manifold)
    elif sde_type == 'VE':
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales,manifold=manifold)
    elif sde_type == 'subVP':
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales,manifold=manifold)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")
    return sde


def load_loss_fn(config,manifold=None,encoder=None):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x,manifold)
    sde_adj = load_sde(config.sde.adj)
    
    loss_fn = get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=reduce_mean, continuous=True, 
                                likelihood_weighting=False, eps=config.train.eps,manifold=manifold,encoder=encoder)
    return loss_fn


def load_sampling_fn(config_train, config_module, config_sample, device,manifold,batch_size=None):
    if batch_size is None:
        batch_size = 10000
    sde_x = load_sde(config_train.sde.x,manifold)
    sde_adj = load_sde(config_train.sde.adj)
    max_node_num  = config_train.data.max_node_num

    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device


    get_sampler = get_pc_sampler

    if config_train.data.data in ['QM9', 'ZINC250k']:
        shape_x = (batch_size, max_node_num, config_train.data.max_feat_num)
        shape_adj = (batch_size, max_node_num, max_node_num)
    else:
        shape_x = (config_train.data.batch_size, max_node_num, config_train.data.max_feat_num)
        shape_adj = (config_train.data.batch_size, max_node_num, max_node_num)
        
    sampling_fn = get_sampler(sde_x=sde_x, sde_adj=sde_adj, shape_x=shape_x, shape_adj=shape_adj, 
                                predictor=config_module.predictor, corrector=config_module.corrector,
                                probability_flow=config_sample.probability_flow, 
                                continuous=True, denoise=config_sample.noise_removal, 
                                eps=config_sample.eps, device=device_id,config_module=config_module)
    return sampling_fn


def load_model_params(config,manifold=None):
    config_m = config.model
    max_feat_num = config.data.max_feat_num

    if 'GMH' in config_m.x:
        params_x = {'model_type': config_m.x, 'max_feat_num': max_feat_num, 'depth': config_m.depth, 
                    'nhid': config_m.nhid, 'num_linears': config_m.num_linears,
                    'c_init': config_m.c_init, 'c_hid': config_m.c_hid, 'c_final': config_m.c_final, 
                    'adim': config_m.adim, 'num_heads': config_m.num_heads, 'conv':config_m.conv}
    elif 'poincare' in config_m.x:
        params_x = {'model_type': config_m.x, 'max_feat_num': max_feat_num, 'depth': config_m.depth,
                    'nhid': config_m.nhid,'manifold':manifold,'edge_dim':config_m.edge_dim,
                    'GCN_type':config_m.GCN_type}
    else:
        params_x = {'model_type':config_m.x, 'max_feat_num':max_feat_num, 'depth':config_m.depth, 'nhid':config_m.nhid}

    if 'poincare' in config_m.adj:
        params_adj = {'model_type': config_m.adj, 'max_feat_num': max_feat_num,'max_node_num':config.data.max_node_num,
                      'nhid': config_m.nhid, 'num_layers': config_m.num_layers, 'num_linears': config_m.num_linears,
                      'c_init': config_m.c_init, 'c_hid': config_m.c_hid, 'c_final': config_m.c_final,
                      'adim': config_m.adim, 'num_heads': config_m.num_heads, 'conv': config_m.conv,
                       'manifold': manifold}
    elif 'HScoreNetworkA' == config_m.adj:
        params_adj = {'model_type': config_m.adj, 'max_feat_num': max_feat_num,'max_node_num':config.data.max_node_num,
                      'nhid': config_m.nhid, 'num_layers': config_m.num_layers, 'num_linears': config_m.num_linears,
                      'c_init': config_m.c_init, 'c_hid': config_m.c_hid, 'c_final': config_m.c_final,
                      'adim': config_m.adim, 'num_heads': config_m.num_heads, 'conv': config_m.conv,
                      'manifold': manifold}
    else:
        params_adj = {'model_type':config_m.adj, 'max_feat_num':max_feat_num, 'max_node_num':config.data.max_node_num,
                        'nhid':config_m.nhid, 'num_layers':config_m.num_layers, 'num_linears':config_m.num_linears,
                        'c_init':config_m.c_init, 'c_hid':config_m.c_hid, 'c_final':config_m.c_final,
                        'adim':config_m.adim, 'num_heads':config_m.num_heads, 'conv':config_m.conv}
    return params_x, params_adj


def load_ckpt(config, device, ts=None, return_ckpt=False):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    if ts is not None:
        config.ckpt = ts
    path = f'./checkpoints/{config.data.data}/{config.ckpt}/{config.saved_name}.pth'
    ckpt = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    ckpt_dict= {'config': ckpt['model_config'], 'params_x': ckpt['params_x'], 'x_state_dict': ckpt['x_state_dict'],
                'params_adj': ckpt['params_adj'], 'adj_state_dict': ckpt['adj_state_dict']}
    if config.sample.use_ema:
        ckpt_dict['ema_x'] = ckpt['ema_x']
        ckpt_dict['ema_adj'] = ckpt['ema_adj']
    if return_ckpt:
        ckpt_dict['ckpt'] = ckpt
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    else:
        model = model.to(device)
    return model


def load_eval_settings(data, orbit_on=True):
    # Settings for generic graph generation
    methods = ['degree', 'cluster', 'orbit', 'spectral'] 
    kernels = {'degree':gaussian_emd, 
                'cluster':gaussian_emd, 
                'orbit':gaussian,
                'spectral':gaussian_emd}
    return methods, kernels
