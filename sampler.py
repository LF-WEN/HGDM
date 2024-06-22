import os
import time
import pickle
import math

import geoopt
import ml_collections
import numpy as np
import torch
import wandb
from models.HVAE import HVAE

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, \
    load_ema_from_ckpt, load_sampling_fn, load_eval_settings
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from moses.metrics.metrics import get_all_metrics
import torch.nn.functional as F

# -------- Sampler for generic graph generation tasks --------
class Sampler(object):
    def __init__(self, config):
        super(Sampler, self).__init__()

        self.config = config
        self.device = load_device(config)

    def sample(self,independent=True):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'],
                                              self.device)
        if independent:
            if self.config.wandb.no_wandb:
                mode = 'disabled'
            else:
                mode = 'online' if self.config.wandb.online else 'offline'
            kwargs = {'entity': self.config.wandb.wandb_usr, 'name': self.config.exp_name,
                      'project': self.config.wandb.project,'config': self.config.to_dict(),
                      'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
            wandb.init(**kwargs)

        if hasattr(self.model_x,'manifold'):
            manifold = self.model_x.manifold
        else:
            manifold = None
        print('manifold:', manifold)
        if manifold is not None:
            print('k=:', manifold.k)

        load_seed(self.configt.seed)
        self.train_graph_list, self.test_graph_list = load_data(self.configt, get_graph_list=True)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        # self.log_name = f"{self.config.exp_name}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.log_name}')
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(self.model_x, self.ckpt_dict['ema_x'], self.configt.train.ema)
            self.ema_adj = load_ema_from_ckpt(self.model_adj, self.ckpt_dict['ema_adj'], self.configt.train.ema)

            self.ema_x.copy_to(self.model_x.parameters())   # ema 模型参数复制过去
            self.ema_adj.copy_to(self.model_adj.parameters())

        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device,manifold)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        num_sampling_rounds = math.ceil(len(self.test_graph_list) / self.configt.data.batch_size)
        gen_graph_list = []
        for r in range(num_sampling_rounds):
            t_start = time.time()

            self.init_flags = init_flags(self.train_graph_list, self.configt).to(self.device)

            x, adj = self.sampling_fn(self.model_x, self.model_adj, self.init_flags)

            logger.log(f"Round {r} : {time.time() - t_start:.2f}s")

            samples_int = quantize(adj)
            gen_graph_list.extend(adjs_to_graphs(samples_int, True))

        gen_graph_list = gen_graph_list[:len(self.test_graph_list)]
        # -------- Evaluation --------
        methods, kernels = load_eval_settings(self.config.data.data)
        result_dict = eval_graph_list(self.test_graph_list, gen_graph_list, methods=methods, kernels=kernels)
        result_dict['mean'] = (result_dict['degree']+result_dict['cluster']+result_dict['orbit'])/3
        logger.log(f'MMD_full {result_dict}'
                   f'\n{self.config.sampler.predictor}-{self.config.sampler.corrector}-'
                   f'X:{self.config.sampler.snr_x}-{self.config.sampler.scale_eps_x} A:{self.config.sampler.snr_A}-{self.config.sampler.scale_eps_A}'
                   f'\n{self.config.saved_name}', verbose=False)
        logger.log('=' * 100)
        if independent:
            wandb.log(result_dict, commit=True)
        # -------- Save samples --------
        save_dir = save_graph_list(self.log_folder_name, self.log_name, gen_graph_list)
        with open(save_dir, 'rb') as f:
            sample_graph_list = pickle.load(f)
        # plot_graphs_list(graphs=sample_graph_list, title=f'{self.config.ckpt}', max_num=16,
        #                  save_dir=self.log_folder_name)
        plot_graphs_list(graphs=sample_graph_list, title=f'snr={self.config.sampler.snr_x}_scale={self.config.sampler.scale_eps_x}.png', max_num=16,
                         save_dir=self.log_folder_name)
        return {"degree": result_dict['degree'], 'cluster': result_dict['cluster'],
                       'orbit': result_dict['orbit'],'mean': result_dict['mean']}


# -------- Sampler for molecule generation tasks --------
class Sampler_mol(object):
    def __init__(self, config):
        self.config = config
        self.device = load_device(config)

    def sample(self,independent=True):

        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'],
                                              self.device)
        if self.config.wandb.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online' if self.config.wandb.online else 'offline'

        if independent:
            kwargs = {'entity': self.config.wandb.wandb_usr, 'name': self.config.exp_name,
                      'project': self.config.wandb.project,'config': self.config.to_dict(),
                      'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
            wandb.init(**kwargs)

        if not hasattr(self.configt,'ae_path') or self.configt.ae_path is None:
            Decoder = None
        else:
            checkpoint = torch.load(self.configt.ae_path,map_location=self.device)
            AE_state_dict = checkpoint['ae_state_dict']
            AE_config = ml_collections.ConfigDict(checkpoint['model_config'])
            AE_config.model.dropout = 0
            ae = HVAE(AE_config)
            ae.load_state_dict(AE_state_dict,strict=False)
            for name, param in ae.named_parameters():
                if "encoder" in name or 'decoder' in name:
                    param.requires_grad = False
            Decoder = ae.decoder.to(self.device)
            # if Decoder.manifolds is not None:
            #     manifold = Decoder.manifolds[0]
            #     print(manifold, manifold.k)
            # else:
            #     manifold = None
        if hasattr(self.model_x,'manifold'):
            manifold = self.model_x.manifold
        else:
            manifold = None
        # print('manifold:', manifold)
        # if manifold is not None:
        #     print('k=:', manifold.k)


        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.exp_name}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        if self.config.data.data == 'ZINC250k':
            n_samples = 10000
            batch_size = 5000
        else:
            n_samples = 10000
            batch_size = 10000

        n_iter = int(n_samples/batch_size)
        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device,
                                            manifold,batch_size)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        train_smiles, test_smiles = load_smiles(self.configt.data.data)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)

        self.train_graph_list, _ = load_data(self.configt, get_graph_list=True)  # for init_flags
        with open(f'data/{self.configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
            self.test_graph_list = pickle.load(f)  # for NSPDK MMD

        num_mols=0
        num_mols_wo_correction_total = 0
        gen_smiles_len = 0
        metric_total = {
            'valid':0.,
            'unique':0.,
            'FCD/Test':0.,
            'Novelty':0.,
            'NSPDK MMD':0.
        }
        for r in range(n_iter):
            self.init_flags = init_flags(self.train_graph_list, self.configt, batch_size).to(self.device)
            x, adj = self.sampling_fn(self.model_x, self.model_adj, self.init_flags)
            adj = torch.where(torch.isnan(adj), torch.zeros_like(adj), adj)
            adj = quantize_mol(adj)
            samples_int = np.array(adj.to(torch.int64))

            if Decoder is None:
                x = torch.where(x > 0.5, 1, 0)
            else:
                x = Decoder(x, adj.to(x.device), self.init_flags.unsqueeze(-1))
                x = torch.argmax(x,dim=-1)
                x = F.one_hot(x, num_classes=self.config.data.max_feat_num)
            x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)  # 32, 9, 4 -> 32, 9, 5



            samples_int = samples_int - 1
            samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2
            adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
            gen_mols, num_mols_wo_correction = gen_mol(x, adj, self.configt.data.data)
            num_mols_wo_correction_total += num_mols_wo_correction
            num_mols += len(gen_mols)

            gen_smiles = mols_to_smiles(gen_mols)
            gen_smiles = [smi for smi in gen_smiles if len(smi)]
            gen_smiles_len += len(gen_smiles)
            # -------- Save generated molecules --------
            with open(os.path.join(self.log_dir, f'{self.log_name}.txt'), 'a') as f:
                for smiles in gen_smiles:
                    f.write(f'{smiles}\n')

            # -------- Evaluation --------
            scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device, n_jobs=8, test=test_smiles,
                                     train=train_smiles)

            for metric in ['valid', 'FCD/Test', 'Novelty']:
                metric_total[metric] += scores[metric]
            metric_total['unique'] += scores[f'unique@{len(gen_smiles)}']

            scores_nspdk = eval_graph_list(self.test_graph_list, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
            metric_total['NSPDK MMD'] += scores_nspdk

        for k in metric_total:
            metric_total[k] = metric_total[k]/n_iter
        metric_total[f'unique@{gen_smiles_len}'] = metric_total['unique']

        logger.log(f'Number of molecules: {num_mols}')
        logger.log(f'validity w/o correction: {num_mols_wo_correction_total / num_mols}')
        for metric in ['valid', f'unique@{gen_smiles_len}', 'FCD/Test', 'Novelty','NSPDK MMD']:
            logger.log(f'{metric}: {metric_total[metric]}')
        logger.log(f'\n{self.config.saved_name}\n{self.config.sampler.predictor}-{self.config.sampler.corrector}-'
                   f'X:{self.config.sampler.snr_x}-{self.config.sampler.scale_eps_x} A:{self.config.sampler.snr_A}-{self.config.sampler.scale_eps_A}',
                   verbose=False)

        logger.log('=' * 100)
        if independent:
            wandb.log({"Number of molecules": num_mols, 'validity w/o correction': num_mols_wo_correction_total / num_mols,
                       'valid': metric_total['valid'], f'unique@{gen_smiles_len}': metric_total[f'unique@{gen_smiles_len}'],
                       'FCD/Test': metric_total['FCD/Test'], 'Novelty': metric_total['Novelty'],
                       'NSPDK MMD':metric_total['NSPDK MMD']}, commit=True)
        else:
            return {"Number of molecules": num_mols, 'validity w/o correction': num_mols_wo_correction_total / num_mols,
                       'valid': metric_total['valid'], f'unique@{gen_smiles_len}': metric_total[f'unique@{gen_smiles_len}'],
                       'FCD/Test': metric_total['FCD/Test'], 'Novelty': metric_total['Novelty'],
                       'NSPDK MMD':metric_total['NSPDK MMD']}
