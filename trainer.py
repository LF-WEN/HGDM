import os
import time

import geoopt
import ml_collections

import configs
import wandb
from tqdm import tqdm, trange
import numpy as np
import torch

from models.HVAE import HVAE
from sampler import Sampler_mol, Sampler
from utils.graph_utils import node_flags
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
                         load_ema, load_loss_fn, load_batch
from utils.logger import Logger, set_log, start_log, train_log


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device(config)
        self.train_loader, self.test_loader = load_data(self.config)

    def train_ae(self, ts=None):
        if ts is not None:
            self.config.exp_name = ts
        else:
            ts = self.config.exp_name
        if self.config.wandb.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online' if self.config.wandb.online else 'offline'
        kwargs = {'entity': self.config.wandb.wandb_usr, 'name': self.config.exp_name, 'project': self.config.wandb.project,
                  'config': self.config.to_dict(),
                  'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
        wandb.init(**kwargs)

        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # -------- Load models, optimizers, ema --------
        self.model, self.optimizer, self.scheduler = load_model_optimizer(self.config.to_dict(), self.config,
                                                                                self.device)

        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        # self.ema_model = load_ema(self.model, decay=self.config.train.ema)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)
        # -------- Training --------
        best_mean_test_loss = 1e10

        for epoch in trange(0, (self.config.train.num_epochs), desc='[Epoch]', position=1, leave=False):

            self.total_train_loss = []
            self.total_test_loss = []
            self.test_kl_loss = []
            self.test_edge_loss = []
            self.test_rec_loss = []
            t_start = time.time()

            self.model.train()

            for step, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, adj = load_batch(batch, self.device)
                loss,kl_loss,edge_loss = self.model(x,adj)
                loss = loss+self.config.train.kl_regularization*kl_loss+1e-2*edge_loss
                if torch.isnan(loss):
                    raise ValueError
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)

                self.optimizer.step()

                # -------- EMA update --------
                # self.ema_model.update(self.model.parameters())
                self.total_train_loss.append(loss.item())

            if self.config.train.lr_schedule:
                self.scheduler.step()

            self.model.eval()
            for _, test_batch in enumerate(self.test_loader):
                x, adj = load_batch(test_batch, self.device)

                with torch.no_grad():
                    # self.ema_model.store(self.model.parameters())
                    # self.ema_model.copy_to(self.model.parameters())

                    rec_loss,kl_loss,edge_loss = self.model(x,adj)
                    loss = rec_loss+self.config.train.kl_regularization*kl_loss+1e-2*edge_loss
                    self.total_test_loss.append(loss.item())
                    self.test_rec_loss.append(rec_loss.item())
                    self.test_kl_loss.append(kl_loss.item())
                    self.test_edge_loss.append(edge_loss.item())

                    # self.ema_model.restore(self.model.parameters())

            mean_total_train_loss = np.mean(self.total_train_loss)
            mean_total_test_loss = np.mean(self.total_test_loss)
            mean_test_rec_loss = np.mean(self.test_rec_loss)
            mean_test_kl_loss = np.mean(self.test_kl_loss)
            mean_test_edge_loss = np.mean(self.test_edge_loss)

            if self.config.model.model == 'HGCN' and self.config.model.learnable_c:
                self.model.show_curvatures()
            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval - 1 and best_mean_test_loss>mean_total_test_loss:

                best_mean_test_loss = mean_total_test_loss
                torch.save({
                    'model_config': self.config.to_dict(),
                    'ae_state_dict': self.model.state_dict(),
                    # 'ema_ae': self.ema_model.state_dict(),
                }, f'./checkpoints/{self.config.data.data}/{self.config.exp_name}/{self.ckpt}.pth')
            wandb.log({"total_test_loss": mean_total_test_loss,'total_train_loss': mean_total_train_loss,'test_edge_loss':mean_test_edge_loss,
                       'test_kl_loss': mean_test_kl_loss,'test_rec_loss':mean_test_rec_loss}, commit=True)
            if epoch % self.config.train.print_interval == self.config.train.print_interval - 1:
                logger.log(f'{epoch + 1:03d} | {time.time() - t_start:.2f}s | '
                           f'total train loss: {mean_total_train_loss:.3e} | '
                           f'total test loss: {mean_total_test_loss:.3e} | '
                           f'test rec loss:{mean_test_rec_loss:.3e} | '
                           f'test kl loss:{mean_test_kl_loss:.3e} | '
                           f'test edge loss:{mean_test_edge_loss:.3e} |', verbose=False)
        print(' ')
        return self.ckpt

    def train_sde(self, ts=None):
        if ts is not None:
            self.config.exp_name = ts
        else:
            ts = self.config.exp_name
        if self.config.ae_path is None:
            Encoder = None
            if self.config.model.manifold == 'Euclidean':
                manifold = None
            else:
                manifold_class = {'PoincareBall': geoopt.PoincareBall(self.config.model.c),
                                  'Lorentz': geoopt.Lorentz(self.config.model.c),
                                  }
                manifold = manifold_class[self.config.model.manifold]
        else:
            checkpoint = torch.load(self.config.ae_path,map_location=self.config.device)
            AE_state_dict = checkpoint['ae_state_dict']
            AE_config = ml_collections.ConfigDict(checkpoint['model_config'])
            AE_config.model.dropout = 0
            # AE_config.model.pred_edge = True
            # torch.save({
            #     'model_config': AE_config.to_dict(),
            #     'ae_state_dict': AE_state_dict,
            #     'ema_ae': checkpoint['ema_ae'],
            # }, f'./checkpoints/{AE_config.data.data}/{AE_config.exp_name}/{AE_config.exp_name}.pth')
            # exit(0)
            ae = HVAE(AE_config)
            ae.load_state_dict(AE_state_dict,strict=False)
            for name, param in ae.named_parameters():
                if "encoder" in name or 'decoder' in name:
                    param.requires_grad = False
            Encoder = ae.encoder.to(self.device)
            manifold = Encoder.manifold

        self.params_x, self.params_adj = load_model_params(self.config,manifold=manifold)

        if self.config.wandb.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online' if self.config.wandb.online else 'offline'
        kwargs = {'entity': self.config.wandb.wandb_usr, 'name': self.config.exp_name, 'project': self.config.wandb.project,
                  'config': self.config.to_dict(),
                  'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
        wandb.init(**kwargs)

        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # -------- Load models, optimizers, ema --------
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(self.params_x, self.config,self.device)
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(self.params_adj, self.config,
                                                                                        self.device)
        total = sum([param.nelement() for param in self.model_x.parameters()]+
                    [param.nelement() for param in self.model_adj.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        self.loss_fn = load_loss_fn(self.config,manifold=manifold,encoder=Encoder)

        # -------- Training --------
        for epoch in trange(0, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):

            self.train_x = []
            self.train_adj = []
            self.test_x = []
            self.test_adj = []
            t_start = time.time()

            self.model_x.train()
            self.model_adj.train()

            # list = []
            # for _, train_b in tqdm(enumerate(self.train_loader)):
            #     x, adj = load_batch(train_b, self.device)
            #     flags = node_flags(adj)
            #     posterior = Encoder(x, adj, flags)
            #     m = Encoder.manifold
            #     x = posterior.mode()    # (b,N,dim)
            #     dist0 = m.dist0(x,keepdim=True)      # (b,N,1)
            #     degree = adj.gt(1e-5).sum(-1,keepdim=True)    # (b,N,1)
            #     data_cat = torch.cat([dist0[degree>0][:,None],degree[degree>0][:,None]],dim=-1)
            #     list.append(data_cat.view(-1,2))   #(b,K,2)
            #     print(data_cat.shape)
            # data = torch.cat(list,dim=0)
            # torch.save(data,'geodesic_degree_enz.pt')
            # exit()
            # checkpoint = torch.load(self.config.ae_path, map_location=self.device)
            # AE_state_dict = checkpoint['ae_state_dict']
            # AE_config = ml_collections.ConfigDict(checkpoint['model_config'])
            # ae = HVAE(AE_config)
            # ae.load_state_dict(AE_state_dict, strict=False)
            # enc = ae.encoder.to(self.device)
            # list = []
            # for _, train_b in tqdm(enumerate(self.train_loader)):
            #     x, adj = load_batch(train_b, self.device)
            #     node_mask = node_flags(adj)
            #     emb = enc(x, adj, node_mask).mode()
            #     degree = adj.gt(1e-5).sum(-1, keepdim=True)
            #     temp = torch.cat([degree[degree>0][:,None], emb[(degree>0).squeeze()]],dim=-1)
            #     print(temp.shape)
            #     list.append(temp)
            # torch.save(torch.cat(list, dim=0), 'qm9_emb1.pt')
            # exit(0)  # Todo

            for _, train_b in enumerate(self.train_loader):
                x, adj = load_batch(train_b, self.device)
                loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, x, adj)
                if torch.isnan(loss_x):
                    raise ValueError('NaN')
                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()
                loss_x.backward()
                loss_adj.backward()
                torch.nn.utils.clip_grad_norm_(self.model_x.parameters(), self.config.train.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model_adj.parameters(), self.config.train.grad_norm)

                self.optimizer_x.step()
                self.optimizer_adj.step()

                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())
                self.ema_adj.update(self.model_adj.parameters())

                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())

            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()

            self.model_x.eval()
            self.model_adj.eval()
            for _, test_b in enumerate(self.test_loader):   
                
                x, adj = load_batch(test_b, self.device)
                loss_subject = (x, adj)

                with torch.no_grad():
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())

                    loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())

                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)

            # -------- Log losses --------
            if epoch % self.config.train.print_interval == self.config.train.print_interval - 1:
                logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                            f'test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | '
                            f'train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | ', verbose=False)
                wandb.log({"Test x": mean_test_x, 'test adj': mean_test_adj,
                           'train x': mean_train_x, 'train adj': mean_train_adj, 'epoch': epoch + 1}, commit=True)
            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''
                torch.save({ 
                    'model_config': self.config,
                    'params_x' : self.params_x,
                    'params_adj' : self.params_adj,
                    'x_state_dict': self.model_x.state_dict(), 
                    'adj_state_dict': self.model_adj.state_dict(),
                    'ema_x': self.ema_x.state_dict(),
                    'ema_adj': self.ema_adj.state_dict()
                    }, f'./checkpoints/{self.config.data.data}/{self.ckpt}/{self.ckpt + save_name}.pth')
                torch.save({
                    'model_config': self.config,
                    'params_x': self.params_x,
                    'params_adj': self.params_adj,
                    'x_state_dict': self.model_x.state_dict(),
                    'adj_state_dict': self.model_adj.state_dict(),
                    'ema_x': self.ema_x.state_dict(),
                    'ema_adj': self.ema_adj.state_dict()
                }, f'./checkpoints/{self.config.data.data}/{self.ckpt}/{self.ckpt}.pth')

                device, ckpt, snr_x, scale_eps_x = self.device, self.ckpt, 0.1, 1
                if self.config.data.data == 'QM9':
                    config = getattr(configs, 'default_qm9_sample').get_config(device,ckpt,snr_x,scale_eps_x)
                    eval_dict = Sampler_mol(config).sample(independent=False)
                elif self.config.data.data == 'ZINC250k':
                    config = getattr(configs, 'default_zinc_sample').get_config(device,ckpt,snr_x,scale_eps_x)
                    eval_dict = Sampler_mol(config).sample(independent=False)
                elif self.config.data.data == 'ego_small':
                    config = getattr(configs, 'default_ego_small_sample').get_config(device, ckpt, snr_x, scale_eps_x)
                    eval_dict = Sampler(config).sample(independent=False)
                elif self.config.data.data == 'community_small':
                    config = getattr(configs, 'default_community_small_sample').get_config(device, ckpt, snr_x, scale_eps_x)
                    eval_dict = Sampler(config).sample(independent=False)
                elif self.config.data.data == 'ENZYMES':
                    config = getattr(configs, 'default_enzymes_sample').get_config(device, ckpt, snr_x, scale_eps_x)
                    eval_dict = Sampler(config).sample(independent=False)
                elif self.config.data.data == 'grid':
                    config = getattr(configs, 'default_grid_sample').get_config(device, ckpt, snr_x, scale_eps_x)
                    eval_dict = Sampler(config).sample(independent=False)
                else:
                    raise AssertionError

                eval_dict['epoch'] = epoch + 1
                wandb.log(eval_dict, commit=True)
                logger.log(f'[EPOCH {epoch + 1:04d}] Saved! \n'+str(eval_dict), verbose=False)

            if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                tqdm.write(f'[EPOCH {epoch+1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | '
                            f'test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e}')
        print(' ')
        return self.ckpt
