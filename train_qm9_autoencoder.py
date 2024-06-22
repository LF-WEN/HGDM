import configs.qm9_configs as configs
from trainer import Trainer



# def create_folders(args):
#     try:
#         os.makedirs('../outputs')
#     except OSError:
#         pass
#
#     try:
#         os.makedirs('outputs/' + args.exp_name)
#     except OSError:
#         pass
# create_folders(config)

if __name__ == "__main__":
    config = getattr(configs, 'ae_hgat_ponicare').get_config()
    """
    ae_hgcn_ponicare
    ae_hgat_ponicare
    ae_gcn
    ae_gat
    """
    trainer = Trainer(config)
    trainer.train_ae()