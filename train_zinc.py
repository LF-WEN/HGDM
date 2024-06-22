import configs.zinc_config as configs
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
    config = getattr(configs, 'diff_hgat_ponicare_ae').get_config()
    """
    diff_hgcn_ponicare
    diff_hgcn_ponicare_ae
    diff_hgat_ponicare
    diff_hgat_ponicare_ae
    diff_gcn
    diff_gcn_ae
    diff_gat
    diff_gat_ae
    """

    trainer = Trainer(config)
    trainer.train_sde()