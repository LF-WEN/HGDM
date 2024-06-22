# import configs.community_small_config as configs
import configs.ego_small_config as configs
# import configs.enzymes_config as configs
# import configs.grid_config as configs
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
    config = getattr(configs, 'ae_hgat_poincare').get_config()
    """
    ae_hgat_poincare
    """
    trainer = Trainer(config)
    trainer.train_ae()