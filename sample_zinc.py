import sys

import configs
from sampler import Sampler_mol
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
    config = getattr(configs, 'default_zinc_sample').get_config(*sys.argv[1:])
    """
    default_zinc_sample
    """
    sampler = Sampler_mol(config)
    sampler.sample()
