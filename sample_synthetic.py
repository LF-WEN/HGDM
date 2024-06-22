import sys

import configs
from sampler import Sampler_mol, Sampler
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

    config = getattr(configs, sys.argv[1]).get_config(*sys.argv[2:])

    sampler = Sampler(config)
    sampler.sample()
