# Dependencies

Please use the following command to install the requirements:

```
pip install -r requirements.txt
```

For molecule generation, additionally run the following command:

```
conda install -c conda-forge rdkit=2020.09.1.0
pip install git+https://github.com/fabriziocosta/EDeN.git
```

# Preparations

To preprocess the molecular graph datasets for training models, run the following command:

```
python data/preprocess.py --dataset ${dataset_name}
python data/preprocess_for_nspdk.py --dataset ${dataset_name}
```

For the evaluation of generic graph generation tasks, run the following command to compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html):

```
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```

# Configurations

The configurations are provided on the `configs/` directory.

# Pretrained checkpoints

We provide checkpoints of the pretrained models on the `checkpoints/` directory, which are used in the experiments.

For the VAE checkpoints:

- `ego_small/ae/ae.pth`
- `community_small/ae/ae.pth`
- `ENZYMES/ae/ae.pth`
- `grid/ae/ae.pth`
- `QM9/ae/ae.pth`
- `ZINC250k/ae/ae.pth`

For the score model checkpoints:

- `ego_small/score_model/score_model.pth`
- `community_small/score_model/score_model.pth`
- `ENZYMES/score_model/score_model.pth`
- `grid/score_model/score_model.pth`
- `QM9/score_model/score_model.pth`
- `ZINC250k/score_model/score_model.pth`

# Training

We provide the commands for  training a hyperbolic score model on several datasets.

For example, to train a VAE on QM9

```
python qm9_autoencoder.py
```

and then train the score model,

```
python qm9.py
```

To train on generic graph datasets, please import the appropriate config and comment out unrelated configuration librariesin `train_synthetic.py` and `train_synthetic_autoencoder.py`.

For example: 

```
# import configs.community_small_config as configs
import configs.ego_small_config as configs
# import configs.enzymes_config as configs
# import configs.grid_config as configs
```

# Sampling scripts

We provide sampling scripts of our models on the `script/` directory

- `script/sample_community_small.sh`
- `script/sample_ego_small.sh`
- `script/sample_enzymes.sh`
- `script/sample_grid.sh`
- `script/sample_qm9.sh`
- `script/sample_zinc.sh`

To sample on qm9:

```
sh script/sample_qm9.sh
```

