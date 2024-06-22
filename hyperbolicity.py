import os
import pickle as pkl
import sys
import time

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from data.data_generators import load_dataset


def hyperbolicity_sample(G, num_samples=50000):
    curr_time = time.time()
    hyps = []
    for i in tqdm(range(num_samples)):
        try:
            curr_time = time.time()
            node_tuple = np.random.choice(G.nodes(), 4, replace=False)
            s = []
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    ok = True
    if len(hyps) == 0:
        ok = False
        return ok,None
    return ok,max(hyps)


if __name__ == '__main__':
    # dataset_name = 'community_small'  # =1
    # dataset_name = 'ego_small'  # (0.2525)
    # dataset_name = 'ENZYMES'  #1.2619
    # dataset_name = 'grid'   #9.76
    dataset_name = 'qm9_test_nx'  #0.7091
    # dataset_name = 'zinc250k_test_nx'  #0.9734
    graph_list = load_dataset(file_name=dataset_name)
    # length = len(graph_list)
    # graph_list = graph_list[:length//10]


    # graph = nx.from_scipy_sparse_matrix(data['adj_train'])
    # print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    hyp_l = []
    for i in tqdm(graph_list):
        ok,res = hyperbolicity_sample(i,2000)
        if not ok:
            continue
        hyp_l.append(res)
        print(f'graph:{res}')
    hyp = torch.Tensor(hyp_l).mean()
    print('Hyp: ', hyp)

