import os
import scanpy as sc
import numpy as np
from running.tl import load_config
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from running import RunningPipeline
from running.run_factors import topicvi, topicvi_denovo_finding
import topicvi as tv

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6

warnings.filterwarnings("ignore")

result_dir = '../results/subsampled/'
data_ids = os.listdir(result_dir)

def value_valid_trans(X):
    idx = np.isinf(X) | np.isnan(X)
    X[idx] = 0
    return X

def load_results(data_id, method):
    dir = os.path.join(result_dir, data_id, method, 'results.npz')
    return np.load(dir, allow_pickle=True)

methods = os.listdir(os.path.join(result_dir, data_ids[0]))
methods = [i for i in methods if '.' not in i]

data_ids = os.listdir(result_dir)
# data_ids = [
#     'HLCA_Basal_l3', 'HLCA_DC_l3', 'HLCA_Lymphoid_lf', 'HLCA_Secretory_lf'
# ]
#________
# main runing
for data_id in data_ids:
    adata = sc.read_h5ad(os.path.join(result_dir, data_id, 'adata.h5ad'))
    for method in [topicvi, topicvi_denovo_finding]:
        print(f"##### {data_id} --- {method.__name__}")
        print(adata.shape)
        config = load_config(os.path.join(result_dir, data_id, 'running_config.yaml'))
        # config['extra_kwargs']['topicvi'] = {
        #     'data_kwargs': dict(label_key=None),
        #     'train_kwargs': dict(
        #         pretrain_model = config['extra_kwargs']['topicvi']['train_kwargs']['pretrain_model'],
        #         plan_kwargs = dict(cl_weight=5)
        #     ),
        # }
        # config['extra_kwargs']['topicvi_denovo_finding'] = config['extra_kwargs']['topicvi']
        # config['train_kwargs']['max_epochs'] = 1000
        # tv.utils.write_config(config, os.path.join(result_dir, data_id, 'running_config.yaml'))
        rp = RunningPipeline(method, adata, config)
        rp(verbose=False, save_model=True, check_runned=False)
