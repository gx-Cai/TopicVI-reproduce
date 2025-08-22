import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
import warnings
import torch
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import scib
from topicvi.model import inverse_davies_bouldin_score
from topicvi.utils import load_results, write_config, load_config
from topicvi.preprocess import preprocess_adata

def subsample_from_annote(
    annote: pd.DataFrame,
    level,
    cell_type,
    n_cells=None,
    sample_strategy="random",
    next_level=None,
    replace=False,
):
    """
    Subsample n samples from each annotation level.
    """

    subsampled = annote.query(f"{level} == '{cell_type}'")
    if subsampled.shape[0] == 0:
        raise ValueError(f"No cell type {cell_type} found in level {level}.")

    if n_cells is None:
        n_cells = subsampled.shape[0] // 10
    elif n_cells > subsampled.shape[0]:
        n_cells = subsampled.shape[0]

    if sample_strategy == "random":
        return subsampled.sample(n=n_cells, replace=replace).index
    elif sample_strategy == "balanced":
        next_level = (
            level[:-1] + str(int(level[-1]) + 1) 
            if next_level is None else next_level
        )
        n_unique = subsampled[next_level].nunique()
        n_cells = n_cells // n_unique
        print(
            f"Subsample {n_cells} cells from each {next_level}, total {n_cells*n_unique} cells."
        )
        try:
            seletced_cells = (
                subsampled.groupby(next_level)
                .apply(
                    lambda x: x.sample(n=n_cells, replace=replace).index,
                    # include_groups=False,
                )
                .values
            )
            return np.concatenate(seletced_cells)
        except Exception as e:
            print(
                f"{subsampled[next_level].value_counts()}"
            )
            raise
    else:
        raise ValueError(f"Unknown sample strategy: {sample_strategy}")


def make_umap(adata, use_rep, key_added, **kwargs):
    sc.pp.neighbors(adata, use_rep=use_rep, key_added=f'nei_{use_rep}', **kwargs)
    sc.tl.umap(adata, neighbors_key = f'nei_{use_rep}')
    adata.obsm[key_added] = adata.obsm['X_umap']
    del adata.obsm['X_umap']


def make_clusters_obsm(
        adata, use_rep, label_key,
        resolution = None, 
        use_label_key_for_resolution = False,
        use_metrics = 'db'
    ):
    from sklearn.metrics import calinski_harabasz_score
    def ch_score_adata(adata, label_key, cluster_key, use_rep = 'X_pca', **metric_kwargs):
        return calinski_harabasz_score(adata.obsm[use_rep], adata.obs[cluster_key])

    metric_func = inverse_davies_bouldin_score if use_metrics == 'db' else ch_score_adata
    if f'umap_{use_rep}' not in adata.obsm.keys():
        make_umap(adata, use_rep=use_rep, key_added=f'umap_{use_rep}')
    if resolution is None:
        if use_label_key_for_resolution:
            scib.clustering.cluster_optimal_resolution(
                adata, 
                label_key=label_key, 
                cluster_key=f'leiden_{use_rep}', 
                use_rep=use_rep,
                cluster_function=sc.tl.leiden,
                resolutions=np.linspace(0.1, 1.0, 10),
            )
        else:
            scib.clustering.cluster_optimal_resolution(
                adata, 
                label_key=None, 
                cluster_key=f'leiden_{use_rep}', 
                use_rep=use_rep,
                cluster_function=sc.tl.leiden,
                resolutions=np.linspace(0.1, 1.0, 10),
                metric=metric_func
            )
    else:
        sc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_{use_rep}', neighbors_key=f'nei_{use_rep}')
    
    sc.pl.embedding(adata,basis=f'umap_{use_rep}', color=[f'leiden_{use_rep}', label_key], frameon=False)
    print(
        ari_score(adata.obs[label_key], adata.obs[f'leiden_{use_rep}']),
        nmi_score(adata.obs[label_key], adata.obs[f'leiden_{use_rep}'])
    )


def setting_config(adata, config):
    config['train_kwargs']['max_epochs'] = 500
    config['train_kwargs']['batch_size'] = 1024
    config['model_kwargs']['n_topics'] = adata.shape[1] // config['model_kwargs']['n_clusters'] // 10 + 10
    config['extra_kwargs']['amortized_lda'] = {'train_kwargs': dict(early_stopping_monitor = 'elbo_train')}
    config['extra_kwargs']['pycogaps'] = {'train_kwargs': dict(nIterations = 5000)}
    config['extra_kwargs']['spectra'] = {'data_kwargs': dict(label_key = None), 'train_kwargs': dict(use_gpu=False)}
    config['extra_kwargs']['topicvi'] = {
        'train_kwargs': dict(pretrain_model = os.path.join(config['save_dir'], 'topicvi', 'pretrain')),
        'data_kwargs': dict(label_key = None)
    }
    config['model_kwargs']['n_topics'] = config['model_kwargs']['n_clusters']*2 + max(10, adata.shape[1] // 200)
    if adata.shape[0] > 100000:
        config['extra_kwargs']['scanvi_seed_label'] = {
            'train_kwargs': dict(plan_kwargs = dict(lr=0.0001), max_epochs=200) # To avoid nan bugs. when training.
        }
    config['extra_kwargs']['topicvi_denovo_finding'] = {
        'data_kwargs': dict(label_key=None),
        'train_kwargs': dict(
            pretrain_model = config['extra_kwargs']['topicvi']['train_kwargs']['pretrain_model'],
        ),
    }
    return config

