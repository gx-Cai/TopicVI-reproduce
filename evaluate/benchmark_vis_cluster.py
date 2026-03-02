import os
import scanpy as sc
import numpy as np
import pandas as pd
import scib
from running.tl import load_config

import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['pdf.fonttype'] = 42

def inverse_davies_bouldin_score(
        adata, label_key, cluster_key, 
        use_rep='X_pca',
        **metric_kwargs
    ):
    from sklearn.metrics import davies_bouldin_score
    if adata.obs[cluster_key].nunique() == 1:
        return 0
    return 1 / davies_bouldin_score(
        adata.obsm[use_rep], 
        adata.obs[cluster_key]
    )

def max_f1(label_true, label_pred, ):
    unique_clusters = np.unique(label_pred)
    result = []
    for target_label in np.unique(label_true):
        max_score = 0
        for cluster in unique_clusters:
            y_pred = label_pred == cluster
            y_true = label_true == target_label
            score = f1_score(y_true, y_pred)
            if score > max_score:
                max_score = score
        result.append(max_score)
    return np.mean(result)

def load_results(data_id, method):
    dir = os.path.join(result_dir, data_id, method, 'results.npz')
    return np.load(dir, allow_pickle=True)

result_dir = '../results/subsampled/'
data_ids = os.listdir(result_dir)

cluster_metrics = {}
cmethod=[
    'topicvi', 'topicvi_denovo_finding', 
    'scanvi_seed_label', 'dcjcomm', 'harmony', 
    'expimap', 'scvi'
]

for data_id in data_ids:
    adata = sc.read_h5ad(os.path.join(result_dir, data_id, 'adata.h5ad'))
    config = load_config(os.path.join(result_dir, data_id, 'running_config.yaml'))
    label_true = adata.obs[config['data_kwargs']['label_key']]

    sc.pp.neighbors(adata, use_rep='X_scanvi_emb')
    sc.tl.umap(adata)

    f, axes = plt.subplots(3, 3, figsize=(15,15))
    axes = axes.flatten()
    sc.pl.umap(
        adata, color=config['data_kwargs']['label_key'], legend_loc='on data',
        ax = axes[0],
        show=False, frameon=False,
        title = 'Ground Truth'
    )
    for method in cmethod:
        results = load_results(data_id, method)
        if 'labels' in results:
            label = results['labels']
        elif 'label' in results:
            label = results['label']
        
        if 'embedding' in results:
            adata.obsm[method] = results['embedding']
        elif 'loading' in results:
            adata.obsm[method] = results['loading']
        
        nmi = normalized_mutual_info_score(label_true, label)
        ari = adjusted_rand_score(label_true, label)
        f1 = max_f1(label_true, label)

        ax = axes[cmethod.index(method)+1]
        adata.obs[method] = label
        if label.dtype == 'float':
            adata.obs[method] = adata.obs[method].astype(int)
        adata.obs[method] = adata.obs[method].astype('category')
        sc.pl.umap(
            adata, color=method, legend_loc='on data',
            ax = ax,
            show=False, frameon=False
        )
        ax.text(
            s = f'ARI: {ari:.2f}\nF1: {f1:.2f}',
            x = 0.05,
            y = 0.9,
            transform=ax.transAxes,
            fontsize=12,
        )

        cluster_metrics[(data_id, method)] = {'nmi': nmi, 'ari': ari, 'f1': f1}

        if adata.obsm[method] is not None:
            scib.clustering.cluster_optimal_resolution(
                adata, 
                label_key=None,#config['data_kwargs']['label_key'], 
                cluster_key=method+'_leiden', 
                use_rep=method,
                cluster_function=sc.tl.leiden,
                resolutions=np.linspace(0.1, 1.0, 10),
                verbose = False,
                metric=inverse_davies_bouldin_score
            )
            label = adata.obs[method+'_leiden']
            nmi = normalized_mutual_info_score(label_true, label)
            ari = adjusted_rand_score(label_true, label)
            f1 = max_f1(label_true, label)
            cluster_metrics[(data_id, method+'_leiden')] = {'nmi': nmi, 'ari': ari, 'f1': f1}

    axes[-1].axis('off')
    f.savefig(f'../assets/part1/umap_{data_id}_subsampled.pdf', dpi=300, bbox_inches='tight')
    plt.close(f)

cluster_metrics = pd.DataFrame(cluster_metrics).T
drop_method = ['harmony', 'expimap', 'scvi']
cluster_metrics = cluster_metrics.drop(index = [i for i in cluster_metrics.index if i[1] in drop_method])
cluster_metrics.to_csv('../assets/part1/cluster_metrics.csv')
cluster_metrics.reset_index().drop(columns='level_0').groupby('level_1').mean()


def transfer_to_rank_score(cluster_metrics):
    rank_metrics = cluster_metrics.reset_index().rename(
        columns = dict(level_0='data_id', level_1='method')
    )
    rank_metrics['nmi'] = rank_metrics.groupby('data_id')['nmi'].rank(ascending=False)
    rank_metrics['ari'] = rank_metrics.groupby('data_id')['ari'].rank(ascending=False)
    rank_metrics['f1'] = rank_metrics.groupby('data_id')['f1'].rank(ascending=False)
    rank_metrics = rank_metrics.set_index(['data_id', 'method'])
    return rank_metrics

ordered_map = {
    'harmony_leiden': 'Harmony + leiden',
    'expimap_leiden': 'expiMap + leiden',
    'scvi_leiden': 'SCVI + leiden',
    'dcjcomm': 'DcjComm',
    # 'dcjcomm_leiden': 'DcjComm + leiden',
    'scanvi_seed_label': 'SCANVI',
    # 'scanvi_seed_label_leiden': 'SCANVI + leiden',
    'topicvi': 'TopicVI',
    # 'topicvi_leiden': 'TopicVI + leiden',
    'topicvi_denovo_finding': 'TopicVI (denovo)',
    # 'topicvi_denovo_finding_leiden': 'TopicVI (denovo) + leiden',
}
# order = list(ordered_map.keys())
cluster_metrics = cluster_metrics[[i[1] in list(ordered_map.keys()) for i in cluster_metrics.index]]
cluster_metrics = transfer_to_rank_score(cluster_metrics)
cluster_metrics.drop(columns=['nmi'], inplace=True)
order = cluster_metrics.reset_index().groupby('method').apply(lambda x: x.set_index('data_id').sum(axis=1)).sum(axis=1).sort_values(ascending=False).index

sns.set(
    style='whitegrid', 
    font_scale=1.2, 
    rc={
        'axes.edgecolor': '0.2',
        'axes.linewidth': 0.8,
        # 'grid.color': '0.9',
        # 'grid.linestyle': '--',
        'legend.frameon': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        # 'axes.labelsize': 12,
        # 'xtick.labelsize': 10,
        # 'ytick.labelsize': 10,
        'font.family': 'Arial',
    }
)

f, axes = plt.subplots(1, 2, figsize=(6, 5))
cluster_metrics.unstack()['ari'][order].plot(
    kind='box', 
    ax=axes[0],
    showmeans=True,
    meanprops = dict(marker='D', markerfacecolor='r', markeredgecolor='r', alpha = 0.25),
    vert=False
)
axes[0].set_xlabel('ARI')
axes[0].set_yticklabels([ordered_map[i] for i in order])

cluster_metrics.unstack()['f1'][order].plot(
    kind='box', 
    ax=axes[1],
    showmeans=True,
    meanprops = dict(marker='D', markerfacecolor='r', markeredgecolor='r', alpha = 0.25),
    vert=False
)
axes[1].set_yticklabels([])
axes[1].set_xlabel('isolated labels F1 score')
axes[1].grid(axis='y')
axes[0].grid(axis='y')
# f.set_title('Average rank across 8 datasets')
# axes[0].set_title('Average rank across 8 datasets', ha = 'left', x=0)
f.savefig('../assets/boxplot_cluster_metrics.pdf', dpi=300, bbox_inches='tight')

# vertical version
f, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=False)
order = order[::-1]
cluster_metrics.unstack()['ari'][order].plot(
    kind='box',
    ax=axes[0],
    showmeans=True,
    meanprops = dict(marker='D', markerfacecolor='r', markeredgecolor='r', alpha = 0.25),
    vert=True
)
axes[0].set_ylabel('ARI')
axes[0].grid(axis='x')
axes[0].set_xlabel('')
axes[0].set_xticklabels([])

axes[0].set_title('Average rank across 8 datasets', ha = 'left', x=0)
cluster_metrics.unstack()['f1'][order].plot(
    kind='box',
    ax=axes[1],
    showmeans=True,
    meanprops = dict(marker='D', markerfacecolor='r', markeredgecolor='r', alpha = 0.25),
    vert=True
)
axes[1].set_ylabel('isolated labels F1 score')
axes[1].set_xticks(np.arange(1, len(order)+1)+0.25)
axes[1].set_xticklabels([ordered_map.get(i, i) for i in order], rotation=90, ha='right')
axes[1].grid(axis='x')
axes[1].set_xlabel('')
f.savefig('../assets/part1/boxplot_cluster_metrics_vertical.pdf', dpi=300, bbox_inches='tight')



### zheng68k dataset
result_dir = '../results/'
data_id = 'zheng68k_sorted'
methods = [
    'amortized_lda',
    'dcjcomm',
    'expimap',
    'harmony',
    'ldvae',
    'muvi',
    'pycogaps',
    'pyliger',
    'scanvi_seed_label',
    'scgpt_zero_shot',
    'schpf',
    'scvi',
    'spectra',
    'spike_slab_lda',
    'topicvi',
    'topicvi_denovo_finding',
    'tree_spike_slab_lda'
]

adata = sc.read_h5ad(os.path.join(result_dir, data_id, 'adata.h5ad'))
config = load_config(os.path.join(result_dir, data_id, 'running_config.yaml'))
label_true = adata.obs[config['data_kwargs']['label_key']]

cluster_metrics = {}

for method in methods:
    try:
        results = load_results(data_id, method)
    except:
        continue
    if 'embedding' in results:
        adata.obsm[method] = results['embedding']
    elif 'loading' in results:
        adata.obsm[method] = results['loading']

    if adata.obsm[method] is not None:
        scib.clustering.cluster_optimal_resolution(
            adata, 
            label_key=None,#config['data_kwargs']['label_key'], 
            cluster_key=method+'_leiden', 
            use_rep=method,
            cluster_function=sc.tl.leiden,
            resolutions=np.linspace(0.1, 1.0, 10),
            verbose = False,
            metric=inverse_davies_bouldin_score
        )
        label = adata.obs[method+'_leiden']
        nmi = normalized_mutual_info_score(label_true, label)
        ari = adjusted_rand_score(label_true, label)
        f1 = max_f1(label_true, label)
        cluster_metrics[(data_id, method+'_leiden')] = {'nmi': nmi, 'ari': ari, 'f1': f1}

    if 'labels' in results:
        label = results['labels']
    elif 'label' in results:
        label = results['label']
    else:
        continue
    
    nmi = normalized_mutual_info_score(label_true, label)
    ari = adjusted_rand_score(label_true, label)
    f1 = max_f1(label_true, label)

    cluster_metrics[(data_id, method)] = {'nmi': nmi, 'ari': ari, 'f1': f1}

cluster_metrics = pd.DataFrame(cluster_metrics).T
cluster_metrics = cluster_metrics.reset_index().drop(columns='level_0').rename(columns={'level_1':'method'}).set_index('method')


index_map = {
    'amortized_lda': 'Amortized LDA',
    'dcjcomm': 'DcjComm',
    'expimap': 'expiMap',
    'harmony': 'Harmony',
    'ldvae': 'LDVAE',
    'muvi': 'MuVI',
    'pycogaps': 'Cogaps',
    'pyliger': 'Liger',
    'scanvi_seed_label': 'scANVI',
    'scgpt_zero_shot': 'scGPT (zero-shot)',
    'schpf': 'scHPF',
    'scvi': 'scVI',
    'spectra': 'Spectra',
    'spike_slab_lda': 'Spike-and-slab LDA',
    'topicvi': 'TopicVI',
    'topicvi_denovo_finding': 'TopicVI (denovo)',
    'tree_spike_slab_lda': 'Tree Spike-and-slab LDA'
}

def index_cleaner(i):
    if i.endswith('_leiden'):
        i = i.replace('_leiden', '')
        return index_map.get(i, i) + ' + leiden'
    else:
        return index_map.get(i, i)

f1_ = cluster_metrics['f1'].sort_values()
ax = f1_.plot(
    kind='barh', 
    figsize=(7, 7), 
    xlabel='isolated label score F1 score',
    color = [
        sns.color_palette('YlOrBr', as_cmap=True)(i) for i in f1_
    ],
    width=0.7
)
for val in f1_.values:
    ax.text(
        val+0.005, 
        ax.get_yticks()[list(f1_.values).index(val)], 
        f'{val:.2f}', 
        va='center',
        fontsize=13,
        fontdict=dict(color='.45')
)
plt.grid(axis='y')
ax.set_yticklabels([index_cleaner(i) for i in f1_.index])
ax.set_ylabel('Methods')
plt.savefig('../assets/zheng68k_sorted/zheng68k_cluster_metrics.pdf', dpi=300, bbox_inches='tight')