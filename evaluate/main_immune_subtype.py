import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
import warnings
import torch
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from scib_metrics.benchmark import Benchmarker, BatchCorrection
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import scib

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module='pandas')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='torchmetrics')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='scarches')

from topicvi import *
sys.path.append('../external/')

from running import RunningPipeline
from running.run_factors import *
from running.run_presentation import *
from running.tl import load_config
from utils.metrics import TopicMetrics
from utils.prior import *
from model import inverse_davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['grid.alpha'] = 0.6
sc.set_figure_params(frameon=False, dpi=150, dpi_save=300, format='pdf')
sc.settings.figdir = '../assets/'
cluster_paltte = sns.color_palette()
ground_truth_palette = sns.color_palette('Set2', n_colors=8)+['#9edae5', '#f9696a', '#c49c94']

def load_results(data_id, method):
    dir = os.path.join(base_dir, data_id, method, 'results.npz')
    return np.load(dir, allow_pickle=True)

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

def display_topic_max_overlap(prior, topic, n=5):
    overlap = {}
    for k, v in prior.items():
        o = len(set(v) & set(topic)) / len(set(v) | set(topic))
        overlap[k] = o
    overlap = pd.Series(overlap)
    overlap.sort_values(ascending=False, inplace=True)

    with plt.style.context({'axes.linewidth': 0}):
        ax = sns.barplot(x=overlap[0:n], y=overlap[0:n].index, palette='Reds_r', alpha=0.5)
        sns.despine(left=True, bottom=True, ax=ax, top=True, right=True)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        texts = [i.get_text() for i in ax.get_yticklabels()]
        ax.set_yticklabels([])
        for i, txt in enumerate(texts):
            ax.text(
                0.02, i, txt,
                ha = 'left', va = 'center', fontsize=12,
                fontweight='semibold', color='black'
            )

def visualize_embeddings_and_nmi(show, save=False):
    f, axes = plt.subplots(2, 2, figsize=(8, 8))
    if show == 'leiden':
        topicvi_key = 'leiden_topicvi'
        scanvi_key = 'leiden_scanvi'
    else:
        topicvi_key = 'topicvi_pred'
        scanvi_key = 'scanvi_pred'

    display_cltVsground(
        title='TopicVI (leiden)' if show == 'leiden' else 'TopicVI', 
        ax0=axes[0, 0], ax1=axes[0, 1], key=topicvi_key,
        basis='umap_topicvi'
    )
    display_cltVsground(
        title='SCANVI (leiden)' if show == 'leiden' else 'SCANVI',
        ax0=axes[1, 0], ax1=axes[1, 1], key=scanvi_key,
        basis='umap_scanvi'
    )
    if save:
        f.savefig(f'../assets/{data_id}_topicvi_scanvi_{show}.pdf', bbox_inches='tight', dpi=300)

def display_cltVsground(title, ax0, ax1, key, basis):
    sc.pl.embedding(
        adata,basis=basis, color=key,
        legend_loc='on data', 
        title=title,
        show=False, 
        ax=ax0,
        palette = cluster_paltte,
        legend_fontoutline=1
    )
    ax0.text(
        s = "NMI: {:.2f}".format(nmi_score(adata.obs[label_key], adata.obs[key])),
        x = 0.15, y = 0.9, ha = 'center', va = 'center', 
        transform=ax0.transAxes,
        fontsize=10, color = '.5'
    )
    sc.pl.embedding(
        adata,basis=basis, color=label_key,
        title='Original Annotation', show=False, ax=ax1,
        palette = ground_truth_palette,
    )

def display_all3(title, key1, key2, basis):
    f, axes = plt.subplots(1, 3, figsize=(12, 4))

    sc.pl.embedding(
        adata,basis=basis, color=key1,
        legend_loc='on data', 
        title=title,
        show=False, 
        ax=axes[0],
        palette = cluster_paltte,
        legend_fontoutline=1
    )
    axes[0].text(
        s = "NMI: {:.2f}".format(nmi_score(adata.obs[label_key], adata.obs[key1])),
        x = 0.15, y = 0.9, ha = 'center', va = 'center', 
        transform=axes[0].transAxes,
        fontsize=10, color = '.5'
    )
    sc.pl.embedding(
        adata,basis=basis, color=key2,
        title=title+' (leiden)', show=False, ax=axes[1],
        palette = cluster_paltte,
        legend_fontoutline=1,
        legend_loc='on data',
    )
    axes[1].text(
        s = "NMI: {:.2f}".format(nmi_score(adata.obs[label_key], adata.obs[key2])),
        x = 0.15, y = 0.9, ha = 'center', va = 'center',
        transform=axes[1].transAxes,
        fontsize=10, color = '.5'
    )
    sc.pl.embedding(
        adata,basis=basis, color=label_key,
        title='Original Annotation', show=False, ax=axes[2],
        palette = ground_truth_palette,
    )

    return f

base_dir = '../results/'
data_ids = os.listdir(base_dir)
if '.ipynb_checkpoints' in data_ids:
    data_ids.remove('.ipynb_checkpoints')
print('\n'.join(data_ids))

data_id = 'Mariana_24NC'

config = load_config(os.path.join(base_dir, data_id, 'running_config.yaml'))
adata = sc.read_h5ad(os.path.join(base_dir, data_id, 'adata.h5ad'))
label_key = config['data_kwargs'].get('label_key',)
batch_key = config['data_kwargs']['batch_key']

print('\n'.join(list(adata.obs[label_key].unique())))
print('\n'.join(list(adata.uns['annotation']['clusters'].keys())))

scvi_result = load_results(data_id, 'scvi')
topicvi_result = load_results(data_id, 'topicvi')
adata.obs['topicvi_pred'] = pd.Categorical(topicvi_result['labels'].astype(str))
print(
    ari_score(adata.obs[label_key], topicvi_result['labels']),
    nmi_score(adata.obs[label_key], topicvi_result['labels'])
)
scanvi_result = load_results(data_id, 'scanvi_seed_label')

clean_map = {
    'Basophils': 'CD34+', 
    'CD4+ T': 'CD4+ T', 
    'CD8+ T': 'CD8+ T', 
    'Classical Monocytes': 'monocytes',
    'Cytotoxic': 'CD8+ Cytotoxic T', 
    'Granulocytes': 'CD34+', 
    'HSC': 'HSC', 
    'Intermediate monocytes': 'monocytes',
    'Memory CD8+ T cells': 'CD8+ T', 
    'Myeloid Dendritic cells': 'CD34+', 
    'Naive B cells': 'B cells',
    'Naive CD8+ T cells': 'Naive CD8+ T', 
    'Natural killer': 'NK cells', 'Natural killer  cells': 'NK cells',
    'Neutrophils': 'CD34+', 'Non-classical monocytes': 'monocytes', 'Plasma B cells': 'B cells',
    'Plasmacytoid Dendritic cells': 'CD34+', 'Pre-B cells': 'B cells', 'Pro-B cells': 'B cells',
    'Progenitor cells': 'Others', 
    'Th': 'Th', 
    'Th0': 'Th', 
    'Treg': 'Treg',
    'ISG expressing immune cells': 'CD34+',
}

# clean_map = {
#     'Classical Monocytes': 'Monocytes', 
#     'Dendritic': 'Dendritic cells', 
#     'Macrophages': 'Macrophages', 
#     'Mast': 'Mast cells', 
#     'Mast cells': 'Mast cells', 
#     'Myeloid Dendritic cells': 'Dendritic cells', 
#     'Plasmacytoid Dendritic cells': 'Dendritic cells',
# }

adata.obs['scanvi_pred'] = pd.Categorical(pd.Series(scanvi_result['labels']).map(clean_map))
# adata.obs['scanvi_pred'] = pd.Categorical(pd.Series(scanvi_result['labels']))
print(
    ari_score(adata.obs[label_key], scanvi_result['labels']),
    nmi_score(adata.obs[label_key], scanvi_result['labels'])
)

adata.obsm['scvi'] = scvi_result['embedding']
adata.obsm['scanvi'] = scanvi_result['embedding']
adata.obsm['topicvi'] = topicvi_result['embedding']

# scan topics
tp = TopicMetrics(
    adata,
    topic_comp=topicvi_result['factors'],
    topic_prop=topicvi_result['loading'],
    topk=50,
    coherence_norm=False,
    coherence_quantile=0.75
)
print(tp.get_metrics())

topics = tp.get_topics()
priors1 = adata.uns['annotation']['background']
priors2 = adata.uns['annotation']['clusters']
prior = {**priors1, **priors2}
most_related = {}
for i, topic in enumerate(topics):
    overlap = {}
    for k, v in prior.items():
        o = len(set(v) & set(topic)) / len(set(v) | set(topic))
        overlap[k] = o
    # print max overlap
    max_keys = sorted(overlap, key=overlap.get, reverse=True)[0:3]
    most_related[i]={k: overlap[k] for k in max_keys}
# ---

# adata.obs['scanvi_pred'] = scanvi_result['labels'].astype(str)
# adata.obs['topicvi_pred'] = topicvi_result['labels'].astype(str)

make_clusters_obsm(adata,'scvi', label_key)
make_clusters_obsm(adata,'scanvi', label_key)
make_clusters_obsm(adata,'topicvi', label_key) #, use_label_key_for_resolution=True

visualize_embeddings_and_nmi('leiden', )
visualize_embeddings_and_nmi('pred', )

f, axes = plt.subplots(1, 2, figsize=(8, 4))
display_cltVsground(
    'SCVI (leiden)',
    ax0=axes[0], ax1=axes[1],
    key='leiden_scvi',
    basis='umap_scvi'
)

fdata = sc.AnnData(
    X = topicvi_result['loading'],
    obs = adata.obs[['cell_type', 'leiden_topicvi', 'topicvi_pred']],
    var = pd.DataFrame(
        index=[f'topic_{i}' for i in range(topicvi_result['factors'].shape[0])],
        data=topicvi_result['factors'],
        columns=adata.var_names
    )
)
fdata.obsm['X_umap'] = adata.obsm['umap_topicvi']

sc.tl.rank_genes_groups(fdata, groupby='topicvi_pred',)
sc.pl.rank_genes_groups_heatmap(
    fdata, n_genes=3, cmap='RdYlBu_r',
    standard_scale='var',
    min_logfoldchange=0.25,
    # save='_pbmc10_topicvi_heatmap.pdf',
)

sc.pl.umap(
    fdata, 
    color=[f'topic_{i}' for i in range(30)],
    ncols=6,
    cmap='RdYlBu_r',
    vmax='p99',
    vmin='p01',
)

if data_id == 'pbmc10k':
    sc.pl.umap(
        fdata, #basis='umap_topicvi', 
        color=[f'topic_{i}' for i in [11, 0, 4]],
        frameon=False,
        ncols=3,
        vmin='p05', vmax='p99',
        cmap='RdYlBu_r',
        colorbar_loc='bottom',
        title=['Topic 11\n(Apoptosis)', 'Topic 0\n(Myeloid)', 'Topic 4\n(Interleukins Signals)'],
        save='_pbmc10_topic_Cluster7.pdf',
    )

    dfvis = pd.merge(
        pd.Series(fdata.obs_vector('topic_17'), index = fdata.obs_names, name='topic_17'),
        fdata.obs, left_index=True, right_index=True
    ).query('leiden_topicvi == "4" | leiden_topicvi == "0"')
    dfvis['leiden_topicvi'] = dfvis['leiden_topicvi'].astype(str)

    from scipy.stats import mannwhitneyu
    p_val = mannwhitneyu(
        dfvis.query('leiden_topicvi == "0"')['topic_17'],
        dfvis.query('leiden_topicvi == "4"')['topic_17'],
    ).pvalue

    f, ax = plt.subplots(1, 1, figsize=(4, 2))
    sns.boxplot(
        data = dfvis,
        x = 'topic_17',
        y = 'leiden_topicvi',
        palette={k:fdata.uns['leiden_topicvi_colors'][int(k)] for k in ['0', '4']},
        saturation=0.5,
        ax = ax,
        showmeans=True,
        meanprops={'marker':'o', 'markerfacecolor':'.75', 'markeredgecolor':'black'},
        flierprops={'marker':'.', 'markersize':5, 'markerfacecolor':'.25', 'markeredgecolor':'.25'},
    )
    ax.set_xlabel('Topic 17\n(Interferon Signaling)')
    ax.set_ylabel('Leiden Clusters')
    ax.annotate(
            '',
            (0.71, 0.),
            (0.71, 1),
            xycoords="data",
            horizontalalignment="left",
            verticalalignment="top",
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-",
                color='k',
                lw=1,
                connectionstyle = "bar,fraction=0.1",
            ),
        )
    ax.text(
        s = 'p-value < 0.001',
        x = 0.74, y = 0.5,
        ha = 'left', va = 'center',
        fontsize=10,
        color = 'k',
        rotation = 270
    )
    f.savefig(f'../assets/{data_id}_interferon_signaling.pdf', bbox_inches='tight', dpi=300)

    topic = topics[17]
    display_topic_max_overlap(prior, topic)

if data_id == 'zheng68k_sorted':
    # 
    # postive: 13 11 27 2 5 9
    # negative: 4 17 15 12 0 8 6

    #
    dcjcomm_result = load_results(data_id, 'dcjcomm')
    adata.obs['dcjcomm_pred'] = pd.Categorical(dcjcomm_result['label'].astype(int))
    adata.obsm['dcjcomm'] = dcjcomm_result['loading']
    make_clusters_obsm(adata,'dcjcomm', label_key)
    f = display_all3('DCJComm', 'dcjcomm_pred', 'leiden_dcjcomm', 'umap_dcjcomm')
    f.savefig(f'../assets/{data_id}_dcjcomm.pdf', bbox_inches='tight', dpi=300)

    topicvi_denovo_result = load_results(data_id, 'topicvi_denovo_finding')
    adata.obs['topicvi_denovo_pred'] = pd.Categorical(topicvi_denovo_result['labels'].astype(int))
    adata.obsm['topicvi_denovo'] = topicvi_denovo_result['embedding']
    make_clusters_obsm(adata,'topicvi_denovo', label_key)
    display_all3('TopicVI (denovo)', 'topicvi_denovo_pred', 'leiden_topicvi_denovo', 'umap_topicvi_denovo')

    expimap_result = load_results(data_id, 'expimap')
    adata.obsm['expimap'] = expimap_result['embedding']
    make_clusters_obsm(adata,'expimap', label_key)
    f, axes = plt.subplots(1, 2, figsize=(8, 4))
    display_cltVsground(
        'ExpiMap (leiden)',
        ax0=axes[0], ax1=axes[1],
        key='leiden_expimap',
        basis='umap_expimap'
    )
    f.savefig(f'../assets/{data_id}_expimap.pdf', bbox_inches='tight', dpi=300)
    
    # adata.obsm['X_pca']
    make_clusters_obsm(adata,'scvi', label_key,)
    f, axes = plt.subplots(1, 2, figsize=(8, 4))
    display_cltVsground(
        'SCVI (leiden)',
        ax0=axes[0], ax1=axes[1],
        key='leiden_scvi',
        basis='umap_scvi'
    )
    f.savefig(f'../assets/{data_id}_scvi.pdf', bbox_inches='tight', dpi=300)

    topics_annotate = {
        # postive
        '13': 'Antigen Processing',
        '11': 'Toll-like Receptor',
        '5': 'Interferon Signaling',
        # '9': 'NTRK1 Signaling',
        # negative
        '4': 'Pperoxisome',
        '17': 'Potassium Channels',
        '15': 'G Alpha (S) Signaling',
        # '12': 'Rab Regulation Of Trafficking',
    }

    f, axes = plt.subplots(2, 3, figsize=(10, 8))
    for i, (topic, title) in enumerate(topics_annotate.items()):
        sc.pl.umap(
            fdata, color=f'topic_{topic}',
            title=f"Topic {topic}\n"+title, show=False, ax=axes[i//3, i%3],
            cmap='RdYlBu_r', ncols=6, vmin='p05', vmax='p99',
            colorbar_loc=None #'none',
        )
    f.savefig(f'../assets/{data_id}_topic_annotate.pdf', bbox_inches='tight', dpi=300)
    # 0,2,4,6,9
    Tsub = ['0', '6', '9', '2', '4',]
    # search for the topics that diff in Tsub clusters
    cluster_vect = fdata.obs['topicvi_pred'].isin(Tsub)
    cluster_obs = fdata.obs[cluster_vect]
    def make_topic_loading(i,):
        topic_loading = fdata.obs_vector(f'topic_{i}')[cluster_vect]
        topic_loading = pd.DataFrame(
                {
                    'loading': topic_loading,
                    'cluster': cluster_obs['topicvi_pred'].astype(str),
                }, 
                index=cluster_obs.index
            )
        return topic_loading
    
    for i, topic in enumerate(topics):
        topic_loading = make_topic_loading(i,)
        corr = topic_loading.groupby('cluster').mean()
        if (score:=(corr.std() / corr.mean()).item()) > 0.5:
            print(i, score, most_related[i])

    # ridge plot
    vistopic = 18
    topic_annotate = 'Cell Cycle'
    topic_loading = make_topic_loading(vistopic)
    
    g = sns.FacetGrid(
        topic_loading, row='cluster', hue='cluster', 
        aspect=3, height=1.5, palette={i:cluster_paltte[int(i)] for i in Tsub},
        sharex=True, row_order=Tsub
    )
    g.map(sns.kdeplot, 'loading', clip_on=False, shade=True, alpha=0.8, lw=1.5,
          )
    # g.map(plt.axhline, y=0, lw=1, color='black')
    g.refline(y=0, linewidth=2, linestyle='-', color=None, clip_on=False)
    g.set_titles('')
    g.set(yticks=[], ylabel='')
    g.despine(left=True, bottom=True)
    
    def label(x, color, label):
        ax = plt.gca()
        maxx = g.axes[-1, -1].get_xlim()[1]
        ax.text(maxx*0.83, 0.3,'Cluster' + label, fontweight='semibold', color=color, 
                ha='left', va='bottom', fontsize=12)

    g.map(label, 'loading')
    g.fig.subplots_adjust(hspace=-0.5)
    g.fig.set_figheight(2.5)
    g.set_xlabels('Topic Loading', fontsize=10)
    # with a title
    g.fig.suptitle(f'Topic {vistopic}\n({topic_annotate})', fontsize=12)
    g.savefig(f'../assets/{data_id}_topic_{vistopic}.pdf', bbox_inches='tight', dpi=300)