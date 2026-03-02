import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import scib
from running.tl import load_config
from topicvi.metrics import TopicMetrics
from tqdm import tqdm
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score

sns.set_style('white')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['pdf.fonttype'] = 42

warnings.filterwarnings("ignore")

index_name_cleaner = {
    "topicvi": "TopicVI",
    "scvi": "SCVI",
    "scanvi_seed_label": "SCANVI",
    "expimap": "expiMap",
    "muvi": "MuVI",
    "spectra": "Spectra",
    "ldvae": "LDVAE",
    "pyliger": "LIGER",
    "harmony": "Harmony",
    "pycogaps": "CoGAPS",
    "topicvi_denovo_finding": "TopicVI (denovo)",
    "dcjcomm": "DcjComm",
    "scgpt_zero_shot": "scGPT (zero-shot)",
    "spike_slab_lda": "Spike-slab LDA",
    "tree_spike_slab_lda": "Tree-spike-slab LDA",
    'schpf': 'scHPF',
    'amortized_lda': 'Amortized LDA',
}

result_dir = '../results/subsampled/'
data_ids = os.listdir(result_dir)

def load_results(data_id, method, base_dir=result_dir):
    dir = os.path.join(base_dir, data_id, method, 'results.npz')
    return np.load(dir, allow_pickle=True)

methods = [
    'amortized_lda', 'dcjcomm', 'expimap',
    'harmony', 'ldvae', 'muvi', 'pycogaps',
    'pyliger', 'scanvi_seed_label',
    'scgpt_zero_shot', 'schpf', 'scvi',
    'spectra', 'spike_slab_lda', 'topicvi',
    'topicvi_denovo_finding', 'tree_spike_slab_lda'
]

# ------------
# explainability of topics.


explains = {}

for data_id in (bar:=tqdm(data_ids, leave=False)):    
    bar.set_description(data_id)
    adata = sc.read_h5ad(os.path.join(result_dir, data_id, 'adata.h5ad'))
    config = load_config(os.path.join(result_dir, data_id, 'running_config.yaml'))
    priors = adata.uns['annotation']['clusters']
    priors.update(adata.uns['annotation']['background'])
    
    result = dict()
    for m in methods:
        if os.path.exists(os.path.join(result_dir, data_id, m, 'results.npz')) is False:
            continue
        mres = load_results(data_id, m)
        if m == 'expimap':
            continue # the expiMap topics are just priors, thus overlap is always 1
        else:
            if 'loading' not in mres.keys(): continue
            if 'loading' in mres.keys():
                loading = mres['loading']
            if mres['factors'].shape[0] != adata.n_vars:
                factors = mres['factors']
            else:
                factors = mres['factors'].T
            adata_ = adata.copy()
        if (fmin:=np.nanmin(factors)) < 0:
            factors = factors - fmin
        factors[np.isnan(factors)] = 0
        # not used topics, all zero
        valid_topics = np.where(factors.sum(axis=1) > 0)[0]
        factors = factors[valid_topics, :]
        loading = loading[:, valid_topics]
        if loading.shape[0] != adata.n_obs: 
            print("Size not match", m)
            continue
        
        tm = TopicMetrics(
            adata_, layer='normalized',
            topic_comp=factors,
            topic_prop=loading,
            topk=50,
            coherence_norm=False
        )
        topics = tm.get_topics()
        if m == 'topicvi':
            topics = topics[0:-5] # only consider the aligned topics

        for ti, topic in enumerate(topics):
            overlap = {}
            for k, v in priors.items():
                o = len(set(v) & set(topic)) / len(set(v) | set(topic))
                overlap[k] = o
            max_keys = sorted(overlap, key=overlap.get, reverse=True)[0]
            max_overlap = overlap[max_keys]
            result[(data_id, m, ti)] = max_keys, max_overlap
        
    bar.update(1)
    explains.update(result)

explains = pd.DataFrame(explains).T
explains.reset_index(inplace=True)
explains.columns = ['data_id', 'method', 'topic_idx','topic', 'overlap']
explains['overlap'] = explains['overlap'].astype(float)
explains['With_prior'] = explains['method'].isin(
    ['topicvi', 'spectra', 'muvi']
)

# ---------
# diffential loading significance

deresults = {}
ntop = 1
for data_id in data_ids:
    adata = sc.read_h5ad(os.path.join(result_dir, data_id, 'adata.h5ad'))
    config = load_config(os.path.join(result_dir, data_id, 'running_config.yaml'))
    deresults[data_id] = {}
    for method in methods:
        mres = load_results(data_id, method, base_dir=result_dir)
        if 'loading' not in mres.keys() or mres['loading'].shape[0] != adata.n_obs:
            continue
        
        loading = mres['loading']
        if method == 'topicvi':
            loading = loading[:,0:-5] # only consider the aligned topics
        topic_data = sc.AnnData(
            X = loading,
            obs = adata.obs,
            var = pd.DataFrame(index=[f'topic_{i}' for i in range(loading.shape[1])])
        )
        sc.tl.rank_genes_groups(
            topic_data, groupby=config['data_kwargs']['label_key'], 
        )
        detopics = sc.get.rank_genes_groups_df(topic_data, group=None)
        detopics = (
            detopics.groupby('names')
            .apply(lambda df: df.sort_values('pvals_adj').head(ntop), include_groups=False)
            .reset_index()
            .drop(columns=['level_1'])
        )
        
        if detopics['names'].nunique() < topic_data.n_vars:
            # fill with zero score topics
            existing_topics = set(detopics['names'].unique())
            all_topics = set(topic_data.var_names)
            missing_topics = all_topics - existing_topics
            if len(missing_topics) > 0:
                missing_df = pd.DataFrame({
                    'names': list(missing_topics) * ntop,
                    'scores': [0]*len(missing_topics) * ntop,
                    'logfoldchanges': [0]*len(missing_topics) * ntop,
                    'pvals': [1]*len(missing_topics) * ntop,
                    'pvals_adj': [1]*len(missing_topics) * ntop,
                    'group': ['unknown']*len(missing_topics) * ntop,
                })
                detopics = pd.concat([detopics, missing_df], ignore_index=True)

        # detopics['n_topics'] = topic_data.n_vars
        deresults[data_id][method] = detopics

    deresults[data_id] = (
        pd.concat(deresults[data_id].values(), keys=deresults[data_id].keys())
        .reset_index()
        .rename(columns={'level_0':'method'})
        # .drop(columns=['level_1'])
    )
    deresults[data_id]['data_id'] = data_id

deresults = pd.concat(deresults.values()).drop(columns=['level_1'])

average_topics = (
    deresults
    .groupby(['data_id', 'method'])
    .apply(
        lambda df: df['scores'].abs().median(),
        include_groups=False
    )
    .reset_index(name='values')
)

average_topics['With_prior'] = average_topics['method'].isin(
    ['topicvi', 'spectra', 'muvi']
)

# ---- a bar plot merge two
order = average_topics.groupby('method')['values'].mean().sort_values(ascending=False).index

sns.set(
    style='whitegrid', 
    font_scale=1.2, 
    rc={
        'axes.edgecolor': '0.2',
        'axes.linewidth': 0.8,
        'grid.color': '0.9',
        'grid.linestyle': '--',
        'legend.frameon': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        # 'axes.labelsize': 12,
        # 'xtick.labelsize': 10,
        # 'ytick.labelsize': 10,
        'font.family': 'Arial',
    }
)

f, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
f.subplots_adjust(wspace=0.5)

sns.barplot(
    data=explains,
    x='overlap', y='method', hue='With_prior',
    order=order,
    edgecolor = '.3',
    # alpha=.5,
    errwidth=1,
    errorbar=("ci", 95),
    ax=axes[0],
    orient='h',
    palette = 'Set2'
)
# axes[0].invert_yaxis()
axes[0].invert_xaxis()
# set y tick labels into right side
axes[0].yaxis.tick_right()
axes[0].set_ylabel('')
axes[0].set_xlabel('Max overlap with prior gene sets')
axes[0].set_yticklabels(
    [index_name_cleaner[i.get_text()] for i in axes[0].get_yticklabels()]
)
leg = axes[0].legend(frameon=False, title=None, )
leg.get_texts()[0].set_text('Without prior')
leg.get_texts()[1].set_text('With prior')

sns.barplot(
    data=average_topics,
    x='values', y='method', hue='With_prior',
    order=order,
    edgecolor = '.3',
    errwidth=1,
    errorbar=("ci", 95),
    ax=axes[1],
    orient='h',
    palette = 'Set2'

)
axes[1].set_ylabel('')
axes[1].set_xlabel('DE Scores in cell types ($|t-statistics|$)')
axes[1].set_yticklabels([])
axes[1].legend_.remove()

f.savefig('../assets/barplot_topic_explainability_merged.pdf', bbox_inches='tight')
