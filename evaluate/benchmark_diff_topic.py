import pandas as pd
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt
import seaborn as sns
import topicvi
from topicvi.utils import load_results, load_config


sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6

result_dir = '../results/subsampled/'
data_ids = os.listdir(result_dir)

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


methods = [
    'amortized_lda', 'dcjcomm', 'expimap',
    'harmony', 'ldvae', 'muvi', 'pycogaps',
    'pyliger', 'scanvi_seed_label',
    'scgpt_zero_shot', 'schpf', 'scvi',
    'spectra', 'spike_slab_lda', 'topicvi',
    'topicvi_denovo_finding', 'tree_spike_slab_lda'
]

deresults = {}
data_id = data_ids[0]
method = 'topicvi'
ntop = 1
for data_id in data_ids:
    adata = sc.read_h5ad(os.path.join(result_dir, data_id, 'adata.h5ad'))
    config = load_config(os.path.join(result_dir, data_id, 'running_config.yaml'))
    deresults[data_id] = {}
    for method in methods:
        mres = load_results(data_id, method, base_dir=result_dir)
        if 'loading' not in mres.keys() or mres['loading'].shape[0] != adata.n_obs:
            continue

        topic_data = sc.AnnData(
            X = mres['loading'],
            obs = adata.obs,
            var = pd.DataFrame(index=[f'topic_{i}' for i in range(mres['loading'].shape[1])])
        )
        sc.tl.rank_genes_groups(
            topic_data, groupby=config['data_kwargs']['label_key'], 
        )
        detopics = sc.get.rank_genes_groups_df(topic_data, group=None)
        # detopics = detopics.query('scores > 0') # pvals_adj < 0.05 & 
        # detopics = detopics.groupby('group', observed=False).apply(lambda df: df.sort_values('pvals_adj').head(3), include_groups=False)
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

all_deresults = pd.concat(deresults.values()).drop(columns=['level_1'])

average_topics = (
    all_deresults
    .groupby(['data_id', 'method'])
    .apply(
        lambda df: df['scores'].abs().median(),
        include_groups=False
    )
    .reset_index(name='values')
)


f, ax  = plt.subplots(dpi=300, figsize=(6,4))
(
    average_topics.pivot(index='data_id', columns='method', values='values')
    # sort columns by median
    [average_topics.groupby('method')['values'].mean().sort_values(ascending=False).index]
    .plot(
        kind='box', 
        flierprops={'marker': 'x', 'markersize': 5}, 
        showmeans=True,
        meanprops = dict(marker='D', markerfacecolor='r', markeredgecolor='r', alpha = 0.25),
        ax=ax,
    )
)
ax.set_xticklabels(
    [index_name_cleaner[tick.get_text()] for tick in ax.get_xticklabels()],
    rotation=90, ha='right'
)
ax.set_ylabel('Average of $|t-statistics|$ in each dataset (per topic)')
f.savefig('../assets/part1/boxplot_diff_topic.pdf', dpi=300, bbox_inches='tight')
