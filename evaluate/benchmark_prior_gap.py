import os
import scanpy as sc
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from topicvi import TopicDict
from topicvi.utils import array_nlargest, load_config
from scipy import stats
from sklearn.preprocessing import minmax_scale, StandardScaler

sns.set_style('whitegrid')
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

def cal_topic_explained_varaince(w, X, ):
    w_norm = w / np.linalg.norm(w)
    topic_scores = np.dot(w_norm.T, X)  # shape: (1, C)
    X_topic = np.outer(w_norm, topic_scores)  # shape: (G, C)
    explained_ratio = np.linalg.norm(X_topic, 'fro')**2 / np.linalg.norm(X, 'fro')**2
    return explained_ratio

def cal_topic_explained_varaince_cell_type(w, adata, label_key):
    w_norm = w.T / w.sum(axis=1)  
    X = adata.X.toarray() 
    topic_scores = np.dot(X, w_norm)
        
    labels = adata.obs[label_key].values
    result = []
    for i in labels.unique():
        tstat, pval = stats.ttest_ind(topic_scores[labels==i], topic_scores[labels!=i], axis=0, equal_var=False)
        result.append(tstat)
    
    result = np.array(result) # shape: (num_labels, num_topics)
    max_score = np.abs(result).max(axis=0) # shape: (num_topics, )
    return max_score

def make_prior_gene_weights(prior_name, adata):
    var_names = adata.var_names
    if prior_name in adata.uns['annotation']['background']:
        prior_genes = adata.uns['annotation']['background'][prior_name]
    elif prior_name in adata.uns['annotation']['clusters']:
        prior_genes = adata.uns['annotation']['clusters'][prior_name]
    else:
        raise ValueError(f'Prior name {prior_name} not found in annotation.')
    prior_weights = np.zeros(len(var_names))
    for i in prior_genes:
        if i in var_names:
            prior_weights[var_names.tolist().index(i)] = 1
    return prior_weights

method = 'topicvi'
merged_explains = []
merged_prior_explains = []
merged_max_overlap = []
for data_id in data_ids:
    res = load_results(data_id, method)
    w = res['factors']
    adata = sc.read_h5ad(os.path.join(result_dir, data_id, 'adata.h5ad'))
    config = load_config(os.path.join(result_dir, data_id, 'running_config.yaml'))
    label_key = config['data_kwargs']['label_key']
    X = adata.X.toarray()

    # as each run, there are 5 topics that are not supervised, we remove them for fair comparison
    explain_var = cal_topic_explained_varaince_cell_type(w,adata, label_key=label_key)[0:-5]

    topics = TopicDict(
        topic_list=[[adata.var_names[j] for j in i] for i in array_nlargest(arr=w, n=50)][0:-5]
    )
    topics.compare_prior_overlap(
        {**adata.uns['annotation']['background'],
        **adata.uns['annotation']['clusters']},
    )

    max_overlap = pd.concat(
        [pd.Series(i) for i in topics.get_topic_annotes(1)]
    )
    max_overlap_val = max_overlap.values

    corr_priors = max_overlap.index.to_list()
    prior_w = np.array([
        make_prior_gene_weights(i, adata) for i in corr_priors
    ])
    prior_explains = cal_topic_explained_varaince_cell_type(prior_w, adata, label_key=label_key)

    # base_mean, base_std = np.mean(explain_var), np.std(explain_var)
    # explain_var = (explain_var - base_mean) / base_std
    # prior_explains = (prior_explains - base_mean) / base_std
    
    # normalize to [0, 1]
    explain_var_concat = np.concatenate([explain_var, prior_explains])
    explain_var_concat = minmax_scale(explain_var_concat)
    explain_var = explain_var_concat[0:len(explain_var)]
    prior_explains = explain_var_concat[len(explain_var):]

    merged_explains.extend(explain_var)
    merged_prior_explains.extend(prior_explains)
    merged_max_overlap.extend(max_overlap_val)
    
scale = 0.75
f, ax = plt.subplots(1, 1, figsize=(4*scale, 3*scale), dpi=150)

ax = sns.scatterplot(
    x=merged_max_overlap,
    y=merged_explains,
    # color="#597ac8",
    alpha=0.6
)
sns.regplot(
    x= merged_max_overlap,
    y= merged_explains,
    scatter=False,
    # color="#597ac8",
    ax=ax
)
ax.set_xlabel('Max Overlap with Prior Gene Sets')
ax.set_ylabel('Cell Type DE score\n(scaled $|t-statistics|$)')

f.savefig(
    '../assets/part1/topic_explained_var_vs_max_overlap.pdf',
    bbox_inches='tight'
)


dfvis = pd.DataFrame({
    'TopicVI': merged_explains,
    'PriorGeneSet': merged_prior_explains,
    'MaxOverlap': merged_max_overlap
})

# cut based on max overlap; 0.1, 0.2, ..., 1.0
dfvis['bins'] = pd.qcut(dfvis['MaxOverlap'], 10)
dfvis = dfvis.groupby('bins').median()

palette = sns.color_palette()
f, ax = plt.subplots(1, 1, figsize=(4.2*scale, 3*scale), dpi=150)
sns.barplot(
    data=dfvis['TopicVI'],
    ax=ax,
    ci=False,
    width=0.8,
    color=palette[0],
    edgecolor=None,
    linewidth=0,
    label='TopicVI'
)
sns.barplot(
    data=dfvis['PriorGeneSet'],
    ax=ax,
    ci=False,
    width=0.45,
    color=palette[1],
    edgecolor=None,
    linewidth=0,
    label='Prior Gene Set'
)
ax.set_xticks([0, 9])
ax.set_xticklabels([1, 10])
ax.annotate(
    text='',
    xy=(0.9, -0.08), xycoords='axes fraction',
    xytext=(0.1, -0.08), textcoords='axes fraction',
    arrowprops=dict(arrowstyle='-|>', color='black')
)
ax.legend(frameon=False, bbox_to_anchor=(1.05,1.15), ncol=2)
ax.set_xlabel('Bins of Max Overlap with Prior Gene Sets')
ax.set_ylabel('Cell Type DE score\n(scaled $|t-statistics|$)')

f.savefig(
    '../assets/part1/topic_explained_var_vs_max_overlap_bar.pdf',
    bbox_inches='tight'
)
