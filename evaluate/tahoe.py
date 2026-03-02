import scanpy as sc
import numpy as np
import pandas as pd
import topicvi as tv
import torch
from sklearn.preprocessing import normalize, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

sc.set_figure_params(dpi=150, frameon=False)
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['pdf.fonttype'] = 42


file_dir = r'D:\Data\tahoe_drug_perturb\tahoe100_plate1_A-172_valid_drugconc.h5ad'
data = sc.read_h5ad(file_dir)

gene_names_cleaned = {
    'RTFDC1': 'RTF1',
    'ZUFSP': 'ZUP1',
    'ASUN': 'INTS13',
    'FAM60A': 'SINHCAF',
    'RP5-890E16.2': 'NFE2L1-DT',
    'KIAA1107': 'BTBD8',
    'FAM58A': 'CCNQ',
    'C11orf84': 'SPINDOC',
}

topic32_weight = pd.read_table('../assets/ZhaoSim2021/topic32_weights.tsv', index_col=0)['score']
topic32_weight.index = topic32_weight.index.to_series().replace(gene_names_cleaned)
topic12_weight = pd.read_table('../assets/ZhaoSim2021/topic12_weights.tsv', index_col=0).iloc[:,0]

sc.pp.normalize_total(data)
sc.pp.log1p(data)
sc.pp.filter_cells(data, min_genes=200)
sc.pp.filter_genes(data, min_cells=3)
sc.pp.pca(data, n_comps=50)
sc.pp.neighbors(data, n_neighbors=15)
sc.tl.umap(data)

assert np.all(topic32_weight.index.isin(data.var_names))
assert np.all(topic12_weight.index.isin(data.var_names))

def normalized_by_DMSO(x):
    dmsos = x[data.obs['drug'] == 'DMSO_TF']
    mean, std = dmsos.mean(), dmsos.std()
    return (x - mean) / std

data.obs['topic_32'] = normalized_by_DMSO(data[:, topic32_weight.index].X @ topic32_weight)
data.obs['topic_12'] = normalized_by_DMSO(data[:, topic12_weight.index].X @ topic12_weight)


sc.pl.umap(
    data, color=['topic_12','topic_32'],
    cmap='RdBu_r', vmax='p99', vmin='p1',
    ncols=1
    # save='_topic32.pdf',
)

sc.tl.rank_genes_groups(data, 'drug', method='t-test', reference='DMSO_TF', pts = True)

deg_filtered = sc.get.rank_genes_groups_df(data, group=None).query(
    '(logfoldchanges > 1 | logfoldchanges < -1)'
    ' & pvals_adj < 0.05'
)


result = pd.DataFrame(
    {
        'N_DEGs': deg_filtered['group'].value_counts(),
        'Topic32_Mean': data.obs.groupby('drug')['topic_32'].mean(),
        'Topic12_Mean': data.obs.groupby('drug')['topic_12'].mean(),
    }
)
result.sort_values('N_DEGs', ascending=False,)

scale = 0.75
plt.figure(figsize=(4*scale,4*scale))
ax = sns.scatterplot(
    data=result,
    y='Topic12_Mean',
    x='Topic32_Mean',
    size='N_DEGs',
    hue='N_DEGs',
    palette='Reds',
    sizes=(20, 200),
    edgecolor='.25',
)
ax.legend(
    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False,
    title='Number\nof DEGs'
)
ax.set_ylabel('Mean Topic 12 Score')
ax.set_xlabel('Mean Topic 32 Score')
plt.savefig('../assets/tahoe_A-172_topic12_topic32_vs_DEGs.pdf', bbox_inches='tight')


## ----

subdata = data[:, topic12_weight.index].copy()
sc.pp.scale(subdata)

# plot bar
selected_cells = subdata.obs.query('drug == "Bortezomib"', ).index
dfvis = subdata[selected_cells].to_df()
dfvis = dfvis.mean().sort_values(ascending=False)


ax = dfvis.plot(
    kind='barh',
    figsize=(3,6),
    title='Bortezomib treated A-172 cells\nTopic 12 genes',
    color='skyblue',
    edgecolor='k',
    width=0.8,
)
ax.grid(axis='y')
ax.set_ylabel('')
ax.set_xlabel('Expression (z-score)')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

base_x = 0.7
for i, tick in enumerate(ax.get_yticklabels()):
    gene_name = tick.get_text()
    val = dfvis[gene_name]

    if val > 0:
        tick.set_x( base_x - 0.2)
    else:
        tick.set_x(base_x )
        tick.set_horizontalalignment('left')
    
    if val > 0.05 or val < -0.15:
        tick.set_color('k')
    else:
        tick.set_color('.5')
plt.savefig('../assets/ccle_drug_response/bortezomib_A-172_topic12_genes.pdf', bbox_inches='tight')