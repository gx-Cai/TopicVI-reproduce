import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import scib
from scib_metrics.benchmark import Benchmarker, BatchCorrection
from running.tl import load_config
from topicvi import *
from metrics import TopicMetrics
from tqdm import tqdm
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6

warnings.filterwarnings("ignore")
metric_name_cleaner = {
    "silhouette_label": "Silhouette label",
    "silhouette_batch": "Silhouette batch",
    "isolated_labels": "Isolated labels",
    "nmi_ari_cluster_labels_leiden_nmi": "Leiden NMI",
    "nmi_ari_cluster_labels_leiden_ari": "Leiden ARI",
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "clisi_knn": "cLISI",
    "ilisi_knn": "iLISI",
    "kbet_per_label": "KBET",
    "graph_connectivity": "Graph connectivity",
    "pcr_comparison": "PCR comparison",
    "Gene Topic Coherence": "Gene-topic coherence",
}
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
minmax_scale = lambda x: (x - x.min()) / (x.max() - x.min())

def value_valid_trans(X):
    idx = np.isinf(X) | np.isnan(X)
    X[idx] = 0
    return X

def load_results(data_id, method):
    dir = os.path.join(result_dir, data_id, method, 'results.npz')
    return np.load(dir, allow_pickle=True)

methods = os.listdir(os.path.join(result_dir, data_ids[0]))
methods = [i for i in methods if '.' not in i]

#________
# main runing
from running import RunningPipeline
from running.run_factors import topicvi, topicvi_denovo_finding

for data_id in os.listdir(result_dir):
    adata = sc.read_h5ad(os.path.join(result_dir, data_id, 'adata.h5ad'))
    for method in [topicvi, topicvi_denovo_finding]:
        print(f"##### {data_id} --- {method.__name__}")
        print(adata.shape)
        config = load_config(os.path.join(result_dir, data_id, 'running_config.yaml'))
        # config['model_kwargs']['n_topics'] = config['model_kwargs']['n_clusters'] * 2 + 5
        config['extra_kwargs']['topicvi'] = {
            'data_kwargs': dict(label_key=None),
            'train_kwargs': dict(
                pretrain_model = config['extra_kwargs']['topicvi']['train_kwargs']['pretrain_model'],
                plan_kwargs = dict(cl_weight=5)
            ),
            # 'model_kwargs': dict(
            #     # topic_decoder_params = dict(
            #     #     n_topics_without_prior = 5 #config['model_kwargs']['n_topics'] // 2
            #     # ),
            #     cluster_decoder_params = dict(
            #         center_penalty_weight = 2
            #     )
            # )
        }
        config['extra_kwargs']['topicvi_denovo_finding'] = config['extra_kwargs']['topicvi']
        config['train_kwargs']['max_epochs'] = 1000

        rp = RunningPipeline(method, adata, config)
        rp(verbose=False, save_model=True, check_runned=False)

# --- 
# main evaluation

all_result = {}
methods = ['topicvi', 'topicvi_denovo_finding']

for data_id in (bar:=tqdm(data_ids, leave=False)):    
    bar.set_description(data_id)
    if data_id in all_result.keys(): continue

    adata = sc.read_h5ad(os.path.join(result_dir, data_id, 'adata.h5ad'))
    config = load_config(os.path.join(result_dir, data_id, 'running_config.yaml'))

    obsm_keys = []
    for m in methods:
        try:
            if os.path.exists(os.path.join(result_dir, data_id, m, 'results.npz')) is False:
                continue
            mres = load_results(data_id, m)
            if 'embedding' in mres.keys():
                adata.obsm[m] = value_valid_trans(mres['embedding'])
            else:
                adata.obsm[m] = value_valid_trans(mres['loading'])
            obsm_keys.append(m)
        except Exception as e:
            print(f'Error loading {m} for {data_id}')
            print(e)
            continue
    
    if config['data_kwargs'].get('batch_key', None):
        bench = Benchmarker(
            adata=adata,
            batch_key=config['data_kwargs'].get('batch_key', None),
            label_key=config['data_kwargs'].get('label_key', None),
            embedding_obsm_keys = obsm_keys,
        )
    else:
        adata.obs['batch'] = 'batch'
        bench = Benchmarker(
            adata=adata,
            embedding_obsm_keys = obsm_keys,
            batch_key='batch',
            label_key=config['data_kwargs'].get('label_key', None),
            batch_correction_metrics = BatchCorrection(*([False] * 5))
        )
        
    bench.benchmark()
    emb_res = bench._results.T.drop(index=['Metric Type'])

    result = dict()
    for m in methods:
        if os.path.exists(os.path.join(result_dir, data_id, m, 'results.npz')) is False:
            continue
        mres = load_results(data_id, m)
        if m == 'expimap':
            loading = mres['embedding']
            factors = mres['factors'].T
            adata = adata[:, mres['factor_genes']]
        else:
            if 'loading' not in mres.keys(): continue
            if 'loading' in mres.keys():
                loading = mres['loading']
            if mres['factors'].shape[0] != adata.n_vars:
                factors = mres['factors']
            else:
                factors = mres['factors'].T
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
        
        result[m] = TopicMetrics(
            adata, layer='normalized',
            topic_comp=factors,
            topic_prop=loading,
            topk=50,
            coherence_norm=False
        ).get_metrics()
    result = pd.DataFrame(result).T
    
    all_result[data_id] = pd.merge(result, emb_res, left_index=True, right_index=True, how='outer')

all_result = pd.concat(list(all_result.values()), keys = all_result.keys())

# update the result
prev_result = pd.read_csv('../assets/part1/performance_HLCA_subsampled.csv', index_col=[0, 1])
prev_result.update(all_result)
all_result = prev_result
all_result.to_csv('../assets/performance_HLCA_subsampled.csv')

# all_result = pd.read_csv('../assets/part1/performance_HLCA_subsampled.csv', index_col=[0, 1])
# -------- VISUALIZATION

# all_result = pd.read_csv('../assets/performance_HLCA_subsampled.csv', index_col=[0, 1])
all_result = all_result.astype(float)
# minmax scale based on each sample
all_result = (
    all_result
    .reset_index()
    .groupby('level_0')
    .apply(lambda x: minmax_scale(x.set_index('level_1')))
)

all_result['Topic Performance'] = all_result.iloc[:, 0:4].mean(axis=1)
all_result['Bio conservation'] = all_result.iloc[:, 4:9].mean(axis=1)
all_result['Batch correction'] = all_result.iloc[:, 9:14].mean(axis=1)

Overall_metric = all_result[['Topic Performance', 'Batch correction', 'Bio conservation']].reset_index().rename(columns = dict(level_0='Sample', level_1='Method'))
# The weights. determain the order.
Overall_metric['Total'] = Overall_metric['Bio conservation'] * 0.375 + Overall_metric['Batch correction'] * 0.25 + Overall_metric['Topic Performance'].fillna(0) * 0.375
Overall_metric['Method'] = Overall_metric['Method'].replace(index_name_cleaner)
dfvis = Overall_metric.melt(
    id_vars = ['Sample', 'Method'],
    value_vars = ['Topic Performance', 'Batch correction', 'Bio conservation'],
    var_name = 'Metric'
)
#  ----- Figure. 1
g = sns.catplot(
    data=dfvis, kind="bar",
    y="Method", x="value", col="Metric",
    errorbar="sd", 
    palette="Reds", 
    alpha=.5, height=6, 
    sharex = False,
    edgecolor = '.7',
    err_kws={'linewidth': 2},
    order=Overall_metric.groupby('Method')['Total'].mean().sort_values().index,
)
g.set_xlabels('')
g.set_titles('{col_name}')
g.savefig('../assets/barplot_3metric_overall.pdf', dpi=300)
plt.show()

# Figure. 2
scale = 1.5
plt.figure(figsize=(5*scale, 4*scale), dpi=300)
sns.barplot(
    data = Overall_metric.dropna(),
    x = 'Total', y = 'Method', 
    palette='Reds',
    alpha=.5,
    edgecolor = '.7',
    order=Overall_metric.dropna().groupby('Method')['Total'].mean().sort_values().index,
    errwidth=2
)
plt.savefig('../assets/barplot_total_overall.pdf', dpi=300, bbox_inches='tight')
plt.show()
# Figure. 3.
# Bubble.

# x: Batch correction
# y: Bio conservation
# size: Topic Performance

def lighten_color(color, amount=0.5):
    import colorsys
    import matplotlib.colors as mc
    c = color

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]*0.8)

f = plt.figure(figsize=(6, 6), dpi=300)

bubble_vis = Overall_metric.dropna().groupby(['Method']).mean(numeric_only=True).reset_index()
color_map = dict(zip(bubble_vis['Method'], sns.color_palette('tab20', n_colors=len(bubble_vis['Method']))))
color_map = {k: lighten_color(v, 0.75) for k, v in color_map.items()}
ax = sns.scatterplot(
    data = bubble_vis,
    x = 'Batch correction', y = 'Bio conservation', 
    size = 'Topic Performance',
    hue = 'Method',
    sizes = (50, 750),
    alpha=.9,
    edgecolor = '.7',
    palette=color_map
)
# show error bar
for i, m in enumerate(bubble_vis['Method']):
    x = Overall_metric[Overall_metric['Method'] == m]['Batch correction']
    y = Overall_metric[Overall_metric['Method'] == m]['Bio conservation']
    ax.errorbar(
        x = x.mean(), y = y.mean(),
        xerr = x.std(), yerr = y.std(),
        fmt='', 
        color = color_map[m],
        alpha=0.5,
        # set the error bar style
    )

# show special points: 
# baseline: scvi scanvi
ax.scatter(
    x = Overall_metric.query('Method == "SCANVI"')['Batch correction'].mean(),
    y = Overall_metric.query('Method == "SCANVI"')['Bio conservation'].mean(),
    marker = 'x',
    color = '.5',
    # s = 100,
    label = 'SCANVI'
)

ax.scatter(
    x = Overall_metric.query('Method == "SCVI"')['Batch correction'].mean(),
    y = Overall_metric.query('Method == "SCVI"')['Bio conservation'].mean(),
    marker = 'o',
    color = 'white',
    edgecolor = '.5',
    label = 'SCVI',
    linewidth=2
)

leg = ax.legend()
leg.set_visible(False)
# reset the legend
# leg.get_legend_handler_map()
leg.legendHandles[0].get_label() # "Method"
new_leg1 = ax.legend(
    handles = leg.legendHandles[1:14],
    bbox_to_anchor=(1.05, 1), 
    loc=2, borderaxespad=0.,
    edgecolor = 'white',
    ncol=1,
    title='Method',
    title_fontproperties = {'weight':'semibold'},
    alignment='left'
)
new_leg2 = ax.legend(
    handles = leg.legendHandles[15:19],
    bbox_to_anchor=(1.75, 1), 
    loc=2, borderaxespad=0., 
    edgecolor = 'white',
    ncol=1,
    title='Topic Performance',
    title_fontproperties = {'weight':'semibold'},
    alignment='left'
)
new_leg3 = ax.legend(
    handles = leg.legendHandles[19:],
    bbox_to_anchor=(1.75, 0.5), 
    loc=2, borderaxespad=0., 
    edgecolor = 'white',
    ncol=1,
    title='Baseline',
    title_fontproperties = {'weight':'semibold'},
)

f.add_artist(new_leg1)
f.add_artist(new_leg2)
# debug: the added legend is out of the plot
f.savefig('../assets/bubble_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# --- Figure 4. topic performance
topic_perf = (
    all_result.iloc[:, 0:4]
    .dropna()
    .reset_index()
    .rename(columns=dict(level_0='Sample', level_1='Method'))
)
topic_metric = topic_perf.groupby('Method').mean(numeric_only=True)
topic_metric['Overall'] = topic_metric.mean(axis=1)
topic_metric = topic_metric.sort_values('Overall')
topic_perf['Method'] = topic_perf['Method'].replace(index_name_cleaner)
g = sns.catplot(
    data=topic_perf.melt(id_vars=['Sample', 'Method'], var_name='Metric'), 
    kind="bar",
    y="Method", x="value", col="Metric",
    errorbar="sd", 
    palette="Reds", 
    alpha=.5, height=6, 
    sharex = False,
    edgecolor = '.7',
    err_kws={'linewidth': 2},
    order=topic_metric.index.map(index_name_cleaner.get),
)
g.set_xlabels('')
g.set_titles('{col_name}')
plt.show()
g.savefig('../assets/barplot_topic_performance.pdf', dpi=300)
##--- Figure 5. plottabel

def plot_results_table(
    df,
    save_dir = None
):
    """
    df:
        with columns: Total
        with index: metric_type
    """
    import plottable
    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.plots import bar
    import matplotlib
    assert 'metric_type' in df.index
    num_embeds = df.shape[0] - 1
    # change the saturation of the color map
    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=sns.diverging_palette(250, 20, as_cmap=True), num_stds=2.5)  # noqa: E731

    # Sort by total score
    plot_df = df.drop('metric_type', axis=0)
    plot_df = plot_df.sort_values(by="Total", ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index
    # Split columns by metric type, using df as it doesn't have the new method col
    score_cols = df.columns[df.loc['metric_type'] == 'Aggregate']
    other_cols = df.columns.difference(score_cols)
    column_definitions = [
        ColumnDefinition(
            "Method", 
            width=1.5, 
            textprops={"ha": "left", "weight": "bold"}),
    ]
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn(plot_df[col]),
            group=df.loc['metric_type', col],
            formatter="{:.2f}",
        )
        for i, col in enumerate(other_cols)
    ]
    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": sns.diverging_palette(145, 300, s=60, as_cmap=True),#matplotlib.cm.Reds,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
                "textprops": {"fontsize": 10},
            },
            group=df.loc['metric_type', col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols)
    ]
    # Allow to manipulate text post-hoc (in illustrator)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
    
    plt.show()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "scib_results.svg"), facecolor=ax.get_facecolor(), dpi=300)

    return tab

vistab = all_result.reset_index().groupby('level_1').mean(numeric_only=True)
vistab['Total'] = vistab['Bio conservation'] * 0.375 + vistab['Batch correction'] * 0.25 + vistab['Topic Performance'].fillna(0) * 0.375
vistab.loc['metric_type', :] = ['Topic Performance'] * 4 + ['Bio conservation'] * 5 + ['Batch correction'] * 5 + ['Aggregate'] * 4
vistab.rename(columns=metric_name_cleaner, index=index_name_cleaner, inplace=True)
plot_results_table(
    vistab, 
    save_dir='../assets/'
)

### Cluster Metrics

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
    for method in cmethod:
        results = load_results(data_id, method)
        if 'labels' in results:
            label = results['labels']
        elif 'label' in results:
            label = results['label']
            
        nmi = normalized_mutual_info_score(label_true, label)
        ari = adjusted_rand_score(label_true, label)

        cluster_metrics[(data_id, method)] = {'nmi': nmi, 'ari': ari}

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
            cluster_metrics[(data_id, method+'_leiden')] = {'nmi': nmi, 'ari': ari}

cluster_metrics = pd.DataFrame(cluster_metrics).T
drop_method = ['harmony', 'expimap', 'scvi']
cluster_metrics = cluster_metrics.drop(index = [i for i in cluster_metrics.index if i[1] in drop_method])
cluster_metrics.to_csv('../assets/part1/cluster_metrics.csv')
cluster_metrics.reset_index().drop(columns='level_0').groupby('level_1').mean()

ordered_map = {
    'harmony_leiden': 'Harmony + leiden',
    'expimap_leiden': 'expiMap + leiden',
    'scvi_leiden': 'SCVI + leiden',
    'dcjcomm': 'DcjComm',
    'dcjcomm_leiden': 'DcjComm + leiden',
    'scanvi_seed_label': 'SCANVI',
    'scanvi_seed_label_leiden': 'SCANVI + leiden',
    'topicvi': 'TopicVI',
    'topicvi_leiden': 'TopicVI + leiden',
    'topicvi_denovo_finding': 'TopicVI (denovo)',
    'topicvi_denovo_finding_leiden': 'TopicVI (denovo) + leiden',
}
# order = list(ordered_map.keys())
order = cluster_metrics.reset_index().groupby('level_1').apply(lambda x: x.set_index('level_0').sum(axis=1)).sum(axis=1).sort_values().index

f, axes = plt.subplots(1, 2, figsize=(6, 6))
cluster_metrics.unstack()['nmi'][order].plot(
    kind='box', 
    ax=axes[0],
    showmeans=True,
    meanprops = dict(marker='D', markerfacecolor='r', markeredgecolor='r', alpha = 0.25),
    vert=False
)
axes[0].set_xlabel('NMI')
axes[0].set_yticklabels([ordered_map[i] for i in order])
cluster_metrics.unstack()['ari'][order].plot(
    kind='box', 
    ax=axes[1],
    showmeans=True,
    meanprops = dict(marker='D', markerfacecolor='r', markeredgecolor='r', alpha = 0.25),
    vert=False
)
axes[1].set_yticklabels([])
axes[1].set_xlabel('ARI')

f.savefig('../assets/boxplot_cluster_metrics.pdf', dpi=300, bbox_inches='tight')

# stastics test
# pairwise test
# from scipy.stats import wilcoxon

# dftest = cluster_metrics.unstack()['nmi']
# wilcoxon(dftest['topicvi'], dftest['scanvi_seed_label'])
# wilcoxon(dftest['topicvi_leiden'], dftest['scanvi_seed_label_leiden'])

# dftest = cluster_metrics.unstack()['ari']
# wilcoxon(dftest['topicvi'], dftest['scanvi_seed_label'])
# wilcoxon(dftest['topicvi_leiden'], dftest['scanvi_seed_label_leiden'])

# ------------
# explainability of topics.
# methods.remove('expimap')
methods = os.listdir(os.path.join(result_dir, data_ids[0]))
methods = [i for i in methods if '.' not in i]
methods.remove('topicvi_seed_labels')
methods.remove('expimap') # all overlap with prior

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
            loading = mres['embedding']
            factors = mres['factors'].T
            adata_ = adata[:, mres['factor_genes']].copy()
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
    explains.update(result)
explains = pd.DataFrame(explains).T
explains.reset_index(inplace=True)
explains.columns = ['data_id', 'method', 'topic_idx','topic', 'overlap']
explains['overlap'] = explains['overlap'].astype(float)
explains['With_prior'] = explains['method'].isin(
    ['topicvi', 'spectra', 'muvi']
)

explains.groupby('method')['overlap'].mean()

f, ax = plt.subplots(figsize=(5, 5), dpi=300)
ax = sns.barplot(
    data=explains,
    y='overlap', x='method', hue='With_prior',
    order=explains.groupby('method')['overlap'].mean().sort_values().index,
    edgecolor = '.7',
    alpha=.5,
    errwidth=1,
    errorbar=("ci", 95),
)
ax.set_xticklabels(
    [index_name_cleaner[i.get_text()] for i in ax.get_xticklabels()], 
    rotation=90
)
ax.set_xlabel('')
ax.set_ylabel('Max overlap with prior gene sets')
leg = ax.legend(frameon=False, title=None, )
leg.get_texts()[0].set_text('Without prior')
leg.get_texts()[1].set_text('With prior')
f.savefig('../assets/barplot_topic_explainability.pdf', bbox_inches='tight')