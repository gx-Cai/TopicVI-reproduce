import os
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['pdf.fonttype'] = 42

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

def load_results(data_id, method):
    dir = os.path.join(result_dir, data_id, method, 'results.npz')
    return np.load(dir, allow_pickle=True)

methods = os.listdir(os.path.join(result_dir, data_ids[0]))
methods = [i for i in methods if '.' not in i]
 
# -------- VISUALIZATION

all_result = pd.read_csv('../assets/part1/performance_HLCA_subsampled.csv', index_col=[0, 1])
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
Overall_metric['Total_Geo'] = (Overall_metric['Bio conservation'] * Overall_metric['Batch correction'] * Overall_metric['Topic Performance']) ** (1/3)

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
vistab.dropna(inplace=True)
vistab['Total'] = (vistab['Bio conservation'] * vistab['Batch correction'] * vistab['Topic Performance']) ** (1/3)

# ---
vistab.drop(columns = ['Topic Performance', 'Bio conservation', 'Batch correction'], inplace=True)
vistab.loc['metric_type', :] = ['Topic Performance'] * 4 + ['Bio conservation'] * 5 + ['Batch correction'] * 5 + ['Aggregate'] * 1
vistab.rename(columns=metric_name_cleaner, index=index_name_cleaner, inplace=True)
plot_results_table(
    vistab, 
    save_dir='../assets/'
)

# ---

scale = 1
# Normalize values for colormap
norm = plt.Normalize(Overall_metric['Total_Geo'].min(), Overall_metric['Total_Geo'].max())
cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
colors = {
    idx: cmap(norm(val)) for idx, val in Overall_metric.groupby('Method')['Total_Geo'].mean().items()
}

f = plt.figure(figsize=(2*scale, 4*scale), dpi=150)
ax = sns.barplot(
    data = Overall_metric.dropna(),
    x = 'Total_Geo', y = 'Method', 
    palette=colors,
    # alpha=.5,
    edgecolor = '.7',
    order=Overall_metric.dropna().groupby('Method')['Total_Geo'].mean().sort_values(ascending=False).index,
    errwidth=1
)

ax.set_ylabel('')
ax.set_xlabel('Total Score')
f.savefig('../assets/part1/total_bar.pdf', bbox_inches='tight')