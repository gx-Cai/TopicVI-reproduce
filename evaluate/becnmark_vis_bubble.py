# x: Batch correction
# y: Bio conservation
# size: Topic Performance

import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import adjustText as aT

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6

def minmax_scale(x):
    return (x - x.min()) / (x.max() - x.min())

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
dfvis = Overall_metric.melt(
    id_vars = ['Sample', 'Method'],
    value_vars = ['Topic Performance', 'Batch correction', 'Bio conservation'],
    var_name = 'Metric'
)



bubble_vis = Overall_metric.dropna().groupby(['Method']).mean(numeric_only=True).reset_index()
# geometric mean
bubble_vis['Total_Geo'] = (bubble_vis['Bio conservation'] * bubble_vis['Batch correction'] * bubble_vis['Topic Performance']) ** (1/3)

scale = 1.75
f = plt.figure(figsize=(3*scale, 3*scale), dpi=300)
# color_map = dict(zip(bubble_vis['Method'], sns.color_palette('tab20', n_colors=len(bubble_vis['Method']))))
# color_map = {k: lighten_color(v, 0.75) for k, v in color_map.items()}
color_map = dict(
    zip(
    bubble_vis['Method'], 
    [sns.color_palette('Reds', as_cmap=True)(i) for i in minmax_scale(bubble_vis['Total_Geo'])]
    )
)

ax = sns.scatterplot(
    data = bubble_vis,
    x = 'Batch correction', y = 'Bio conservation', 
    size = 'Topic Performance',
    hue = 'Method',
    sizes = (50, 1200),
    alpha=.9,
    edgecolor = '.7',
    # color show the Total_Geo
    palette=color_map,
)
# show error bar
# for i, m in enumerate(bubble_vis['Method']):
#     x = Overall_metric[Overall_metric['Method'] == m]['Batch correction']
#     y = Overall_metric[Overall_metric['Method'] == m]['Bio conservation']
#     ax.errorbar(
#         x = x.mean(), y = y.mean(),
#         xerr = x.std(), yerr = y.std(),
#         fmt='', 
#         color = color_map[m],
#         alpha=0.5,
#     )

# add text
texts = []
for i, row in bubble_vis.iterrows():
    texts.append(
        ax.text(
            row['Batch correction'], row['Bio conservation'], row['Method'],
            fontsize=10,
            weight='semibold',
            color='k' if row['Method'] in ['TopicVI', 'TopicVI (denovo)'] else '.55',
        )
    )
aT.adjust_text(
    texts,
    # arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
    expand=(1.75, 2),
    # force_text=0.5,
    force_points=0.5,
    # lim=1000,
    ax=ax,
    add_objects = f,
)

leg = ax.legend()
leg.set_visible(False)

# new_leg1 = ax.legend(
#     handles = leg.legend_handles[1:14],
#     bbox_to_anchor=(1.05, 1), 
#     loc=2, borderaxespad=0.,
#     edgecolor = 'white',
#     ncol=1,
#     title='Method',
#     title_fontproperties = {'weight':'semibold'},
#     alignment='left'
# )


new_leg2 = ax.legend(
    handles = leg.legend_handles[15:19],
    bbox_to_anchor=(1.05, 0.85), 
    loc=2, borderaxespad=0., 
    edgecolor = 'white',
    ncol=1,
    title='Topic\nPerformance',
    title_fontproperties = {'weight':'semibold'},
    # alignment='left',
    labelspacing=1,  # 控制每个图例项之间的垂直间距（默认是 0.5）
    handletextpad=1.8,
    handlelength=1.0        # 控制 marker 的长度
)

color_bar = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=bubble_vis['Total_Geo'].min(), vmax=bubble_vis['Total_Geo'].max()))
color_bar.set_array(bubble_vis['Total_Geo'])
cax = f.add_axes([0.95, 0.1, 0.03, 0.35])  # 绝对位置，单位是 figure 的比例
cbar = f.colorbar(
    color_bar, 
    # fraction=0.03, pad=0.04, 
    orientation='vertical',
    location='right',
    cax=cax
)
cbar.set_label('Overall Performance\n(Geometric Mean)', fontsize=10, weight='semibold')
cbar.ax.tick_params(labelsize=10)
cbar.outline.set_edgecolor('white')
cbar.outline.set_linewidth(1.5)
cbar.outline.set_alpha(0.6)

ax.set_xlabel('Batch Correction', fontsize=12, weight='semibold')
ax.set_ylabel('Bio Conservation', fontsize=12, weight='semibold')

f.add_artist(new_leg2)
f.savefig('../assets/heat_bubble_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()

