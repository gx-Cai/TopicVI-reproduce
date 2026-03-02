import torch
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import topicvi
from tqdm import trange
from matplotlib.colors import LinearSegmentedColormap
import adjustText as aT
import re

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6

model_anatomy = torch.load('../results/GSE233208_ADDS/spatial_model.torch', weights_only=False)
model_disease = torch.load('../results/GSE233208_ADDS/topicvi/supervised_disease/model.torch', weights_only=False)
model_anatomy.store_topics_info()
model_disease.store_topics_info()

prior = model_anatomy.adata.uns['annotation']
prior = {**prior['background'], **prior['clusters']}

topics_anatomy = topicvi.TopicDict.transfer_from_adata(model_anatomy.adata)
topics_disease = topicvi.TopicDict.transfer_from_adata(model_disease.adata)
topics_anatomy.compare_prior_overlap(prior)
topics_disease.compare_prior_overlap(prior)

prior_overlaps = pd.DataFrame(
    index = list(prior),
    columns = ['anatomy', 'disease'],
    data=0
)

for scan in ['anatomy', 'disease']:
    topics = {'anatomy': topics_anatomy, 'disease': topics_disease}[scan]
    for ti in trange(len(topics)):
        ati:pd.Series = topics.scan_topic_overlap(ti)
        for pi in ati.index:
            if prior_overlaps.loc[pi, scan] < ati[pi]:
                prior_overlaps.loc[pi, scan] = ati[pi]

prior_overlaps = prior_overlaps[~(prior_overlaps == 0).all(axis=1)]
prior_overlaps['discrepancy'] = np.abs(prior_overlaps['anatomy'] - prior_overlaps['disease'])
prior_overlaps.to_csv('../assets/spatial/prior_overlaps.csv')
# from grey to color
palette = LinearSegmentedColormap.from_list("grey_radian", [".75", "#3366cc"])

scale = 1.5
f, ax = plt.subplots(dpi=150, figsize=(4*scale,4*scale))
ax.scatter(
    x = prior_overlaps['anatomy'],
    y = prior_overlaps['disease'],
    c = prior_overlaps['discrepancy'],
    cmap = palette,
    edgecolor = 'none'
)

x = np.linspace(0, 1, 10)  # from 0 to 10
ax.plot(x, x, label='y = x', color='.25', linestyle='-')
ax.text(x = 0.85, y=0.9, s ='$y=x$', fontdict=dict(rotation=45, fontsize=14, color = '.25'))
ax.set_ylabel('Overlaps (supervised by disease)', fontsize=12)
ax.set_xlabel('Overlaps (supervised by anatomy annotation)', fontsize=12)

# items = prior_overlaps['discrepancy'].sort_values(ascending=False).index[0: ntop]

items = [
    'Biosynthesis Of N-glycan Precursor (Dolichol LLO) And Transfer To Protein R-HSA-446193',
    'Intrinsic Pathway For Apoptosis R-HSA-109606',
    'RNA Polymerase II Transcription Termination R-HSA-73856',
    'Glycosaminoglycan Metabolism R-HSA-1630316',
    'MTOR Signaling R-HSA-165159',
    'Transport Of Mature Transcript To Cytoplasm R-HSA-72202',
    'mRNA 3-End Processing R-HSA-72187',
    'NCAM Signaling For Neurite Out-Growth R-HSA-375165',
    'RHOD GTPase Cycle R-HSA-9013405',
    'Cell Junction Organization R-HSA-446728',
    'Neurotransmitter Release Cycle R-HSA-112310',
    'Transcription-Coupled Nucleotide Excision Repair (TC-NER) R-HSA-6781827',
    'Telomere Maintenance R-HSA-157579'
]

patten = re.compile(r'([\s\S]+) R-HSA-\d+')

texts = []
for it in items:
    if len(patten.findall(it)) > 0:
        s = patten.findall(it)[0]
    else:
        s = it
    texts.append(
        ax.text(
            x = prior_overlaps.loc[it, 'anatomy'],
            y = prior_overlaps.loc[it, 'disease'],
            s = s,
            fontsize=9
        )
    )

aT.adjust_text(
    texts,
    ax=ax,
    only_move={'points': 'xy', 'text': 'xy'},
    arrowprops=dict(arrowstyle='->', color='gray'),
    force_text=0.5,
    ensure_inside_axes = False
)

f.savefig('../assets/ADDS_topics_alignment.pdf', bbox_inches='tight')
