import pandas as pd
import numpy as np
import os
from pathlib import Path
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['pdf.fonttype'] = 42

def cal_topic_score(adata: AnnData, topic_weight: pd.DataFrame, z_scale=True) -> pd.DataFrame:
    common_genes = adata.var_names.intersection(topic_weight.index)
    assert len(common_genes) > 0, "No common genes between adata and topic_weight."
    if len(common_genes) < topic_weight.shape[0]:
        print(f"Warning: Only {len(common_genes)} out of {topic_weight.shape[0]} genes are common.")
    topic_score = pd.DataFrame(
        np.dot(adata[:, common_genes].X, topic_weight.loc[common_genes, :]),
        index=adata.obs_names,
        columns=topic_weight.columns
    )
    if z_scale:
        topic_score = (topic_score - topic_score.mean()) / topic_score.std()
    return topic_score

data_base_dir = Path('D:/Data/')

# 1. load ccle data
if False:
    ccle_data_dir = data_base_dir / 'CCLE'
    meta_data_file = ccle_data_dir / 'Model.csv'
    ccle_data_file = ccle_data_dir / 'OmicsExpressionRawReadCountHumanAllGenes.csv'

    meta = pd.read_csv(meta_data_file, index_col=0)
    data = pd.read_csv(ccle_data_file, index_col='ModelID')
    data.drop(columns=['Unnamed: 0'], inplace=True)
    data_extra_cols = [data.pop(i) for i in ['ProfileID', 'is_default_entry']]
    data.columns = data.columns.str.split(' ').str[0]
    data.columns = data.columns.str.split('.').str[0]

    adata = AnnData(
        X = data.values,
        obs = meta.loc[data.index, :],
        var = pd.DataFrame(index=data.columns)
    )

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=200)

    for col in adata.obs.columns:
        if adata.obs[col].isna().all():
            adata.obs.drop(columns=[col], inplace=True)

    adata.obs['SerumFreeMedia'] = adata.obs['SerumFreeMedia'].astype('str')
    adata.obs['PediatricModelType'] = adata.obs['PediatricModelType'].astype('str')

    # adata.write_h5ad(data_base_dir / 'CCLE' / 'ccle_raw_filtered.h5ad', compression='gzip')

adata = sc.read_h5ad(data_base_dir / 'CCLE' / 'ccle_log1p_filtered.h5ad')
adata = adata[adata.obs.query('OncotreePrimaryDisease != "Non-Cancerous"').index, :].copy()
# cal the topic score.
topic32_weight = pd.read_csv('../assets/ZhaoSim2021/topic32_weights.tsv', sep='\t', index_col=0)
topic12_weight = pd.read_csv('../assets/ZhaoSim2021/topic12_weights.tsv', sep='\t', index_col=0)
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
topic32_weight.index = topic32_weight.index.to_series().replace(gene_names_cleaned)

assert topic12_weight.index.isin(adata.var_names).all() and topic32_weight.index.isin(adata.var_names).all()

adata.obs['topic32_score'] = cal_topic_score(adata, topic32_weight)['score']
adata.obs['topic12_score'] = cal_topic_score(adata, topic12_weight)['0']

plt.figure(dpi=150)
dfvis = (
    adata.obs.query("`OncotreeCode` == 'GB'")
    .set_index('CellLineName')['topic32_score']
    .sort_values(ascending=False)
)

dfvis.index = dfvis.index.str.replace('CCLF_NEURO', 'CCLF')
ax = (
    dfvis
    .plot(
        kind='bar', 
        figsize=(10,3),
        width=1,
        color='skyblue',
        edgecolor=None,
        linewidth=0,
    )
)
ax.set_ylabel('Topic 32 Score')
ax.set_xlabel('')
ax.set_title('CCLE GBM Cell Lines Topic 32 Scores')
ax.grid(axis='x', visible=False)
# highlight specific cell lines
highlight_ccl = ['A-172']
# move ticks to x = 0, for value over zero, put under the x-axis, else put above the x-axis
for i, tick in enumerate(ax.get_xticklabels()):
    if dfvis.iloc[i] > 0:
        tick.set_y(0.55)
    else:
        tick.set_y(0.63,)
        tick.set_verticalalignment('bottom')
    if np.abs(dfvis.iloc[i]) > 1:
        tick.set_color('k')
    else:
        tick.set_color('gray')
    if tick.get_text() in highlight_ccl:
        tick.set_fontweight('bold')
plt.savefig('../assets/ccle_drug_response/ccle_gbm_topic32_scores_barplot.pdf', bbox_inches='tight')

# --- drug response data ---
drug_response_dir = data_base_dir / 'DrugResponse'
os.listdir(drug_response_dir)

### AUC data from:
# PRISMRepurposing19Q4 / secondary-screen-dose-response-curve-parameters.csv
# GDSC / sanger-dose-response.csv
# SangerDrugCombo / SangerCombo2022ComboFit.csv
# CTRPv2.0 / v20.data.curves_post_qc.txt; v20.meta.per_cell_line.txt; v20.meta.per_compound.txt

prism_drug_response = pd.read_csv(drug_response_dir / 'PRISMRepurposing19Q4' / 'secondary-screen-dose-response-curve-parameters.csv')
prism_drug_response = prism_drug_response[['depmap_id', 'auc', 'name', 'moa', 'target']]
prism_drug_response = prism_drug_response.pivot_table(
    index='depmap_id', columns='name', values='auc'
)


gdsc_drug_response = pd.read_csv(drug_response_dir / 'GDSC' / 'sanger-dose-response.csv')
gdsc_drug_response = gdsc_drug_response[['ARXSPAN_ID', 'DRUG_NAME', 'AUC_PUBLISHED']]
gdsc_drug_response = gdsc_drug_response.pivot_table(
    index='ARXSPAN_ID', columns='DRUG_NAME', values='AUC_PUBLISHED'
)

sanger_drug_response = pd.read_csv(drug_response_dir / 'SangerDrugCombo' / 'SangerCombo2022ComboFit.csv')
sanger_drug_response = sanger_drug_response[['ModelID', 'lib_name', 'AUC']]
sanger_drug_response = sanger_drug_response.pivot_table(
    index='ModelID', columns='lib_name', values='AUC'
)

ctrp_drug_response = pd.read_csv(drug_response_dir / 'CTRPv2.0' / 'v20.data.curves_post_qc.txt', sep='\t')
ctrp_drug_response_meta_cell = pd.read_csv(drug_response_dir / 'CTRPv2.0' / 'v20.meta.per_cell_line.txt', sep='\t')
ctrp_drug_response_meta_compound = pd.read_csv(drug_response_dir / 'CTRPv2.0' / 'v20.meta.per_compound.txt', sep='\t')
ctrp_drug_response_meta_expr = pd.read_csv(drug_response_dir / 'CTRPv2.0' / 'v20.meta.per_experiment.txt', sep='\t')
ctrp_drug_response = pd.merge(
    ctrp_drug_response[['experiment_id', 'master_cpd_id', 'area_under_curve']],
    ctrp_drug_response_meta_expr[['experiment_id', 'master_ccl_id']],
    on='experiment_id'
)
ctrp_drug_response = pd.merge(
    ctrp_drug_response,
    ctrp_drug_response_meta_compound[['master_cpd_id', 'cpd_name']],
    on='master_cpd_id'
)
ctrp_drug_response = pd.merge(
    ctrp_drug_response,
    ctrp_drug_response_meta_cell[['master_ccl_id', 'ccl_name']],
    on='master_ccl_id'
)
ctrp_drug_response = ctrp_drug_response[['cpd_name', 'ccl_name', 'area_under_curve']]
ctrp_drug_response.to_csv(drug_response_dir / 'CTRPv2.0' / 'ctrp_drug_response_processed.csv', index=False)
ctrp_drug_response = ctrp_drug_response.pivot_table(
    index='ccl_name', columns='cpd_name', values='area_under_curve'
)
new_index = ctrp_drug_response.index.map({v:k for k,v in adata.obs['StrippedCellLineName'].to_dict().items()})
ctrp_drug_response.index = new_index
ctrp_drug_response = ctrp_drug_response[~ctrp_drug_response.index.isna()]



## --- 
from scipy.stats import spearmanr, pearsonr

def scan_significant_correlation(x, y, verbose=False):
    r, p = spearmanr(x, y)
    if verbose or (p < 0.05 and abs(r) > 0.2):
        print(f"Spearmanr: {r:.3f}, p-value: {p:.3e}")
    r, p = pearsonr(x, y)
    if verbose or (p < 0.05 and abs(r) > 0.2):
        print(f"Pearsonr: {r:.3f}, p-value: {p:.3e}")

def plot_topic_regressions(dfvis, y):
    f, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    reg_line_kws = {'line_kws':{'alpha':0.5, 'color':'.25'}}
    sns.regplot(data=dfvis, x='topic32_score', y=y, ax=axes[0], **reg_line_kws)
    sns.regplot(data=dfvis, x='topic12_score', y=y, ax=axes[1], **reg_line_kws)

    stat_sp, p_val_sp = spearmanr(dfvis['topic32_score'], dfvis[y])
    stat_pe, p_val_pe = pearsonr(dfvis['topic32_score'], dfvis[y])
    axes[0].text(1, 1, f'Spearmanr: {stat_sp:.3f} (p={p_val_sp:.3f})\nPearsonr: {stat_pe:.3f} (p={p_val_pe:.3f})', ha='right', va='top', transform=axes[0].transAxes)
    axes[0].set_ylabel(y.replace('_', ' of '))
    axes[0].set_xlabel('Topic 32 Score')

    stat_sp, p_val_sp = spearmanr(dfvis['topic12_score'], dfvis[y])
    stat_pe, p_val_pe = pearsonr(dfvis['topic12_score'], dfvis[y])
    axes[1].set_ylabel('')
    axes[1].text(1, 1, f'Spearmanr: {stat_sp:.3f} (p={p_val_sp:.3f})\nPearsonr: {stat_pe:.3f} (p={p_val_pe:.3f})', ha='right', va='top', transform=axes[1].transAxes)
    axes[1].set_xlabel('Topic 12 Score')

    return f, axes

overlap_base = set(prism_drug_response.index) & set(gdsc_drug_response.index) & set(ctrp_drug_response.index)

def zscore_auc(df:pd.Series, target_index=None):
    overlaps = df.index.intersection(overlap_base)
    # dfz = (df - df.loc[overlaps].median()) / df.loc[overlaps].std()
    q = 0.01
    min_val = df.loc[overlaps].quantile(q)
    max_val = df.loc[overlaps].quantile(1-q)
    dfz = (df - min_val) / (max_val - min_val)
    if target_index is not None:
        dfz = dfz.loc[target_index]
    return dfz

test_data_id = 'prism'

target_drugs = [
    'etoposide',  
    'lorlatinib'
    # 'Tazemetostat',
    # 'RO4929097',
    # 'Ispinesib',
    # 'panobinostat',
    # 'Ana-12'
]

valid_GBM_ccl = adata.obs.query("`OncotreeCode` == 'GB'").index
adata.obs.loc[valid_GBM_ccl, :].sort_values(by='topic32_score', ascending=False)
adata.obs.loc[valid_GBM_ccl, ['topic32_score', 'topic12_score']].corr()

use_all_ccl = False

if test_data_id == 'prism':
    use_all_ccl = True

    [i for i in target_drugs if i.upper() in prism_drug_response.columns.str.upper()]
    if use_all_ccl:
        prism_overlap_ccl = adata.obs_names.intersection(prism_drug_response.index)
    else:
        prism_overlap_ccl = valid_GBM_ccl.intersection(prism_drug_response.index)

    dfvis = pd.DataFrame(
        {
            'AUC_Etoposide': prism_drug_response.loc[prism_overlap_ccl, 'etoposide'],
            'AUC_Lorlatinib': prism_drug_response.loc[prism_overlap_ccl, 'lorlatinib'],
            'AUC_Panobinostat': prism_drug_response.loc[prism_overlap_ccl, 'panobinostat'],
            'AUC_Tazemetostat': prism_drug_response.loc[prism_overlap_ccl, 'tazemetostat'],
            'AUC_Ispinesib': prism_drug_response.loc[prism_overlap_ccl, 'ispinesib'],
            'topic32_score': adata.obs.loc[prism_overlap_ccl, 'topic32_score'],
            'topic12_score': adata.obs.loc[prism_overlap_ccl, 'topic12_score'],
            'bot10AUC': prism_drug_response.loc[prism_overlap_ccl, :].apply(lambda x: x.sort_values().iloc[0:10].mean(), axis=1)
        }
    )
    dfvis = dfvis.dropna()
    dfvis.corr()
    # for y in ['AUC_Etoposide', 'AUC_Panobinostat', 'AUC_Tazemetostat', 'AUC_Ispinesib', 'bot10AUC']:
    #     plot_topic_regressions(dfvis, y)

    f, axes = plot_topic_regressions(dfvis[dfvis['AUC_Etoposide'] > 1], 'AUC_Etoposide')
    axes[0].set_title(f'{dfvis[dfvis["AUC_Etoposide"] > 1].shape[0]} cell lines (AUC > 1)')
    f.savefig('../assets/ccle_prism_etoposide_topic_regression_filtered.pdf', bbox_inches='tight')
    f, axes = plot_topic_regressions(dfvis, 'AUC_Etoposide')
    axes[0].set_title(f'{dfvis.shape[0]} cell lines (all)')
    f.savefig('../assets/ccle_prism_etoposide_topic_regression_all.pdf', bbox_inches='tight')

    # scan for potential related drugs
    outcome = []
    for drug in prism_drug_response.columns:
        drug_score = prism_drug_response.loc[prism_overlap_ccl, drug]
        min_val = drug_score.quantile(0.01)
        max_val = drug_score.quantile(0.99)
        drug_score_z = (drug_score - min_val) / (max_val - min_val)
        drug_score_z = drug_score_z.dropna()
        topic_score = adata.obs.loc[drug_score_z.index, 'topic32_score']
        r, p = spearmanr(drug_score_z, topic_score)
        r_, p_ = pearsonr(drug_score_z, topic_score)

        if p_ < 0.05 and abs(r_) > 0.2:
            outcome.append((drug, r, p, r_, p_))
            print(f"{drug}: Spearmanr: {r:.3f} (p={p:.3e}); Pearsonr: {r_:.3f} (p={p_:.3e})")
    outcome = pd.DataFrame(outcome, columns=['drug', 'spearmanr', 'spearman_p', 'pearsonr', 'pearson_p'])
    outcome.to_csv('../assets/ccle_prism_topic32_correlation_scan.csv', index=False)

elif test_data_id == 'gdsc':
    [i for i in target_drugs if i.upper() in gdsc_drug_response.columns.str.upper()]
    use_all_ccl = False
    if use_all_ccl:
        gdsc_overlap_ccl = adata.obs_names.intersection(gdsc_drug_response.index)
    else:
        gdsc_overlap_ccl = valid_GBM_ccl.intersection(gdsc_drug_response.index)
    
    dfvis = pd.DataFrame(
        {   
            'AUC_Etoposide': gdsc_drug_response.loc[gdsc_overlap_ccl, 'ETOPOSIDE'],
            'AUC_Panobinostat': gdsc_drug_response.loc[gdsc_overlap_ccl, 'PANOBINOSTAT'],
            'bot10AUC': gdsc_drug_response.loc[gdsc_overlap_ccl, :].apply(lambda x: np.mean(x.sort_values().iloc[0:10]), axis=1),
            'topic32_score': adata.obs.loc[gdsc_overlap_ccl, 'topic32_score'],
            'topic12_score': adata.obs.loc[gdsc_overlap_ccl, 'topic12_score']
        }
    )

    dfvis = dfvis.dropna()
    dfvis.corr()

    plot_topic_regressions(dfvis, 'AUC_Etoposide')

elif test_data_id == 'sanger':
    [i for i in target_drugs if i.upper() in sanger_drug_response.columns.str.upper()]
    
    if use_all_ccl:
        sanger_overlap_ccl = adata.obs_names.intersection(sanger_drug_response.index)
    else:
        sanger_overlap_ccl = valid_GBM_ccl.intersection(sanger_drug_response.index)

    dfvis = pd.DataFrame(
        {
            'topic32_score': adata.obs.loc[sanger_overlap_ccl, 'topic32_score'],
            'topic12_score': adata.obs.loc[sanger_overlap_ccl, 'topic12_score'],
            'bot10AUC': sanger_drug_response.loc[sanger_overlap_ccl, :].apply(lambda x: x.sort_values().iloc[0:10].mean(), axis=1)
        }
    )
    dfvis = dfvis.dropna()
    dfvis.corr()
    plot_topic_regressions(dfvis, 'bot10AUC')

elif test_data_id == 'ctrp':
    [i for i in target_drugs if i.upper() in ctrp_drug_response.columns.str.upper()]

    if use_all_ccl:
        ctrp_overlap_ccl = adata.obs_names.intersection(ctrp_drug_response.index)
    else:
        ctrp_overlap_ccl = valid_GBM_ccl.intersection(ctrp_drug_response.index)

    dfvis = pd.DataFrame(
        {
            'AUC_Etoposide': ctrp_drug_response.loc[ctrp_overlap_ccl, 'etoposide'],
            'AUC_RO4929097': ctrp_drug_response.loc[ctrp_overlap_ccl, 'RO4929097'],
            'bot10AUC': ctrp_drug_response.loc[ctrp_overlap_ccl, :].apply(lambda x: x.sort_values().iloc[0:10].mean(), axis=1),
            'topic32_score': adata.obs.loc[ctrp_overlap_ccl, 'topic32_score'],
            'topic12_score': adata.obs.loc[ctrp_overlap_ccl, 'topic12_score']
        }
    )
    dfvis = dfvis.dropna()
    dfvis.corr()
    for y in ['AUC_Etoposide', 'AUC_RO4929097', 'bot10AUC']:
        plot_topic_regressions(dfvis, y)


##--- etoposide merged

dfvis = [
    zscore_auc(prism_drug_response['etoposide'], prism_overlap_ccl).rename('PRISM'),
    zscore_auc(gdsc_drug_response['ETOPOSIDE'], gdsc_overlap_ccl).rename('GDSC'),
    zscore_auc(ctrp_drug_response['etoposide'], ctrp_overlap_ccl).rename('CTRP'),
]
dfvis = pd.concat(dfvis, axis=1)

dfvis_consensus = dfvis.median(axis=1)
# filter outlier
# dfvis_consensus.drop(index=['ACH-000269'], inplace=True)
dfvis_consensus = pd.DataFrame({
    'zscore_AUC (etoposide)': dfvis_consensus,
    'topic32_score': adata.obs.loc[dfvis_consensus.index, 'topic32_score'],
    'topic12_score': adata.obs.loc[dfvis_consensus.index, 'topic12_score'],
}).dropna()
dfvis = dfvis_consensus.copy()
# f, axes = plot_topic_regressions(dfvis_consensus, 'zscore_AUC (etoposide)')
f, ax = plt.subplots(1, 1, figsize=(3, 3))
y = 'zscore_AUC (etoposide)'
reg_line_kws = {'line_kws':{'alpha':0.5, 'color':'.25'}}
sns.regplot(data=dfvis, x='topic32_score', y=y, ax=ax, **reg_line_kws)

stat_sp, p_val_sp = spearmanr(dfvis['topic32_score'], dfvis[y])
stat_pe, p_val_pe = pearsonr(dfvis['topic32_score'], dfvis[y])
ax.text(0.95, 0.25, f'Spearmanr: {stat_sp:.3f} (p={p_val_sp:.3f})\nPearsonr: {stat_pe:.3f} (p={p_val_pe:.3f})', ha='right', va='top', transform=ax.transAxes)
ax.set_ylabel(y.replace('_', ' of '))
ax.set_xlabel('Topic 32 Score')
ax.set_title(f'{dfvis_consensus.shape[0]} GBM cell lines')

f.savefig('../assets/ccle_etoposide_topic_regression_consensus.pdf', bbox_inches='tight')

### Other fromat
# Repurposing / Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv; Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv

repurposing_drug_response = pd.read_csv(drug_response_dir / 'Repurposing' / 'Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv', index_col=0)
drug_names = pd.read_csv(drug_response_dir / 'Repurposing' / 'Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv', index_col=0)
drug_names = drug_names[['IDs', 'Drug.Name']].set_index('IDs')['Drug.Name']
drug_names.drop_duplicates(inplace=True)
repurposing_drug_response.index = repurposing_drug_response.index.map(drug_names)
repurposing_drug_response = repurposing_drug_response[~repurposing_drug_response.index.isna()]
repurposing_drug_response = repurposing_drug_response.T

if use_all_ccl:
    repurposing_overlap_ccl = adata.obs_names.intersection(repurposing_drug_response.index)
else:
    repurposing_overlap_ccl = valid_GBM_ccl.intersection(repurposing_drug_response.index)

print('valid_celllines:',repurposing_overlap_ccl.size)

all_valids = {}
for i in target_drugs:
    if i.upper() not in repurposing_drug_response.columns:
        continue
    all_valids[i] = repurposing_drug_response.loc[repurposing_overlap_ccl, i.upper()]

all_valids['bot10LFC'] = repurposing_drug_response.loc[repurposing_overlap_ccl, :].apply(lambda x: x.sort_values().iloc[0:10].mean(), axis=1)
all_valids['topic32_score'] = adata.obs.loc[repurposing_overlap_ccl, 'topic32_score']
all_valids['topic12_score'] = adata.obs.loc[repurposing_overlap_ccl, 'topic12_score']
dfvis = pd.DataFrame(all_valids)#.dropna()
dfvis.iloc[:, 0:5].boxplot()

for y in [i for i in target_drugs if i.upper() in repurposing_drug_response.columns]+['bot10LFC']:
    plot_topic_regressions(dfvis, y)
