import os
import scanpy as sc
import numpy as np
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BatchCorrection
from running.tl import load_config
from topicvi.metrics import TopicMetrics
from tqdm import tqdm


def load_results(data_id, method):
    dir = os.path.join(result_dir, data_id, method, 'results.npz')
    return np.load(dir, allow_pickle=True)

def value_valid_trans(X):
    idx = np.isinf(X) | np.isnan(X)
    X[idx] = 0
    return X


result_dir = '../results/subsampled/'
data_ids = os.listdir(result_dir)

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
prev_result.update(all_result.astype(float))
all_result = prev_result
all_result.to_csv('../assets/performance_HLCA_subsampled.csv')
