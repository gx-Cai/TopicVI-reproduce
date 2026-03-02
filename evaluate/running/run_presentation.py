import os
import pickle
import sys
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

sys.path.append("../../external/")

def scgpt_zero_shot(
    adata, 
    train_kwargs=dict(),
    model_kwargs=dict(pretrained_model_path=None),
    data_kwargs=dict(),
    save_model=None,
):
    """    
    Modified from.
    https://github.com/bowang-lab/scGPT/blob/main/tutorials/zero-shot/Tutorial_ZeroShot_Reference_Mapping.ipynb
    """
    import scgpt as scg
    from pathlib import Path

    model_dir = Path(model_kwargs["pretrained_model_path"])
    adata.X = adata.X.toarray()
    adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col="index",
        max_length = train_kwargs.get("max_length", 1200),
        batch_size = 64,
        obs_to_save = None,
        device = train_kwargs.get("device", "cuda"),
        use_fast_transformer = train_kwargs.get("use_fast_transformer", True),
        return_new_adata = False
    )

    if save_model is not None:
        print("Not saving the scGPT model, as is not trained.")

    return {
        "embedding": adata.obsm["X_scGPT"]
    }

def harmony(
    adata, 
    train_kwargs=dict(),
    model_kwargs=dict(),
    data_kwargs=dict(),
    save_model=None,
):
    sc.external.pp.harmony_integrate(
        adata,
        basis='X_pca',
        key=data_kwargs.get('batch_key', 'batch'),
        # copy=False,
    )
    return {
        "embedding": adata.obsm["X_pca_harmony"]
    }

def scvi(
    adata, 
    train_kwargs=dict(),
    model_kwargs=dict(),
    data_kwargs=dict(),
    save_model=None,
):
    import scvi
    from scvi.model import SCVI

    SCVI.setup_anndata(
        adata,
        batch_key=data_kwargs.get('batch_key', None),
        labels_key=None,
        size_factor_key=data_kwargs.get('size_factor_key', None),
        layer=data_kwargs.get('layer', None),
        categorical_covariate_keys=data_kwargs.get('categorical_covariate_keys', None),
        continuous_covariate_keys=data_kwargs.get('continuous_covariate_keys', None),
    )
    model_kwargs = filter_kwargs(model_kwargs, SCVI.__init__)
    model = scvi.model.SCVI(adata, **model_kwargs)
    model.train(**train_kwargs)

    adata.obsm['X_scvi'] = model.get_latent_representation()
    if save_model is not None:
        model.save(save_model, overwrite=True, save_anndata=False)

    return {
        "embedding": adata.obsm["X_scvi"]
    }

def scanvi_seed_label(
    adata,
    train_kwargs=dict(n_seed=None),
    model_kwargs=dict(),
    data_kwargs=dict(annotation_key='annotation'),
    save_model=None,
):
    import scvi
    from scvi.model import SCANVI

    def get_score(normalized_adata, gene_set):
        """Returns the score per cell given a dictionary of + and - genes

        Parameters
        ----------
        normalized_adata
        anndata dataset that has been log normalized and scaled to mean 0, std 1
        gene_set
        a dictionary with two keys: 'positive' and 'negative'
        each key should contain a list of genes
        for each gene in gene_set['positive'], its expression will be added to the score
        for each gene in gene_set['negative'], its expression will be subtracted from its score

        Returns
        -------
        array of length of n_cells containing the score per cell
        """
        score = np.zeros(normalized_adata.n_obs)
        if sp.issparse(expression := normalized_adata[:, list(gene_set["positive"])].X):
            expression = expression.toarray()
        score += expression.mean(axis=-1)
        if sp.issparse(expression := normalized_adata[:, list(gene_set["negative"])].X):
            expression = expression.toarray()
        score -= expression.mean(axis=-1)
        return score

    def get_cell_mask(normalized_adata, gene_set, n_cells = None):
        """Calculates the score per cell for a list of genes, then returns a mask for
        the cells with the highest 50 scores.

        Parameters
        ----------
        normalized_adata
        anndata dataset that has been log normalized and scaled to mean 0, std 1
        gene_set
        a dictionary with two keys: 'positive' and 'negative'
        each key should contain a list of genes
        for each gene in gene_set['positive'], its expression will be added to the score
        for each gene in gene_set['negative'], its expression will be subtracted from its score

        Returns
        -------
        Mask for the cells with the top 50 scores over the entire dataset
        """
        if n_cells is None:
            n_cells = normalized_adata.n_obs // 60
        score = get_score(normalized_adata, gene_set)
        cell_idx = score.argsort()[-n_cells:]
        mask = np.zeros(normalized_adata.n_obs)
        mask[cell_idx] = 1
        return mask.astype(bool)

    n_seed = train_kwargs.get("n_seed", adata.shape[0] // 100)
    annotation_key = data_kwargs.get("annotation_key")
    annotation = adata.uns[annotation_key]['clusters']
    all_markers = set()
    cell_type_gene_markers = {}
    for ct in annotation.keys():
        cell_type_gene_markers[ct] = {}
        cell_type_gene_markers[ct]['positive'] = [i for i in annotation[ct] if i in adata.var_names]
        all_markers.update(cell_type_gene_markers[ct]['positive'])
    for ct in annotation.keys():
        cell_type_gene_markers[ct]['negative'] = [i for i in all_markers if i not in cell_type_gene_markers[ct]['positive']]
    
    adata.obs['seed_label'] = 'unknown'
    normalized = adata.copy()
    normalized.X = adata.layers['normalized']
    print(cell_type_gene_markers, n_seed)

    cell_type_mask = {
        cell_type: get_cell_mask(normalized, gene_markers, n_cells=n_seed) 
        for cell_type, gene_markers in cell_type_gene_markers.items()
    }
    for cell_type, mask in cell_type_mask.items():
        adata.obs.loc[mask, 'seed_label'] = cell_type
    adata.obs['seed_label'] = adata.obs['seed_label'].astype('category')
    print("Prepare the seed labels for SCANVI finished.")

    SCANVI.setup_anndata(
        adata,
        batch_key=data_kwargs.get('batch_key', None),
        labels_key='seed_label',
        size_factor_key=data_kwargs.get('size_factor_key', None),
        unlabeled_category='unknown',
        categorical_covariate_keys=data_kwargs.get('categorical_covariate_keys', None),
        continuous_covariate_keys=data_kwargs.get('continuous_covariate_keys', None),
    )
    model_kwargs = filter_kwargs(model_kwargs, SCANVI.__init__)
    model = scvi.model.SCANVI(adata, **model_kwargs)
    model.train(**train_kwargs)
    adata.obsm['X_scanvi'] = model.get_latent_representation()
    pred = model.predict()
    
    if save_model is not None:
        model.save(save_model, overwrite=True, save_anndata=False)
    
    return {
        "embedding": adata.obsm["X_scanvi"],
        'labels': pred
    }

def expimap(
    adata,
    train_kwargs=dict(),
    model_kwargs=dict(),
    data_kwargs=dict(),
    save_model=None,
):
    """
    Follow the intructions in the scarches documentation.
    https://docs.scarches.org/en/latest/expimap_surgery_pipeline_basic.html
    """
    import scarches as sca
    from scarches.models import expiMap
    import warnings
    import torch
    
    torch.use_deterministic_algorithms(False, warn_only=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='scarches')

    cache_dir = '.cache'
    os.makedirs(cache_dir, exist_ok=True)
    annotation = adata.uns[data_kwargs.get('annotation_key', 'annotation')]
    annotation = {**annotation['clusters'], **annotation['background']}
    # write to a gmt file 
    gmt_file = os.path.join(cache_dir, 'run_expimap_annotations.gmt')
    with open(gmt_file, 'w') as f:
        for k, v in annotation.items():
            line = '\t'.join(list(v))
            f.write(f"{k}\t{line}\n")
    print(f"Annotation file -cache written to {gmt_file}, all {len(annotation)} items.")

    sca.utils.add_annotations(
        adata, 
        gmt_file,
        min_genes=5,
        clean=True,
    )

    select_terms = adata.varm['I'].sum(0)>5
    adata.uns['terms'] = np.array(adata.uns['terms'])[select_terms].tolist()
    adata.varm['I'] = adata.varm['I'][:, select_terms]
    print("filter terms less than 5 genes, remaining:", len(adata.uns['terms']))

    if model_kwargs.get('n_ext_m_'):
        print("constrained node is not valid when train reference, parameter `n_ext_m` is deleted", model_kwargs.pop('n_ext_m_')
    )
    condition_key = data_kwargs.get('condition_key') or data_kwargs.get('batch_key')
    if condition_key is None:
        adata.obs['condition'] = 'condition'
        condition_key = 'condition'
    
    ###
    # adata.X = adata.X.astype(int)    

    intr_cvae = sca.models.EXPIMAP(
        adata=adata,
        condition_key=condition_key,
        hidden_layer_sizes=model_kwargs.get('hidden_layer_sizes', [256]*3),
        **model_kwargs
    )

    early_stopping_kwargs = {
        "early_stopping_metric": "val_unweighted_loss", # val_unweighted_loss
        "threshold": 0,
        "patience": 50,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    early_stopping_kwargs.update(train_kwargs.get('early_stopping_kwargs', {}))

    intr_cvae.train(
        n_epochs=train_kwargs.get('n_epochs', 400),
        alpha_epoch_anneal=train_kwargs.get('alpha_epoch_anneal', 100),
        alpha=model_kwargs.get('alpha', 0.7),
        alpha_kl=model_kwargs.get('alpha_kl', 0.5),
        weight_decay=model_kwargs.get('weight_decay', 0.0),
        early_stopping_kwargs=early_stopping_kwargs,
        use_early_stopping=True,
        monitor_only_val=False,
        seed=train_kwargs.get('seed', 0),
    )

    embedding = intr_cvae.get_latent(only_active=True) 

    intr_cvae.update_terms()
    intr_cvae.latent_directions()
    factors = []
    for i, activated in enumerate(intr_cvae.nonzero_terms()):
        if not activated:
            continue
        termname = adata.uns['terms'][i]
        f = intr_cvae.term_genes(termname).set_index('genes')['weights']
        f.name = termname
        factors.append(f)
    factors = pd.concat(factors, axis=1)

    if save_model is not None:
        intr_cvae.save(save_model, overwrite=True, save_anndata=False)
        
    shutil.rmtree(cache_dir)
    return {
        "embedding": embedding,
        "factors": factors.values,
        "factor_genes": factors.index.values,
        "factor_terms": factors.columns.values,
    }

