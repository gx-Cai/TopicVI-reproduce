import os
import dill as pickle
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from inspect import signature
import torch


def filter_kwargs(kwargs, target_function):
    return {
        k: v for k, v in kwargs.items() if k in signature(target_function).parameters
    }

sys.path.append("../../external/")


def amortized_lda(
    adata,
    train_kwargs=dict(),
    data_kwargs=dict(layers=None),
    model_kwargs=dict(n_topics=20, n_hidden=128),
    save_model=None,
):
    import scvi
    
    data_kwargs = filter_kwargs(data_kwargs, scvi.model.AmortizedLDA.setup_anndata)
    scvi.model.AmortizedLDA.setup_anndata(adata, **data_kwargs)
    model = scvi.model.AmortizedLDA(
        adata, 
        n_topics=model_kwargs.get("n_topics", 20), 
        n_hidden=model_kwargs.get("n_hidden", 128),
        cell_topic_prior=model_kwargs.get("cell_topic_prior", None),
        topic_feature_prior=model_kwargs.get("topic_feature_prior", None),
    )
    model.train(**train_kwargs)

    embedding = model.get_latent_representation()
    factors = model.get_feature_by_topic().loc[adata.var_names, :].values

    if save_model:
        model.save(
            save_model,
            overwrite=True,
            save_anndata=False,
        )

    return {"loading": embedding, "factors": factors}


def ldvae(
    adata,
    train_kwargs=dict(),
    data_kwargs=dict(layers=None),
    model_kwargs=dict(n_topics=20, n_hidden=128),
    save_model=None,
):
    import scvi
    
    # data_kwargs = filter_kwargs(data_kwargs, scvi.model.LinearSCVI.setup_anndata)
    batch_key = data_kwargs.get("batch_key", None)
    if batch_key:
        if adata.obs[batch_key].value_counts().min() <= 1:
            batch_key = None
            print("Warning: batch_key has less than 2 samples, dropped.")

    scvi.model.LinearSCVI.setup_anndata(
        adata, 
        batch_key=batch_key,
        labels_key=data_kwargs.get("label_key", None),
        layer=data_kwargs.get("layer", None),
    )
    model = scvi.model.LinearSCVI(
        adata, 
        n_latent=model_kwargs.get("n_topics", 20), 
        n_hidden=model_kwargs.get("n_hidden", 128),
        n_layers=model_kwargs.get("n_layers", 1),
        dropout_rate=model_kwargs.get("dropout_rate", 0.1),
        gene_likelihood=model_kwargs.get("gene_likelihood", "nb"),
        latent_distribution=model_kwargs.get("latent_distribution", "normal"),
    )
    model.train(**train_kwargs)

    embedding = model.get_latent_representation()
    factors = model.get_loadings().loc[adata.var_names, :].values

    if save_model:
        model.save(
            save_model,
            overwrite=True,
            save_anndata=False,
        )

    return {"loading": embedding, "factors": factors}


def pycogaps(
    adata,
    train_kwargs=dict(),
    data_kwargs=dict(log1p=True),
    model_kwargs=dict(nSets=None, n_topics=20),
    save_model=None,
):
    from PyCoGAPS.parameters import CoParams, setParams
    from PyCoGAPS.pycogaps_main import CoGAPS

    adata.X = adata.layers['normalized']
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    params = CoParams(adata=adata)

    default_model_kwargs = {
        "nIterations": train_kwargs.get("nIterations", 5000),
        "seed": train_kwargs.get("seed", 42),
        "nPatterns": model_kwargs.get("n_topics", 20),
        "useSparseOptimization": model_kwargs.get("useSparseOptimization", True),
        "distributed": model_kwargs.get("distributed", "single-cell"),
    }
    print(default_model_kwargs)

    setParams(params, default_model_kwargs)
    
    if ns := model_kwargs.get("nSets"):
        params.setDistributedParams(nSets=ns)

    result = CoGAPS(adata, params)
    obs_loading = result.obs.filter(like="Pattern")
    var_loading = result.var.filter(like="Pattern")

    if save_model:
        with open(os.path.join(save_model, "model.pkl"), "wb") as f:
            pickle.dump(result.uns, f)

    return {"loading": obs_loading.values, "factors": var_loading.values}


def pyliger(
    adata,
    train_kwargs=dict(),
    data_kwargs=dict(batch_key="batch"),
    model_kwargs=dict(n_topics=20),
    save_model=None,
):
    import pyliger

    adata.var.index.name = "gene_symbols"
    adata.obs.index.name = "sample_id"
    n_topics = model_kwargs.get("n_topics", 20)
    batch_key = data_kwargs.get("batch_key")
    if ((batch_count:= adata.obs[batch_key].value_counts()) < n_topics).any():
        replace_batch = batch_count.reset_index().query('count<@n_topics')[batch_key].values.tolist()
        adata.obs[batch_key] = adata.obs[batch_key].map(
            {i: i if i not in replace_batch else 'mannual combined' 
             for i in batch_count.index
            }
    )
        
    if ((batch_count:= adata.obs[batch_key].value_counts()) < n_topics).any():
        replace_batch = batch_count.reset_index().query('count<@n_topics')[batch_key].values.tolist()
        print("Warning: ",replace_batch, "batches with too little samples. dropped.")
        adata = adata[adata.obs.query(f"{batch_key} not in @replace_batch").index, :].copy()

    data_list = [
        adata[adata.obs[batch_key] == batch, :].copy()
        for batch in adata.obs[batch_key].unique()
    ]
    for dat in data_list:
        dat.uns["sample_name"] = dat.obs[batch_key][0]
        
    liger_obj = pyliger.create_liger(data_list, remove_missing=False)
    # unknown bugs: AttributeError: 'Liger' object has no attribute 'var_genes'
    liger_obj.var_genes = adata.var_names.tolist()
    pyliger.normalize(liger_obj, remove_missing=False)
    # For evaluation purposes, we will not use the select_genes function.
    # pyliger.select_genes(liger_obj)
    pyliger.scale_not_center(liger_obj) 
    pyliger.optimize_ALS(
        liger_obj, k=n_topics, 
        rand_seed = np.random.randint(0, 100) # To avoid sometimes "Singular matrix" Error. 
    )
    if batch_count.min() < 20: 
        knn_k = 10
    else:
        knn_k = 20
    pyliger.quantile_norm(liger_obj, knn_k = knn_k)

    adatas = liger_obj.adata_list
    result = sc.concat(adatas, join="inner")
    embedding = result.obsm["H_norm"]
    factors = []
    for adt in adatas:
        factors.append((adt.varm["V"] + adt.varm["W"]).T)
    factors = np.sum(factors, axis=0)

    return {"loading": embedding, "factors": factors}


def spectra(
    adata,
    train_kwargs=dict(use_gpu=True),
    data_kwargs=dict(label_key="cell_type", annotation_key="annotation"),
    model_kwargs=dict(),
    save_model=None,
):
    import Spectra
    from Spectra import Spectra_util as spc_tl
    
    if label_name := data_kwargs.get("label_key"):
        assert (
            label_name in adata.obs.columns
        ), f"Label name {label_name} not found in adata.obs.columns"
        print(adata.obs[label_name].value_counts())
        
    if train_kwargs.get("use_gpu"):
        import Spectra.Spectra_gpu as spc
    else:
        import Spectra as spc

    if data_kwargs.get("annotation_key"):
        annotations = adata.uns[data_kwargs["annotation_key"]]
        background_gs = annotations.get("background", {})
        annotations = annotations['clusters']
        annotations['global'] = background_gs
    else:
        annotations = Spectra.default_gene_sets.load()

    print(annotations.keys())
    assert "global" in annotations.keys(), "global key not found in annotations"
    
    if label_name is not None:
        # check the annotations and label
        ann_keys = list(annotations.keys())
        ann_keys.remove("global")
        label_vals = adata.obs[label_name].unique()
        for l in label_vals:
            if l not in ann_keys:
                label_name = None
                break
        if label_name is not None:
            for ann in ann_keys:
                if ann not in label_vals:
                    annotations.pop(ann)

    annotations = spc_tl.check_gene_set_dictionary(
        adata,
        annotations,
        obs_key=label_name,
        use_cell_types=(label_name is not None),
        global_key="global",
    )

    default_train_kwargs = dict(
        use_weights=True,
        # varies depending on data and gene sets, try between 0.5 and 0.001
        lam=model_kwargs.get("lam", 0.01),
        delta=model_kwargs.get("delta", 0.001),
        kappa=model_kwargs.get("kappa", None),
        rho=model_kwargs.get("rho", 0.001),
        use_cell_types=label_name if label_name else False,
        n_top_vals=model_kwargs.get("n_top_vals", 50),
        num_epochs=train_kwargs.get("num_epochs", 10000),
        early_stop_patience=train_kwargs.get(
            "early_stopping_patience", 50
        ),  ### Modified for early stopping.
    )

    model = spc.est_spectra(
        adata=adata,
        gene_set_dictionary=annotations,
        use_highly_variable=False,  # have performed in preprocess step
        cell_type_key=label_name,
        **default_train_kwargs,
    )

    if save_model:
        model.save(os.path.join(save_model, "model.pt"))

    return {
        "loading": adata.obsm["SPECTRA_cell_scores"],
        "factors": adata.uns["SPECTRA_factors"],
    }


def spike_slab_lda(
    adata, train_kwargs=dict(), data_kwargs=dict(), model_kwargs=dict(), save_model=None
):
    from larch.util.modelhub import SpikeSlab
    from larch.util.util import setup_anndata
    from scipy.sparse import csr_matrix

    adata.layers["counts"] = csr_matrix(adata.X).copy()
    setup_anndata(adata, layer="counts")

    default_model_kwargs = dict(
        n_latent=model_kwargs.get("n_topics", 20),
        pip0_rho=model_kwargs.get("pip0_rho", 0.1),
        kl_weight_beta=model_kwargs.get("kl_weight_beta", 1),
        kl_weight=model_kwargs.get("kl_weight", 1),
        a0=model_kwargs.get("a0", 1e-4),
    )

    model = SpikeSlab(adata, **default_model_kwargs)
    model.train(deterministic=True, **train_kwargs)

    if save_model:
        model.save(save_model, overwrite=True, save_anndata=False)

    with torch.no_grad():
        decoder = model.module.decoder
        beta = decoder.get_beta(
            decoder.spike_logit, decoder.slab_mean, decoder.slab_lnvar, decoder.bias_d
        )
        beta = torch.exp(torch.clamp(beta, -10, 10))
        beta = beta.cpu().numpy()
        topics_np = model.get_latent_representation(
            deterministic=True, output_softmax_z=True
        )
    # topic proportions (after softmax)
    return {"loading": topics_np, "factors": beta}


def tree_spike_slab_lda(
    adata, train_kwargs=dict(), data_kwargs=dict(), model_kwargs=dict(), save_model=None
):
    from larch.util.modelhub import TreeSpikeSlab
    from larch.util.util import setup_anndata
    from scipy.sparse import csr_matrix

    adata.layers["counts"] = csr_matrix(adata.X).copy()
    setup_anndata(adata, layer="counts")

    default_model_kwargs = dict(
        tree_depth=model_kwargs.get("tree_depth", 5),
        pip0_rho=model_kwargs.get("pip0", 0.1),
        kl_weight_beta=model_kwargs.get("kl_weight_beta", 1),
        kl_weight=model_kwargs.get("kl_weight", 1),
        a0=model_kwargs.get("a0", 1e-4),
    )

    model = TreeSpikeSlab(adata, **default_model_kwargs)
    model.train(deterministic=True, **train_kwargs)

    if save_model:
        model.save(save_model, overwrite=True, save_anndata=False)

    with torch.no_grad():
        decoder = model.module.decoder
        beta = decoder.get_beta(
            decoder.spike_logit, decoder.slab_mean, decoder.slab_lnvar, decoder.bias_d
        )
        beta = torch.exp(torch.clamp(beta, -10, 10))
        beta = torch.mm(decoder.A, beta)
        beta = beta.cpu().numpy()
        topics_np = model.get_latent_representation(
            deterministic=True, output_softmax_z=True
        )

    return {"loading": topics_np, "factors": beta}


def dcjcomm(
    adata,
    train_kwargs=dict(maxIter=100),
    data_kwargs=dict(),
    model_kwargs=dict(n_topics=20, n_clusters=10, xita=2),
    save_model=None,
):
    import rpy2.robjects as ro
    import anndata2ri
    anndata2ri.activate()
    r = ro.r

    r.source("../external/DcjComm_nmf.R")
    main_func = r["NMFbased"]
    n_topics = model_kwargs.get("n_topics", 20)
    n_clutsers = model_kwargs.get("n_clusters", 10)
    X = adata.X.T.copy().tocsr()
    sc.pp.neighbors(adata)
    # get the nearest neighbors
    W = adata.obsp["distances"]

    # run the model
    nmf_result = main_func(X, W, n_topics, n_clutsers, train_kwargs.get("maxIter", 100))
    # module = r["moduleNodesSelection"](
    #     nmf_result["U_final"], model_kwargs.get("xita", 2)
    # )
    if save_model:
        with open(os.path.join(save_model, "model.pkl"), "wb") as f:
            pickle.dump(nmf_result, f)

    return {
        "loading": nmf_result["V_final"].T,
        "factors": nmf_result["U_final"],
        "fators_weights": nmf_result["S_final"],
        "label": nmf_result["Label"].flatten(),
    }


def schpf(
    adata,
    train_kwargs=dict(batchsize=256),
    data_kwargs=dict(),
    model_kwargs=dict(n_topics=20),
    save_model=None,
):
    import schpf
    from schpf import run_trials, scHPF

    if sp.issparse(adata.X):
        X = adata.X.tocoo()
    else:
        X = sp.coo_matrix(adata.X)

    model = run_trials(
        X=X,
        nfactors=model_kwargs.get("n_topics", 20),
        batchsize=train_kwargs.get("batchsize", 256),
        ntrials=train_kwargs.get("ntrials", 5),
        min_iter=train_kwargs.get("min_iter", 30),
        max_iter=train_kwargs.get("max_iter", 1000),
        check_freq=train_kwargs.get("check_freq", 10),
        epsilon=train_kwargs.get("epsilon", 1e-3),
        better_than_n_ago=train_kwargs.get("better_than_n_ago", 5),
        beta_theta_simultaneous=train_kwargs.get("beta_theta_simultaneous", False),
        loss_smoothing=train_kwargs.get("loss_smoothing", 1),
        model_kwargs=filter_kwargs(model_kwargs, scHPF.__init__),
    )

    cell_score = model.cell_score()
    gene_score = model.gene_score().T

    if save_model:
        with open(os.path.join(save_model, "model.pkl"), "wb") as f:
            pickle.dump(model, f)

    return {"loading": cell_score, "factors": gene_score}


def muvi(
    adata,
    train_kwargs=dict(),
    data_kwargs=dict(annotation_key='annotation'),
    model_kwargs=dict(n_topics=20),
    save_model=None,
):
    import muvi

    gene_sets = adata.uns[data_kwargs.get("annotation_key")]
    gene_sets = {**gene_sets['background'], **gene_sets['clusters']}
    gene_sets_mask = muvi.fs.from_dict(gene_sets).to_mask(adata.var_names.values.tolist())
    adata.varm['gene_sets_mask'] = gene_sets_mask.T

    model = muvi.tl.from_adata(
        adata, 
        prior_mask_key='gene_sets_mask',
        n_factors=model_kwargs.get("n_topics", 20),
        prior_confidence=model_kwargs.get("prior_confidence", "low"),
        view_names=model_kwargs.get("view_names", None),
        likelihoods=model_kwargs.get("likelihoods", None),
        reg_hs=model_kwargs.get("reg_hs", True),
        nmf=model_kwargs.get("nmf", False),
        pos_transform=model_kwargs.get("pos_transform", "relu"),
        normalize=model_kwargs.get("normalize", True),
        device=train_kwargs.get("device", "cuda"),
    )
    print(model.n_samples)
    model.fit(
        batch_size=train_kwargs.get("batch_size", 1024),
        n_epochs=train_kwargs.get("n_epochs", 500),
        n_particles=train_kwargs.get("n_particles", 10),
        learning_rate=train_kwargs.get("learning_rate", 0.005),
        optimizer=train_kwargs.get("optimizer", "clipped"),
        scale_elbo=train_kwargs.get("scale_elbo", True),
        early_stopping=train_kwargs.get("early_stopping", True),
        callbacks=train_kwargs.get("callbacks", None),
        seed=train_kwargs.get("seed", None),
        # EarlyStoppingCallback kwargs.
        min_epochs=train_kwargs.get("min_epochs", 100),
        tolerance=train_kwargs.get("tolerance", 0.0001),
        patience=train_kwargs.get("patience", 10),
        # verbose=False
    )

    if save_model:
        with open(os.path.join(save_model, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
            
    return {
        "loading": model.get_factor_scores(),
        "factors": model.get_factor_loadings()['view_0'].T,
        "factor_names": model.factor_names,
    }


def topicvi(
    adata, 
    train_kwargs=dict(pretrain_model=None),
    data_kwargs=dict(),
    model_kwargs=dict(n_topics=20),
    save_model=None,
):

    from topicvi.model.module import TopicVI
    from topicvi.model import inverse_davies_bouldin_score
    from topicvi.prior import clean_prior_dict
    import scib
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='torchmetrics')
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='pytorch') 

    running_mode = model_kwargs.get('running_mode', 'unsupervised')
    topic_decoder_params = model_kwargs.get('topic_decoder_params', dict())
    cluster_decoder_params = model_kwargs.get('cluster_decoder_params', dict())
    n_topics = model_kwargs.get('n_topics', 20)
    n_clusters = model_kwargs.get('n_clusters', 10)
    topicvi_kwargs = model_kwargs.get('topicvi_kwargs', dict())
    pretrain_kwargs = model_kwargs.get('pretrain_kwargs', dict())
    max_init_cells = model_kwargs.get('max_init_cells', 10000)

    cell_type_key = data_kwargs.get('label_key')
    batch_key = data_kwargs.get('batch_key')
    annotation_key = data_kwargs.get('annotation_key')
    size_factor_key = data_kwargs.get('size_factor_key')
    setup_kwargs = data_kwargs.get('setup_kwargs', dict())

    if running_mode == 'unsupervised':
        if (default_cluster_key:=data_kwargs.get('default_cluster_key')) is None:
            scib.clustering.cluster_optimal_resolution(
                adata, 
                label_key=None, 
                cluster_key='default_cluster', 
                use_rep='X_pca',
                cluster_function=sc.tl.leiden,
                metric=inverse_davies_bouldin_score,
                resolutions=np.linspace(0.1, 1, 10) #[0.1,0.2,0.3,]
            )
        else:
            adata.obs['default_cluster'] = adata.obs[default_cluster_key]
    
    TopicVI.setup_anndata(
        adata,
        batch_key=batch_key,
        labels_key=cell_type_key if running_mode != 'unsupervised' else 'default_cluster',
        size_factor_key=size_factor_key,
        run_cluster_kwargs = dict(
            max_cells = max_init_cells
        ),
        **setup_kwargs
    )

    annotation = adata.uns[annotation_key]

    model = TopicVI(
        adata,
        n_topics=n_topics, 
        n_labels=n_clusters,
        prior_genesets=clean_prior_dict(annotation['background'], adata),
        cluster_prior_genesets=clean_prior_dict(annotation['clusters'], adata),
        mode = running_mode,
        topic_decoder_params=topic_decoder_params,
        cluster_decoder_params=cluster_decoder_params,
        **topicvi_kwargs
    )

    pretrain_model=train_kwargs.pop('pretrain_model', None)
    try:
        model.load_pretrained_model(pretrain_model)
    except Exception as e:
        print(
            'No pretrained model found or something error occured' 
            'start training from scratch.', e
        )
        model.pretrain(
            save = pretrain_model,
            setup_kwargs=dict(
                batch_key = batch_key,
                labels_key = cell_type_key if running_mode != 'unsupervised' else None,
            ),
            **pretrain_kwargs
        )
    
    if gene_emb_dir:=train_kwargs.pop('gene_emb_dir', None):
        gene_embedding = np.load(gene_emb_dir, allow_pickle=True).tolist()
        model.load_gene_embedding(
            gene_embedding['gene_emb'],
            gene_embedding['gene_ids'],
            fix = False
        )

    model.train(**train_kwargs)
    model.store_topics_info()
    if save_model:
        model.save(save_model, overwrite=True, save_anndata=False)
    
    return {
        "loading": model.get_topic_by_sample(),
        "factors": model.get_topic_by_genes(),
        'embedding': model.get_latent_representation(),
        'labels': model.get_cluster_assignment(),
    }


def topicvi_denovo_finding(
    adata, 
    train_kwargs=dict(pretrain_model=None, gene_emb_dir=None),
    data_kwargs=dict(),
    model_kwargs=dict(n_topics=20),
    save_model=None,
):
    n_topics = model_kwargs.get('n_topics', 20)
    topic_decoder_params = model_kwargs.get('topic_decoder_params', dict())
    topic_decoder_params.update(dict(n_topics_without_prior=n_topics))
    model_kwargs.update(dict(topic_decoder_params=topic_decoder_params))

    return topicvi(
        adata, 
        train_kwargs=train_kwargs, 
        data_kwargs=data_kwargs, 
        model_kwargs=model_kwargs,
        save_model=save_model
    )


def topicvi_seed_labels(
    adata, 
    train_kwargs=dict(pretrain_model=None, gene_emb_dir=None),
    data_kwargs=dict(),
    model_kwargs=dict(n_topics=20),
    save_model=None,
):
    sys.path.append('../../src/')

    from model.module import TopicVI

    annotation_key = data_kwargs.get('annotation_key', 'annotation')
    annotation = adata.uns[annotation_key]['clusters']
    TopicVI.seed_label_from_topic(
        adata, annotation, layer='normalized', key_added='seed_label'
    )
    data_kwargs.update(dict(label_key='seed_label'))
    model_kwargs.update(dict(running_mode='supervised'))
    model_kwargs.update(dict(n_clusters=len(annotation)))
    return topicvi(
        adata, 
        train_kwargs=train_kwargs, 
        data_kwargs=data_kwargs, 
        model_kwargs=model_kwargs,
        save_model=save_model
    )