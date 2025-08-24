# -*- coding: utf-8 -*-
"""
SpatialEx / SpatialEx+ training utilities.

This module defines three trainer classes designed for histology-to-omics
translation on spatial transcriptomics data:

- :class:`Train_SpatialEx`: Baseline trainer. Trains two independent SpatialEx
  models (one per slice), then evaluates cross-panel prediction quality via
  cosine similarity, SSIM, PCC, and CMD.
- :class:`Train_SpatialExP`: SpatialEx+ trainer. Extends the baseline with two
  regression mapping heads (AB/BA) in a cycle-style setup to improve
  cross-panel translation.
- :class:`Train_SpatialExP_Big`: Large-scale SpatialEx+ trainer tailored for
  million-cell data by tiling pseudo-spots and iterating in mini-batches.

Requirements
------------
Two :class:`anndata.AnnData` objects are expected. Each should provide:

- ``.obsm['he']``: Histology-derived embeddings for each cell/spot.
- ``.X``: Gene expression matrix (cells × genes).
- ``.var_names``: Gene names aligned with network outputs.

Notes
-----
The trainers rely on project modules:

- Models: ``model.Model``, ``model.Model_Plus``, ``model.Model_Big``,
  ``model.Regression``.
- Preprocessing: ``preprocess`` (aliased as ``pp``) for hypergraph building and
  data loading.
- Utilities: ``utils.create_optimizer``, ``utils.Generate_pseudo_spot``,
  ``utils.Compute_metrics`` (metrics are used downstream by users).

All classes expose a single high-level entrypoint :meth:`train` which performs
training and (when applicable) evaluation/saving.

"""

import os
import torch
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from . import preprocess as pp
from .utils import create_optimizer, Generate_pseudo_spot
from .model import Model, Model_Plus, Model_Big, Regression


warnings.filterwarnings("ignore")


class Train_SpatialEx:
    """
    Baseline trainer for the **SpatialEx** framework on two slices.

    This trainer independently optimizes two slice-specific models (one per
    slice) on hypergraph-based mini-batches, then produces **cross-panel**
    predictions (e.g., predict panel B on slice 1 using model trained on slice 2).
    At evaluation time, predictions are aggregated to cell level and can be
    saved to CSV files.

    Parameters
    ----------
    adata1 : :class:`anndata.AnnData`
        Slice 1. Must contain histology embeddings in ``.obsm['he']`` and
        expression matrix in ``.X``; gene names in ``.var_names``.
    adata2 : :class:`anndata.AnnData`
        Slice 2 with the same structure as ``adata1``.
    num_layers : int, optional, default=2
        Number of HGNN layers used by the backbone.
    hidden_dim : int, optional, default=512
        Hidden width of intermediate feature representations.
    epochs : int, optional, default=500
        Number of training epochs.
    seed : int, optional, default=0
        Random seed for reproducibility.
    device : torch.device, optional
        Device for training. Defaults to ``cuda`` if available, else ``cpu``.
    weight_decay : float, optional, default=0.0
        Weight decay for optimizer.
    optimizer : str, optional, default="adam"
        Optimizer type. One of ``"adam"``, ``"adamw"``, ``"adadelta"``,
        ``"radam"``, or ``"sgd"``.
    batch_size : int, optional, default=4096
        Effective batch size used when constructing hypergraphs.
    encoder : str, optional, default="hgnn"
        Backbone encoder key (currently HGNN-based modeling).
    lr : float, optional, default=1e-3
        Learning rate.
    loss_fn : str, optional, default="mse"
        Reconstruction loss key.
    num_neighbors : int, optional, default=7
        K in KNN graph building for the hypergraph.
    graph_kind : str, optional, default="spatial"
        Hypergraph/graph type to build (``"spatial"`` or ``"he"``).
    prune : int, optional, default=10000
        Spatial tiling window for dataloader construction.
    save : bool, optional, default=True
        Whether to save cross-panel predictions as CSVs in ``./results``.
        (Files: ``HE_to_omics_panel1b.csv`` and ``HE_to_omics_panel2a.csv``)

    Attributes
    ----------
    adata1, adata2 : :class:`anndata.AnnData`
        References to input slices.
    num_layers, hidden_dim, epochs, seed : int
        Basic trainer hyperparameters.
    device : torch.device
        Target computation device.
    weight_decay : float
        Weight decay coefficient for optimizer.
    batch_size : int
        Graph building batch size.
    encoder : str
        Backbone key (for future extensibility).
    lr : float
        Learning rate.
    loss_fn : str
        Loss key (e.g., ``"mse"``).
    num_neighbors : int
        KNN parameter for hypergraph building.
    graph_kind : str
        Graph construction mode.
    prune : int
        Tiling window side length (pixels).
    save : bool
        Output saving flag.

    in_dim1, in_dim2 : int
        Input feature dimensions derived from ``.obsm['he']`` of each slice.
    out_dim1, out_dim2 : int
        Output dimensions equal to number of genes per slice.
    module_HA, module_HB : :class:`SpatialEx.model.Model`
        Two slice-specific SpatialEx models (one per slice).
    models : list[nn.Module]
        List of models passed to the optimizer.
    optimizer : torch.optim.Optimizer
        Optimizer instance created via :func:`utils.create_optimizer`.
    slice1_dataloader, slice2_dataloader : torch.utils.data.DataLoader
        Hypergraph-based dataloaders created via :func:`preprocess.Build_dataloader`.

    Methods
    -------
    train()
        Run the training loop, generate cross-panel predictions, and optionally
        save CSV outputs under ``./results``.
    """

    def __init__(self,
                 adata1,
                 adata2,
                 num_layers=2,
                 hidden_dim=512,
                 epochs=500,
                 seed=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 weight_decay=0,
                 optimizer="adam",
                 batch_size=4096,
                 encoder="hgnn",
                 lr=0.001,
                 loss_fn="mse",
                 num_neighbors=7,
                 graph_kind='spatial',
                 prune=10000,
                 save=True
                 ):
        self.adata1 = adata1
        self.adata2 = adata2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.encoder = encoder

        self.lr = lr
        self.loss_fn = loss_fn
        self.num_neighbors = num_neighbors
        self.graph_kind = graph_kind
        self.prune = prune
        self.save = save

        self.in_dim1 = self.adata1.obsm['he'].shape[1]
        self.in_dim2 = self.adata2.obsm['he'].shape[1]
        self.out_dim1 = self.adata1.n_vars
        self.out_dim2 = self.adata2.n_vars

        self.module_HA = Model(self.num_layers, self.in_dim1, self.hidden_dim, self.out_dim1, self.loss_fn, self.device)
        self.module_HB = Model(self.num_layers, self.in_dim2, self.hidden_dim, self.out_dim2, self.loss_fn, self.device)
        self.models = [self.module_HA, self.module_HB]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)

        H1 = pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, batch_size, False, 'spatial', 'crs')
        self.slice1_dataloader = pp.Build_dataloader(adata1, graph=H1, graph_norm='hpnn', feat_norm=False,
                                                     prune=[prune, prune], drop_last=False)
        H2 = pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, batch_size, False, 'spatial', 'crs')
        self.slice2_dataloader = pp.Build_dataloader(adata2, graph=H2, graph_norm='hpnn', feat_norm=False,
                                                     prune=[prune, prune], drop_last=False)

    def train(self):
        """
        Train two slice-specific SpatialEx models and evaluate cross-panel prediction.

        Workflow
        --------
        1. Set random seed and switch both models to train mode.
        2. Iterate over paired batches from ``slice1_dataloader`` and
           ``slice2_dataloader``; compute reconstruction losses and optimize.
        3. Switch models to eval mode and generate cross-panel predictions:
           - Predict panel **B** on slice 1 using ``module_HB``.
           - Predict panel **A** on slice 2 using ``module_HA``.
        4. Aggregate by cell and (optionally) save CSV files to ``./results``.

        Returns
        -------
        None
            Prints progress and optionally writes prediction CSV files:
            ``HE_to_omics_panel1b.csv`` (slice 1 as B) and
            ``HE_to_omics_panel2a.csv`` (slice 2 as A).
        """
        pp.set_random_seed(self.seed)
        self.module_HA.train()
        self.module_HB.train()
        print('\n')
        print('=================================== Start training =========================================')
        epoch_iter = tqdm(range(self.epochs))
        for epoch in epoch_iter:
            batch_iter = zip(self.slice1_dataloader, self.slice2_dataloader)
            for data1, data2 in batch_iter:
                graph1, he1, panel_1a, selection1 = data1[0]['graph'].to(self.device), data1[0]['he'].to(self.device), \
                    data1[0]['exp'].to(self.device), data1[0]['selection']
                graph2, he2, panel_2b, selection2 = data2[0]['graph'].to(self.device), data2[0]['he'].to(self.device), \
                    data2[0]['exp'].to(self.device), data2[0]['selection']
                agg_mtx1, agg_exp1 = data1[0]['agg_mtx'].to(self.device), data1[0]['agg_exp'].to(self.device)
                agg_mtx2, agg_exp2 = data2[0]['agg_mtx'].to(self.device), data2[0]['agg_exp'].to(self.device)

                loss1, _ = self.module_HA(graph1, he1, agg_exp1, agg_mtx1, selection1)
                loss2, _ = self.module_HB(graph2, he2, agg_exp2, agg_mtx2, selection2)
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_iter.set_description(f"#Epoch: {epoch}: train_loss: {loss.item():.2f}")

        # ========================= Testing / Prediction =========================
        self.module_HA.eval()
        self.module_HB.eval()
        # Panel B on slice 1
        panel_1b = []
        obs_list = []
        for data in self.slice1_dataloader:
            graph, he, obs = data[0]['graph'].to(self.device), data[0]['he'].to(self.device), data[0]['obs']
            panelB1 = self.module_HB.predict(he, graph).detach().cpu().numpy()
            panel_1b.append(panelB1)
            obs_list = obs_list + obs
        panel_1b = np.vstack(panel_1b)
        panel_1b = pd.DataFrame(panel_1b)
        panel_1b.columns = self.adata1.var_names
        panel_1b['obs_name'] = obs_list
        panel_1b = panel_1b.groupby('obs_name').mean()

        # Panel A on slice 2
        panel_2a = []
        obs_list = []
        for data in self.slice2_dataloader:
            graph, he, obs = data[0]['graph'].to(self.device), data[0]['he'].to(self.device), data[0]['obs']
            panel2A = self.module_HA.predict(he, graph).detach().cpu().numpy()
            panel_2a.append(panel2A)
            obs_list = obs_list + obs
        panel_2a = np.vstack(panel_2a)
        panel_2a = pd.DataFrame(panel_2a)
        panel_2a.columns = self.adata2.var_names
        panel_2a['obs_name'] = obs_list
        panel_2a = panel_2a.groupby('obs_name').mean()

        if self.save:
            save_path = './results/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            panel_1b.to_csv(save_path + 'HE_to_omics_panel1b.csv')
            panel_2a.to_csv(save_path + 'HE_to_omics_panel2a.csv')
            print(f'The results have been sucessfully saved in {save_path}')


class Train_SpatialExP:
    """
    SpatialEx+ trainer with bidirectional regression mapping (cycle-style).

    Compared to the baseline, this trainer introduces two regression modules
    (AB/BA) to translate between panels, enabling **indirect** cross-panel
    prediction and encouraging consistency between direct and mapped outputs.

    Parameters
    ----------
    adata1 : :class:`anndata.AnnData`
        Slice 1 with ``.obsm['he']`` and ``.X``.
    adata2 : :class:`anndata.AnnData`
        Slice 2 with ``.obsm['he']`` and ``.X``.
    seed : int, optional, default=0
        Random seed.
    device : torch.device, optional
        Training device (defaults to GPU if available).
    weight_decay : float, optional, default=0.0
        Weight decay.
    optimizer : str, optional, default="adam"
        Optimizer type.
    batch_size : int, optional, default=4096
        Graph-building batch size.
    encoder : str, optional, default="hgnn"
        Backbone key.
    hidden_dim : int, optional, default=512
        Hidden width for HGNN/MLP blocks.
    num_layers : int, optional, default=2
        Number of HGNN layers.
    epochs : int, optional, default=1000
        Number of training epochs.
    lr : float, optional, default=1e-3
        Learning rate.
    loss_fn : str, optional, default="mse"
        Reconstruction loss key.
    num_neighbors : int, optional, default=7
        K in KNN for hypergraph.
    graph_kind : str, optional, default="spatial"
        Graph type (``"spatial"`` or ``"he"``).
    save : bool, optional, default=True
        Save outputs (``omics1.npy``, ``omics2.npy``) under ``./results``.

    Attributes
    ----------
    H1, H2 : torch.sparse.FloatTensor
        Hypergraphs for slice 1/2 (symmetric normalized if requested upstream).
    HE1, HE2 : torch.FloatTensor
        Histology embeddings per cell for slice 1/2.
    panelA1, panelB2 : torch.FloatTensor
        Ground-truth expression matrices (slice-specific panels).
    in_dim1, in_dim2 : int
        Input feature dims from HE embeddings.
    out_dim1, out_dim2 : int
        Output gene dims.
    module_HA, module_HB : :class:`SpatialEx.model.Model_Plus`
        SpatialEx+ backbones for each slice.
    rm_AB, rm_BA : :class:`SpatialEx.model.Regression`
        Regression heads mapping A→B and B→A.
    models : list[nn.Module]
        All models to optimize jointly.
    optimizer : torch.optim.Optimizer
        Optimizer over all model parameters.
    save : bool
        Whether to persist numpy outputs.

    Methods
    -------
    train()
        Run end-to-end training with direct reconstruction and cycle-style
        regression mapping, then save indirect predictions to ``./results``.
    """

    def __init__(self,
                 adata1,
                 adata2,
                 seed=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 weight_decay=0,
                 optimizer="adam",
                 batch_size=4096,
                 encoder="hgnn",
                 hidden_dim=512,
                 num_layers=2,
                 epochs=1000,
                 lr=0.001,
                 loss_fn="mse",
                 num_neighbors=7,
                 graph_kind='spatial',
                 save=True
                 ):
        self.adata1 = adata1
        self.adata2 = adata2
        # Basic hyperparams
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.loss_fn = loss_fn
        self.save = save

        # Spatial params
        self.num_neighbors = num_neighbors
        self.graph_kind = graph_kind

        H1 = pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, batch_size, False, 'spatial', 'crs')
        self.H1 = pp.sparse_mx_to_torch_sparse_tensor(H1).to(self.device)
        H2 = pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, batch_size, False, 'spatial', 'crs')
        self.H2 = pp.sparse_mx_to_torch_sparse_tensor(H2).to(self.device)

        self.HE1, self.HE2 = torch.Tensor(adata1.obsm['he']).to(self.device), torch.Tensor(adata2.obsm['he']).to(self.device)
        self.panelA1, self.panelB2 = torch.Tensor(adata1.X).to(self.device), torch.Tensor(adata2.X).to(self.device)

        self.in_dim1 = adata1.obsm['he'].shape[1]
        self.in_dim2 = adata2.obsm['he'].shape[1]
        self.out_dim1 = adata1.n_vars
        self.out_dim2 = adata2.n_vars

        self.module_HA = Model_Plus(in_dim=self.in_dim1, hidden_dim=self.hidden_dim, out_dim=self.out_dim1, num_layers=self.num_layers,
                                   platform='Visium').to(self.device)
        self.module_HB = Model_Plus(in_dim=self.in_dim2, hidden_dim=self.hidden_dim, out_dim=self.out_dim2, num_layers=self.num_layers,
                                   platform='Visium').to(self.device)

        self.rm_AB = Regression(self.out_dim1, self.out_dim2, self.out_dim2).to(self.device)
        self.rm_BA = Regression(self.out_dim2, self.out_dim1, self.out_dim1).to(self.device)
        self.models = [self.module_HA, self.module_HB, self.rm_AB, self.rm_BA]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)

    def train(self):
        """
        Train SpatialEx+ with direct reconstruction and cycle-style regression.

        Workflow
        --------
        1. Optimize :attr:`module_HA` on ``(HE1, H1) → panelA1`` and
           :attr:`module_HB` on ``(HE2, H2) → panelB2`` using reconstruction loss.
        2. Cross-predict: ``module_HA(HE2, H2) → panelA2_direct``,
           ``module_HB(HE1, H1) → panelB1_direct``.
        3. Enforce cycle mappings using regression heads:
           - :attr:`rm_AB`: map A→B (both from cross-predicted A2 and from ground-truth A1).
           - :attr:`rm_BA`: map B→A (both from cross-predicted B1 and from ground-truth B2).
        4. After training, export **indirect** omics predictions as NumPy arrays.

        Returns
        -------
        None
            Saves two arrays to ``./results``:
            - ``omics1.npy``: B-like array inferred for slice 1 via A→B mapping.
            - ``omics2.npy``: A-like array inferred for slice 2 via B→A mapping.
        """
        pp.set_random_seed(self.seed)
        self.module_HA.train()
        self.module_HB.train()
        self.rm_AB.train()
        self.rm_BA.train()
        print('\n')
        print('=================================== Start training =========================================')
        for epoch in tqdm(range(self.epochs)):
            loss1, _ = self.module_HA(self.HE1, self.H1, self.panelA1)
            loss2, _ = self.module_HB(self.HE2, self.H2, self.panelB2)

            panelA2 = self.module_HA.predict(self.HE2, self.H2, grad=False)
            panelB1 = self.module_HB.predict(self.HE1, self.H1, grad=False)
            loss3, _ = self.rm_AB(panelA2, self.panelB2)
            loss4, _ = self.rm_BA(panelB1, self.panelA1)

            loss5, _ = self.rm_AB(self.panelA1, panelB1)
            loss6, _ = self.rm_BA(self.panelB2, panelA2)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ========================= Testing / Export =========================
        self.module_HA.eval()
        self.module_HB.eval()
        self.rm_AB.eval()
        self.rm_BA.eval()

        # Indirect predictions via regression heads
        panelA1_direct = self.module_HA.predict(self.HE1, self.H1, grad=False)
        omics1_indirect = self.rm_AB.predict(panelA1_direct)

        panelB2_direct = self.module_HB.predict(self.HE2, self.H2, grad=False)
        omics2_indirect = self.rm_BA.predict(panelB2_direct)

        omics1_indirect = omics1_indirect.detach().cpu().numpy()
        omics2_indirect = omics2_indirect.detach().cpu().numpy()

        if self.save:
            save_path = './results/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            np.save(save_path + 'omics1.npy', omics1_indirect)
            np.save(save_path + 'omics2.npy', omics2_indirect)
            print(f'The results have been sucessfully saved in {save_path}')


class Train_SpatialExP_Big:
    """
    Large-scale SpatialEx+ trainer for million-cell datasets.

    This trainer tiles each slice into pseudo-spots and iterates in
    :attr:`batch_num` mini-batches to control memory footprint. It jointly
    optimizes a large hypergraph backbone and bidirectional regression heads.

    Parameters
    ----------
    adata1 : :class:`anndata.AnnData`
        First slice with HE embeddings and expression matrix.
    adata2 : :class:`anndata.AnnData`
        Second slice with HE embeddings and expression matrix.
    num_layers : int, optional, default=2
        Number of HGNN layers in the large backbone.
    hidden_dim : int, optional, default=512
        Hidden width in MLP/HGNN components.
    epochs : int, optional, default=500
        Number of training epochs.
    seed : int, optional, default=0
        Random seed.
    device : torch.device, optional
        Computation device (defaults to GPU if available).
    weight_decay : float, optional, default=0.0
        Optimizer weight decay.
    optimizer : str, optional, default="adam"
        Optimizer type.
    batch_size : int, optional, default=4096
        Graph-building batch size for HE features.
    batch_num : int, optional, default=10
        Number of mini-batches to split spots per epoch.
    encoder : str, optional, default="hgnn"
        Backbone key (for consistency with other trainers).
    lr : float, optional, default=1e-3
        Learning rate.
    loss_fn : str, optional, default="mse"
        Reconstruction loss key.
    num_neighbors : int, optional, default=7
        K in KNN for hypergraph construction.
    graph_kind : str, optional, default="spatial"
        Graph construction mode.
    save : bool, optional, default=True
        Save outputs under ``./results/Big``.

    Attributes
    ----------
    in_dim1, in_dim2 : int
        Input dims from HE embeddings (per slice).
    out_dim1, out_dim2 : int
        Output gene dims per slice.
    agg_mtx1, agg_mtx2 : scipy.sparse.csr_matrix
        Aggregation matrices mapping cells→spots for each slice.
    spot_A1, spot_B2 : torch.FloatTensor
        Aggregated (spot-level) expressions used as supervision.
    HE1, HE2 : torch.FloatTensor
        HE embeddings for slices 1/2.
    panelA1, panelB2 : torch.FloatTensor
        Cell-level expressions for slices 1/2.
    model_big : :class:`SpatialEx.model.Model_Big`
        Large-scale hypergraph backbone handling both slices jointly.
    model_AB, model_BA : :class:`SpatialEx.model.Regression`
        Regression heads enforcing A→B and B→A mappings.
    models : list[nn.Module]
        Modules optimized together.
    optimizer : torch.optim.Optimizer
        Joint optimizer.
    batch_num : int
        Number of mini-batches per epoch.
    save : bool
        Whether to persist NumPy outputs.

    Methods
    -------
    train()
        Mini-batch training over pseudo-spots, then export indirect predictions
        for both slices under ``./results/Big``.
    """

    def __init__(self,
                 adata1,
                 adata2,
                 num_layers=2,
                 hidden_dim=512,
                 epochs=500,
                 seed=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 weight_decay=0,
                 optimizer="adam",
                 batch_size=4096,
                 batch_num=10,
                 encoder="hgnn",
                 lr=0.001,
                 loss_fn="mse",
                 num_neighbors=7,
                 graph_kind='spatial',
                 save=True
                 ):
        self.adata1 = adata1
        self.adata2 = adata2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.batch_num = batch_num
        self.encoder = encoder

        self.lr = lr
        self.loss_fn = loss_fn
        self.num_neighbors = num_neighbors
        self.graph_kind = graph_kind
        self.save = save

        self.in_dim1 = self.adata1.obsm['he'].shape[1]
        self.in_dim2 = self.adata2.obsm['he'].shape[1]
        self.out_dim1 = self.adata1.n_vars
        self.out_dim2 = self.adata2.n_vars

        H1 = pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, batch_size, False, 'spatial', 'crs')
        _, _, adata1 = Generate_pseudo_spot(adata1, all_in=True)
        spot_id = adata1.obs['spot'].values
        head = spot_id[~pd.isna(adata1.obs['spot'])].astype(int)
        tail = np.where(~pd.isna(adata1.obs['spot']))[0]
        values = np.ones_like(tail)
        self.agg_mtx1 = sp.coo_matrix((values, (head, tail)), shape=(head.max() + 1, adata1.n_obs)).tocsr()
        self.spot_A1 = torch.Tensor(self.agg_mtx1 @ adata1.X)

        H2 = pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, batch_size, False, 'spatial', 'crs')
        _, _, adata2 = Generate_pseudo_spot(adata2, all_in=True)
        spot_id = adata2.obs['spot'].values
        head = spot_id[~pd.isna(adata2.obs['spot'])].astype(int)
        tail = np.where(~pd.isna(adata2.obs['spot']))[0]
        values = np.ones_like(tail)
        self.agg_mtx2 = sp.coo_matrix((values, (head, tail)), shape=(head.max()+1, adata2.n_obs)).tocsr()
        self.spot_B2 = torch.Tensor(self.agg_mtx2 @ adata2.X)

        self.HE1, self.HE2 = torch.Tensor(adata1.obsm['he']), torch.Tensor(adata2.obsm['he'])
        self.panelA1, self.panelB2 = torch.Tensor(adata1.X), torch.Tensor(adata2.X)

        self.model_big = Model_Big([H1, H2], [self.in_dim1, self.in_dim2], [self.out_dim1, self.out_dim2], num_layers=self.num_layers,
                                   hidden_dim=self.hidden_dim, device=self.device).to(self.device)
        self.model_AB = Regression(self.out_dim1, int(self.out_dim1/2), self.out_dim2).to(self.device)
        self.model_BA = Regression(self.out_dim2, int(self.out_dim1/2), self.out_dim1).to(self.device)
        self.models = [self.model_big, self.model_AB, self.model_BA]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)

    def train(self):
        """
        Train SpatialEx+ at scale via pseudo-spot tiling and mini-batches.

        Workflow
        --------
        1. Split each slice into :attr:`batch_num` spot mini-batches per epoch.
        2. For each batch:
           - Build sub-aggregation matrices (:math:`\\text{cells} \\rightarrow \\text{spots}`).
           - Supervise spot-level reconstruction on both slices.
           - Predict cross-panel cell-level panels and enforce AB/BA regression
             consistency at spot level (including cycle consistency).
        3. After training, run full-dataset prediction in mini-batches and export
           indirect panels for both slices.

        Returns
        -------
        None
            Writes two arrays to ``./results/Big``:
            - ``big_panel_B1.npy``: Indirect B-like panel for slice 1.
            - ``big_panel_A2.npy``: Indirect A-like panel for slice 2.
        """
        batch_num = self.batch_num
        obs_index1 = list(range(self.agg_mtx1.shape[0]))
        obs_index2 = list(range(self.agg_mtx2.shape[0]))
        batch_size1 = int(self.agg_mtx1.shape[0]/batch_num)
        batch_size2 = int(self.agg_mtx2.shape[0]/batch_num)
        for epoch in range(self.epochs):
            random.shuffle(obs_index1)
            random.shuffle(obs_index2)
            batch_iter = tqdm(range(batch_num), leave=False)
            for batch_idx in batch_iter:
                torch.cuda.empty_cache()
                tgt_spot1 = obs_index1[batch_idx*batch_size1:(batch_idx+1)*batch_size1]
                tgt_cell1 = self.agg_mtx1[tgt_spot1].tocoo().col
                sub_agg_mtx1 = self.agg_mtx1[tgt_spot1][:,tgt_cell1]
                sub_agg_mtx1 = pp.sparse_mx_to_torch_sparse_tensor(sub_agg_mtx1).to(self.device)
                spot_A1_batch = self.spot_A1[tgt_spot1].to(self.device)

                tgt_spot2 = obs_index2[batch_idx*batch_size2:(batch_idx+1)*batch_size2]
                tgt_cell2 = self.agg_mtx2[tgt_spot2].tocoo().col
                sub_agg_mtx2 = self.agg_mtx2[tgt_spot2][:,tgt_cell2]
                sub_agg_mtx2 = pp.sparse_mx_to_torch_sparse_tensor(sub_agg_mtx2).to(self.device)
                spot_B2_batch = self.spot_B2[tgt_spot2].to(self.device)

                loss1, loss2 = self.model_big([tgt_cell1, tgt_cell2], [self.HE1, self.HE2], [spot_A1_batch, spot_B2_batch], [sub_agg_mtx1, sub_agg_mtx2])

                x_prime = self.model_big.predict([tgt_cell1, tgt_cell2], [self.HE1, self.HE2], exchange=True, which='both', grad=False)
                panel_A2, panel_B1 = x_prime[0], x_prime[1]
                loss3, _ = self.model_AB(panel_A2, spot_B2_batch, sub_agg_mtx2)
                loss4, _ = self.model_BA(panel_B1, spot_A1_batch, sub_agg_mtx1)
                loss5, _ = self.model_AB(self.panelA1[tgt_cell1].to(self.device), torch.spmm(sub_agg_mtx1, panel_B1), sub_agg_mtx1)
                loss6, _ = self.model_BA(self.panelB2[tgt_cell2].to(self.device), torch.spmm(sub_agg_mtx2, panel_A2), sub_agg_mtx2)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_iter.set_description(
                f"#Epoch {epoch}, loss1: {round(loss1.item(), 2)}, loss2: {round(loss2.item(), 2)}, loss3: {round(loss3.item(), 2)}, loss4: {round(loss4.item(), 2)}, loss5: {round(loss5.item(), 2)}, loss6: {round(loss6.item(), 2)}")

        # ========================= Full-dataset Inference / Export =========================
        obs_index1 = list(range(self.HE1.shape[0]))
        obs_index2 = list(range(self.HE2.shape[0]))
        batch_size1 = int(np.ceil(self.HE1.shape[0]/batch_num))
        batch_size2 = int(np.ceil(self.HE2.shape[0]/batch_num))
        batch_iter = tqdm(range(batch_num), leave=False)

        indirect_panel_B1_list = []
        indirect_panel_A2_list = []
        tgt_id1_list = []
        tgt_id2_list = []
        self.model_big.eval()
        self.model_AB.eval()
        self.model_BA.eval()
        for batch_idx in batch_iter:
            tgt_id1 = obs_index1[batch_idx*batch_size1:min((batch_idx+1)*batch_size1, self.HE1.shape[0])]
            tgt_id2 = obs_index2[batch_idx*batch_size2:min((batch_idx+1)*batch_size2, self.HE2.shape[0])]

            x_prime = self.model_big.predict([tgt_id1, tgt_id2], [self.HE1, self.HE2], exchange=False, which='both')
            panel_A1_predict, panel_B2_predict = x_prime[0], x_prime[1]

            indirect_panel_B1 = self.model_AB.predict(panel_A1_predict)
            indirect_panel_A2 = self.model_BA.predict(panel_B2_predict)

            tgt_id1_list = tgt_id1_list + tgt_id1
            tgt_id2_list = tgt_id2_list + tgt_id2

            indirect_panel_A2_list.append(indirect_panel_A2.detach().cpu().numpy())
            indirect_panel_B1_list.append(indirect_panel_B1.detach().cpu().numpy())

        indirect_panel_A2_list = np.vstack(indirect_panel_A2_list)
        indirect_panel_B1_list = np.vstack(indirect_panel_B1_list)

        if self.save:
            save_path = './results/Big/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            np.save(save_path + 'big_panel_B1.npy', indirect_panel_B1_list)
            np.save(save_path + 'big_panel_A2.npy', indirect_panel_A2_list)
            print(f'The results have been sucessfully saved in {save_path}')
