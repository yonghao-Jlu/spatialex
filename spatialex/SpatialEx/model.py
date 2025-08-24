# -*- coding: utf-8 -*-
"""
Model components for SpatialEx / SpatialEx+.

This module defines the core neural building blocks used across trainers:

- :class:`HGNN`: Hypergraph message passing with lightweight MLP stacking.
- :class:`DGI`: Deep Graph Infomax head built on :class:`HGNN`.
- :class:`HyperSAGE`: A hypergraph variant of GraphSAGE with neighbor
  aggregation over hyperedges (precomputed attrs for speed).
- :class:`DGI_SAGE`: DGI objective paired with :class:`HyperSAGE`.
- :class:`Predictor_spot` / :class:`Predictor`: HE→omics predictor with optional
  spot-level aggregation loss.
- :class:`Model`, :class:`Model_Plus`, :class:`Model_Big`: end-to-end backbones
  used by the trainers (baseline, plus, and large-scale variants).
- :class:`Regression`: Lightweight mapper between panels (A→B / B→A).
- :class:`Classifier` with :class:`focal_loss`: an auxiliary classifier head.
- :class:`Predictor_dgi`: DGI regularizer module.

Notes
-----
Docstrings follow NumPy/Sphinx conventions so Sphinx ``autodoc`` can render
API reference cleanly with ``:members:`` and ``:undoc-members:``.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .utils import create_activation


class HGNN(nn.Module):
    """
    Hypergraph neural network block with optional hidden MLP stack.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    num_hidden : int
        Hidden width for intermediate linear layers when `num_layers >= 2`.
    out_dim : int
        Output feature dimension.
    num_layers : int
        Number of linear + activation stages (>=1).
    dropout : float
        Dropout rate applied before each linear layer.
    activation : str
        Activation key consumed by :func:`utils.create_activation`.

    Attributes
    ----------
    num_layers : int
        Depth of the block.
    activation : nn.Module
        Activation module created via :func:`utils.create_activation`.
    mlp : nn.ModuleList
        Optional hidden linear stack when `num_layers > 2`.
    W1, W2 : nn.Linear
        First/last linear layers (present depending on `num_layers`).
    dropout : nn.Dropout
        Dropout op shared across layers.

    Notes
    -----
    Each stage computes ``X <- H @ Linear(Dropout(X))`` followed by activation.
    """

    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation):

        super(HGNN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = create_activation(activation)
        self.mlp = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.W1 = nn.Linear(in_dim, out_dim)
        elif num_layers == 2:
            self.W1 = nn.Linear(in_dim, num_hidden)
            self.W2 = nn.Linear(num_hidden, out_dim)
        elif self.num_layers > 2:
            for i in range(self.num_layers - 2):
                self.mlp.append(nn.Linear(num_hidden, num_hidden))

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.FloatTensor, shape (N, F_in)
            Node features.
        H : torch.sparse.FloatTensor, shape (N, N) or (N, E)
            Sparse operator (adjacency or incidence-derived operator).

        Returns
        -------
        torch.FloatTensor, shape (N, F_out)
            Updated node features.
        """
        if self.num_layers == 1:
            X = torch.sparse.mm(H, self.W1(self.dropout(X)))
            X = self.activation(X)
        elif self.num_layers == 2:
            X = torch.sparse.mm(H, self.W1(self.dropout(X)))
            X = self.activation(X)
            X = torch.sparse.mm(H, self.W2(self.dropout(X)))
        else:
            X = torch.sparse.mm(H, self.W1(self.dropout(X)))
            X = self.activation(X)
            for i in range(self.num_layers - 2):
                X = torch.sparse.mm(H, self.mlp[i](self.dropout(X)))
                X = self.activation(X)
            X = torch.sparse.mm(H, self.W2(self.dropout(X)))

        return X


class DGI(nn.Module):
    """
    Deep Graph Infomax objective head built on :class:`HGNN`.

    Parameters
    ----------
    n_in : int
        Input feature dimension.
    n_hid : int
        Hidden width inside :class:`HGNN`.
    n_h : int
        Output feature dimension for :class:`HGNN`.
    activation : str
        Activation key for :class:`HGNN`.

    Methods
    -------
    forward(seq1, seq2, adj) -> (h1, h2, c)
        Compute positive/negative embeddings and global summary vector.
    embed(seq, adj) -> (h_1, c)
        Inference-time embeddings (detached).
    """

    def __init__(self, n_in, n_hid, n_h, activation):
        super(DGI, self).__init__()

        self.hgnn = HGNN(in_dim=n_in,
                         num_hidden=n_hid,
                         out_dim=n_h,
                         num_layers=1,
                         dropout=0.1,
                         activation='prelu')

    def forward(self, seq1, seq2, adj):
        """
        Compute positive/negative node embeddings and global summary.

        Parameters
        ----------
        seq1 : torch.FloatTensor, shape (N, F)
            Original features (positive).
        seq2 : torch.FloatTensor, shape (N, F)
            Shuffled features (negative).
        adj : torch.sparse.FloatTensor
            Sparse graph operator.

        Returns
        -------
        h1 : torch.FloatTensor, shape (N, H)
            Positive node embeddings.
        h2 : torch.FloatTensor, shape (N, H)
            Negative node embeddings.
        c : torch.FloatTensor, shape (1, H)
            Global summary (mean of ``h1``).
        """
        h1 = self.hgnn(seq1, adj)  # 每个细胞正确的表征
        c = torch.mean(h1, dim=0)  # 正确的全局表征
        h2 = self.hgnn(seq2, adj)  # 每个细胞错误的表征
        c = c.unsqueeze(0)
        return h1, h2, c

    def embed(self, seq, adj):
        """
        Embed nodes and return detached tensors.

        Parameters
        ----------
        seq : torch.FloatTensor, shape (N, F)
            Input features.
        adj : torch.sparse.FloatTensor
            Graph operator.

        Returns
        -------
        h_1 : torch.FloatTensor, shape (N, H)
            Node embeddings (detached).
        c : torch.FloatTensor, shape (H,)
            Global summary vector (detached).
        """
        h_1 = self.hgnn(seq, adj)
        c = torch.mean(h_1, dim=0)
        return h_1.detach(), c.detach()


class HyperSAGE(nn.Module):
    """
    Hypergraph-SAGE with neighbor aggregation via incidence structure.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden width for intermediate layers.
    out_dim : int
        Output feature dimension.
    num_layers : int
        Number of message-passing layers.
    dropout : float
        Dropout rate.
    device : torch.device
        Device for sparse ops and parameters.

    Notes
    -----
    This module expects precomputed graph attributes:
    ``graph_attr = {'graph', 'num_nodes', 'num_edges', 'num_neighbors'}``.
    """

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 device):
        super(HyperSAGE, self).__init__()
        self.device = device
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        if self.num_layers > 1:
            self.weight_list = [nn.Parameter(torch.FloatTensor(2 * in_dim, hidden_dim)).to(self.device), ]
            for _ in range(self.num_layers - 2):
                self.weight_list.append(nn.Parameter(torch.FloatTensor(2 * hidden_dim, hidden_dim)).to(self.device))
            self.weight_list.append(nn.Parameter(torch.FloatTensor(2 * hidden_dim, out_dim)).to(self.device))
        else:
            self.weight_list = [nn.Parameter(torch.FloatTensor(2 * in_dim, out_dim)).to(self.device), ]
        for weight in self.weight_list:
            torch.nn.init.xavier_uniform_(weight)

    def forward(self, node_feat, neighbor_list, graph_attr):
        """
        Train-time forward using precomputed neighbor lists and attrs.

        Parameters
        ----------
        node_feat : torch.FloatTensor, shape (N, F)
            Node features (at **current** layer's index space).
        neighbor_list : list[tuple(np.ndarray, torch.Tensor)]
            Multi-layer neighbor tuples; each element is ``(ids, map)`` where
            ``map`` maps global ids to current layer indexing.
        graph_attr : dict
            Dictionary containing precomputed tensors for hypergraph ops.

        Returns
        -------
        torch.FloatTensor, shape (N_layer0, out_dim)
            Updated features in the top layer's index space.
        """
        '''训练过程中会提前计算好一些图属性以加速训练'''
        for layer in range(self.num_layers):
            neighbor_agg_emb = self.Aggregate_neighbors(neighbor_list[self.num_layers - layer - 1][0],
                                                        neighbor_list[self.num_layers - layer][0], node_feat,
                                                        graph_attr)
            map_dict = neighbor_list[self.num_layers - layer][1]
            tgt_index = map_dict[neighbor_list[self.num_layers - 1 - layer][0]]  # 全局索引映射回上一层索引
            feat_input = torch.hstack([node_feat[tgt_index], neighbor_agg_emb])
            node_feat = F.leaky_relu(torch.mm(self.dropout(feat_input), self.weight_list[layer]))
        return node_feat

    def predict(self, node_feat, neighbor_list, graph_attr):
        """
        Inference-time forward that recomputes needed hypergraph attrs.

        Parameters
        ----------
        node_feat : torch.FloatTensor
            Node features at the base resolution.
        neighbor_list : list
            Neighbor tuples up to top layer.
        graph_attr : dict
            Hypergraph attributes (same keys as in :meth:`forward`).

        Returns
        -------
        torch.FloatTensor
            Node embeddings at the top layer.
        """
        '''需要重新计算超图相关属性'''
        neighbor_agg_emb = self.Aggregate_neighbors(neighbor_list[self.num_layers - 1][0],
                                                    neighbor_list[self.num_layers][0], node_feat, graph_attr)
        map_dict = neighbor_list[self.num_layers][1]
        tgt_index = map_dict[neighbor_list[self.num_layers - 1][0]]
        feat_input = torch.hstack([node_feat[tgt_index], neighbor_agg_emb])
        node_feat = F.leaky_relu(torch.mm(self.dropout(feat_input), self.weight1))

        neighbor_agg_emb = self.Aggregate_neighbors(neighbor_list[self.num_layers - 2][0],
                                                    neighbor_list[self.num_layers - 1][0], node_feat, graph_attr)
        map_dict = neighbor_list[self.num_layers - 1][1]
        tgt_index = map_dict[neighbor_list[self.num_layers - 2][0]]
        feat_input = torch.hstack([node_feat[tgt_index], neighbor_agg_emb])
        node_feat = F.leaky_relu(torch.mm(self.dropout(feat_input), self.weight2))
        return node_feat

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx, cuda=False):
        """
        Convert a SciPy sparse matrix to a torch sparse FloatTensor.

        Parameters
        ----------
        sparse_mx : scipy.sparse.spmatrix
        cuda : bool, default=False
            If True, move tensor to ``self.device``.

        Returns
        -------
        torch.sparse.FloatTensor
        """
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        if cuda:
            return torch.sparse.FloatTensor(indices, values, shape).to(self.device)
        else:
            return torch.sparse.FloatTensor(indices, values, shape)

    def sparse_diags(self, data):
        """
        Create a sparse diagonal tensor on ``self.device``.

        Parameters
        ----------
        data : torch.FloatTensor, shape (N,)

        Returns
        -------
        torch.sparse.FloatTensor, shape (N, N)
        """
        indices = torch.arange(data.shape[0])
        indices = torch.vstack([indices, indices]).to(self.device)
        return torch.sparse_coo_tensor(indices, data, (data.shape[0], data.shape[0]))

    def initialize(self, hyper_graph):
        """
        Precompute hypergraph statistics used by aggregation.

        Parameters
        ----------
        hyper_graph : scipy.sparse.spmatrix, shape (N, E)
            Incidence matrix.

        Returns
        -------
        num_nodes : torch.FloatTensor, shape (E,)
            #nodes per hyperedge.
        num_edges : torch.FloatTensor, shape (N,)
            #hyperedges per node.
        num_neighbors : torch.FloatTensor, shape (N,)
            #neighbors per node (via node-by-node connectivity).
        """
        hyper_graph = self.sparse_mx_to_torch_sparse_tensor(hyper_graph).to(self.device)
        num_nodes = hyper_graph.sum(0).to_dense()  # 计算每个超边包含的节点数量
        num_edges = hyper_graph.sum(1).to_dense()  # 每个节点被几个超边包含

        node_by_node = torch.spmm(hyper_graph, hyper_graph.T)
        indices = node_by_node.indices()
        data = torch.ones(indices.shape[-1])
        node_by_node = torch.sparse_coo_tensor(indices, data.to(self.device), node_by_node.shape)  # 构建节点和节点之间的连接矩阵
        num_neighbors = node_by_node.sum(1).to_dense()  # 计算每个节点的邻居节点数量
        return num_nodes, num_edges, num_neighbors

    def Aggregate_neighbors(self, tgt_idx, src_idx, node_emb, graph_attr=None):
        """
        Aggregate neighbor information through hyperedges.

        Parameters
        ----------
        tgt_idx : array-like
            Target node indices at current layer.
        src_idx : array-like
            Source node indices (neighbors) at previous layer.
        node_emb : torch.FloatTensor, shape (N_src, F)
            Node embeddings at `src_idx`.
        graph_attr : dict
            Must contain keys: ``'graph'``, ``'num_nodes'``, ``'num_edges'``,
            ``'num_neighbors'``.

        Returns
        -------
        neighbor_agg_emb : torch.FloatTensor, shape (N_tgt, F)
            Aggregated neighbor features for target nodes.

        Notes
        -----
        Steps:
        1) Project source features to hyperedges (normalized by edge size).
        2) Project hyperedge features back to target nodes and normalize by
           #neighbors and #edges incident to each target node.
        """
        '''
        前向传播每次需要聚合邻居节点
        tgt_idx,            [list],             本层需要更新的目标节点
        src_idx,            [list],             本层目标节点的邻居节点
        node_emb,           [torch.Tensor, 2D], 前一层的节点表征
        model,              [str]             ,
        '''

        hyper_graph, num_nodes, num_edges, num_neighbors = graph_attr['graph'], graph_attr['num_nodes'], graph_attr[
            'num_edges'], graph_attr['num_neighbors']

        '''将源节点特征聚合到超边上'''
        tgt_edge = np.unique(hyper_graph[tgt_idx].tocoo().col)  # 相关超边，只有稀疏array支持索引
        edge_cardinality = num_nodes[tgt_edge]
        edge_cardinality_inv = 1.0 / edge_cardinality
        edge_cardinality = self.sparse_diags(edge_cardinality)
        edge_cardinality_inv = self.sparse_diags(edge_cardinality_inv)

        edge_agg_mtx = self.sparse_mx_to_torch_sparse_tensor(hyper_graph[src_idx][:, tgt_edge].T,
                                                             cuda=True)  # 目标节点参与的超边*邻接节点的聚合矩阵
        edge_emb = torch.spmm(torch.spmm(edge_cardinality_inv, edge_agg_mtx), node_emb)  # 将节点特征聚合到超边上

        '''将超边特征聚合到目标节点上'''
        num_neighbor_inv = 1.0 / num_neighbors[tgt_idx]
        num_neighbor_inv = self.sparse_diags(num_neighbor_inv)  # 算目标节点所有通过超边邻接的节点数量
        num_edge_inv = 1.0 / num_edges[tgt_idx]
        num_edge_inv = self.sparse_diags(num_edge_inv)

        tgt_by_edge = self.sparse_mx_to_torch_sparse_tensor(hyper_graph[tgt_idx][:, tgt_edge], cuda=True)
        neighbor_agg_emb = torch.spmm(num_edge_inv,
                                      num_neighbor_inv @ torch.spmm(torch.spmm(tgt_by_edge, edge_cardinality),
                                                                    edge_emb))
        return neighbor_agg_emb


class DGI_SAGE(nn.Module):
    """
    DGI objective coupled with :class:`HyperSAGE`.

    Parameters
    ----------
    num_layers : int
        #layers for :class:`HyperSAGE`.
    dropout : float
        Dropout rate.
    device : torch.device
        Device for computation.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden width for MLP and :class:`HyperSAGE`.
    """

    def __init__(
            self,
            num_layers,
            dropout,
            device,
            in_dim: int,
            hidden_dim: int,
    ):
        super(DGI_SAGE, self).__init__()

        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim)).to(device)
        self.gnn = HyperSAGE(in_dim=hidden_dim,
                             hidden_dim=hidden_dim,
                             out_dim=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             device=device)
        self.b_xent = nn.CosineEmbeddingLoss()

    def forward(self, node_feat, neighbor_list, graph_attr):
        """
        Compute DGI loss for :class:`HyperSAGE`.

        Parameters
        ----------
        node_feat : torch.FloatTensor, shape (N, F)
        neighbor_list : list
            As required by :meth:`HyperSAGE.forward`.
        graph_attr : dict
            Precomputed attributes for hypergraph ops.

        Returns
        -------
        torch.Tensor
            DGI cosine-embedding loss.
        """
        feat = self.mlp(node_feat)

        nb_nodes = node_feat.shape[0]
        idx = torch.randperm(nb_nodes)
        feat_shuffled = feat[idx, :]

        h1 = self.gnn(feat, neighbor_list, graph_attr)
        h2 = self.gnn(feat_shuffled, neighbor_list, graph_attr)
        c = torch.mean(h1, dim=0).unsqueeze(0)

        lbl_1 = torch.ones(len(neighbor_list[0][0])).to(self.device)
        lbl_2 = -torch.ones(len(neighbor_list[0][0])).to(self.device)
        loss = self.b_xent(h1, c, lbl_1) + self.b_xent(h2, c, lbl_2)
        return loss


class Predictor_spot(nn.Module):
    """
    HE→omics predictor with optional spot-level aggregation loss.

    Parameters
    ----------
    in_dim : int
        Input HE feature dimension.
    hidden_dim : int
        Hidden width.
    out_dim : int
        Output gene dimension.
    num_layers : int
        #layers for :class:`HGNN`.
    dropout : float, default=0.1
        Dropout rate (internally set to 0 in this implementation).
    loss_fn : {'mse'}, default='mse'
        Reconstruction loss.
    activation : str, default='prelu'
        Activation key for :class:`HGNN`.
    agg : bool, default=True
        If True, compute loss on aggregated (spot-level) outputs using
        ``agg_mtx`` and ``selection``.

    Methods
    -------
    forward(graph, he_rep, x, agg_mtx=None, selection=None)
        Return (loss, per-cell prediction, latent).
    predict(graph, he_rep)
        Return only per-cell prediction (no loss).
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            dropout: float = 0.1,
            loss_fn='mse',
            activation='prelu',
            agg=True,
    ):
        super(Predictor_spot, self).__init__()

        dropout = 0
        self.agg = agg
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim)
        )

        self.mod = HGNN(
            in_dim=hidden_dim,
            num_hidden=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )

        self.linear = nn.Linear(hidden_dim, out_dim)

        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            print("not implement")

    def forward(self, graph, he_rep, x, agg_mtx=None, selection=None):
        """
        Compute loss and prediction.

        Parameters
        ----------
        graph : torch.sparse.FloatTensor
            Sparse operator for propagation.
        he_rep : torch.FloatTensor, shape (N, F)
            HE features.
        x : torch.FloatTensor
            Supervision target (spot-level if `agg=True`, else cell-level).
        agg_mtx : torch.sparse.FloatTensor or None
            Spot aggregation matrix (spots × selected-cells).
        selection : np.ndarray[bool] or None
            Mask for cells within ROI contributing to loss.

        Returns
        -------
        loss : torch.Tensor
        x_prime : torch.FloatTensor, shape (N, G)
        enc : torch.FloatTensor, shape (N, H)
        """
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))
        if self.agg:
            loss = self.criterion(torch.sparse.mm(agg_mtx, x_prime[selection]), x)
        else:
            loss = self.criterion(x_prime, x)
        return loss, x_prime, enc

    def predict(self, graph, he_rep):
        """
        Predict per-cell gene panel.

        Parameters
        ----------
        graph : torch.sparse.FloatTensor
        he_rep : torch.FloatTensor, shape (N, F)

        Returns
        -------
        torch.FloatTensor, shape (N, G)
        """
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))
        return x_prime


class Model(nn.Module):
    """
    Baseline SpatialEx backbone (per-slice).

    This model combines a per-cell predictor (:class:`Predictor_spot`) and a
    DGI regularizer (:class:`Predictor_dgi`). The forward computes a weighted
    reconstruction loss (optionally aggregated at spot level) plus DGI.

    Parameters
    ----------
    num_layers : int, default=2
        #HGNN layers inside the predictor.
    in_dim : int, default=2048
        Input HE dimension.
    hidden_dim : int, default=512
        Hidden width.
    out_dim : int, default=150
        Output gene dimension.
    loss_fn : {'mse'}, default='mse'
        Reconstruction loss.
    device : str or torch.device, default='cpu'
        Target device.

    Methods
    -------
    forward(graph, he_rep, exp, agg_mtx, selection) -> (loss, x_prime)
        Train-time objective combining predictor and DGI losses.
    predict(he_representations, graph, grad=False) -> torch.Tensor
        Inference-only per-cell prediction.
    """

    def __init__(self,
                 num_layers=2,
                 in_dim=2048,
                 hidden_dim=512,
                 out_dim=150,
                 loss_fn="mse",
                 device="cpu"
                 ):
        super(Model, self).__init__()
        self.predictor = Predictor_spot(
            in_dim=in_dim,  # 超图训练
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            loss_fn=loss_fn)

        self.dgi_model = Predictor_dgi(in_dim=in_dim,  # dgi模型
                                       hidden_dim=hidden_dim,
                                       out_dim=out_dim)

        self.predictor.to(device)
        self.dgi_model.to(device)

    def forward(self, graph, he_rep, exp, agg_mtx, selection):
        """
        Forward training objective.

        Parameters
        ----------
        graph : torch.sparse.FloatTensor
        he_rep : torch.FloatTensor, shape (N, F)
        exp : torch.FloatTensor
            Supervision tensor (spot-level if aggregated).
        agg_mtx : torch.sparse.FloatTensor
            Spot aggregation matrix.
        selection : np.ndarray[bool]
            Mask of cells within ROI contributing to loss.

        Returns
        -------
        loss : torch.Tensor
            Sum of reconstruction and DGI losses.
        x_prime : torch.FloatTensor, shape (N, G)
            Per-cell prediction.
        """
        loss_pre, x_prime, _ = self.predictor(graph, he_rep, exp, agg_mtx, selection)
        loss_dgi = self.dgi_model(graph, he_rep)
        loss = loss_pre + loss_dgi
        return loss, x_prime

    def predict(self, he_representations, graph, grad=False):
        """
        Predict per-cell gene panel (no loss).

        Parameters
        ----------
        he_representations : torch.FloatTensor, shape (N, F)
        graph : torch.sparse.FloatTensor
        grad : bool, default=False
            If True, keep gradient graph.

        Returns
        -------
        torch.FloatTensor, shape (N, G)
        """
        if not grad:
            with torch.no_grad():
                x_prime = self.predictor.predict(graph, he_representations)
        else:
            x_prime = self.predictor.predict(graph, he_representations)
        return x_prime


class Model_Plus(nn.Module):
    """
    SpatialEx+ backbone with optional DGI regularization.

    Parameters
    ----------
    in_dim : int
        Input HE dimension.
    hidden_dim : int
        Hidden width.
    out_dim : int
        Output gene dimension.
    num_layers : int
        #HGNN layers.
    dropout : float, default=0.1
        Dropout rate for :class:`HGNN`.
    activation : str, default='prelu'
        Activation key.
    use_dgi : bool, default=True
        If True, apply DGI auxiliary loss.
    loss_fn : {'mse'}, default='mse'
        Reconstruction loss key.
    platform : {'Xenium','Visium'}, default='Xenium'
        Governs whether loss is computed at cell-level or spot-level.

    Methods
    -------
    forward(x, adj, y, agg_mtx=None) -> (loss, x_prime)
    predict(x, adj, grad=False) -> torch.Tensor
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 activation='prelu',
                 use_dgi: bool = True,
                 loss_fn: str = 'mse',
                 platform: str = 'Xenium'):
        super(Model_Plus, self).__init__()

        self.platform = platform
        self.use_dgi = use_dgi
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.BatchNorm1d(hidden_dim))
        self.hgnn = HGNN(in_dim=hidden_dim,
                         num_hidden=hidden_dim,
                         out_dim=hidden_dim,
                         num_layers=num_layers,
                         dropout=dropout,
                         activation=activation)
        self.predictor = nn.Linear(hidden_dim, out_dim)

        if self.use_dgi:
            self.dgi = DGI(hidden_dim, hidden_dim, out_dim, 'prelu')
            self.b_xent = nn.CosineEmbeddingLoss()

        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()

    def forward(self, x, adj, y, agg_mtx=None):
        """
        Compute reconstruction + optional DGI loss.

        Parameters
        ----------
        x : torch.FloatTensor, shape (N, F)
            HE features.
        adj : torch.sparse.FloatTensor
            Graph operator.
        y : torch.FloatTensor
            Supervision (cell- or spot-level depending on `platform`).
        agg_mtx : torch.sparse.FloatTensor or None
            Aggregation matrix when platform != 'Visium'.

        Returns
        -------
        loss : torch.Tensor
        x_prime : torch.FloatTensor, shape (N, G)
        """
        x = self.mlp(x)
        h = F.leaky_relu(self.hgnn(x, adj))
        x_prime = F.leaky_relu(self.predictor(h))
        if self.platform == 'Visium':
            loss = self.criterion(x_prime, y)
        else:
            loss = self.criterion(torch.mm(agg_mtx, x_prime), y)
        if self.use_dgi:
            nb_nodes = x.shape[0]
            x_shuffle = x[torch.randperm(nb_nodes)]
            h1, h2, c = self.dgi(x, x_shuffle, adj)
            lbl_1 = torch.ones(nb_nodes).to(x.device)
            lbl_2 = -torch.ones(nb_nodes).to(x.device)
            loss = loss + self.b_xent(h1, c, lbl_1) + self.b_xent(h2, c, lbl_2)
        return loss, x_prime

    def predict(self, x, adj, grad=False):
        """
        Predict per-cell gene panel.

        Parameters
        ----------
        x : torch.FloatTensor, shape (N, F)
        adj : torch.sparse.FloatTensor
        grad : bool, default=False

        Returns
        -------
        torch.FloatTensor, shape (N, G)
        """
        if not grad:
            with torch.no_grad():
                x = self.mlp(x)
                h = F.leaky_relu(self.hgnn(x, adj))
                x_prime = F.leaky_relu(self.predictor(h))
        else:
            x = self.mlp(x)
            h = F.leaky_relu(self.hgnn(x, adj))
            x_prime = F.leaky_relu(self.predictor(h))
        return x_prime


class Regression(nn.Module):
    """
    Lightweight regression mapper between panels (A→B or B→A).

    Parameters
    ----------
    in_dim : int
        Input dimension (source panel).
    hidden_dim : int
        Hidden width.
    out_dim : int
        Output dimension (target panel).
    platform : {'Xenium','Visium'}, default='Xenium'
        Reserved for potential platform-specific behavior.

    Methods
    -------
    forward(x, y=None, agg_mtx=None) -> (loss, mapped) or mapped
        If `y` is provided, compute MSE loss (optionally on aggregated output).
    predict(x, grad=False) -> torch.Tensor
        Inference-only mapping.
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            platform: str = 'Xenium',
    ):
        super(Regression, self).__init__()

        self.platform = platform
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.LeakyReLU(0.1),
        )
        self.b_xent = nn.MSELoss()

    def forward(self, x, y=None, agg_mtx=None):
        """
        Forward mapping and optional loss.

        Parameters
        ----------
        x : torch.FloatTensor, shape (N, F_in)
            Source panel representation.
        y : torch.FloatTensor or None
            Target supervision. If None, returns mapped output only.
        agg_mtx : torch.sparse.FloatTensor or None
            Optional aggregation matrix for spot-level supervision.

        Returns
        -------
        If `y` is None:
            mapped : torch.FloatTensor, shape (N, F_out)
        Else:
            loss : torch.Tensor
            mapped : torch.FloatTensor, shape (N, F_out)
        """
        x = self.mlp(x)
        if y is None:
            return x
        if agg_mtx != None:
            loss = self.b_xent(torch.spmm(agg_mtx, x), y)
        else:
            loss = self.b_xent(x, y)
        return loss, x

    def predict(self, x, grad=False):
        """
        Predict-only mapping.

        Parameters
        ----------
        x : torch.FloatTensor, shape (N, F_in)
        grad : bool, default=False

        Returns
        -------
        torch.FloatTensor, shape (N, F_out)
        """
        if not grad:
            with torch.no_grad():
                x = self.mlp(x)
        else:
            x = self.mlp(x)
        return x


class Classifier(torch.nn.Module):
    """
    Auxiliary classifier with focal loss.

    Parameters
    ----------
    dim_input : int
        Input feature dimension.
    dim_hidden : int
        Hidden width.
    dim_output : int
        #classes.
    alpha : float or list[float]
        Class weighting (see :class:`focal_loss`).
    device : torch.device
        Device for loss tensors.

    Methods
    -------
    forward(x, y) -> loss
    predict(x) -> logits
    """

    def __init__(self,
                 dim_input,
                 dim_hidden,
                 dim_output,
                 alpha,
                 device):
        super(Classifier, self).__init__()
        self.hidden1 = nn.Linear(dim_input, dim_hidden)
        self.hidden2 = nn.Linear(dim_hidden, dim_hidden)
        self.hidden3 = nn.Linear(dim_hidden, dim_output)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.xavier_uniform_(self.hidden2.weight)
        torch.nn.init.xavier_uniform_(self.hidden3.weight)

        self.criterion = focal_loss(alpha=alpha, device=device)

    def forward(self, x, y):
        """
        Compute focal loss.

        Parameters
        ----------
        x : torch.FloatTensor, shape (N, F)
        y : torch.LongTensor, shape (N,)
            Class labels.

        Returns
        -------
        torch.Tensor
            Focal loss value.
        """
        x = self.add_gaussian_noise(x)
        h = F.leaky_relu(F.dropout(self.hidden1(x)))
        h = F.leaky_relu(F.dropout(self.hidden2(h)))
        h = F.leaky_relu(self.hidden3(h))
        loss = self.criterion(h, y)
        return loss

    def predict(self, x):
        """
        Compute logits (no loss).

        Parameters
        ----------
        x : torch.FloatTensor, shape (N, F)

        Returns
        -------
        torch.FloatTensor, shape (N, C)
            Class logits.
        """
        h = F.leaky_relu(F.dropout(self.hidden1(x)))
        h = F.leaky_relu(F.dropout(self.hidden2(h)))
        h = F.leaky_relu(self.hidden3(h))
        return h

    def add_gaussian_noise(self, x, mean=0, std=0.1):
        """
        Add Gaussian noise for regularization.

        Parameters
        ----------
        x : torch.FloatTensor
        mean : float, default=0
        std : float, default=0.1

        Returns
        -------
        torch.FloatTensor
        """
        noise = torch.randn_like(x) * std + mean
        return x + noise.to(x.device)


class Model_Big(nn.Module):
    """
    Large-scale SpatialEx+ backbone jointly handling two slices.

    Parameters
    ----------
    hyper_graph : list[scipy.sparse.spmatrix]
        Hypergraph incidence matrices for slice A and B.
    in_dim : list[int]
        Input dims [in_dim_A, in_dim_B].
    out_dim : list[int]
        Output dims [out_dim_A, out_dim_B].
    num_layers : int
        #layers for :class:`HyperSAGE`.
    hidden_dim : int
        Hidden width.
    device : torch.device
        Device for ops.
    use_dgi : bool, default=True
        If True, apply DGI losses per slice.

    Attributes
    ----------
    graph_attr1, graph_attr2 : dict
        Precomputed hypergraph stats for each slice.
    node_by_node1, node_by_node2 : scipy.sparse.spmatrix
        Node-by-node connectivity (H H^T) used to get neighbors.

    Methods
    -------
    forward(tgt_id, node_feat, x, agg_mtx=None, return_prime=False)
        Return per-slice losses (and optionally predictions).
    predict(tgt_id, node_feat, exchange=False, which='both', grad=False)
        Per-cell predictions for selected ids; exchange=True swaps panels.
    set_graph_attr(hyper_graph)
        Reset graph attributes when H changes.
    """

    def __init__(self,
                 hyper_graph,
                 in_dim,
                 out_dim,
                 num_layers,
                 hidden_dim,
                 device,
                 use_dgi=True):
        super(Model_Big, self).__init__()

        self.use_dgi = use_dgi
        self.device = device
        self.gnn_layers = num_layers
        self.graph_attr1 = self.initialize_graph_attr(hyper_graph[0])
        self.graph_attr2 = self.initialize_graph_attr(hyper_graph[1])
        self.node_by_node1 = hyper_graph[0] @ hyper_graph[0].T
        self.node_by_node2 = hyper_graph[1] @ hyper_graph[1].T

        self.mlp1 = nn.Sequential(nn.Linear(in_dim[0], hidden_dim),
                                  nn.LeakyReLU(0.1),
                                  nn.BatchNorm1d(hidden_dim))
        self.mlp2 = nn.Sequential(nn.Linear(in_dim[1], hidden_dim),
                                  nn.LeakyReLU(0.1),
                                  nn.BatchNorm1d(hidden_dim))

        self.SAGE_HA = HyperSAGE(in_dim=hidden_dim,
                                 hidden_dim=hidden_dim,
                                 out_dim=hidden_dim,
                                 num_layers=num_layers,
                                 dropout=0.1,
                                 device=device)

        self.SAGE_HB = HyperSAGE(in_dim=hidden_dim,
                                 hidden_dim=hidden_dim,
                                 out_dim=hidden_dim,
                                 num_layers=num_layers,
                                 dropout=0.1,
                                 device=device)

        self.predicter1 = nn.Linear(hidden_dim, out_dim[0])
        self.predicter2 = nn.Linear(hidden_dim, out_dim[1])

        if self.use_dgi:
            self.dgi1 = DGI_SAGE(num_layers=1,
                                 dropout=0.1,
                                 device=device,
                                 in_dim=in_dim[0],
                                 hidden_dim=hidden_dim)

            self.dgi2 = DGI_SAGE(num_layers=1,
                                 dropout=0.1,
                                 device=device,
                                 in_dim=in_dim[1],
                                 hidden_dim=hidden_dim)
        self.criterion = nn.MSELoss()

    def initialize_graph_attr(self, hyper_graph):
        """
        Precompute per-slice hypergraph statistics.

        Parameters
        ----------
        hyper_graph : scipy.sparse.spmatrix, shape (N, E)

        Returns
        -------
        graph_attr : dict
            Keys: 'graph', 'num_nodes', 'num_edges', 'num_neighbors'.
        """
        graph_attr = {}
        graph_attr['graph'] = hyper_graph.copy()
        hyper_graph = self.sparse_mx_to_torch_sparse_tensor(hyper_graph).to(self.device)
        num_nodes = hyper_graph.sum(0).to_dense()  # 计算每个超边包含的节点数量
        num_edges = hyper_graph.sum(1).to_dense()  # 每个节点被几个超边包含

        node_by_node = torch.spmm(hyper_graph, hyper_graph.T)
        indices = node_by_node.indices()
        data = torch.ones(indices.shape[-1])
        node_by_node = torch.sparse_coo_tensor(indices, data.to(self.device), node_by_node.shape)  # 构建节点和节点之间的连接矩阵
        num_neighbors = node_by_node.sum(1).to_dense()  # 计算每个节点的邻居节点数量
        graph_attr['num_nodes'] = num_nodes
        graph_attr['num_edges'] = num_edges
        graph_attr['num_neighbors'] = num_neighbors
        return graph_attr

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx, cuda=False):
        """
        Convert SciPy sparse matrix to torch sparse FloatTensor.

        Parameters
        ----------
        sparse_mx : scipy.sparse.spmatrix
        cuda : bool, default=False

        Returns
        -------
        torch.sparse.FloatTensor
        """
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        if cuda:
            return torch.sparse.FloatTensor(indices, values, shape).to(self.device)
        else:
            return torch.sparse.FloatTensor(indices, values, shape)

    def get_neighbors(self, tgt_idx, node_by_node):
        """
        Build multi-layer neighbor lists for :class:`HyperSAGE`.

        Parameters
        ----------
        tgt_idx : np.ndarray
            Seed node indices for the top layer.
        node_by_node : scipy.sparse.spmatrix
            Sparse node-by-node connectivity.

        Returns
        -------
        list[tuple(np.ndarray, torch.Tensor)]
            Each element ``(ids, mapping)`` where `mapping` maps global ids
            into contiguous indices of current layer.
        """
        neighbor_list = [(tgt_idx, None), ]
        for _ in range(self.gnn_layers):
            tgt_idx = np.unique(node_by_node[tgt_idx].tocoo().col)  # 找到邻居节点
            mapped_indices = torch.arange(tgt_idx.shape[0], device=self.device, dtype=torch.int32)
            mapping = torch.zeros(node_by_node.shape[0], dtype=torch.int32, device=self.device)
            mapping[tgt_idx] = mapped_indices
            neighbor_list.append((tgt_idx, mapping))
        return neighbor_list

    def forward(self, tgt_id, node_feat, x, agg_mtx=None, return_prime=False):
        """
        Joint forward for both slices (mini-batch of pseudo-spots).

        Parameters
        ----------
        tgt_id : list[np.ndarray]
            Target node ids for slice A and B.
        node_feat : list[torch.FloatTensor]
            HE features for both slices.
        x : list[torch.FloatTensor]
            Spot-level supervision for both slices.
        agg_mtx : list[torch.sparse.FloatTensor]
            Spot aggregation matrices (spots × cells in batch) for both slices.
        return_prime : bool, default=False
            If True, also return per-cell predictions for both slices.

        Returns
        -------
        loss1, loss2 : torch.Tensor
            Loss values for slice A and B.
        (optional) x_prime1, x_prime2 : torch.FloatTensor
            Per-cell predictions when `return_prime=True`.
        """
        tgt_id1, node_feat1, x1, agg_mtx1 = tgt_id[0], node_feat[0], x[0], agg_mtx[0]
        neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
        enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
        enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
        x_prime1 = self.predicter1(enc)
        loss1 = self.criterion(torch.spmm(agg_mtx1, x_prime1), x1)
        if self.use_dgi:
            loss1 = loss1 + self.dgi1(node_feat1[neighbor_list[-2][0]].to(self.device), neighbor_list[:2],
                                      self.graph_attr1)  # 完全复现之前的

        tgt_id2, node_feat2, x2, agg_mtx2 = tgt_id[1], node_feat[1], x[1], agg_mtx[1]
        neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node2)
        enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
        enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr2)
        x_prime2 = self.predicter2(enc)
        loss2 = self.criterion(torch.spmm(agg_mtx2, x_prime2), x2)
        if self.use_dgi:
            loss2 = loss2 + self.dgi2(node_feat2[neighbor_list[-2][0]].to(self.device), neighbor_list[:2],
                                      self.graph_attr2)
        if return_prime:
            return loss1, loss2, x_prime1, x_prime2

        return loss1, loss2

    def predict(self, tgt_id, node_feat, exchange=False, which='both', grad=False):
        """
        Per-cell prediction on selected node ids.

        Parameters
        ----------
        tgt_id : list or np.ndarray
            If `which='both'`, a pair of id arrays [ids_A, ids_B].
            Else a single id array for the requested panel.
        node_feat : list or torch.FloatTensor
            HE features aligned with `tgt_id`.
        exchange : bool, default=False
            If True and `which='both'`, exchange panels (A→B, B→A).
        which : {'panelA','panelB','both'}, default='both'
            Select which panel(s) to predict.
        grad : bool, default=False
            Keep computation graph if True.

        Returns
        -------
        torch.FloatTensor or list[torch.FloatTensor]
            Predictions for the requested panel(s).
        """
        if not grad:
            with torch.no_grad():
                if which == 'panelA':
                    neighbor_list = self.get_neighbors(tgt_id, self.node_by_node1)
                    enc = self.mlp1(node_feat[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HA(node_feat, neighbor_list, self.graph_attr1)
                    x_prime = self.predicter1(enc)
                elif which == 'panelB':
                    neighbor_list = self.get_neighbors(tgt_id, self.node_by_node2)
                    enc = self.mlp2(node_feat[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HB(node_feat, neighbor_list, self.graph_attr2)
                    x_prime = self.predicter2(enc)
                elif which == 'both':
                    if not exchange:
                        x_prime = []
                        tgt_id1, node_feat1 = tgt_id[0], node_feat[0]
                        neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
                        enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                        enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
                        x_prime.append(self.predicter1(enc))

                        tgt_id2, node_feat2 = tgt_id[1], node_feat[1]
                        neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node2)
                        enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
                        enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr2)
                        x_prime.append(self.predicter2(enc))
                    else:
                        x_prime = []
                        tgt_id1, node_feat1 = tgt_id[1], node_feat[1]
                        neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node2)
                        enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                        enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr2)
                        x_prime.append(self.predicter1(enc))

                        tgt_id2, node_feat2 = tgt_id[0], node_feat[0]
                        neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node1)
                        enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
                        enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr1)
                        x_prime.append(self.predicter2(enc))
                else:
                    print('Please specify the panel you want to predict: panelA/panelB/both.')
        else:
            if which == 'panelA':
                neighbor_list = self.get_neighbors(tgt_id, self.node_by_node1)
                enc = self.mlp1(node_feat[neighbor_list[-1][0]].to(self.device))
                enc = self.SAGE_HA(node_feat, neighbor_list, self.graph_attr1)
                x_prime = self.predicter1(enc)
            elif which == 'panelB':
                neighbor_list = self.get_neighbors(tgt_id, self.node_by_node2)
                enc = self.mlp2(node_feat[neighbor_list[-1][0]].to(self.device))
                enc = self.SAGE_HB(node_feat, neighbor_list, self.graph_attr2)
                x_prime = self.predicter2(enc)
            elif which == 'both':
                if not exchange:
                    x_prime = []
                    tgt_id1, node_feat1 = tgt_id[0], node_feat[0]
                    neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node1)
                    enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr1)
                    x_prime.append(self.predicter1(enc))

                    tgt_id2, node_feat2 = tgt_id[1], node_feat[1]
                    neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node2)
                    enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr2)
                    x_prime.append(self.predicter2(enc))
                else:
                    x_prime = []
                    tgt_id1, node_feat1 = tgt_id[1], node_feat[1]
                    neighbor_list = self.get_neighbors(tgt_id1, self.node_by_node2)
                    enc = self.mlp1(node_feat1[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HA(enc, neighbor_list, self.graph_attr2)
                    x_prime.append(self.predicter1(enc))

                    tgt_id2, node_feat2 = tgt_id[0], node_feat[0]
                    neighbor_list = self.get_neighbors(tgt_id2, self.node_by_node1)
                    enc = self.mlp2(node_feat2[neighbor_list[-1][0]].to(self.device))
                    enc = self.SAGE_HB(enc, neighbor_list, self.graph_attr1)
                    x_prime.append(self.predicter2(enc))
            else:
                print('Please specify the panel you want to predict: panelA/panelB/both.')
        return x_prime

    def set_graph_attr(self, hyper_graph):
        """
        Reset cached hypergraph attributes (when H changes).

        Parameters
        ----------
        hyper_graph : list[scipy.sparse.spmatrix]
            New incidence matrices for the two slices.
        """
        self.graph_attr1 = self.initialize_graph_attr(hyper_graph[0])
        self.graph_attr2 = self.initialize_graph_attr(hyper_graph[1])
        self.node_by_node1 = hyper_graph[0] @ hyper_graph[0].T
        self.node_by_node2 = hyper_graph[1] @ hyper_graph[1].T


class focal_loss(nn.Module):
    """
    Multi-class focal loss.

    Parameters
    ----------
    alpha : float or list[float], default=0.25
        Class weighting. If list, length must equal `num_classes`.
    gamma : float, default=2
        Focusing parameter.
    num_classes : int, default=3
        Number of classes when `alpha` is scalar.
    size_average : bool, default=False
        If True, average the loss; else sum.
    device : str or torch.device, default='cpu'
        Device for tensors.

    References
    ----------
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=False, device='cpu'):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            num_classes = len(alpha)
            self.alpha = torch.Tensor(alpha).to(device)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        Compute focal loss.

        Parameters
        ----------
        preds : torch.FloatTensor, shape (N, C)
            Class logits (unnormalized).
        labels : torch.LongTensor, shape (N,)
            Ground-truth classes.

        Returns
        -------
        torch.Tensor
            Loss value (reduced by sum or mean).
        """
        preds = preds.view(-1, preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Predictor(nn.Module):
    """
    Simpler HE→omics predictor (cell-level loss).

    Parameters
    ----------
    in_dim : int
        Input HE dimension.
    hidden_dim : int
        Hidden width.
    out_dim : int
        Output gene dimension.
    num_layers : int
        #HGNN layers.
    dropout : float, default=0.1
        Dropout rate (internally set to 0).
    loss_fn : {'mse'}, default='mse'
        Reconstruction loss.
    activation : str, default='prelu'
        Activation key.

    Methods
    -------
    forward(H, he_rep, x) -> (loss, x_prime, enc)
    predict(H, he_rep) -> x_prime
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            dropout: float = 0.1,
            loss_fn='mse',
            activation='prelu',
    ):
        super(Predictor, self).__init__()

        dropout = 0
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim))
        self.mod = HGNN(
            in_dim=hidden_dim,
            num_hidden=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation)
        self.linear = nn.Linear(hidden_dim, out_dim)
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            print("not implement")

    def forward(self, H, he_rep, x):
        """
        Compute loss and prediction at cell level.

        Parameters
        ----------
        H : torch.sparse.FloatTensor
            Graph operator.
        he_rep : torch.FloatTensor, shape (N, F)
            HE features.
        x : torch.FloatTensor, shape (N, G)
            Cell-level targets.

        Returns
        -------
        loss : torch.Tensor
        x_prime : torch.FloatTensor, shape (N, G)
        enc : torch.FloatTensor, shape (N, H)
        """
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, H)
        x_prime = self.linear(F.leaky_relu(enc))
        loss = self.criterion(x_prime, x)
        return loss, x_prime, enc

    def predict(self, H, he_rep):
        """
        Predict per-cell gene panel.

        Parameters
        ----------
        H : torch.sparse.FloatTensor
        he_rep : torch.FloatTensor, shape (N, F)

        Returns
        -------
        torch.FloatTensor, shape (N, G)
        """
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, H)
        x_prime = self.linear(F.leaky_relu(enc))
        return x_prime


class Predictor_dgi(nn.Module):
    """
    DGI regularization module used by :class:`Model`.

    Parameters
    ----------
    in_dim : int
        Input HE dimension.
    hidden_dim : int
        Hidden width.
    out_dim : int
        Output dimension for :class:`DGI`.

    Methods
    -------
    forward(H, x) -> loss
        Compute DGI loss with a shuffled copy of `x`.
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int
    ):
        super(Predictor_dgi, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim)
        )
        self.dgi = DGI(hidden_dim, hidden_dim, out_dim, 'prelu')
        self.b_xent = nn.CosineEmbeddingLoss()

    def forward(self, H, x):
        """
        Compute DGI loss.

        Parameters
        ----------
        H : torch.sparse.FloatTensor
            Graph operator.
        x : torch.FloatTensor, shape (N, F_in)
            HE features.

        Returns
        -------
        torch.Tensor
            DGI loss value.
        """
        h = self.mlp(x)

        nb_nodes = x.shape[0]
        idx = torch.randperm(nb_nodes)
        shuf_fts = h[idx, :]

        lbl_1 = torch.ones(nb_nodes).to(x.device)
        lbl_2 = -torch.ones(nb_nodes).to(x.device)

        h1, h2, c = self.dgi(h, shuf_fts, H)

        loss = self.b_xent(h1, c, lbl_1) + self.b_xent(h2, c, lbl_2)

        return loss
