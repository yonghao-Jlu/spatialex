import os
import random
from functools import partial

from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim as optim
import torchvision.models as models
from tqdm import tqdm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
import matplotlib.pyplot as plt

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def structural_similarity_on_graph_data(x, y, adj, K1=0.01, K2=0.03, alpha=1, beta=1, gamma=1, sigma=1.5,
                                        use_sample_covariance=True):
    assert x.shape == y.shape

    K3 = K2 / np.sqrt(2)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if K3 < 0:
        raise ValueError("K3 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")

    R = x.max() - x.min()
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    C3 = (K3 * R) ** 2

    num_neighbor_list = adj.getnnz(axis=1)
    if use_sample_covariance:
        cov_norm = num_neighbor_list / (num_neighbor_list - 1 + 1e-6)  # 计算方差的norm
    else:
        cov_norm = 1 / (num_neighbor_list + 1e-6)
    cov_norm = cov_norm[:, np.newaxis]

    ux = adj @ x
    uy = adj @ y
    uxx = adj @ (x * x)
    uyy = adj @ (y * y)
    uxy = adj @ (x * y)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    A1 = 2 * ux * uy + C1
    A2 = 2 * np.sqrt(np.clip(vx * vy, 0, None)) + C2
    A3 = vxy + C3
    B1 = ux * ux + uy * uy + C1
    B2 = vx + vy + C2
    B3 = np.sqrt(np.clip(vx * vy, 0, None)) + C3
    S = (A1 / B1) ** alpha * (A2 / B2) ** beta * (A3 / B3) ** gamma
    return S.mean(0)


def Compute_metrics(x, x_prime, metric='cosine_similarity', reduce='mean', graph=None):
    metric = metric.lower()
    if metric == 'cosine_similarity':
        dot_product = np.sum(x_prime * x, axis=0)
        norm1 = np.linalg.norm(x_prime, axis=0)
        norm2 = np.linalg.norm(x, axis=0)
        metric = dot_product / (norm1 * norm2 + 1e-6)
    elif metric == 'rmse':
        mse = np.mean((x_prime - x) ** 2, axis=0)
        metric = np.sqrt(mse)
    elif metric == 'pcc':
        x_center = x - np.mean(x, axis=0)
        y_center = x_prime - np.mean(x_prime, axis=0)
        denominator = np.sqrt(np.sum(x_center * x_center, axis=0) * np.sum(y_center * y_center, axis=0))
        metric = np.sum(x_center * y_center, axis=0) / (denominator + 1e-6)
    elif metric == 'ssim':
        print("x shape is ", x.shape[0])
        if x.shape[0] < 200000:
            print("cell number is less than 200000")
            metric = structural_similarity_on_graph_data(x, x_prime, graph)
        else:
            print("cell number is greater than 200000")
            idx_list = list(range(0, x.shape[0]))
            random.shuffle(idx_list)
            batch_size = 200000
            batch_num = int(np.ceil(x.shape[0] / batch_size))
            batch_size = int(np.ceil(x.shape[0] / batch_num))
            ssim_sum = np.zeros(x.shape[-1])
            print('To avoid memory overflow, the data is splited into ' + str(batch_size) + ' cells batches.')
            for i in tqdm(range(batch_num)):
                tgt_cells = idx_list[i * batch_size: min((i + 1) * batch_size, x.shape[0])]
                tgt_cells_potential = graph[tgt_cells].tocoo().col
                tgt_cells = list(set(tgt_cells).union(set(tgt_cells_potential)))
                metric = structural_similarity_on_graph_data(x[tgt_cells], x_prime[tgt_cells],
                                                             graph[tgt_cells][:, tgt_cells])
                ssim_sum = ssim_sum + metric
            metric = ssim_sum / batch_num
    elif metric == 'cmd':
        x = x + np.random.normal(0, 1e-8, x.shape)
        x_prime = x_prime + np.random.normal(0, 1e-8, x_prime.shape)
        if x.shape[1] < 10000:
            corr_pred = np.corrcoef(x_prime, dtype=np.float32, rowvar=0)
            corr_true = np.corrcoef(x, dtype=np.float32, rowvar=0)
            x = np.trace(corr_pred.dot(corr_true))
            y = np.linalg.norm(corr_pred, 'fro') * np.linalg.norm(corr_true, 'fro')
            metric = 1 - x / (y + 1e-8)
        else:
            idx_list = list(range(0, x.shape[1]))
            random.shuffle(idx_list)
            batch_size = 10000
            batch_num = int(np.ceil(x.shape[1] / batch_size))
            batch_size = int(np.ceil(x.shape[1] / batch_num))
            cmd_list = []
            print('To avoid memory overflow, the data is splited into ' + str(batch_size) + ' cells batches.')
            for i in tqdm(range(batch_num)):
                tgt_cells = idx_list[i * batch_size: min((i + 1) * batch_size, x.shape[1])]
                corr_pred = np.corrcoef(x_prime[:, tgt_cells], dtype=np.float32, rowvar=0)
                corr_true = np.corrcoef(x[:, tgt_cells], dtype=np.float32, rowvar=0)
                numerator = np.trace(corr_pred.dot(corr_true))
                denominator = np.linalg.norm(corr_pred, 'fro') * np.linalg.norm(corr_true, 'fro')
                metric = 1 - numerator / (denominator + 1e-8)
                cmd_list.append(metric)
            metric = np.array(cmd_list)
    else:
        print('Not implemented!')
        return np.nan, np.nan

    if reduce == 'mean':
        metric_reduce = metric.mean()
    elif reduce == 'sum':
        metric_reduce = metric.sum()
    elif reduce == 'median':
        metric_reduce = np.median(metric)
    return metric, metric_reduce


def Compute_MoransI(adata, adj, store_key=None):
    n = adata.n_obs
    x_bar = np.mean(adata.X, axis=0)
    x = adata.X - x_bar

    numerator = np.sum((adj @ x) * x, axis=0)
    denominator = np.sum(x ** 2, axis=0)
    MoransI = (n / np.sum(adj)) * (numerator / (denominator + 1e-6))

    if isinstance(store_key, str):
        adata.var[store_key] = MoransI
    return MoransI


def Plot_imputed_gene(adata, gene_name, downsample=50000, save_path=None, dpi=300):
    fig, axs = plt.subplots(1, 2, figsize=(75, 20), gridspec_kw={'wspace': 0.01})
    if downsample > 0 and downsample < adata.n_obs:
        adata = adata[np.random.permutation(np.arange(adata.n_obs))[:downsample]]  # 下采样数量
    vmax = adata[:, gene_name].X.max()
    sc1 = axs[0].scatter(adata.obs['x_centroid'], adata.obs['y_centroid'], c=adata[:, gene_name].X, cmap='viridis',
                         vmin=0, vmax=vmax, s=30)
    axs[0].axis('equal')
    axs[0].axis('off')
    axs[0].set_title(gene_name + ' Ground Truth', fontsize=40)

    adata.X = adata.obsm['x_prime']
    sc2 = axs[1].scatter(adata.obs['x_centroid'], adata.obs['y_centroid'], c=adata[:, gene_name].X, cmap='viridis',
                         vmin=0, vmax=vmax, s=30)
    # plt.colorbar(sc2, label='Expression level')
    axs[1].axis('equal')
    axs[1].axis('off')
    axs[1].set_title(gene_name + ' Imputed', fontsize=40)

    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.5])  # 调整位置 [left, bottom, width, height]
    cbar = fig.colorbar(sc2, cax=cbar_ax)
    cbar.set_label('Expression level', fontsize=40)  # 设置颜色条标签和字体大小
    cbar.ax.tick_params(labelsize=30)

    if isinstance(save_path, str):
        plt.savefig(save_path + '/' + gene_name + '.png', dpi=dpi)
    plt.show()


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    if isinstance(model, list):
        parameters = []
        for each in model:
            parameters.append({'params': each.parameters()})
    else:
        parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def create_ImageEncoder(model_name='resnet50', pretrained=True, frozen=True):
    model_name = model_name.lower()
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 不要分类头
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(pretrained=True)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(pretrained=True)

    elif model_name == 'vit_l_32':
        model = models.vit_l_32(pretrained=True)

    elif model_name == 'vit_h_14':
        model = models.vit_h_14(pretrained=True)

    elif model_name == 'uni':

        local_dir = os.path.join(DEFAULT_DATA_DIR,
                                 "pyFile/Xenium_modality_impute/inputs/vit_large_patch16_224.dinov2.uni_mass100k/")  # "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
        # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

    elif model_name == 'conch':
        model = None

    else:
        assert False

    if frozen:
        model.eval()
    return model


def create_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'prelu':
        return nn.PReLU()
    else:
        return None


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device='cpu'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)


def Generate_pseudo_spot(adata, key=['x_centroid', 'y_centroid'], platform='visium', all_in=False):
    # print('Please make sure that the selected coordinates start from 0 and have units in um.')
    # print('It is recommended that the count expression be provided in adata.')
    x, y = adata.obs[key[0]], adata.obs[key[1]]
    spatial = np.vstack([x, y]).T

    platform = platform.lower()
    if platform == 'visium':  # 定义 x 和 y 的范围
        x_interval = 100
        y_interval = 100 * np.sqrt(3)
    else:
        print('Only support visium platform currently!')
        return

    x_start, x_end = 0, x.max()
    y_start, y_end = 0, y.max()
    spot_x1 = np.arange(x_start, x_end + x_interval, x_interval)
    spot_y1 = np.arange(y_start, y_end + y_interval, y_interval)
    spot_x1, spot_y1 = np.meshgrid(spot_x1, spot_y1)
    spot_x1 = spot_x1.reshape(-1)
    spot_y1 = spot_y1.reshape(-1)

    x_start, x_end = 50, x.max()
    y_start, y_end = y_interval / 2, y.max()
    spot_x2 = np.arange(x_start, x_end + x_interval, x_interval)
    spot_y2 = np.arange(y_start, y_end + y_interval, y_interval)
    spot_x2, spot_y2 = np.meshgrid(spot_x2, spot_y2)
    spot_x2 = spot_x2.reshape(-1)
    spot_y2 = spot_y2.reshape(-1)

    spot1 = np.vstack([spot_x1, spot_y1]).T
    spot2 = np.vstack([spot_x2, spot_y2]).T
    spot = np.vstack([spot1, spot2])  # 到这步是生成了矩形点阵

    tree = KDTree(spot)
    distances, indices = tree.query(spatial)
    if all_in:
        in_spot = np.array([True] * adata.n_obs)
    else:
        in_spot = distances < 55 / 2.0  # 这里没有包含细胞的spot自动就会被删除掉
    print(in_spot.sum(), ' cells are included in its nearest spot!')
    indices[~in_spot] = -1
    adata.obs['spot'] = indices  # 存储对应的spot的id

    count = pd.DataFrame(adata[in_spot].X)  # 聚合得到每个spot的基因表达
    count.columns = adata.var_names
    count['spot_id'] = indices[in_spot]
    spot_count = count.groupby('spot_id').sum()

    spot_count['old_id'] = spot_count.index
    spot_count['id'] = np.arange(spot_count.shape[0])
    spot_count.index = spot_count['id']
    map_dict = dict(zip(spot_count['old_id'], spot_count['id']))
    adata.obs['spot'] = adata.obs['spot'].map(map_dict)

    spot_coor = pd.DataFrame(spot[spot_count['old_id']])
    spot_coor.index = spot_count.index
    spot_coor.columns = ['x_centroid', 'y_centroid']

    return spot_coor, spot_count, adata


def Estimate_boundary(x, y, x_bin=250, deg=4):
    print('Estimating y boundary')
    bin_idx_list = x // x_bin
    max_list_y = []
    max_list_x = []
    for bin_idx in tqdm(np.arange(bin_idx_list.max())):
        selection = bin_idx_list == bin_idx
        if selection.sum() == 0:
            continue;
        y_bin, x_bin = y[selection], x[selection]
        ymax_idx = np.argmax(y_bin)
        max_list_y.append(y_bin[ymax_idx])
        max_list_x.append(x_bin[ymax_idx])

    coeffs = np.polyfit(max_list_x, max_list_y, deg=deg)
    poly = np.poly1d(coeffs)
    y_estimate = poly(x)
    return poly, y_estimate


def Estimate_gap(adata, array_key_prefix='array_', image_key='spatial'):
    num_array_col = np.unique(adata.obs[array_key_prefix + 'row']).shape[0]
    random_col_list = np.unique(adata.obs[array_key_prefix + 'row'])[np.random.randint(0, num_array_col, 10)]
    gap_list = []
    for random_col in random_col_list:
        adata_col = adata[adata.obs[array_key_prefix + 'row'] == random_col]
        adata_col = adata_col[np.argsort(adata_col.obs[array_key_prefix + 'col'].values)]
        gap = np.median(adata_col.obsm[image_key][:, 0][1:] - adata_col.obsm[image_key][:, 0][:-1])
        gap_list.append(gap)
    gap = np.median(gap)
    return gap
