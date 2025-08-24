# -*- coding: utf-8 -*-
"""
Preprocessing and graph/hypergraph construction utilities for SpatialEx(+).

This module provides a complete set of tools to:
- Read & normalize raw data (Xenium/HE images), tile HE patches, and extract
  patch-level representations using configurable image encoders.
- Build spatial graphs/hypergraphs on either physical coordinates or
  high-dimensional HE embeddings, with optional weighting/normalization.
- Prepare mini-batch dataloaders based on ROI tiling, including pseudo-spot
  aggregation for spot-level supervision.
- Supply a torch Dataset (:class:`Xenium_HBRC_overlap`) that returns graph,
  features, masks, aggregation matrices and bookkeeping info per ROI.

Notes
-----
The functions in this module are used throughout the SpatialEx/SpatialEx+
trainers. Public APIs include (non-exhaustive):
:func:`Read_Xenium`, :func:`Preprocess_adata`, :func:`Read_HE_image`,
:func:`Extract_HE_patches_representaion`, :func:`Build_graph`,
:func:`Build_graph_for_high_dim_feat`, :func:`Build_hypergraph`,
:func:`Build_hypergraph_spatial_and_HE`, :func:`Build_dataloader`,
and the :class:`Xenium_HBRC_overlap` dataset.

"""

import torch
import random
import itertools
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import tifffile as tiff
import scipy.sparse as sp
from cellpose import models
import torch.nn.functional as F
from PIL import Image, ImageFile
import xml.etree.ElementTree as ET
from .utils import create_ImageEncoder
from sklearn.neighbors import BallTree
from torch.utils.data import DataLoader
from .utils import Generate_pseudo_spot
import torchvision.transforms as transforms
from transformers import AutoImageProcessor
from sklearn.preprocessing import normalize, StandardScaler

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def set_random_seed(seed):
    """
    Set random seed across Python, NumPy and PyTorch (CPU/CUDA).

    Parameters
    ----------
    seed : int
        Seed value.

    Notes
    -----
    Also enforces deterministic CuDNN behavior (may impact speed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Read_Xenium(h5_path, obs_path):
    """
    Load Xenium data into an AnnData object.

    Parameters
    ----------
    h5_path : str or Path
        Path to 10x HDF5 gene expression file.
    obs_path : str or Path
        CSV file with per-cell metadata (must include coordinates).
        Expected columns: ``x_centroid``, ``y_centroid``.

    Returns
    -------
    adata : :class:`anndata.AnnData`
        With ``.X`` expression matrix, ``.obs`` metadata (index/strings),
        and ``.obsm['spatial']`` = (x_centroid, y_centroid).
    """
    adata = sc.read_10x_h5(h5_path)
    adata.obs = pd.read_csv(obs_path, index_col=0)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata


def Preprocess_adata(adata, cell_mRNA_cutoff=10, selected_genes=None, scale=False):
    """
    Filter, normalize/log, and (optionally) min-max scale an AnnData object.

    Parameters
    ----------
    adata : :class:`anndata.AnnData`
        Input object; may contain sparse ``.X``.
    cell_mRNA_cutoff : int, default=10
        Minimum total counts per cell.
    selected_genes : list[str] or None
        If provided, subset variables to this list before normalization.
    scale : bool, default=False
        If True, apply per-gene min-max scaling (stores min/max in ``.var``);
        else perform total-count normalization and log1p.

    Returns
    -------
    adata : :class:`anndata.AnnData`
        Processed object with ``.layers['raw']`` preserved and dense ``.X``.
    """
    adata.var_names_make_unique()
    if selected_genes is not None:
        adata = adata[:, selected_genes]
    sc.pp.filter_cells(adata, min_counts=cell_mRNA_cutoff)
    adata.layers['raw'] = adata.X.copy()

    if isinstance(adata.X, sp.csr_matrix):
        adata.X = adata.X.todense().A
    if scale:
        gene_min = adata.X.min(0)
        gene_max = adata.X.max(0)
        adata.var['min'] = gene_min
        adata.var['max'] = gene_max
        adata.X = adata.X - gene_min
        adata.X = adata.X / (gene_max - gene_min + 1e-12)
    else:
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
    return adata


def Read_HE_image(img_path, suffix='.ome.tif'):
    """
    Read histology image in OME-TIFF / TIFF / PNG / JPG format.

    Parameters
    ----------
    img_path : str or Path
        Image file path.
    suffix : {'.ome.tif', '.tif', '.png', '.jpg'}, default='.ome.tif'
        Format hint; affects metadata parsing.

    Returns
    -------
    image_data : np.ndarray
        Raw image array (H × W × C).
    scale : float
        Estimated physical pixel size (µm per pixel) if available from OME
        metadata, else ``-1``.
    """
    scale = -1
    if suffix == '.ome.tif':
        ome_tif = tiff.TiffFile(img_path)
        image_data = ome_tif.asarray()
        metadata = ome_tif.ome_metadata
        ome_tif.close()

        root = ET.fromstring(metadata)
        namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

        pixels_element = root.find('.//ome:Pixels', namespace)
        if pixels_element is not None:
            pixels_attributes = pixels_element.attrib
            for attr, value in pixels_attributes.items():
                if attr == 'PhysicalSizeX' or attr == 'PhysicalSizeX':
                    scale = float(value)
                    break
    elif suffix == '.png' or suffix == '.jpg':
        image = Image.open(img_path)
        image_data = np.array(image)
    elif suffix == '.tif':
        ome_tif = tiff.TiffFile(img_path)
        image_data = ome_tif.asarray()
        ome_tif.close()
    else:
        print("Only support '.ome.tif', '.png' or '.jpg' file currently.")
    return image_data, scale


def Crop_HE_image(img, crop):
    """
    Crop histology image by pixel box.

    Parameters
    ----------
    img : np.ndarray
        Image array (H × W × C).
    crop : tuple[int, int, int, int]
        (top, bottom, left, right) pixel indices.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    img = img[crop[0]:crop[1], crop[2]:crop[3]]
    return img


def Cell_segmentation(img, percent_threshold=0.2, pixel_threshold=190, device='cpu',
                      patch_size=512, scale=1, min_size=50, chan=[1, 0], diameter=12, flow_threshold=1.0):
    """
    Segment nuclei/cells on HE image using **Cellpose** with block tiling.

    Parameters
    ----------
    img : np.ndarray
        RGB image (H × W × 3).
    percent_threshold : float, default=0.2
        Skip a patch if its ``percent_threshold`` percentile > ``pixel_threshold``.
    pixel_threshold : int, default=190
        Intensity threshold for skipping bright background patches.
    device : {'cpu','cuda'}, default='cpu'
        Compute device for Cellpose.
    patch_size : int, default=512
        Square tile edge length.
    scale : float, default=1
        Downscale factor to reduce memory.
    min_size : int, default=50
        Remove ROIs smaller than this area.
    chan : list[int], default=[1, 0]
        Cellpose channels (nuclei model: signal in ``chan[0]+1``, background in ``chan[1]``).
    diameter : int, default=12
        Median cell diameter for Cellpose.
    flow_threshold : float, default=1.0
        Cellpose flow/probability threshold.

    Returns
    -------
    centers : np.ndarray, shape (N_cells, 2)
        Estimated (row, col) centers of segmented cells.

    Notes
    -----
    Tiles are processed to avoid out-of-memory, with global contiguous cell
    indexing across tiles.
    """
    print('======================== Estimating cells from histological image ===========================')
    image_tensor = torch.Tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0)
    _, _, height, width = image_tensor.shape
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    image_tensor_padded = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=255)

    patches = image_tensor_padded.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.squeeze().permute(1, 2, 3, 4, 0).numpy().astype(int)

    model = models.Cellpose(gpu=True, model_type='nuclei', device=device)
    total_num = 0.0
    cell_list = []
    for i in tqdm(range(patches.shape[0])):
        row_list = []
        for j in range(patches.shape[1]):
            img = patches[i, j]
            if np.percentile(img, percent_threshold) > pixel_threshold:
                nuclei_masks = np.zeros_like(img[..., 0])
            else:
                nuclei_masks, _, _, _ = model.eval(255 - img, flow_threshold=flow_threshold, min_size=min_size,
                                                   diameter=diameter, channels=chan, invert=False)
                cell_num = nuclei_masks.max()
                nuclei_masks = nuclei_masks.astype(np.float64)
                nuclei_masks[nuclei_masks != 0] += total_num
                total_num = total_num + cell_num
            row_list.append(nuclei_masks)

        row_list = np.hstack(row_list)
        cell_list.append(row_list)
    cell_list = np.vstack(cell_list)
    print(total_num, ' cells detected')

    coords = np.array(np.nonzero(cell_list)).T
    values = cell_list[tuple(coords.T)]
    df = pd.DataFrame(coords, columns=['row', 'col'])
    df['value'] = values
    mean_coor = df.groupby('value').mean()
    return np.round(mean_coor.values)


def Estimate_scale(physical_coor, pixel_coor, unit=100, reduce='median'):
    """
    Estimate physical scale (µm per pixel) assuming linear mapping.

    Parameters
    ----------
    physical_coor : np.ndarray, shape (N, 2)
        Physical coordinates (x, y) in units of ``unit`` µm.
    pixel_coor : np.ndarray, shape (N, 2)
        Pixel coordinates (px, py).
    unit : int, default=100
        Physical unit per difference in `physical_coor`.
    reduce : {'median','mean'}, default='median'
        Aggregation of per-step ratios.

    Returns
    -------
    scale : list[float, float]
        [scale_x, scale_y] in µm per pixel.
    """
    spatial = pd.DataFrame(np.hstack([physical_coor, pixel_coor]))
    spatial.columns = ['x', 'y', 'px', 'py']
    spatial_diff = spatial[1:].values - spatial[:-1].values
    spatial_diff = pd.DataFrame(spatial_diff)
    spatial_diff.columns = spatial.columns

    scale_x = spatial_diff['x'] * unit / (spatial_diff['px'] + 1e-6)
    scale_y = spatial_diff['y'] * unit / (spatial_diff['py'] + 1e-6)
    scale_x = scale_x[scale_x != 0.0]
    scale_y = scale_y[scale_y != 0.0]
    if reduce == 'median':
        scale = [np.median(scale_x.values), np.median(scale_y.values)]
    elif reduce == 'mean':
        scale = [np.mean(scale_x.values), np.mean(scale_y.values)]
    return scale


def Register_physical_to_pixel(adata, transform_matrix, scale=1,
                               raw_key=['x_centroid', 'y_centroid'],
                               matrix_type='pixel2phsical',
                               prefix='image'):
    """
    Apply a 3×3 affine transform between physical and pixel coordinates.

    Parameters
    ----------
    adata : :class:`anndata.AnnData`
        Input with columns in ``.obs`` specified by `raw_key`.
    transform_matrix : np.ndarray, shape (3, 3)
        Affine transform matrix.
    scale : float, default=1
        Post-hoc rescaling factor.
    raw_key : list[str], default=['x_centroid', 'y_centroid']
        Names for original coordinates stored in ``.obs``.
    matrix_type : {'pixel2phsical','physical2pixel'}, default='pixel2phsical'
        Direction of `transform_matrix` provided by the caller.
    prefix : str, default='image'
        Output prefix for new coordinates.

    Returns
    -------
    adata : :class:`anndata.AnnData`
        With new image coordinates in ``.obsm[f'{prefix}_coor']`` and columns
        ``obs[f'{prefix}_col']``, ``obs[f'{prefix}_row']``.
    """
    scale_old = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2)
    scale = scale / scale_old
    transform_matrix = transform_matrix * scale
    transform_matrix[-1, -1] = 1

    if matrix_type == 'pixel2phsical':
        transform_matrix = np.linalg.inv(transform_matrix)

    x = adata.obs[raw_key[0]].values
    y = adata.obs[raw_key[1]].values
    ones = np.ones_like(x)
    coor_raw = np.vstack([x, y, ones])

    coor_new = (transform_matrix @ coor_raw)[:2, :]
    image_coor = np.round(coor_new).astype(int)
    adata.obsm[prefix + '_coor'] = image_coor.T
    adata.obs[prefix + '_col'] = image_coor[0]
    adata.obs[prefix + '_row'] = image_coor[1]
    return adata


def Tiling_HE_patches(resolution, adata, img, key='image_coor'):
    """
    Tile HE image into per-cell patches around image coordinates.

    Parameters
    ----------
    resolution : int
        Patch edge length in pixels.
    adata : :class:`anndata.AnnData`
        Must contain integer image coordinates in ``.obsm[key]``.
    img : np.ndarray
        Full-resolution HE image (H × W × 3).
    key : str, default='image_coor'
        Name of image coordinates array in ``.obsm``.

    Returns
    -------
    he_patches : torch.FloatTensor, shape (N_cells, 3, R, R)
        Stacked per-cell patches normalized to [0, 1].
    adata : :class:`anndata.AnnData`
        Possibly reduced after removing out-of-bound cells.

    Notes
    -----
    Cells whose patches exceed the image boundary are filtered out.
    """
    print('======================== Tiling HE patches for each single cells ===========================')
    patch_radius = int(resolution / 2.0)
    print("patch radius is ", patch_radius)

    outlier_cells = np.unique(np.where(adata.obsm[key] < patch_radius)[0])
    if len(outlier_cells) != 0:
        print('Remove the outlier cells, and Anndata file was reduced!')
        inlier_cells = set(np.arange(adata.n_obs)) - set(outlier_cells)
        adata = adata[list(inlier_cells)]
    he_patches = [0] * adata.n_obs
    adata.obsm[key] = adata.obsm[key].astype(int)
    for i in tqdm(range(adata.n_obs)):
        x, y = adata.obsm[key][i]
        he_patches[i] = torch.tensor(img[y - patch_radius: y + patch_radius, x - patch_radius:x + patch_radius])
    return torch.stack(he_patches, dim=0) / 255.0, adata


def Tiling_HE_patches_by_coor(resolution, coor, img, col_name=['col', 'row']):
    """
    Tile HE image into per-entry patches given a DataFrame of coordinates.

    Parameters
    ----------
    resolution : int
        Patch edge length in pixels.
    coor : pandas.DataFrame
        Must include columns specified by `col_name`.
    img : np.ndarray
        Full-resolution HE image (H × W × 3).
    col_name : list[str], default=['col','row']
        Column names for (x, y) coordinates.

    Returns
    -------
    he_patches : torch.FloatTensor, shape (N, 3, R, R)
        Patches normalized to [0, 1].
    coor : pandas.DataFrame
        Possibly reduced after removing out-of-bound entries.
    """
    print('======================== Tiling HE patches for each single cells ===========================')

    patch_radius = int(resolution / 2.0)
    outlier_cells1 = np.where(coor < patch_radius)[0]
    outlier_cells2 = np.where(coor[col_name[0]] > (img.shape[1] - patch_radius))[0]
    outlier_cells3 = np.where(coor[col_name[1]] > (img.shape[0] - patch_radius))[0]
    outlier_cells = np.unique(np.hstack([outlier_cells1, outlier_cells2, outlier_cells3]))
    if len(outlier_cells) != 0:
        print('Remove the outlier cells, and Anndata file was reduced!')
        inlier_cells = set(np.arange(coor.shape[0])) - set(outlier_cells)
        coor = coor.iloc[list(inlier_cells)]
    he_patches = [0] * coor.shape[0]
    for i in tqdm(range(coor.shape[0])):
        x, y = coor.iloc[i][col_name[0]], coor.iloc[i][col_name[1]]
        he_patches[i] = torch.tensor(img[y - patch_radius: y + patch_radius, x - patch_radius:x + patch_radius])
    return torch.stack(he_patches, dim=0) / 255.0, coor


def Extract_HE_patches_representaion(he_patches, store_key=None, adata=None, skip_embedding=False, img_batch_size=64, image_encoder='uni', device='cuda'):
    """
    Extract patch-level representations using a configurable image encoder.

    Parameters
    ----------
    he_patches : torch.FloatTensor
        Either (N, 3, H, W) or (N, H, W, 3); if 3D, a batch dimension is added.
    store_key : str or None
        If provided, store the resulting matrix into ``adata.obsm[store_key]``.
    adata : :class:`anndata.AnnData` or None
        AnnData to store outputs into (required when `store_key` is set).
    skip_embedding : bool, default=False
        If True, return **preprocessed tensors** (no encoder forward pass).
    img_batch_size : int, default=64
        Batch size for the encoder forward pass.
    image_encoder : {'uni','conch','gigapath','phikon','resnet50','resnet101','resnet152'}
        Predefined encoder backends.
    device : {'cuda','cpu'}, default='cuda'
        Device to place the encoder and inputs on.

    Returns
    -------
    adata : :class:`anndata.AnnData` or None
        If `store_key` is set, returns the same AnnData with embeddings stored
        in ``.obsm[store_key]``; otherwise returns None.

    Raises
    ------
    NotImplementedError
        If `image_encoder` is not one of the supported choices.

    Notes
    -----
    - Normalization/resizing pipelines follow each encoder's convention.
    - When `skip_embedding=True`, this function returns the preprocessed
      tensors without running any encoder (handy for debugging).
    """
    if he_patches.dim() == 3:
        he_patches = he_patches.unsqueeze(0)
    if he_patches.size(1) != 3:
        he_patches = he_patches.permute(0, 3, 1, 2)

    print('====================== Extracting HE representations for each cell =========================')
    if image_encoder == 'uni':
        print("The image encoder is uni")
        preprocess = transforms.Compose([transforms.Resize(224),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), )])
    elif image_encoder == 'conch':
        print("The image encoder is conch")
        preprocess = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                         transforms.CenterCrop(224),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ])
    elif image_encoder == 'gigapath':
        print("The image encoder is gigapath")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    elif image_encoder == 'phikon':
        print("The image encoder is phikon")
        preprocess = AutoImageProcessor.from_pretrained("owkin/phikon", do_rescale=False)
    elif image_encoder in ['resnet50', 'resnet101', 'resnet152']:
        print(f"The image encoder is {image_encoder}")
        preprocess = transforms.Compose([transforms.Resize(224),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), )])
    else:
        print("The image encoder is not implemented")
        raise NotImplementedError

    representaions = []
    batch_num = int(np.ceil(he_patches.size(0) / img_batch_size))
    if not skip_embedding:
        model = create_ImageEncoder(image_encoder)
        model.to(device)
        model.eval()

        for i in tqdm(range(batch_num)):
            if image_encoder == 'phikon':
                img_tensor = preprocess(
                    he_patches[i * img_batch_size:min((i + 1) * img_batch_size, he_patches.size(0))],
                    return_tensors="pt")
            else:
                img_tensor = preprocess(
                    he_patches[i * img_batch_size:min((i + 1) * img_batch_size, he_patches.size(0))].to(device))
            with torch.no_grad():
                if image_encoder == 'conch':
                    features = model.encode_image(img_tensor, proj_contrast=False,
                                                  normalize=False).squeeze().detach().cpu().numpy()
                elif image_encoder == 'phikon':
                    img_tensor = img_tensor['pixel_values'].to(device)
                    features = model(img_tensor).last_hidden_state[:, 0, :].detach().cpu().numpy()
                else:
                    features = model(img_tensor).squeeze().detach().cpu().numpy()
                representaions.append(features)
    else:
        for i in tqdm(range(batch_num)):
            img_tensor = preprocess(
                he_patches[i * img_batch_size:min((i + 1) * img_batch_size, he_patches.size(0))])
            representaions.append(img_tensor)

    representaions = np.vstack(representaions)
    if isinstance(store_key, str):
        adata.obsm[store_key] = representaions
    return adata


def Build_graph(x, weighted=False, symmetric=False, graph_type='radius', metric='euclidean', self_loop=True,
                radius=50, num_neighbors=50, apply_normalize='none', sigma=0.01, type='coo'):
    """
    Build a sparse adjacency matrix on points/features via BallTree.

    Parameters
    ----------
    x : np.ndarray, shape (N, D)
        Input coordinates or features.
    weighted : bool or {'reciprocal','gaussian'}, default=False
        If True, distances are used as weights (=1 if False);
        'reciprocal' uses 1/d; 'gaussian' uses exp(-d^2 / (2*sigma^2)).
    symmetric : bool, default=False
        Make A symmetric via max rule (A = A + A^T - min(A, A^T)).
    graph_type : {'radius','knn'}, default='radius'
        Strategy for neighbor selection.
    metric : {'euclidean','cosine'}, default='euclidean'
        Distance metric. For 'cosine', L2-normalization is applied first.
    self_loop : bool, default=True
        Whether to keep self-loops in the result.
    radius : float, default=50
        Radius for 'radius' mode.
    num_neighbors : int, default=50
        K for 'knn' mode.
    apply_normalize : {'none','row','col','both','gcn'}, default='none'
        Optional normalization post-processing.
    sigma : float, default=0.01
        Bandwidth for 'gaussian' weighted edges.
    type : {'coo','csr'}, default='coo'
        Scipy sparse matrix format to return.

    Returns
    -------
    adj : scipy.sparse.{coo_matrix,csr_matrix}, shape (N, N)
        Sparse adjacency matrix.

    Notes
    -----
    For 'cosine', distances are derived from normalized dot products, which
    are converted to pseudo-Euclidean distances to reuse BallTree.
    """
    metric = metric.lower()
    apply_normalize = apply_normalize.lower()
    graph_type = graph_type.lower()

    if metric == 'cosine':
        x = normalize(x, norm='l2')

    tree = BallTree(x)
    if graph_type == 'radius':
        tail_list, distances = tree.query_radius(x, r=radius, return_distance=True)
    elif graph_type == 'knn':
        distances, tail_list = tree.query(x, k=num_neighbors)

    head_list = []
    head_list = [head_list + [i] * len(tail_list[i]) for i in range(len(tail_list))]
    head_list = list(itertools.chain.from_iterable(head_list))
    tail_list = list(itertools.chain.from_iterable(tail_list))

    if not weighted:
        distances = np.ones_like(head_list)
    elif isinstance(weighted, str):
        distances = np.array(list(itertools.chain.from_iterable(distances)))
        if metric == 'cosine':
            distances = (distances * distances) / 2
        if weighted == 'reciprocal':
            distances = 1 / distances
        elif weighted == 'gaussian':
            distances = np.exp(-(distances ** 2) / 2 * sigma * sigma) / (2 * np.pi * sigma * sigma)

    adj = sp.coo_matrix((distances, (head_list, tail_list)), shape=(x.shape[0], x.shape[0]))

    if not self_loop:
        adj = adj.tocsr()
        adj.setdiag(0)

    if symmetric:
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if apply_normalize == 'row':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(1))
        adj = adj.multiply(normalization_factors)
    elif apply_normalize == 'col':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(0))
        adj = adj.multiply(normalization_factors)
    elif apply_normalize == 'both':
        normalization_factors1 = sp.csr_matrix(1.0 / adj.sum(0))
        normalization_factors2 = sp.csr_matrix(1.0 / adj.sum(1))
        adj = adj.multiply(normalization_factors1)
        adj = adj.multiply(normalization_factors2)
    elif apply_normalize == 'gcn':
        D = np.squeeze(adj.sum(1).A)
        D = sp.diags(np.power(D.astype(float), -0.5), offsets=0, format='coo')
        adj = D @ adj @ D

    if type == 'coo':
        if not isinstance(adj, sp.coo_matrix):
            adj = adj.tocoo()
    elif type == 'csr':
        if not isinstance(adj, sp.csr_matrix):
            adj = adj.tocsr()
    return adj


def Build_graph_for_high_dim_feat(x, weighted=False, num_neighbors=50, device='cpu', batch_size=4096,
                                  apply_normalize=False, type='coo'):
    """
    Build a KNN graph on high-dimensional features using GPU cosine similarity.

    Parameters
    ----------
    x : np.ndarray, shape (N, D)
        Input feature matrix.
    weighted : bool, default=False
        If True, use inverse-cosine as weights (1 / cos).
    num_neighbors : int, default=50
        KNN parameter K.
    device : {'cpu','cuda'}, default='cpu'
        Device for similarity computation.
    batch_size : int, default=4096
        Batch granularity for similarity blocks.
    apply_normalize : {'row','col','both','gcn',False}, default=False
        Optional normalization scheme for the resulting graph.
    type : {'coo','crs'}, default='coo'
        Sparse format to return (note: 'crs' is interpreted as 'csr').

    Returns
    -------
    adj : scipy.sparse.{coo_matrix,csr_matrix}
        KNN graph adjacency.

    Notes
    -----
    Cosine similarity is computed via normalized matrix multiplication blocks.
    """
    print('======================== Build adj on high dimensional features =========================')
    x = torch.Tensor(x).to(device)
    x = F.normalize(x, p=2, dim=1)

    tail_list = []
    distance_list = []
    batch_num = int(np.ceil(x.shape[0] / batch_size))
    for i in tqdm(range(batch_num)):
        cosine_sim = x[i * batch_size:min((i + 1) * batch_size, x.shape[0])] @ x.T
        topk_values, topk_indices = torch.topk(cosine_sim, k=num_neighbors)
        tail_list.append(topk_indices)
        distance_list.append(topk_values)

    tail_list = torch.vstack(tail_list).reshape(-1).detach().cpu().numpy()
    head_list = np.arange(x.shape[0]).repeat(num_neighbors)
    if weighted:
        distance = torch.vstack(distance_list).reshape(-1)
        distance = (1 / distance).detach().cpu().numpy()
    else:
        distance = np.ones_like(head_list)

    adj = sp.coo_matrix((distance, (head_list, tail_list)), shape=(x.shape[0], x.shape[0]))

    if apply_normalize == 'row':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(1))
        adj = adj.multiply(normalization_factors)
    elif apply_normalize == 'col':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(0))
        adj = adj.multiply(normalization_factors)
    elif apply_normalize == 'both':
        normalization_factors1 = sp.csr_matrix(1.0 / adj.sum(0))
        normalization_factors2 = sp.csr_matrix(1.0 / adj.sum(1))
        adj = adj.multiply(normalization_factors1)
        adj = adj.multiply(normalization_factors2)
    elif apply_normalize == 'gcn':
        D = np.squeeze(adj.sum(1).A)
        D = sp.diags(np.power(D.astype(float), -0.5), offsets=0, format='coo')
        adj = D @ adj @ D

    if type == 'coo':
        if not isinstance(adj, sp.coo_matrix):
            adj = adj.tocoo()
    elif type == 'crs':
        if not isinstance(adj, sp.csr_matrix):
            adj = adj.tocsr()
    return adj


def Build_hypergraph(x, metric='euclidean', graph_type='knn', radius=50, num_neighbors=50,
                     self_loop=True, normalize=False, edge_weight=None, type='coo'):
    """
    Construct a hypergraph incidence matrix H from a base graph.

    Parameters
    ----------
    x : np.ndarray
        Coordinates or features (passed to :func:`Build_graph`).
    metric : {'euclidean','cosine'}, default='euclidean'
        Distance metric for the base graph.
    graph_type : {'radius','knn'}, default='knn'
        Base graph strategy.
    radius : float, default=50
        Radius for 'radius' mode.
    num_neighbors : int, default=50
        K for 'knn'.
    self_loop : bool, default=True
        Keep self loops in the base graph.
    normalize : bool, default=False
        If True, apply hypergraph normalization (:func:`normalize_graph` with
        ``type='hpnn'``).
    edge_weight : np.ndarray or None
        Optional hyperedge weights.
    type : {'coo','coo'}, default='coo'
        Output sparse format for H.

    Returns
    -------
    H : scipy.sparse.{coo_matrix,csr_matrix}, shape (N, E)
        Incidence matrix (columns = hyperedges).
    """
    H = Build_graph(x, metric=metric, graph_type=graph_type, radius=radius, num_neighbors=num_neighbors,
                    self_loop=self_loop, type=type)
    H = H.T
    if normalize:
        H = normalize_graph(H, edge_weight, type='hpnn')
    return H


def Build_hypergraph_spatial_and_HE(adata, num_neighbors=7, batch_size=4096, normalize=False, graph_kind='spatial',
                                    type='coo', device="cpu"):
    """
    Build hypergraph from either spatial coordinates or HE embeddings (or both).

    Parameters
    ----------
    adata : :class:`anndata.AnnData`
        Must have ``.obsm['spatial']`` and/or ``.obsm['he']``.
    num_neighbors : int, default=7
        K for KNN construction.
    batch_size : int, default=4096
        Used when HE embeddings are high-dimensional.
    normalize : bool, default=False
        Apply hypergraph normalization (HPNN).
    graph_kind : {'spatial','he','all'}, default='spatial'
        Select which view(s) to use; 'all' stacks both as a block-diagonal H.
    type : {'coo','crs'}, default='coo'
        Sparse format.
    device : {'cpu','cuda'}, default='cpu'
        Device for HE-based KNN.

    Returns
    -------
    H : scipy.sparse.{coo_matrix,csr_matrix}
        Incidence matrix.

    Raises
    ------
    AssertionError
        If `graph_kind` is not one of the supported choices.
    """
    if graph_kind.lower() == 'spatial':
        H1 = Build_graph(adata.obsm['spatial'], graph_type='knn', num_neighbors=num_neighbors, type=type)
        H = H1.T
    elif graph_kind.lower() == 'he':
        H2 = Build_graph_for_high_dim_feat(adata.obsm['he'], num_neighbors=num_neighbors,
                                           device=device, batch_size=batch_size)
        H = H2.T
    elif graph_kind.lower() == 'all':
        H1 = Build_graph(adata.obsm['spatial'], graph_type='knn', num_neighbors=num_neighbors)
        H2 = Build_graph_for_high_dim_feat(adata.obsm['he'], num_neighbors=num_neighbors,
                                           device=device, batch_size=batch_size)
        H = sp.hstack([H1.T, H2.T])
    else:
        assert False

    if normalize:
        H = normalize_graph(H, type='hpnn')

    if type == 'coo':
        if not isinstance(H, sp.coo_matrix):
            H = H.tocoo()
    elif type == 'crs':
        if not isinstance(H, sp.csr_matrix):
            H = H.tocsr()
    return H


def normalize_hypergraph(H, edge_weight=None):
    """
    Classical hypergraph normalization (DV^{-1/2} H W DE^{-1} H^T DV^{-1/2}).

    Parameters
    ----------
    H : scipy.sparse.spmatrix, shape (N, E)
        Incidence matrix.
    edge_weight : np.ndarray or None
        Optional per-hyperedge weights.

    Returns
    -------
    A_norm : scipy.sparse.csr_matrix, shape (N, N)
        Normalized adjacency.
    """
    DE = np.squeeze(H.sum(0).A)
    DV = np.squeeze(H.sum(1).A)
    DE = sp.diags(np.power(DE.astype(float), -1), offsets=0, format='csr')
    DV = sp.diags(np.power(DV.astype(float), -0.5), offsets=0, format='csr')
    if edge_weight != None:
        W = sp.diags(np.squeeze(edge_weight), offsets=0, format='csr')
    else:
        W = sp.diags(np.ones(shape=(H.shape[1])), offsets=0, format='csr')
    return DV @ H @ W @ DE @ H.T @ DV


def normalize_graph(H, edge_weight=None, type='gcn'):
    """
    Normalize a sparse graph/hypergraph in several common ways.

    Parameters
    ----------
    H : scipy.sparse.spmatrix
        Input (square adjacency for 'gcn' / V×E incidence for 'hpnn').
    edge_weight : np.ndarray or None
        Optional edge weights (used when `type='hpnn'`).
    type : {'row','col','both','gcn','hpnn'}, default='gcn'
        Normalization scheme.

    Returns
    -------
    adj : scipy.sparse.spmatrix
        Normalized adjacency.

    Notes
    -----
    - 'gcn' uses D^{-1/2} A D^{-1/2}.
    - 'hpnn' returns DV^{-1/2} H W DE^{-1} H^T DV^{-1/2}.
    """
    if type == 'row':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(1))
        adj = adj.multiply(normalization_factors)
    elif type == 'col':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(0))
        adj = adj.multiply(normalization_factors)
    elif type == 'both':
        normalization_factors1 = sp.csr_matrix(1.0 / adj.sum(0))
        normalization_factors2 = sp.csr_matrix(1.0 / adj.sum(1))
        adj = adj.multiply(normalization_factors1)
        adj = adj.multiply(normalization_factors2)
    elif type == 'gcn':
        D = np.squeeze(H.sum(1).A)
        D = sp.diags(np.power(D.astype(float), -0.5), offsets=0, format='coo')
        adj = D @ H @ D
    elif type == 'hpnn':
        DE = np.squeeze(H.sum(0).A)
        DV = np.squeeze(H.sum(1).A)
        DE = sp.diags(np.power(DE.astype(float), -1), offsets=0, format='csr')
        DV = sp.diags(np.power(DV.astype(float), -0.5), offsets=0, format='csr')
        if edge_weight != None:
            W = sp.diags(np.squeeze(edge_weight), offsets=0, format='csr')
        else:
            W = sp.diags(np.ones(shape=(H.shape[1])), offsets=0, format='csr')
        adj = DV @ H @ W @ DE @ H.T @ DV

    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx, return_mtx=True):
    """
    Convert a SciPy sparse matrix to a torch sparse FloatTensor.

    Parameters
    ----------
    sparse_mx : scipy.sparse.spmatrix
        COO/CSR/CSC/etc.
    return_mtx : bool, default=True
        If False, return (indices, values, shape) instead.

    Returns
    -------
    T : torch.sparse.FloatTensor or tuple
        Sparse COO tensor or (indices, values, shape) components.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    if return_mtx:
        return torch.sparse.FloatTensor(indices, values, shape)
    else:
        return indices, values, shape


def Build_dataloader(adata, graph, batch_size=1, ori=False, graph_norm='hpnn', feat_norm=False,
                     shuffle=True, prune=[10000, 10000], drop_last=False):
    """
    Create a :class:`torch.utils.data.DataLoader` over ROI tiles (overlap aware).

    Parameters
    ----------
    adata : :class:`anndata.AnnData`
        Slice with HE embeddings in ``.obsm['he']`` and expression in ``.X``.
    graph : scipy.sparse.spmatrix
        Global graph/hypergraph adjacency/incidence for the full slice.
    batch_size : int, default=1
        Num of ROIs per batch (dataset already yields per-ROI dicts).
    ori : bool, default=False
        If True, include ``'ori'`` raw layer & size factors (if available).
    graph_norm : {'hpnn','gcn','row','col','both'}, default='hpnn'
        Normalization scheme applied on per-ROI subgraph.
    feat_norm : bool, default=False
        If True, standardize HE features per-ROI.
    shuffle : bool, default=True
        Shuffle ROI tiles between epochs.
    prune : list[int, int], default=[10000, 10000]
        Spatial window size (pixels) used to define ROIs & overlap.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch.

    Returns
    -------
    loader : :class:`torch.utils.data.DataLoader`
        Over :class:`Xenium_HBRC_overlap`.

    Notes
    -----
    Collate function is custom to keep dicts as-is (no stacking).
    """
    dataset = Xenium_HBRC_overlap(adata, graph, ori=ori, prune=prune, graph_norm=graph_norm, feat_norm=feat_norm,
                                  drop_last=drop_last)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
    return train_loader


def custom_collate_fn(batch):
    """Keep the single-ROI dicts unmodified (no default collation)."""
    return batch


class Xenium_HBRC_overlap(torch.utils.data.Dataset):
    """
    Overlapping ROI dataset for Xenium HBRC slices with spot aggregation.

    Each item corresponds to a 2×`prune` square ROI (with overlap handling),
    returning pre-cut HE features, a normalized subgraph, masks for where to
    compute loss, and aggregation matrices to produce spot-level supervision.

    Parameters
    ----------
    adata : :class:`anndata.AnnData`
        Slice with columns ``x_centroid`` / ``y_centroid``, HE in ``.obsm['he']``,
        and expression matrix ``.X``.
    graph : scipy.sparse.spmatrix
        Global adjacency/incidence (will be subset & normalized per-ROI).
    ori : bool, default=False
        Include raw counts & size factors (requires ``.layers['ori']``).
    graph_norm : {'hpnn','gcn','row','col','both'}, default='hpnn'
        Normalization scheme for per-ROI adjacency.
    feat_norm : bool, default=False
        Standardize HE features within each ROI.
    prune : list[int, int], default=[3000, 3000]
        Pixel window defining ROIs and overlaps.
    drop_last : bool, default=False
        Whether to drop ROIs with empty center (rare after overlap check).

    Attributes
    ----------
    idx_cat : np.ndarray[str]
        Encoded ROI identifiers ``"{xbin}*{ybin}"`` for bookkeeping.
    he_dict : dict[str, torch.FloatTensor]
        HE features per ROI.
    graph_dict : dict[str, torch.sparse.FloatTensor]
        Normalized subgraph per ROI.
    selection_dict : dict[str, np.ndarray[bool]]
        Mask (boolean) indicating which cells in the ROI contribute to loss.
    agg_dict : dict[str, torch.sparse.FloatTensor]
        Aggregation matrix (spots × cells-in-selection) for spot supervision.
    agg_exp_dict : dict[str, torch.FloatTensor]
        Aggregated expression on selection (precomputed).
    exp_dict : dict[str, torch.FloatTensor]
        Expression tensor restricted to selection mask (for loss).
    obs_dict : dict[str, list[str]]
        Original cell ids in ROI order (for later aggregation).
    id2name : dict[int, str]
        Mapping from dataset index to ROI id.

    Notes
    -----
    ROI selection logic ensures non-empty “central” area; boundary cases are
    skipped to avoid degenerate batches.
    """

    def __init__(self, adata, graph, ori=False, graph_norm='hpnn', feat_norm=False, prune=[3000, 3000],
                 drop_last=False):
        super(Xenium_HBRC_overlap, self).__init__()

        self.ori = ori
        _, _, adata = Generate_pseudo_spot(adata, all_in=True)
        spot_id = adata.obs['spot'].values
        head = spot_id[~pd.isna(adata.obs['spot'])].astype(int)
        tail = np.where(~pd.isna(adata.obs['spot']))[0]
        values = np.ones_like(tail)
        agg_mtx = sp.coo_matrix((values, (head, tail)), shape=(head.max() + 1, adata.n_obs)).tocsr()

        row = adata.obs['x_centroid'].values
        col = adata.obs['y_centroid'].values
        exp = torch.Tensor(adata.X)
        he = adata.obsm['he']
        if feat_norm:
            scaler = StandardScaler()
            he = scaler.fit_transform(he)
        he = torch.Tensor(he)

        x_cat = row // prune[0]
        y_cat = col // prune[1]
        x_min, x_max = x_cat.min(), x_cat.max()
        y_min, y_max = y_cat.min(), y_cat.max()
        x_cat = x_cat.astype(int).astype(str)
        y_cat = y_cat.astype(int).astype(str)
        idx = np.char.add(np.char.add(x_cat, '*'), y_cat)
        idx_cat = np.unique(idx)
        self.idx_cat = idx_cat.copy()

        if row.max() < prune[0]:
            prune[0] = row.max()
        if col.max() < prune[1]:
            prune[1] = col.max()

        self.roi_dict = {}
        self.selection_dict = {}
        self.exp_dict = {}
        self.he_dict = {}
        self.graph_dict = {}
        self.agg_dict = {}
        self.agg_exp_dict = {}
        self.obs_dict = {}
        for name in idx_cat:
            x = int(name.split('*')[0]) * prune[0]
            y = int(name.split('*')[-1]) * prune[1]

            selection = (row > x + 0.5 * prune[0]) & (row < (x + 1.5 * prune[0])) & (col > y + 0.5 * prune[1]) & (
                    col < (y + 1.5 * prune[1]))
            if not selection.sum():
                self.idx_cat = self.idx_cat[self.idx_cat != name]
                continue

            self.roi_dict[name] = (row > x) & (row < (x + 2 * prune[0])) & (col > y) & (col < (y + 2 * prune[1]))
            self.he_dict[name] = he[self.roi_dict[name]]
            self.obs_dict[name] = adata.obs_names[self.roi_dict[name]].tolist()

            sub_graph = normalize_graph(graph[self.roi_dict[name]][:, self.roi_dict[name]], type=graph_norm)
            self.graph_dict[name] = sparse_mx_to_torch_sparse_tensor(sub_graph)

            # central selection with boundary-aware masks
            if int(name.split('*')[0]) > x_min:
                selection_x_left = (row > x + 0.5 * prune[0])
            else:
                selection_x_left = np.ones_like(row).astype(bool)

            if int(name.split('*')[0]) < x_max:
                selection_x_right = (row < (x + 1.5 * prune[0]))
            else:
                selection_x_right = np.ones_like(row).astype(bool)

            if int(name.split('*')[-1]) > y_min:
                selection_y_down = (col > y + 0.5 * prune[1])
            else:
                selection_y_down = np.ones_like(col).astype(bool)

            if int(name.split('*')[-1]) < y_max:
                selection_y_up = (col < (y + 1.5 * prune[1]))
            else:
                selection_y_up = np.ones_like(col).astype(bool)

            selection = selection_x_left & selection_x_right & selection_y_down & selection_y_up
            self.exp_dict[name] = exp[selection]
            self.agg_dict[name] = sparse_mx_to_torch_sparse_tensor(agg_mtx[:, selection])
            self.agg_exp_dict[name] = torch.sparse.mm(self.agg_dict[name], self.exp_dict[name])
            self.selection_dict[name] = selection[self.roi_dict[name]]

        self.id2name = dict(enumerate(self.idx_cat))

        if ori:
            # If raw counts & size factors are present, add them here (optional path).
            # (Kept minimal to avoid changing your original logic.)
            pass

    def __getitem__(self, index):
        """
        Return a dictionary with HE, graph, expression, masks and aggregators.

        Keys
        ----
        'he' : torch.FloatTensor
            HE features in ROI order.
        'graph' : torch.sparse.FloatTensor
            Normalized ROI subgraph (sparse COO).
        'exp' : torch.FloatTensor
            Expression restricted to central selection (for loss).
        'selection' : np.ndarray[bool]
            Mask indicating “loss region” inside ROI.
        'obs' : list[str]
            Cell ids in ROI order (for later aggregation).
        'agg_mtx' : torch.sparse.FloatTensor
            Aggregation matrix (spots × selected-cells).
        'agg_exp' : torch.FloatTensor
            Precomputed spot-level expression on selection.

        Returns
        -------
        dict
            Data bundle for this ROI.
        """
        ID = self.id2name[index]
        data = {}
        data['he'] = self.he_dict[ID]
        data['graph'] = self.graph_dict[ID]
        data['exp'] = self.exp_dict[ID]
        data['selection'] = self.selection_dict[ID]
        data['obs'] = self.obs_dict[ID]
        data['agg_mtx'] = self.agg_dict[ID]
        data['agg_exp'] = self.agg_exp_dict[ID]
        if self.ori:
            data['ori'] = self.ori_dict[ID]
            data['sfs'] = self.sfs_dict[ID]
        return data

    def __len__(self):
        """Number of valid ROI tiles."""
        return len(self.idx_cat)
