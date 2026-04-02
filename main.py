"""
Final Method: Registration + Feature Similarity Dense Prompts for SAM-Med3D.

Canonical reference: FINAL_APPROACH_02-17-2026.md

Supports 3 prompt modes:
  - dense_only:      Dense prompt (reg+sim fusion) → SAM decoder (no sparse prompts)
  - points_only:     Sample pos/neg points from dense estimates → SAM decoder (no dense mask)
  - dense_and_points: Dense prompt + sampled pos/neg points → SAM decoder (both)

Final configuration (proposed_t007, late fusion, Gaussian-soft registration R):
  fusion_mode=proposed_t007, reg_soft_method=gaussian, reg_net=unigradicon/multigradicon,
  sim_temperature=0.1, entropy_temperature=0.07, late fusion in logit space.
  Default CLI matches scripts_final/run_final_1shot_biag6.sh: use_all_splits, 1-shot,
  pairs_per_query=5, reg_finetune_steps=0 (no IO), dense_only, device=cuda:0.
  Unnormalized fusion weights by default; pass --normalize_weights for that ablation.

Usage:
  python main.py \
      -qp ./data_autoprompt/MSD_Spleen -sp ./data_autoprompt/MSD_Spleen \
      -qmod ct -smod ct \
      --save_path ./results_final/intradataset/MSD_Spleen/1shot
  Place SAM-Med3D weights at checkpoints/sam_med3d_turbo.pth or set SAM_MED3D_CHECKPOINT.
  (Other defaults: num_support=1, pairs_per_query=5, reg_net=unigradicon,
   reg_finetune_steps=0, prompt_mode=dense_only, use_all_splits=True.)
"""

import os
import glob
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import itk
import unigradicon
import icon_registration
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _default_sam_checkpoint() -> str:
    """SAM-Med3D weights: env SAM_MED3D_CHECKPOINT, else <repo>/checkpoints/sam_med3d_turbo.pth."""
    env = os.environ.get("SAM_MED3D_CHECKPOINT")
    if env:
        return env
    return os.path.join(_REPO_ROOT, "checkpoints", "sam_med3d_turbo.pth")


def _resolve_checkpoint_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_REPO_ROOT, path))


from segment_anything.build_sam3D import sam_model_registry3D


# =============================================================================
# Visualization
# =============================================================================

def save_visualization(
    save_dir: str,
    sample_id: str,
    query_img: np.ndarray,   # (D,H,W)
    gt_mask: np.ndarray,     # (D,H,W) binary
    reg_binary: np.ndarray,  # (D,H,W) binary
    fused_prob: np.ndarray,  # (D,H,W) probability [0,1]
    pred_mask: np.ndarray,   # (D,H,W) binary
    dice_val: float,
    reg_dice: float,
):
    """Save a multi-panel visualization for 3 axial slices (25%, 50%, 75% of organ extent)."""
    os.makedirs(save_dir, exist_ok=True)

    # Find slices with GT content
    gt_slices = np.where(gt_mask.sum(axis=(1, 2)) > 0)[0]
    if len(gt_slices) == 0:
        return
    pcts = [0.25, 0.5, 0.75]
    slice_indices = [gt_slices[int(len(gt_slices) * p)] for p in pcts]

    fig, axes = plt.subplots(len(slice_indices), 5, figsize=(25, 5 * len(slice_indices)))
    if len(slice_indices) == 1:
        axes = axes[np.newaxis, :]

    titles = ["Query Image", "Ground Truth", "Reg Binary (prompt)", "Fused Prob (dense prompt)", "Prediction"]
    for row, sl in enumerate(slice_indices):
        axes[row, 0].imshow(query_img[sl], cmap="gray")
        axes[row, 0].set_title(f"{titles[0]} (z={sl})", fontsize=11)

        axes[row, 1].imshow(query_img[sl], cmap="gray")
        axes[row, 1].contour(gt_mask[sl], levels=[0.5], colors="lime", linewidths=1.5)
        axes[row, 1].set_title(titles[1], fontsize=11)

        axes[row, 2].imshow(query_img[sl], cmap="gray")
        axes[row, 2].contour(reg_binary[sl], levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 2].set_title(f"{titles[2]} (reg dice={reg_dice:.4f})", fontsize=11)

        axes[row, 3].imshow(fused_prob[sl], cmap="hot", vmin=0, vmax=1)
        axes[row, 3].contour(gt_mask[sl], levels=[0.5], colors="lime", linewidths=1)
        axes[row, 3].set_title(titles[3], fontsize=11)

        axes[row, 4].imshow(query_img[sl], cmap="gray")
        axes[row, 4].contour(gt_mask[sl], levels=[0.5], colors="lime", linewidths=1.2)
        pred_contour = pred_mask[sl] if pred_mask[sl].max() > 0 else np.zeros_like(pred_mask[sl])
        if pred_contour.max() > 0:
            axes[row, 4].contour(pred_contour, levels=[0.5], colors="red", linewidths=1.5)
        axes[row, 4].set_title(f"{titles[4]} (dice={dice_val:.4f})", fontsize=11)

        for c in range(5):
            axes[row, c].axis("off")

    fig.suptitle(f"Sample: {sample_id}  |  Final Dice: {dice_val:.4f}  |  Reg Dice: {reg_dice:.4f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{sample_id}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Seed
# =============================================================================

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# ITK utilities (numpy <-> ITK for registration on cropped ROIs)
# =============================================================================

def _numpy_to_itk_image(
    arr: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> itk.Image:
    spacing_itk = tuple(float(s) for s in spacing[::-1])
    arr_itk_order = np.transpose(arr.astype(np.float32), (2, 1, 0))
    itk_img = itk.image_from_array(arr_itk_order)
    itk_img.SetSpacing(spacing_itk)
    itk_img.SetOrigin(origin)
    direction_matrix = np.eye(3, dtype=np.float64)
    itk_img.SetDirection(itk.matrix_from_array(direction_matrix))
    return itk_img


def _itk_image_to_numpy(itk_img: itk.Image) -> np.ndarray:
    return itk.GetArrayFromImage(itk_img)


def _register_pair_and_warp_seg(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    moving_segmentation: np.ndarray,
    fixed_modality: str,
    moving_modality: str,
    reg_net: str,
    fixed_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    moving_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    finetune_steps: Optional[int] = None,
) -> np.ndarray:
    if reg_net == "unigradicon":
        net = unigradicon.get_unigradicon()
    elif reg_net == "multigradicon":
        net = unigradicon.get_multigradicon()
    else:
        raise ValueError(f"Unknown reg_net: {reg_net}")

    fixed_itk = _numpy_to_itk_image(fixed_image.astype(np.float32), fixed_spacing)
    moving_itk = _numpy_to_itk_image(moving_image.astype(np.float32), moving_spacing)
    fixed_itk = unigradicon.preprocess(fixed_itk, fixed_modality)
    moving_itk = unigradicon.preprocess(moving_itk, moving_modality)

    with torch.set_grad_enabled(finetune_steps is not None):
        phi_AB, _ = icon_registration.itk_wrapper.register_pair(
            net, moving_itk, fixed_itk, finetune_steps=finetune_steps
        )

    moving_seg_itk = _numpy_to_itk_image(moving_segmentation.astype(np.uint8), moving_spacing)
    moving_seg_itk_uc = itk.CastImageFilter[type(moving_seg_itk), itk.Image[itk.UC, 3]].New()(moving_seg_itk)
    interpolator = itk.NearestNeighborInterpolateImageFunction[itk.Image[itk.UC, 3], itk.D].New()
    interpolator.SetInputImage(moving_seg_itk_uc)

    reference_itk = _numpy_to_itk_image(
        np.zeros(fixed_image.shape, dtype=np.uint8), fixed_spacing
    )
    reference_itk_uc = itk.CastImageFilter[type(reference_itk), itk.Image[itk.UC, 3]].New()(reference_itk)

    warped_seg_itk = itk.resample_image_filter(
        moving_seg_itk_uc,
        transform=phi_AB,
        interpolator=interpolator,
        use_reference_image=True,
        reference_image=reference_itk_uc,
    )
    warped_seg = _itk_image_to_numpy(warped_seg_itk)
    assert warped_seg.shape == fixed_image.shape
    return warped_seg


def run_registration_on_cropped(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    moving_seg: np.ndarray,
    fixed_modality: str,
    moving_modality: str,
    reg_net: str = "unigradicon",
    finetune_steps: Optional[int] = None,
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
) -> np.ndarray:
    return _register_pair_and_warp_seg(
        fixed_image=fixed_img.astype(np.float32),
        moving_image=moving_img.astype(np.float32),
        moving_segmentation=moving_seg.astype(np.uint8),
        fixed_modality=fixed_modality,
        moving_modality=moving_modality,
        reg_net=reg_net,
        fixed_spacing=spacing,
        moving_spacing=spacing,
        finetune_steps=finetune_steps,
    )


# =============================================================================
# Registration aggregation
# =============================================================================

def mask_to_soft_prob(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    mask_binary = (mask > 0.5).astype(np.float32)
    if mask_binary.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    soft = gaussian_filter(mask_binary, sigma=sigma)
    if soft.max() > 0:
        soft = soft / soft.max()
    return soft.astype(np.float32)


def aggregate_registration_outputs(
    warped_segs: List[torch.Tensor],
    mode: str = "majority_vote",
    to_logit: bool = True,
    return_prob: bool = False,
    soft_method: Optional[str] = None,
    soft_sigma: float = 2.0,
    eps: float = 1e-6,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if not warped_segs:
        raise ValueError("warped_segs cannot be empty")
    stacked = []
    for w in warped_segs:
        t = w.float() if isinstance(w, torch.Tensor) else torch.from_numpy(np.asarray(w)).float()
        if t.dim() == 3:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 4:
            t = t.unsqueeze(0)
        stacked.append(t)
    masks = torch.stack(stacked, dim=0).squeeze(1)

    if mode == "majority_vote":
        votes = masks.sum(dim=0)
        prob = (votes >= (masks.shape[0] / 2)).float()
    elif mode == "mean":
        prob = masks.float().mean(dim=0)
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    if prob.dim() == 3:
        prob = prob.unsqueeze(0).unsqueeze(0)
    elif prob.dim() == 4:
        prob = prob.unsqueeze(0)

    if soft_method == "gaussian":
        prob_np = prob.squeeze().cpu().numpy()
        soft_prob = mask_to_soft_prob(prob_np, sigma=soft_sigma)
        prob = torch.from_numpy(soft_prob).float().unsqueeze(0).unsqueeze(0)

    if to_logit:
        prob = torch.clamp(prob, eps, 1 - eps)
        logit = torch.log(prob / (1 - prob))
        return logit, prob  # return both so caller can use prob for weights without redundant sigmoid
    return prob


# =============================================================================
# Feature similarity (cosine, the final method)
# =============================================================================

def get_feature_similarity_map(
    support_feats: torch.Tensor,
    support_masks: torch.Tensor,
    query_feat: torch.Tensor,
    temperature: float = 0.1,
    entropy_temperature: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cosine similarity + entropy uncertainty.
    Returns: (sim_maps (B,D,H,W), entropy_maps (B,D,H,W)).
    """
    B, C, D, H, W = support_feats.shape
    query_feat = query_feat.squeeze(0)
    query_feat = F.normalize(query_feat, dim=0, eps=1e-6)
    out_maps = []
    out_entropy = []
    eps = 1e-8

    for i in range(B):
        sf = support_feats[i]
        sm = support_masks[i].squeeze(0)
        sf = F.normalize(sf, dim=0, eps=1e-6)
        fg_idx = torch.nonzero(sm > 0, as_tuple=False)
        if fg_idx.numel() == 0:
            out_maps.append(torch.zeros(D, H, W, device=query_feat.device, dtype=query_feat.dtype))
            out_entropy.append(torch.zeros(D, H, W, device=query_feat.device, dtype=query_feat.dtype))
            continue
        local_vecs = sf[:, fg_idx[:, 0], fg_idx[:, 1], fg_idx[:, 2]]
        valid = local_vecs.abs().sum(dim=0) > 1e-6
        local_vecs = local_vecs[:, valid]
        if local_vecs.shape[1] == 0:
            out_maps.append(torch.zeros(D, H, W, device=query_feat.device, dtype=query_feat.dtype))
            out_entropy.append(torch.zeros(D, H, W, device=query_feat.device, dtype=query_feat.dtype))
            continue
        query_flat = query_feat.view(C, -1)
        sim_maps = torch.matmul(local_vecs.T, query_flat).view(-1, D, H, W)
        sim_maps = torch.nan_to_num(sim_maps, nan=0.0, posinf=1.0, neginf=-1.0)
        N = sim_maps.shape[0]

        weights = F.softmax(sim_maps / temperature, dim=0)
        sim_agg = (weights * sim_maps).sum(0)

        T_ent = entropy_temperature if entropy_temperature is not None else temperature
        weights_ent = F.softmax(sim_maps / T_ent, dim=0)
        entropy = -(weights_ent * (weights_ent.clamp(min=eps).log())).sum(dim=0)
        log_n = np.log(max(N, 2))
        entropy_norm = (entropy / log_n).clamp(0, 1)

        out_maps.append(sim_agg)
        out_entropy.append(entropy_norm)

    return torch.stack(out_maps), torch.stack(out_entropy)


def fuse_similarity_maps(sim_maps: torch.Tensor) -> torch.Tensor:
    """sim_maps: (N, D, H, W). Returns (D, H, W) in [0,1]."""
    sim_maps = (sim_maps + 1.0) * 0.5
    sim_maps = sim_maps.clamp(0, 1)
    return sim_maps.mean(dim=0).clamp(0, 1)


def similarity_to_logit(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.clamp(prob, eps, 1 - eps)
    return torch.log(prob / (1 - prob))


# =============================================================================
# Fusion block: proposed_t007 (Dynamic Gating with entropy T=0.07)
# =============================================================================

def fusion_block_proposed(
    reg_dense: torch.Tensor,
    sim_dense: torch.Tensor,
    entropy_dense: torch.Tensor,
    reg_prob: Optional[torch.Tensor] = None,
    sim_prob: Optional[torch.Tensor] = None,
    normalize_weights: bool = False,
) -> torch.Tensor:
    """
    Dynamic Gating fusion (proposed_t007). Final method uses normalize_weights=False (unnormalized).
    reg_dense, sim_dense: (1,1,D,H,W) logits (used in the weighted sum).
    entropy_dense: (1,1,D,H,W) in [0,1].
    reg_prob, sim_prob: (1,1,D,H,W) in [0,1]. If provided, used for R and S in the weight formula
        to avoid redundant logit->sigmoid; otherwise R = sigmoid(reg_dense), S = sigmoid(sim_dense).
    W_sim = 1 - entropy; W_reg = R * (S + eps).
    """
    eps = 1e-6
    unc = entropy_dense.to(reg_dense.device)
    R = (reg_prob if reg_prob is not None else torch.sigmoid(reg_dense)).clamp(eps, 1 - eps)
    S = (sim_prob if sim_prob is not None else torch.sigmoid(sim_dense)).clamp(eps, 1 - eps)
    weights_sim = 1 - unc
    weights_reg = R * (S + eps)
    if normalize_weights:
        total = weights_reg + weights_sim + eps
        weights_reg = weights_reg / total
        weights_sim = weights_sim / total
    return weights_reg * reg_dense + weights_sim * sim_dense


# =============================================================================
# Dataset (identical to original)
# =============================================================================

class SupportQueryDataset(Dataset):
    def __init__(
        self,
        query_path: str,
        support_path: str,
        query_mode: str = "Ts",
        support_mode: str = "Tr",
        query_modality: str = "ct",
        support_modality: str = "ct",
        num_support: int = 5,
        crop_size: Tuple[int, int, int] = (128, 128, 128),
        pairs_per_query: int = 10,
        organs: Optional[List[str]] = None,
        use_all_splits: bool = False,
    ):
        self.query_path = query_path
        self.support_path = support_path
        self.query_mode = query_mode
        self.support_mode = support_mode
        self.query_modality = query_modality
        self.support_modality = support_modality
        self.num_support = num_support
        self.crop_size = crop_size
        self.pairs_per_query = pairs_per_query
        self.use_all_splits = use_all_splits

        self.transform = tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name="label", target_shape=crop_size),
        ])

        organ_list = organs if organs is not None else self._common_organs()
        self.organs = [o for o in organ_list if o != "background"]
        self.pairs = self._build_pairs()

    def _common_organs(self) -> List[str]:
        if not os.path.isdir(self.query_path) or not os.path.isdir(self.support_path):
            return []
        q_organs = set(os.listdir(self.query_path)) - {"background", "__pycache__", ".DS_Store"}
        s_organs = set(os.listdir(self.support_path)) - {"background", "__pycache__", ".DS_Store"}
        return sorted(q_organs & s_organs)

    def _get_all_images_for_organ(self, base_path: str, organ: str) -> List[str]:
        all_files = []
        splits = ["Tr", "Ts", "Va"] if self.use_all_splits else []
        organ_dir = os.path.join(base_path, organ)
        if not os.path.isdir(organ_dir):
            return []
        for dataset_subdir in os.listdir(organ_dir):
            subdir_path = os.path.join(organ_dir, dataset_subdir)
            if not os.path.isdir(subdir_path):
                continue
            if self.use_all_splits:
                for split in splits:
                    split_glob = os.path.join(subdir_path, f"images{split}", "*.nii.gz")
                    all_files.extend(glob.glob(split_glob))
            else:
                return []
        return sorted(all_files)

    def _build_pairs(self) -> List[Dict]:
        pairs = []
        for organ in self.organs:
            if self.use_all_splits:
                query_files = self._get_all_images_for_organ(self.query_path, organ)
                support_files = self._get_all_images_for_organ(self.support_path, organ)
            else:
                q_glob = os.path.join(self.query_path, organ, "*", f"images{self.query_mode}", "*.nii.gz")
                s_glob = os.path.join(self.support_path, organ, "*", f"images{self.support_mode}", "*.nii.gz")
                query_files = sorted(glob.glob(q_glob))
                support_files = sorted(glob.glob(s_glob))
            if not query_files or not support_files:
                continue
            for qf in query_files:
                pool = [s for s in support_files if s != qf]
                if not pool:
                    continue
                n = min(self.num_support, len(pool))
                for _ in range(min(self.pairs_per_query, len(pool))):
                    supps = random.sample(pool, n)
                    pairs.append({"organ": organ, "query": qf, "supports": supps})
        random.shuffle(pairs)
        return pairs

    def _image_to_label_path(self, img_path: str) -> str:
        for m in ("Tr", "Va", "Ts"):
            if f"images{m}" in img_path:
                return img_path.replace(f"images{m}", f"labels{m}")
        return img_path.replace("/images", "/labels")

    def _load_itk_to_array(self, path: str) -> np.ndarray:
        img = itk.imread(path, pixel_type=itk.F)
        return np.asarray(itk.GetArrayFromImage(img), dtype=np.float32)

    def _crop_subject(self, img: np.ndarray, label: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        subj = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(img).float().unsqueeze(0)),
            label=tio.LabelMap(tensor=torch.from_numpy(label).float().unsqueeze(0)),
        )
        out = self.transform(subj)
        return out["image"].data, out["label"].data

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.pairs[idx]
        organ = entry["organ"]
        query_path = entry["query"]
        support_paths = entry["supports"]

        query_label_path = self._image_to_label_path(query_path)
        query_img = self._load_itk_to_array(query_path)
        query_gt = self._load_itk_to_array(query_label_path)
        q_img, q_gt = self._crop_subject(query_img, query_gt)

        support_imgs = []
        support_gts = []
        for sp in support_paths:
            sl = self._image_to_label_path(sp)
            si = self._load_itk_to_array(sp)
            sg = self._load_itk_to_array(sl)
            si_t, sg_t = self._crop_subject(si, sg)
            support_imgs.append(si_t)
            support_gts.append(sg_t)

        support_imgs = torch.stack(support_imgs, dim=0)
        support_gts = torch.stack(support_gts, dim=0)

        return {
            "query_image": q_img,
            "query_gt": q_gt,
            "support_images": support_imgs,
            "support_gts": support_gts,
            "organ": organ,
            "query_path": query_path,
        }


# =============================================================================
# SAM prediction functions (3 modes)
# =============================================================================

def sam_predict_from_dense(
    sam_model: torch.nn.Module,
    query_image: torch.Tensor,
    dense_prompt: torch.Tensor,
    crop_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Dense-only: pass dense mask logits, no sparse prompts."""
    low_res = F.interpolate(
        dense_prompt.float(),
        size=(crop_size // 4, crop_size // 4, crop_size // 4),
        mode="trilinear",
    )
    with torch.no_grad():
        img_emb = sam_model.image_encoder(query_image.to(device))
        _, dense_emb = sam_model.prompt_encoder(
            points=None, boxes=None, masks=low_res.to(device),
        )
        low_res_out, _ = sam_model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=None,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        out = F.interpolate(low_res_out, size=query_image.shape[-3:], mode="trilinear", align_corners=False)
        pred = (torch.sigmoid(out) > 0.5).float().squeeze().cpu()
    return pred


def sam_predict_from_points(
    sam_model: torch.nn.Module,
    query_image: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Points-only: pass sparse point prompts, no dense mask."""
    with torch.no_grad():
        img_emb = sam_model.image_encoder(query_image.to(device))
        sparse_emb, dense_emb = sam_model.prompt_encoder(
            points=(point_coords.to(device), point_labels.to(device)),
            boxes=None,
            masks=None,
        )
        low_res_out, _ = sam_model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        out = F.interpolate(low_res_out, size=query_image.shape[-3:], mode="trilinear", align_corners=False)
        pred = (torch.sigmoid(out) > 0.5).float().squeeze().cpu()
    return pred


def sam_predict_from_dense_and_points(
    sam_model: torch.nn.Module,
    query_image: torch.Tensor,
    dense_prompt: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    crop_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Dense + points: pass both dense mask logits and sparse point prompts."""
    low_res = F.interpolate(
        dense_prompt.float(),
        size=(crop_size // 4, crop_size // 4, crop_size // 4),
        mode="trilinear",
    )
    with torch.no_grad():
        img_emb = sam_model.image_encoder(query_image.to(device))
        sparse_emb, dense_emb = sam_model.prompt_encoder(
            points=(point_coords.to(device), point_labels.to(device)),
            boxes=None,
            masks=low_res.to(device),
        )
        low_res_out, _ = sam_model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        out = F.interpolate(low_res_out, size=query_image.shape[-3:], mode="trilinear", align_corners=False)
        pred = (torch.sigmoid(out) > 0.5).float().squeeze().cpu()
    return pred


# =============================================================================
# Point sampling from dense prompt estimates
#
# Key design (fixes the failures from 04_test_tp_points.py):
#   - Multiple points (N_pos + N_neg), not just 1+1 → less over-constraining
#   - Positive from (fused_prob > 0.5) AND reg_binary → dual-confirmed organ
#   - Negative from (fused_prob < neg_threshold) AND NOT reg_binary → dual-confirmed background
#   - FPS for spatial diversity → points spread across the 3D volume
# =============================================================================

def _farthest_point_sampling_3d(
    coords: np.ndarray,
    weights: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Farthest Point Sampling in 3D with weighted seed selection.

    Args:
        coords: (N, 3) array of (d, h, w) voxel coordinates
        weights: (N,) array of scores (higher = preferred for seed)
        num_points: number of points to select

    Returns:
        selected: (num_points, 3) array of selected coordinates
    """
    N = coords.shape[0]
    if N <= num_points:
        return coords.copy()

    selected_idx = [int(np.argmax(weights))]
    remaining = set(range(N)) - {selected_idx[0]}

    while len(selected_idx) < num_points and remaining:
        sel_coords = coords[selected_idx]
        rem_list = list(remaining)
        rem_coords = coords[rem_list]

        dists = cdist(rem_coords, sel_coords)
        min_dists = dists.min(axis=1)

        spatial_score = min_dists / (min_dists.max() + 1e-8)
        hybrid_score = spatial_score * 0.7 + weights[rem_list] * 0.3

        best = np.argmax(hybrid_score)
        best_global = rem_list[best]
        selected_idx.append(best_global)
        remaining.remove(best_global)

    return coords[selected_idx]


def _compute_threshold(fused_prob: np.ndarray, fixed_val: float,
                        percentile_val: float, mode: str, is_upper: bool) -> float:
    """Compute threshold adaptively or as a fixed value.

    Args:
        fused_prob: full 3-D probability volume.
        fixed_val: used when mode == "fixed".
        percentile_val: percentile (0-100) used when mode == "percentile".
        mode: "fixed" or "percentile".
        is_upper: True for positive (upper tail), False for negative (lower tail).
    """
    if mode == "fixed":
        return fixed_val
    return float(np.percentile(fused_prob, percentile_val))


def sample_positive_points_3d(
    fused_prob: np.ndarray,
    reg_binary: np.ndarray,
    num_points: int = 5,
    prob_threshold: float = 0.5,
    pos_percentile: float = 85.0,
    threshold_mode: str = "percentile",
    max_candidates: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample positive points from high-probability foreground confirmed by registration.

    threshold_mode == "fixed":
        Positive region = (fused_prob > prob_threshold) AND reg_binary.
    threshold_mode == "percentile":
        Positive region = (fused_prob > percentile(fused_prob, pos_percentile)) AND reg_binary.

    Falls back to threshold-only, then reg_binary-only.

    Returns: (N, 3) array of (d, h, w) voxel coordinates, N <= num_points.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t = _compute_threshold(fused_prob, prob_threshold, pos_percentile, threshold_mode, is_upper=True)

    pos_mask = (fused_prob > t) & (reg_binary > 0.5)
    if pos_mask.sum() < num_points:
        pos_mask = fused_prob > t
    if pos_mask.sum() < num_points:
        pos_mask = reg_binary > 0.5
    if pos_mask.sum() == 0:
        D, H, W = fused_prob.shape
        return np.array([[D // 2, H // 2, W // 2]], dtype=np.float32)

    coords = np.argwhere(pos_mask).astype(np.float32)
    probs = fused_prob[pos_mask]

    if len(coords) > max_candidates:
        idx = rng.choice(len(coords), max_candidates, replace=False)
        coords = coords[idx]
        probs = probs[idx]

    probs_norm = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
    return _farthest_point_sampling_3d(coords, probs_norm, num_points)


def sample_negative_points_3d(
    fused_prob: np.ndarray,
    reg_binary: np.ndarray,
    num_points: int = 5,
    neg_threshold: float = 0.3,
    neg_percentile: float = 15.0,
    threshold_mode: str = "percentile",
    max_candidates: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample negative points from low-probability background confirmed by registration.

    threshold_mode == "fixed":
        Negative region = (fused_prob < neg_threshold) AND NOT reg_binary.
    threshold_mode == "percentile":
        Negative region = (fused_prob < percentile(fused_prob, neg_percentile)) AND NOT reg_binary.

    Falls back to threshold-only, then NOT-reg_binary-only.

    Returns: (N, 3) array of (d, h, w) voxel coordinates, N <= num_points.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t = _compute_threshold(fused_prob, neg_threshold, neg_percentile, threshold_mode, is_upper=False)

    neg_mask = (fused_prob < t) & (reg_binary < 0.5)
    if neg_mask.sum() < num_points:
        neg_mask = fused_prob < t
    if neg_mask.sum() < num_points:
        neg_mask = reg_binary < 0.5
    if neg_mask.sum() == 0:
        return np.array([[0, 0, 0]], dtype=np.float32)

    coords = np.argwhere(neg_mask).astype(np.float32)
    probs = fused_prob[neg_mask]

    if len(coords) > max_candidates:
        idx = rng.choice(len(coords), max_candidates, replace=False)
        coords = coords[idx]
        probs = probs[idx]

    inv_probs = 1.0 - probs
    inv_norm = (inv_probs - inv_probs.min()) / (inv_probs.max() - inv_probs.min() + 1e-8)
    return _farthest_point_sampling_3d(coords, inv_norm, num_points)


def build_point_prompts(
    fused_prob: np.ndarray,
    reg_binary: np.ndarray,
    num_pos: int = 5,
    num_neg: int = 5,
    pos_threshold: float = 0.5,
    neg_threshold: float = 0.3,
    pos_percentile: float = 85.0,
    neg_percentile: float = 15.0,
    threshold_mode: str = "percentile",
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build point_coords (1, N, 3) and point_labels (1, N) for SAM-Med3D.

    Coordinates are in voxel indices (d, h, w) matching the 128^3 crop.
    Labels: 1 = positive, 0 = negative.

    threshold_mode:
        "fixed"      -- use pos_threshold / neg_threshold directly
        "percentile" -- derive thresholds from percentiles of fused_prob
    """
    rng = np.random.default_rng(42)

    pos_pts = sample_positive_points_3d(
        fused_prob, reg_binary, num_points=num_pos,
        prob_threshold=pos_threshold, pos_percentile=pos_percentile,
        threshold_mode=threshold_mode, rng=rng,
    )
    neg_pts = sample_negative_points_3d(
        fused_prob, reg_binary, num_points=num_neg,
        neg_threshold=neg_threshold, neg_percentile=neg_percentile,
        threshold_mode=threshold_mode, rng=rng,
    )

    all_coords = np.concatenate([pos_pts, neg_pts], axis=0)
    all_labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int64)

    point_coords = torch.from_numpy(all_coords).float().unsqueeze(0)
    point_labels = torch.from_numpy(all_labels).long().unsqueeze(0)
    return point_coords, point_labels


# =============================================================================
# Evaluation metric
# =============================================================================

def compute_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred_b = (pred > 0.5).astype(np.float32)
    gt_b = (gt > 0).astype(np.float32)
    inter = (pred_b * gt_b).sum()
    vol = pred_b.sum() + gt_b.sum()
    if vol == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2 * inter / (vol + eps))


def _axial_axis_from_shape(shape: Tuple[int, ...]) -> int:
    """Return axis index for axial (slice) dimension: the one that differs from the other two (H=W in DHW/HWD)."""
    a, b, c = shape[-3], shape[-2], shape[-1]
    if a != b and a != c:
        return -3
    if b != a and b != c:
        return -2
    if c != a and c != b:
        return -1
    return -3  # all equal, default first of (D,H,W)


def compute_dice_2d_axial(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Compute 2D Dice per slice along the axial (D) axis and return mean. Axial = the axis that doesn't have H=W (D in DHW/HWD)."""
    pred_b = (pred > 0.5).astype(np.float32)
    gt_b = (gt > 0).astype(np.float32)
    axis = _axial_axis_from_shape(pred_b.shape)
    n_slices = pred_b.shape[axis]
    dices = []
    for i in range(n_slices):
        sl = np.take(pred_b, i, axis=axis)
        sl_gt = np.take(gt_b, i, axis=axis)
        inter = (sl * sl_gt).sum()
        vol = sl.sum() + sl_gt.sum()
        if vol == 0:
            dices.append(1.0)
        else:
            dices.append(float(2 * inter / (vol + eps)))
    return float(np.mean(dices)) if dices else 0.0


# =============================================================================
# Main evaluation loop
# =============================================================================

def run_eval(args):
    set_seed(args.seed)
    device = args.device

    dataset = SupportQueryDataset(
        query_path=args.query_path,
        support_path=args.support_path,
        query_mode=args.query_mode,
        support_mode=args.support_mode,
        query_modality=args.query_modality,
        support_modality=args.support_modality,
        num_support=args.num_support,
        crop_size=tuple(args.crop_size),
        pairs_per_query=args.pairs_per_query,
        organs=args.organs.split() if args.organs else None,
        use_all_splits=args.use_all_splits,
    )

    if not dataset.organs:
        raise ValueError(f"No common organs between query_path={args.query_path} and support_path={args.support_path}.")
    if len(dataset) == 0:
        raise ValueError("Dataset built 0 pairs.")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    sam_model.load_state_dict(ckpt["model_state_dict"])
    sam_model.eval()

    norm = tio.ZNormalization(masking_method=lambda x: x > 0)
    dicelist: Dict[str, List[float]] = {o: [] for o in dataset.organs}
    dicelist_2d: Dict[str, List[float]] = {o: [] for o in dataset.organs}
    reg_dicelist: Dict[str, List[float]] = {o: [] for o in dataset.organs}

    prompt_mode = args.prompt_mode
    entropy_temp = 0.07  # proposed_t007

    for batch in tqdm(loader, desc=f"eval ({prompt_mode})"):
        q_img = batch["query_image"].squeeze(0)
        q_gt = batch["query_gt"].squeeze(0)
        s_imgs = batch["support_images"].squeeze(0)
        s_gts = batch["support_gts"].squeeze(0)
        organ = batch["organ"][0]

        q_img_norm = norm(q_img).unsqueeze(0).to(device)
        q_gt_np = q_gt.squeeze().numpy()
        D, H, W = q_img.shape[-3:]
        n_support = s_imgs.shape[0]

        # --- Step 1: Registration ---
        warped_segs = []
        for i in range(n_support):
            mov_img = s_imgs[i].squeeze().numpy()
            mov_seg = (s_gts[i].squeeze().numpy() > 0).astype(np.uint8)
            fix_img = q_img.squeeze().numpy()
            w = run_registration_on_cropped(
                fixed_img=fix_img,
                moving_img=mov_img,
                moving_seg=mov_seg,
                fixed_modality=args.query_modality,
                moving_modality=args.support_modality,
                reg_net=args.reg_net,
                finetune_steps=args.reg_finetune_steps if args.reg_finetune_steps else None,
                spacing=tuple(args.spacing),
            )
            warped_segs.append(torch.from_numpy(w).float().unsqueeze(0).unsqueeze(0))

        reg_dense, reg_prob = aggregate_registration_outputs(
            warped_segs, mode="majority_vote", to_logit=True, return_prob=True,
            soft_method="gaussian", soft_sigma=2.0,
        )
        reg_dense = reg_dense.to(device)
        reg_prob = reg_prob.to(device)

        reg_prob_binary = aggregate_registration_outputs(
            warped_segs, mode="majority_vote", to_logit=False, soft_method=None,
        )
        reg_binary_np = (reg_prob_binary > 0.5).float().squeeze().cpu().numpy()
        reg_dice = compute_dice(reg_binary_np, q_gt_np)
        reg_dicelist[organ].append(reg_dice)

        # --- Step 2: Feature similarity ---
        s_imgs_norm = torch.stack([norm(s_imgs[i]) for i in range(n_support)]).to(device)
        with torch.no_grad():
            q_feat = sam_model.image_encoder(q_img_norm)
            s_feat = sam_model.image_encoder(s_imgs_norm)

        feat_spatial = s_feat.shape[-3:]
        s_gts_down = F.interpolate(s_gts.float().to(device), size=feat_spatial, mode="nearest")

        sim_maps, entropy_maps = get_feature_similarity_map(
            s_feat, s_gts_down, q_feat,
            temperature=0.1,
            entropy_temperature=entropy_temp,
        )
        sim_prob = fuse_similarity_maps(sim_maps)
        entropy_fused = entropy_maps.mean(dim=0)

        sim_prob_up = F.interpolate(
            sim_prob.unsqueeze(0).unsqueeze(0), size=(D, H, W),
            mode="trilinear", align_corners=False,
        )
        entropy_up = F.interpolate(
            entropy_fused.unsqueeze(0).unsqueeze(0), size=(D, H, W),
            mode="trilinear", align_corners=False,
        )
        sim_dense = similarity_to_logit(sim_prob_up.squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0)
        sim_prob_up = sim_prob_up.to(device)

        # --- Step 3: Fusion (proposed_t007 = Dynamic Gating) ---
        dense_prompt = fusion_block_proposed(
            reg_dense, sim_dense, entropy_up,
            reg_prob=reg_prob, sim_prob=sim_prob_up,
            normalize_weights=args.normalize_weights,
        ).to(device)

        fused_prob_np = torch.sigmoid(dense_prompt).squeeze().cpu().numpy()

        # --- Step 4: Predict based on prompt_mode ---
        if prompt_mode == "dense_only":
            pred = sam_predict_from_dense(sam_model, q_img_norm, dense_prompt, args.crop_size[0], device)

        elif prompt_mode == "points_only":
            point_coords, point_labels = build_point_prompts(
                fused_prob_np, reg_binary_np,
                num_pos=args.num_pos_points, num_neg=args.num_neg_points,
                pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold,
                pos_percentile=args.pos_percentile, neg_percentile=args.neg_percentile,
                threshold_mode=args.threshold_mode,
                device=device,
            )
            pred = sam_predict_from_points(sam_model, q_img_norm, point_coords, point_labels, device)

        elif prompt_mode == "dense_and_points":
            point_coords, point_labels = build_point_prompts(
                fused_prob_np, reg_binary_np,
                num_pos=args.num_pos_points, num_neg=args.num_neg_points,
                pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold,
                pos_percentile=args.pos_percentile, neg_percentile=args.neg_percentile,
                threshold_mode=args.threshold_mode,
                device=device,
            )
            pred = sam_predict_from_dense_and_points(
                sam_model, q_img_norm, dense_prompt,
                point_coords, point_labels, args.crop_size[0], device,
            )
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

        pred_np = pred.numpy()
        dice = compute_dice(pred_np, q_gt_np)
        dice_2d = compute_dice_2d_axial(pred_np, q_gt_np)
        dicelist[organ].append(dice)
        dicelist_2d[organ].append(dice_2d)

        if args.save_vis:
            sample_idx = sum(len(v) for v in dicelist.values())
            save_visualization(
                save_dir=os.path.join(args.save_path, "vis"),
                sample_id=f"{organ}_{sample_idx:03d}",
                query_img=q_img.squeeze().numpy(),
                gt_mask=q_gt_np,
                reg_binary=reg_binary_np,
                fused_prob=fused_prob_np,
                pred_mask=pred.numpy(),
                dice_val=dice,
                reg_dice=reg_dice,
            )

        total_done = sum(len(v) for v in dicelist.values())
        if args.max_samples and total_done >= args.max_samples:
            break

    # --- Save results ---
    os.makedirs(args.save_path, exist_ok=True)

    for organ in dicelist:
        scores = dicelist[organ]
        mean_d = np.mean(scores) if scores else 0.0
        std_d = np.std(scores) if scores else 0.0
        print(f"{organ}: mean dice (final) = {mean_d:.4f} +/- {std_d:.4f} (n={len(scores)})")
        if dicelist_2d[organ]:
            mean_2d = np.mean(dicelist_2d[organ])
            print(f"       mean dice 2D (axial) = {mean_2d:.4f}")
        if reg_dicelist[organ]:
            r_mean = np.mean(reg_dicelist[organ])
            print(f"       mean dice (reg)  = {r_mean:.4f}")

    dicelist_float = {k: [float(x) for x in v] for k, v in dicelist.items()}
    with open(os.path.join(args.save_path, "dicelist_3d.json"), "w") as f:
        json.dump(dicelist_float, f, indent=2)
    dicelist_2d_float = {k: [float(x) for x in v] for k, v in dicelist_2d.items()}
    with open(os.path.join(args.save_path, "dicelist_2d.json"), "w") as f:
        json.dump(dicelist_2d_float, f, indent=2)

    reg_dicelist_float = {k: [float(x) for x in v] for k, v in reg_dicelist.items() if v}
    if reg_dicelist_float:
        with open(os.path.join(args.save_path, "reg_dicelist.json"), "w") as f:
            json.dump(reg_dicelist_float, f, indent=2)

    config = vars(args).copy()
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved to {args.save_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Final method: Reg + Sim dense prompts for SAM-Med3D (with point prompt support)")

    p.add_argument("-qp", "--query_path", type=str, required=True)
    p.add_argument("-sp", "--support_path", type=str, required=True)
    p.add_argument("-qmod", "--query_modality", type=str, choices=["ct", "mri"], default="ct")
    p.add_argument("-smod", "--support_modality", type=str, choices=["ct", "mri"], default="ct")
    p.add_argument("-qdm", "--query_mode", type=str, default="Ts", choices=["Tr", "Va", "Ts"])
    p.add_argument("-sdm", "--support_mode", type=str, default="Tr", choices=["Tr", "Va", "Ts"])
    split_grp = p.add_mutually_exclusive_group()
    split_grp.add_argument(
        "--use_all_splits",
        dest="use_all_splits",
        action="store_true",
        help="Pool train/val/test images (default; matches scripts_final 1-shot runs).",
    )
    split_grp.add_argument(
        "--no_use_all_splits",
        dest="use_all_splits",
        action="store_false",
        help="Use only -qdm/-sdm splits (e.g. query test vs support train).",
    )
    p.set_defaults(use_all_splits=True)
    p.add_argument("-ns", "--num_support", type=int, default=1)
    p.add_argument("--organs", type=str, default=None, help="Space-separated organ names")
    p.add_argument("--crop_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--pairs_per_query", type=int, default=5)

    p.add_argument("--reg_net", type=str, default="unigradicon", choices=["unigradicon", "multigradicon"])
    p.add_argument(
        "--reg_finetune_steps",
        type=int,
        default=0,
        help="Instance Optimization (ICON) finetune steps; 0 = none (default, paper setup).",
    )
    p.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5])

    p.add_argument("--prompt_mode", type=str, default="dense_only",
                    choices=["dense_only", "points_only", "dense_and_points"],
                    help="Prompt mode: dense_only (default), points_only, dense_and_points")

    p.add_argument("--num_pos_points", type=int, default=5, help="Number of positive points (default 5)")
    p.add_argument("--num_neg_points", type=int, default=5, help="Number of negative points (default 5)")
    p.add_argument("--threshold_mode", type=str, default="percentile", choices=["fixed", "percentile"],
                   help="How to compute point-sampling thresholds: 'fixed' uses pos/neg_threshold directly; "
                        "'percentile' derives thresholds from the fused probability distribution (adaptive)")
    p.add_argument("--pos_threshold", type=float, default=0.5, help="Fixed probability threshold for positive region (used when threshold_mode=fixed)")
    p.add_argument("--neg_threshold", type=float, default=0.3, help="Fixed probability threshold for negative region (used when threshold_mode=fixed)")
    p.add_argument("--pos_percentile", type=float, default=85.0,
                   help="Percentile of fused_prob above which voxels are positive candidates (used when threshold_mode=percentile). "
                        "E.g. 85 means top 15%% of the probability distribution.")
    p.add_argument("--neg_percentile", type=float, default=15.0,
                   help="Percentile of fused_prob below which voxels are negative candidates (used when threshold_mode=percentile). "
                        "E.g. 15 means bottom 15%% of the probability distribution.")

    p.add_argument("--normalize_weights", action="store_true", default=False,
                   help="Ablation: use normalized fusion (W_reg+W_sim=1). Default is unnormalized (final method).")
    p.add_argument("--save_vis", action="store_true", default=False,
                   help="Save visualization slices for each sample")

    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="SAM-Med3D state dict (.pth). If omitted: SAM_MED3D_CHECKPOINT env, else <repo>/checkpoints/sam_med3d_turbo.pth. "
        "Relative paths are resolved from the directory containing this script.",
    )
    p.add_argument("-mt", "--model_type", type=str, default="vit_b_ori")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--save_path", type=str, default="./results_final/run")
    p.add_argument("--max_samples", type=int, default=None, help="Limit evaluation to N samples (for quick testing)")
    args = p.parse_args()
    if args.checkpoint_path is None:
        args.checkpoint_path = _default_sam_checkpoint()
    else:
        args.checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    return args


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
