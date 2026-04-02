# Uncertainty-Aware Spatio-Semantic Contextual Prompts for Multimodal Medical Segmentation

### Soumitri Chattopadhyay, Basar Demir, Marc Niethammer
### University of California, San Diego  

---

## Abstract

The vast heterogeneity of medical imaging demands developing universal and modality-transferable segmentation models that can ideally work in low-data regimes. Although few-shot cross-domain, in-context learning, and promptable foundational models have emerged as promising data-efficient domain-agnostic solutions, they are all limited either in dimensionality (2D only), scalability (interactive prompting being too slow and iterative), or require re-training for each new task, limiting their general applicability. In this work, we address these limitations and propose a novel framework that harnesses the representational capabilities of foundational models to generate spatial and semantic contextual priors that holistically describe the target structure to be segmented. We also propose a confidence-weighted dynamic gating scheme to fuse these context maps into a single dense prompt, and re-purpose a frozen foundational segmentation model, SAM-Med3D, to predict segmentations using this fused representation instead of sparse points. Our framework is modality-agnostic, training-free, scalable, and enables rapid and robust universal segmentation. We validate our approach on two abdominal CT and MRI datasets under cross-modal and intra-modal settings, and show it outperforms existing state-of-the-art methods by significant margins. 

---

## Setup

```bash
conda create -n medsam python=3.10 -y
conda activate medsam
pip install -r requirements.txt
```

Use a **CUDA** build of PyTorch appropriate for your driver (the pinned `requirements.txt` reflects one working environment; you may need to reinstall `torch` from [pytorch.org](https://pytorch.org) if versions clash).

---

## Data layout

Place datasets under `./data_autoprompt/` (gitignored). Expected layout matches our experiments: per-organ folders with `imagesTr` / `labelsTr`, paired by filename. Follow [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) for data layout instructions. Symlinking an existing tree is fine.

---

## SAM-Med3D checkpoint

1. Download **`sam_med3d_turbo.pth`** from the [SAM-Med3D Hugging Face repo](https://huggingface.co/blueyo0/SAM-Med3D/tree/main) (same file as in the official readme).
2. Save it as **`checkpoints/sam_med3d_turbo.pth`** (or set **`SAM_MED3D_CHECKPOINT`** to an absolute path).

---

## Running

From the repository root (the directory that contains `main.py`):

```bash
python main.py \
  -qp ./data_autoprompt/BTCV -sp ./data_autoprompt/BTCV \
  -qmod ct -smod ct \
  --save_path ./results_final/intradataset/BTCV/1shot
```

Defaults are aligned with our paper runs (e.g. 1-shot, `pairs_per_query=5`, no ICON finetune steps, dense prompts only). Override as needed; see `python main.py --help`.

**Batch jobs (multi-GPU servers):** see `scripts/` — each script `cd`s to the repo root and calls `python main.py ...`. Edit paths or `CUDA_VISIBLE_DEVICES` to match your device.

---


## Acknowledgments

| Component | What we use | Where to get it |
|-----------|-------------|-----------------|
| **SAM-Med3D** | 3D encoder / decoder (`segment_anything/`), **SAM-Med3D-turbo** weights | Code and training recipe: [uni-medical/SAM-Med3D](https://github.com/uni-medical/SAM-Med3D). Checkpoint: [Hugging Face — `sam_med3d_turbo.pth`](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth) (see also their [readme checkpoint section](https://github.com/uni-medical/SAM-Med3D#-checkpoint)). |
| **UniGradICON** | `unigradicon` + `icon_registration` for **UniGradICON** (uni-modal) and **MultigradICON** (cross-modal) networks and pretrained weights | [uncbiag/uniGradICON](https://github.com/uncbiag/uniGradICON) and the [`unigradicon`](https://pypi.org/project/unigradicon/) / [`icon_registration`](https://pypi.org/project/icon-registration/) packages. Weights are loaded by those libraries on first use (see upstream docs). |

---

