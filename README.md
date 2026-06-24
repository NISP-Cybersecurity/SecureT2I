# SecureT2I: No More Unauthorized Manipulation on AI Generated Images from Prompts

**ESORICS 2025** | [Paper](https://github.com/NISP-Cybersecurity/SecureT2I/) | Queen's University

> Xiaodong Wu, Xiangman Li, Qi Li, Jianbing Ni, Rongxing Lu  
> *Queen's University, Kingston, Ontario, Canada*

---

## Overview

Text-guided image manipulation with diffusion models enables flexible, prompt-driven editing but raises serious ethical and copyright concerns when applied to protected images without permission. **SecureT2I** is the first framework to address this problem directly inside the model: rather than placing an easily bypassed external detector, it embeds selective editing restrictions into the model weights via lightweight fine-tuning.

The core idea is to partition images into two sets:
- **Permit set** — images the owner allows to be edited. The model is trained to produce high-quality manipulations as usual.
- **Forbid set** — images the owner wants to protect. The model is trained to produce semantically vague, blurred outputs that suppress meaningful editing.

SecureT2I is **model-agnostic** and works with any diffusion-based manipulation pipeline without architectural changes.

---

## Method

### Problem Formulation

Given a pretrained diffusion manipulation model $f_\theta(\mathbf{x}, \mathbf{p})$, SecureT2I fine-tunes it to satisfy two objectives simultaneously:

- For the **forbid set** $\mathcal{F}$: push outputs toward a vague (low-frequency) version of the original, suppressing recognizable edits.
- For the **permit set** $\mathcal{P}$: preserve editing quality by aligning outputs with the target produced by the original model.

The joint objective is:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{forbid}} \sum_{\mathbf{x}_f \in \mathcal{F}} \mathcal{L}_{\text{forbid}}(f_\theta(\mathbf{x}_f, \mathbf{p}),\, \mathbf{x}^{\text{vague}}) + \lambda_{\text{permit}} \sum_{\mathbf{x}_r \in \mathcal{P}} \mathcal{L}_{\text{permit}}(f_\theta(\mathbf{x}_r, \mathbf{p}),\, \mathbf{x}')$$

### Vague Target Design

The forbid-set target $\mathbf{x}^{\text{vague}}$ is obtained by **resize-based degradation**: compressing the image to 16×16 and upsampling back to the original resolution. This:
1. Removes high-frequency semantic information that manipulation models rely on.
2. Produces bounded, Lipschitz-continuous gradients that prevent training instability.

Compared with Gaussian blur, box filter, motion blur, and other resize factors (8×8, 32×32), **16×16 resize** achieves the best trade-off between suppressing unauthorized edits (WAN\*) and preserving permit-set quality (WAN).

### Training Algorithm

Fine-tuning alternates between forbid-set and permit-set batches for 15 iterations using the Adam optimizer ($\text{lr} = 8 \times 10^{-6}$, $\lambda_{\text{forbid}} = \lambda_{\text{permit}} = 0.5$). Both loss functions use pixel-wise L1 distance.

---

## Supported Models

SecureT2I has been evaluated with three state-of-the-art diffusion-based manipulation methods:

| Model | Directory | Description |
|---|---|---|
| [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP) | `DiffusionCLIP/` | CLIP-guided fine-tuning of DDIM |
| [Asyrp](https://github.com/kwonminki/Asyrp) | `asyrp/` | Asymmetric reverse process via semantic latent space |
| [EffDiff](https://github.com/yandex-research/efficient-straight-through) | `Eff-diff/` | Efficient diffusion-based editing |

---

## Datasets

| Dataset | Images | Description |
|---|---|---|
| **CelebA-HQ** | 30,000 | High-resolution celebrity faces with rich facial attributes |
| **LSUN-Bedroom** | ~3M | Indoor bedroom scenes with diverse layouts and styles |
| **LSUN-Church** | ~126K | Church exteriors across architectural styles and conditions |

Experiments use 100 images per dataset, evaluated across 5 manipulation attributes (e.g., "beards", "snow", "golden").

---

## Installation

```bash
git clone https://github.com/NISP-Cybersecurity/SecureT2I.git
cd SecureT2I
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch (with CUDA recommended), and the packages listed in `requirements.txt`.

Download pretrained model weights (place in `pretrained/`):
- `celeba_hq.ckpt` — CelebA-HQ
- `bedroom.ckpt` — LSUN-Bedroom
- `church.ckpt` — LSUN-Church

---

## Usage

### DiffusionCLIP

```bash
cd DiffusionCLIP

# Fine-tune with SecureT2I (default: attack=True, vague=16x16)
python main.py \
  --config celeba.yml \
  --edit_attr beards \
  --n_train_img 100 \
  --n_iter 15 \
  --lr_clip_finetune 8e-6 \
  --attack True \
  --vague 16x16

# Baselines
python main.py --config celeba.yml --edit_attr beards --max_loss True   # Max Loss
python main.py --config celeba.yml --edit_attr beards --noisy_label True # Noisy Label
python main.py --config celeba.yml --edit_attr beards --retain_label True # Retain Label
python main.py --config celeba.yml --edit_attr beards --retrain True      # Retrain
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--config` | `celeba.yml` | Dataset config (`celeba.yml`, `bedroom.yml`, `church.yml`) |
| `--edit_attr` | `sketch` | Manipulation attribute (e.g., `beards`, `snow`, `golden`) |
| `--attack` | `True` | Enable SecureT2I fine-tuning |
| `--vague` | `motion` | Vagueness method: `8x8`, `16x16`, `32x32`, `gau`, `box`, `motion` |
| `--n_iter` | `15` | Number of fine-tuning iterations |
| `--lr_clip_finetune` | `8e-6` | Learning rate |
| `--model_path` | `pretrained/celeba_hq.ckpt` | Pretrained model weights |

### Asyrp

```bash
cd asyrp

python main.py \
  --config afhq.yml \
  --edit_attr dog_nicolas \
  --run_train True \
  --attack True \
  --n_iter 10
```

### EffDiff

```bash
cd Eff-diff

python main.py \
  --config bedroom.yml \
  --edit_attr golden \
  --attack True \
  --vague 16x16
```

### Evaluation

```bash
# Evaluate generated images (FID, IS, CLIP, WAN, WAN*)
python DiffusionCLIP/metric.py --exp ./runs/<experiment_dir>
```

---

## Results

SecureT2I is compared against three unlearning baselines (Max Loss, Noisy Label, Retain Label) and a Retrain baseline. We report **WAN** (permit set, higher is better) and **WAN\*** (forbid set, higher is better).

### DiffusionCLIP on CelebA-HQ

| Method | WAN (Permit ↑) | WAN\* (Forbid ↑) |
|---|---|---|
| Retrain | 0.67 | 0.11 |
| Max Loss | -0.18 | -0.67 |
| Noisy Label | -0.30 | -0.32 |
| Retain Label | -0.18 | -0.31 |
| **SecureT2I** | **0.44** | **0.16** |

SecureT2I substantially outperforms all unlearning baselines on the permit set while matching or exceeding Retrain on the forbid set across all three datasets and all three manipulation models.

### Generalization to Unseen Images

Evaluated on a held-out 10% subset excluded from fine-tuning, SecureT2I consistently achieves the best WAN and WAN\* scores, confirming robust generalization to images not seen during training.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **FID** ↓ | Fréchet Inception Distance — image quality vs. real distribution |
| **IS** ↑ | Inception Score — diversity and clarity of generated images |
| **CLIP** | Embedding similarity between generated and target images |
| **WAN** ↑ | Weighted Averaged Normalization on permit set (combines FID, IS, CLIP) |
| **WAN\*** ↑ | WAN variant for forbid set measuring alignment with vague target |

WAN definition:

$$\text{WAN} = \frac{-\text{FID}_{\text{norm}} + \text{IS}_{\text{norm}} + \text{CLIP}_{\text{norm}}}{3}$$

$$\text{WAN}^* = \frac{-\text{FID}_{\text{norm}} + \text{IS}_{\text{norm}} - \text{CLIP}_{\text{norm}}}{3}$$

---

## Limitations

- No defense against adversarial perturbations to the input images, which could bypass editing restrictions.
- Prompt-variation robustness is not extensively studied; different phrasings of the same semantic intent may yield inconsistent suppression behavior.

---

## Citation

If you use SecureT2I in your research, please cite:

```bibtex
@inproceedings{wu2025securet2i,
  title     = {SecureT2I: No More Unauthorized Manipulation on AI Generated Images from Prompts},
  author    = {Wu, Xiaodong and Li, Xiangman and Li, Qi and Ni, Jianbing and Lu, Rongxing},
  booktitle = {Proceedings of the 30th European Symposium on Research in Computer Security (ESORICS)},
  year      = {2025}
}
```

---

## Acknowledgments

This work builds upon [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP), [Asyrp](https://github.com/kwonminki/Asyrp), and [EffDiff](https://github.com/yandex-research/efficient-straight-through). We thank the authors of these works for releasing their code.
