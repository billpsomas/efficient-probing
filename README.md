<div align="center">
<h1>Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency</h1>

**Bill Psomas<sup>1</sup>†, Dionysis Christopoulos<sup>2</sup>†, Eirini Baltzi<sup>2</sup>, Ioannis Kakogeorgiou<sup>6</sup>**  
**Tilemachos Aravanis<sup>1</sup>, Nikos Komodakis<sup>3,4,5</sup>, Konstantinos Karantzalos<sup>2</sup>, Yannis Avrithis, Giorgos Tolias<sup>1</sup>**

<sup>1</sup>Visual Recognition Group, FEE, Czech Technical University in Prague <sup>2</sup>National Technical University of Athens
<sup>3</sup>University of Crete <sup>4</sup>Archimedes, Athena RC <sup>5</sup>ACM-FORTH <sup>6</sup>IIT, NCSR “Demokritos”

[![Project Page](https://img.shields.io/badge/-Project_Page-green.svg?colorA=333&logo=html5)](https://vrg.fel.cvut.cz/ep/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2506.10178)
[![arXiv](https://img.shields.io/badge/arXiv-2506.10178-b31b1b.svg)](https://arxiv.org/abs/2506.10178)
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-yellow.svg)](https://openreview.net/pdf?id=PXo0gtT7Al)
[![Code License: Apache 2.0](https://img.shields.io/badge/Code%20License-Apache%202.0-lightgray.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-lightgray.svg)](#)

</div>

Official PyTorch implementation and benchmark results for Efficient Probing.

**TL;DR:** We introduce efficient probing (EP), a lightweight multi-query cross-attention mechanism that improves accuracy of frozen pretrained encoders while yielding interpretable attention maps.

<p align="center">
<img width="75%" alt="EP illustration" src=".github/ep.png">
</p>

## Overview

As fine-tuning becomes impractical at scale, probing is emerging as the preferred evaluation protocol. However, standard linear probing can understate the capability of models whose pre-training optimizes local representations rather than an explicit global representation. This motivates attentive probing, an alternative that uses attention to selectively aggregate patch-level features. Despite growing adoption, attentive probing is still underexplored: existing approaches are often over-parameterized and computationally inefficient. 

In this work, we revisit attentive probing through the lens of the accuracy vs. parameter-efficiency trade-off. We present the first comprehensive study of existing methods, analyzing their design choices and benchmarking their performance. Building on these insights, we propose efficient probing (EP), a lightweight yet effective multi-query cross-attention mechanism that eliminates redundant projections and reduces the number of trainable parameters. Across multiple benchmarks and pre-training paradigms, EP consistently outperforms linear probing and previous attentive probing methods, and remains effective when combined with parameter-efficient fine-tuning. Beyond evaluation, our analysis uncovers emerging properties of EP, including complementary attention maps, which open new directions for leveraging probing beyond protocol design.

## Benchmark

Top-1 accuracy of linear probing (**LP**) vs. efficient probing (**EP**) on frozen encoders. **Pre-tr.** is the pre-training dataset; **Dataset** is the evaluation dataset.

This table is meant to grow. If you evaluate a backbone we have not covered, please open a pull request adding a row — see [Contributing a row](#contributing-a-row).

| | Method | Arch. | Pre-training | Evaluation | LP | EP |
|---|---|---|---|---|---:|---:|
| **MIM** | MAE | ViT-S/16 | IN-1K | IN-1K | 47.4 | **64.6** |
| | MAE | ViT-B/16 | IN-1K | IN-1K | 67.7 | **75.6** |
| | MAE | ViT-L/16 | IN-1K | IN-1K | 76.0 | **79.3** |
| | BEiTv2 | ViT-B/16 | IN-1K | IN-1K | 79.0 | **81.7** |
| | SimMIM | ViT-B/16 | IN-1K | IN-1K | 51.5 | **65.1** |
| | CAPI | ViT-L/14 | IN-1K | IN-1K | 81.5 | **83.6** |
| **JEA** | BYOL | RN-50 | IN-1K | IN-1K | 74.3 | **75.1** |
| | DINO | ViT-B/16 | IN-1K | IN-1K | 77.3 | **77.8** |
| **Hybrid** | iBOT | ViT-B/16 | IN-1K | IN-1K | 78.7 | **79.2** |
| | DINOv2 | ViT-B/14 | LVD-142M | IN-1K | 83.2 | **84.0** |
| | DINOv2 | ViT-L/14 | LVD-142M | IN-1K | 85.2 | **85.6** |
| | Franca | ViT-L/14 | IN-21k | IN-1K | 83.8 | **84.3** |
| | DINOv3 | ViT-B/16 | LVD-1689M | IN-1K | 84.0 | **84.4** |
| | DINOv3 | ViT-L/16 | LVD-1689M | IN-1K | 86.6 | **87.1** |
| **VLM** | CLIP | ViT-L/16 | WIT | IN-1K | 82.3 | **83.4** |
| | SigLIP | ViT-L/16 | WebLI | IN-1K | 84.1<sup>‡</sup> | **86.1** |
| | SigLIP2 | ViT-L/16 | WebLI | IN-1K | 85.2<sup>‡</sup> | **87.0** |
| **GEN** | DiT | DiT-XL/2 | IN-1K | IN-1K | 32.7<sup>‡</sup> | **57.0** |
| | AIMv2 | ViT-L/14 | custom | IN-1K | 84.8<sup>‡</sup> | **85.9** |

Paradigms: **MIM** masked image modelling · **JEA** joint-embedding architectures · **Hybrid** MIM + JEA · **VLM** vision-language models · **GEN** generative models.

**Notes.**

- All numbers are top-1 accuracy at the **best epoch**, not the final one.
- **EP** is the best result over a sweep of query counts *Q* (EP<sub>Q</sub> in the paper). The best *Q* is **not constant across backbones** — it is usually 32, but larger values win for some (e.g. 128 for DiT). Compare rows with this in mind.
- For the **Hybrid** methods, both `--cls_features ep` (patch tokens) and `--cls_features ep_all` (patch + `[CLS]`) were evaluated and the better one is reported, which is `ep_all`. Other rows use `ep`.
- **LP** is the better of the `[CLS]` token (`--cls_features cls`) and global average pooling over patch tokens (`--cls_features pos`). <sup>‡</sup> marks rows where GAP was used, either because the encoder has no `[CLS]` token (DiT, AIMv2) or because it already applies an attention pooling of its own (SigLIP, SigLIP2), making its pooled output an unfair stand-in for `[CLS]`.

### Contributing a row

1. Run LP and EP on your backbone (see [Experiments](#experiments)). Keep the protocol fixed: 90 epochs, LARS, `--blr 0.1`, effective batch size 4096.
2. **LP** — report the better of `--cls_features cls` and `--cls_features pos`. If the encoder has no usable `[CLS]`, use `pos` and mark the value with <sup>‡</sup>.
3. **EP** — sweep `--ep_queries` (32 is a good starting point; try 8/16/64/128 too) and report the best. Also try `--cls_features ep_all` alongside `ep`, and report whichever wins.
4. Report the **best-epoch** accuracy, and open a PR stating the winning *Q* and whether it came from `ep` or `ep_all`, with a link to the training log.

## Emerging Properties

We jointly visualize the attention maps of EP<sub>8</sub>. An emerging property of EP is that its queries specialize in different object regions, yielding complementary and interpretable attention patterns. Queries consistently attend to distinct parts, producing stable semantic correspondences (e.g., tails, beaks, feet) across images and a structured decomposition of visual cues.

<p align="center">
<img width="100%" alt="Complementary attention maps of the 8 EP queries" src=".github/ep8_queries.png">
</p>

## Environment

```bash
pip install -r requirements.txt
```

Optional extras, needed only for specific backbones: `open_clip_torch` (CLIP/SigLIP), `diffusers` (DiT/SiT), `aim` (AIMv2).

> [!IMPORTANT]
> `timm` must stay at the pinned `0.9.16`. From `timm` 1.0.x onwards, `VisionTransformer` passes `scale_attn_norm` to `block_fn`, which the custom `Block` in `models_vit.py` does not accept, so every `models_vit` backbone fails at construction. Installing `open_clip_torch` will silently upgrade `timm` — reinstall the pin afterwards.

## Integration (drop-in EP)

Use Efficient Probing (EP) as a lightweight attentive pooling over patch tokens from a frozen backbone (e.g., ViT). EP learns a small set of queries, attends to tokens with a single key projection, uses identity values (no V/O projections), and averages per-query outputs into one descriptor. It returns both the pooled descriptor and interpretable attention maps.

```python
from poolings.ep import EfficientProbing
# ---- Minimal integration example ----
# In your model.__init__:
   self.ep = EfficientProbing(dim=embed_dim, num_queries=32)  # EP_32

# In your model.forward(...):
#  'tokens' are the outputs of a FROZEN backbone (e.g., ViT):
#  shape (B, 1+N, D) if a [CLS] token exists, else (B, N, D)
#
#  Use only patch tokens (default in our paper/code):
   patch_tokens = tokens[:, 1:, :]          # or 'tokens' if you have no [CLS]
#
#  Optional: include [CLS] among the values by passing all tokens:
#  patch_tokens = tokens                    # uncomment to include [CLS]
#
   pooled = self.ep(patch_tokens)           # pooled: (B, D)
   logits = self.head(pooled)               # your classifier head
```

### Notes

- **Freeze the backbone**; train only `EfficientProbing` and your classification head.
- `num_queries` controls speed/accuracy (e.g., 8, 16, 32). EP averages across queries, so the output stays `(B, D)`.
- **Inputs & shapes:** `tokens` are `(B, N, D)` or `(B, 1+N, D)` if a `[CLS]` token exists.
- **Default usage:** pass patch tokens only (`tokens[:, 1:, :]` when `[CLS]` is present).
- To include `[CLS]` among values, pass **all tokens** instead.
- **Outputs:** `pooled` is `(B, D)` for your head; optional `attn` is `(B, Q, N)` for visualization/analysis.
- **Repro tip:** set seeds to make the learned query initialization reproducible.

## Experiments

### Evaluating MAE ViT-B with Efficient Probing on ImageNet-1k:

```bash
torchrun --nproc_per_node=4 --nnodes=1 \
    main_linprobe.py --amp bfloat16 --num_workers=12 --dataloader_affinity_hack \
    --epochs=90 --accum_iter=1 --optimizer=lars --batch_size=1024 \
    --model vit_base_patch16  --finetune vit_base_patch16_224.mae \
    --dataset_name imagenet1k --nb_classes 1000 --data_path /path/to/imagenet_pytorch \
    --output_dir ./outputs/linprobe_mae_vitb_ep_imagenet1k \
    --cls_features ep --ep_queries 32
```

- To perform standard linear probing (**LP**):
  - Use `--cls_features cls` to utilize the class token from the pre-trained model.
  - Use `--cls_features pos` to utilize the patch tokens (via global average pooling).

- `--ep_queries` sets the number of EP queries (EP<sub>Q</sub> in the paper), e.g. `8`, `16`, `32`. Default: `32`.
  The pooled descriptor stays `(B, D)` regardless, so only the query bank grows.

- To perform full finetuning (**FT**), use the `--finetuning` flag.

#### 🎯 **More poolings, Please!**
- Supported attentive pooling methods (as described in the paper): `abmilp`, `simpool`, `clip`, `siglip`, `aim`, `ep`, `cbam`, `coca`, `cait`, `dinovit`, `jepa`, `dolg`, `cae`
  - These can be passed via the `--cls_features` argument.
  - Note: Appending the suffix `_all` to any pooling type (e.g., `ep_all`) will include both patch tokens and the class token as input to the selected attentive pooling. By default, only patch tokens are used.

#### :globe_with_meridians: **More datasets, Please!**
- Experiment with more datasets in any setup of your choice by adjusting the `--dataset_name`, `--nb_classes`, and `--data_path` arguments accordingly.
  - **Supported datasets**: *ImageNet-1k*, *Places365*, *CIFAR-100*, *StanfordCars*, *Food101*, *FGVCAircraft*, *SUN397*, *DTD*, *OxfordIIITPet*, *CUB200*

####  🛠️ **More models, Please!**
- Try [CAPI](https://github.com/facebookresearch/capi/tree/main) and [DINOv2](https://github.com/facebookresearch/dinov2) pre-trained models (from PyTorch Hub) by adjusting the `--model` argument based on their official repositories.  
  - The `--finetune` argument is **not needed** in this case.

- Try **SimMIM**, **BEiTv2**, and **iBOT** by passing the checkpoint path to the `--finetune` argument.  
  - Pretrained weights are provided via [Google Drive](#).

- Instructions on how to use pre-trained models from **OpenCLIP** are provided in the following subsection.

### Evaluating CLIP ViT-L (pre-trained by openai) with Efficient Probing on ImageNet-1k:

```bash
torchrun --nproc_per_node=4 --nnodes=1 \
    main_linprobe.py --amp bfloat16 --num_workers=12 --dataloader_affinity_hack \
    --epochs=90 --accum_iter=1 --optimizer=lars --batch_size=1024 \
    --model ViT-L-14 --openclip_pretrain openai --openclip \
    --dataset_name imagenet1k --nb_classes 1000 --data_path /path/to/imagenet_pytorch \
    --output_dir ./outputs/linprobe_clip_openai_vitl_ep_imagenet1k \
    --cls_features ep --ep_queries 16
```
- To evaluate alternative pre-trained OpenCLIP models, adjust the `--model` and `--openclip_pretrain` arguments accordingly. Available combinations can be found in the [official OpenCLIP repository](https://github.com/mlfoundations/open_clip).

    Example alternative:
    
    ```bash
    --model ViT-L-16-SigLIP-256 --openclip_pretrain webli --openclip
    ```

## Acknowledgments

This codebase is based on the official [MAE](https://github.com/facebookresearch/mae), [SimMIM](https://github.com/microsoft/SimMIM/tree/main) and [Beyond [cls]](https://github.com/gmum/beyond_cls) implementations.

We thank the authors for open-sourcing them.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


## Citation

If you find this repository useful, please consider giving a star 🌟 and citation:
```
@inproceedings{
psomas2026attention,
title={Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency},
author={Bill Psomas and Dionysis Christopoulos and Eirini Baltzi and Ioannis Kakogeorgiou and Tilemachos Aravanis and Nikos Komodakis and Konstantinos Karantzalos and Yannis Avrithis and Giorgos Tolias},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=PXo0gtT7Al}
}
```
