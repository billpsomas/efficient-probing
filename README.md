# Attention, Please! Revisiting Attentive Probing for Masked Image Modeling

Official PyTorch implementation and pretrained attentive heads for Efficient Probing. [[`arXiv`](https://arxiv.org/pdf/2506.10178)]

**TL;DR:** We introduce efficient probing (EP), a simplified multi-query cross-attention mechanism, enhancing evaluation of self-supervised learning models by improving speed, performance, and interpretability.

<p align="center">
<img width="100%" alt="EP illustration" src=".github/ep.png">
</p>

## Overview

As fine-tuning (FT) becomes increasingly impractical at scale, probing is emerging as the preferred evaluation protocol for self-supervised learning (SSL). Yet, the standard linear probing (LP) fails to adequately reflect the potential of models trained with Masked Image Modeling (MIM), due to the distributed nature of patch tokens. This motivates the need for attentive probing, an alternative that uses attention to selectively aggregate patch-level features. Despite its growing adoption, attentive probing remains under-explored, with existing methods suffering from excessive parameterization and poor computational efficiency. 

In this work, we revisit attentive probing through the lens of the accuracy-efficiency trade-off. We conduct a systematic study of existing methods, analyzing their mechanisms and benchmarking their performance. We introduce efficient probing (EP), a multi-query cross-attention mechanism that eliminates redundant projections, reduces the number of trainable parameters, and achieves up to a 10 $\times$ speed-up over conventional multi-head attention. 

Despite its simplicity, EP outperforms LP and prior attentive probing approaches across seven benchmarks, generalizes well beyond MIM to diverse pre-training paradigms, produces interpretable attention maps, and achieves strong gains in low-shot and layer-wise settings.

## Environment

Dependencies are listed in `requirements.txt`.

## Integration

## Experiments

Evaluating MAE ViT-B with Efficient Probing on ImageNet-1k:

```bash
torchrun --nproc_per_node=4 --nnodes=1 --master_port=29501 \
    /home/psomava1/code/beyond_cls/main_linprobe.py --amp bfloat16 --num_workers=12 --dataloader_affinity_hack \
    --epochs=90 --accum_iter=1 --optimizer=lars --batch_size=1024 \
    --model=ViT-L-16-SigLIP-256 --openclip_pretrain=webli --openclip \
    --dataset_name=imagenet1k --nb_classes 1000 --data_path=/mnt/data/Public_datasets/imagenet/imagenet_pytorch \
    --output_dir=/home/psomava1/code/beyond_cls/outputs/linprobe_siglip_vitl_ep_imagenet1k \
    --cls_features=ep 
```

## Acknowledgments

This codebase is based on the official [MAE](https://github.com/facebookresearch/mae) codebase.

We thank the authors for open-sourcing it.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


## Citation

If you find this repository useful, please consider giving a star 🌟 and citation:
```
@misc{psomas2025attentionpleaserevisitingattentive,
      title={Attention, Please! Revisiting Attentive Probing for Masked Image Modeling}, 
      author={Bill Psomas and Dionysis Christopoulos and Eirini Baltzi and Ioannis Kakogeorgiou and Tilemachos Aravanis and Nikos Komodakis and Konstantinos Karantzalos and Yannis Avrithis and Giorgos Tolias},
      year={2025},
      eprint={2506.10178},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.10178}, 
}
```