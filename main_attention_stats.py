import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset
from tqdm import tqdm

import models_simmim
import models_vit
import util.misc as misc
from util.datasets import build_dataset_v2
from util.misc import AMP_PRECISIONS
from util.pos_embed import interpolate_pos_embed




def get_args_parser():
    parser = argparse.ArgumentParser('MAE attention statistics', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument("--simmim", action="store_true", default=False)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument("--checkpoint_key", default="model", type=str)
    parser.add_argument("--cca_bias", default="none")

    parser.set_defaults(global_pool=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=Path,
                        help='dataset path')

    parser.add_argument('--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--draw_2d_embeddings", action="store_true", default=False)
    parser.add_argument("--amp", default="float16", choices=list(AMP_PRECISIONS.keys()), type=str)


    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    args.dino_aug = False # hack
    _, dataset_val = build_dataset_v2(args, is_pretrain=False)

    print(dataset_val)

    args.distributed = False
    args.gpu = 0
    global_rank = 0

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    if args.output_dir is not None:
        misc.maybe_setup_wandb(args.output_dir, args=args, job_type="attn_stats")


    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    size_patch_kwargs = dict()
    if args.input_size != 224:
        assert args.input_size % 16 == 0, args.input_size
        size_patch_kwargs=dict(
            img_size=args.input_size,
            patch_size=args.input_size // 16
        )

    model_to_kwargs = {
        "vit_tiny_patch16": dict(patch_size=16, embed_dim=192, depth=12, num_heads=12),
        "vit_small_patch16": dict(patch_size=16, embed_dim=384, depth=12, num_heads=12),
        "vit_base_patch16": dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
        "vit_large_patch16": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
        "vit_huge_patch14": dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    }
    if args.simmim:
        model = models_simmim.__dict__[args.model]()
    else:
        model: models_vit.VisionTransformer = models_vit.__dict__[args.model](
            num_classes=1000,
            **size_patch_kwargs
        )


    if args.finetune:
        if Path(args.finetune).exists():
            print("Interpreting", args.finetune, "as path")
            checkpoint_model = torch.load(args.finetune, map_location='cpu')[args.checkpoint_key]

        elif args.finetune.startswith("hub"):
            state_dict = torch.hub.load_state_dict_from_url(
                url=models_vit.HUB_KEY_TO_URL[args.finetune],
            )
            state_dict = state_dict['model']
            for k in list(state_dict.keys()):
                if k.startswith('decoder') or k.startswith('mask_token'):
                    del state_dict[k]
            checkpoint_model = state_dict
        else:
            print("Interpreting", args.finetune, "as timm model")
            from timm.models.vision_transformer import _create_vision_transformer

            model_kwargs = model_to_kwargs[args.model]
            checkpoint_model = _create_vision_transformer(args.finetune, pretrained=True, **model_kwargs).state_dict()

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        assert not any([k.startswith("blocks") for k in msg.missing_keys])


    model.to(device)

    if wandb.run is not None:
        with torch.cuda.amp.autocast(
                enabled=args.amp != "none",
                dtype=AMP_PRECISIONS[args.amp]
        ):
            L_test, Y_test, A_test, M_test = collect_features(
                model, data_loader_val, device, 
                tqdm_desc="attention stats",
            )

        mean_attn_stats = A_test.mean(dim=(0, 2))
        mean_magn_stats = M_test.mean(dim=0)


        cc_attns = mean_attn_stats[:, 0]
        pos_self_attns = mean_attn_stats[:, 1]
        cc_attns_adj = mean_attn_stats[:, 2]
        pos_self_attns_adj = mean_attn_stats[:, 3]
        cls_pos_attns = mean_attn_stats[:, 4] # should complement the cls cls attention
        pos_cls_attns = mean_attn_stats[:, 5]
        cls_pos_entropy = mean_attn_stats[:, 6]
        pos_pos_entropy = mean_attn_stats[:, 7]
        cls_magnitude = mean_magn_stats[:, 0]
        pos_magnitude = mean_magn_stats[:, 1]

        stats_pf = "test_attn"

        for b in range(len(cc_attns)):
            wandb.log({
                f"{stats_pf}/cls_cls_attention": cc_attns[b],
                f"{stats_pf}/pos_self_attention": pos_self_attns[b],
                f"{stats_pf}/cls_cls_attention_adj_for_cls": cc_attns_adj[b],
                f"{stats_pf}/pos_self_attention_adj_for_cls": pos_self_attns_adj[b],
                f"{stats_pf}/cls_pos_attention": cls_pos_attns[b],
                f"{stats_pf}/pos_cls_attention": pos_cls_attns[b],
                f"{stats_pf}/cls_pos_entropy": cls_pos_entropy[b],
                f"{stats_pf}/pos_pos_entropy": pos_pos_entropy[b],
                f"{stats_pf}/cls_magnitude": cls_magnitude[b],
                f"{stats_pf}/pos_magnitude": pos_magnitude[b],
                f"{stats_pf}/vit_block": b,
            })

        tsne = TSNE()
        latent_2d = tsne.fit_transform(L_test.numpy())
        Y_test = Y_test.numpy()
        fig, ax = plt.subplots()

        for label in range(10):
            l_subset = latent_2d[Y_test == label][:25]
            ax.scatter(l_subset[:, 0], l_subset[:, 1], label=label)

        ax.legend()
        wandb.log(
            {"monitoring/tsne": fig}
        )


def collect_features(
        model: models_vit.VisionTransformer, loader: torch.utils.data.DataLoader,
        device,
    tqdm_desc: str = None
):
    model.eval()
    with torch.no_grad():
        features = []
        labels = []
        attns_list = []
        magn_list = []


        for i, (data, target) in enumerate(tqdm(loader, desc=tqdm_desc)):
            with torch.cuda.amp.autocast(
                    enabled=args.amp != "none",
                    dtype=AMP_PRECISIONS[args.amp]
            ):
                z, attns, magnitudes = model.forward_features(data.to(device))

            cls_cls_attns = attns[0, :, :, :, :1]
            pos_self_attns = attns[0, :, :, :, 1:].mean(dim=3, keepdim=True)

            cls_cls_attns_adj = attns[1, :, :, :, :1]
            pos_self_attns_adj = attns[1, :, :, :, 1:].mean(dim=3, keepdim=True)


            cls_pos_attns = attns[2, :, :, :, 1:].mean(dim=3, keepdim=True)
            pos_cls_attns = attns[3, :, :, :, 1:].mean(dim=3, keepdim=True)

            cls_pos_entropy = attns[4, :, :, :, :1]
            pos_pos_entropy = attns[4, :, :, :, 1:].mean(dim=3, keepdim=True)

            attn_stats = torch.cat([cls_cls_attns, pos_self_attns, cls_cls_attns_adj, pos_self_attns_adj, cls_pos_attns, pos_cls_attns, cls_pos_entropy, pos_pos_entropy], dim=3)

            magn_residual = magnitudes[0]
            magn_attended = magnitudes[1]
            magn_stats = magn_attended / (magn_residual + 1e-6)
            cls_magn_stats = magn_stats[:, :, :1]
            pos_magn_stats = magn_stats[:, :, 1:].mean(dim=2, keepdim=True)

            magn_stats = torch.cat([cls_magn_stats, pos_magn_stats], dim=2)


            features.append(z.detach().cpu())
            labels.append(target.detach().short().cpu())

            BSS, L, H, _ = attn_stats.shape
            attn_stats = attn_stats.reshape(BSS, L, H, 8)
            magn_stats = magn_stats.reshape(BSS, L, 2)

            attns_list.append(attn_stats.detach().cpu())
            magn_list.append(magn_stats.detach().cpu())


    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0).long()

    attns_list = torch.cat(attns_list, dim=0)
    magns_list = torch.cat(magn_list, dim=0)

    return features, labels, attns_list, magns_list

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
