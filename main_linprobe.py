
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from fvcore.nn import FlopCountAnalysis

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
import torchvision.datasets as datasets

from util.sun397 import SUN397
from util.cub200 import CUB200

import timm
import open_clip

import models_simmim
import models_capi
import models_more
# assert timm.__version__ == "0.3.2" # version check
# from timm.models.layers import trunc_normal_

import util.misc as misc

from poolings.abmilp import ABMILPHead
from poolings.simpool import SimPool, SimPool_nolinears
from poolings.clip.attention_pool import AttentionPoolLatent
from poolings.clip.attention_pool2d import AttentionPool2d
from poolings.jepa.attentive_pooler import AttentivePooler
from poolings.aim import AttentionPoolingClassifier
from poolings.cbam import CbamPooling
from poolings.coca_pytorch import CrossAttention as CocaPooling
from poolings.other_pool import CAPooling, DinoViTBlockPooling
from poolings.dolg.dolg import SpatialAttention2d
from poolings.cae_att import CAEAttentiveBlock
from poolings.ep import EfficientProbing

from models_vit import CLS_FT_CHOICES
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler, AMP_PRECISIONS
from util.lars import LARS
from util.crop import RandomResizedCrop

import models_vit

from engine_finetune import train_one_epoch, evaluate, knn_classifier, extract_features


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--openclip_pretrain', default='openai', type=str, metavar='PRETRAIN',
                        help='Name of pretrain framework for openclip')
    parser.add_argument("--simmim", action="store_true", default=False)
    parser.add_argument("--openclip", action="store_true", default=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--optimizer', type=str, default="lars", choices=['lars', 'adamw', 'sgd'])
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Methods parameters
    parser.add_argument("--cls_features",
                        choices=CLS_FT_CHOICES,
                        default="cls", help="cls token / positional tokens for classification")
    parser.add_argument("--return_block", type=int, default=None)
    parser.add_argument("--checkpoint_key", default="model", type=str)
    parser.add_argument("--no_cls_token", action='store_true', default=False,
                        help="Disable CLS token (e.g. for I-JEPA). You still have to select appropriate --cls_features"
                        )
    # AbMILP
    parser.add_argument("--abmilp_act", choices=["tanh", "relu"], default="tanh",
                        help="abmilp activation function"
                        )
    parser.add_argument("--abmilp_sa", choices=["none", "map", "both"], default="both",
                        help="how to apply the self-attention in abmilp"
                        )
    parser.add_argument("--abmilp_depth", type=int, default=2, help="depth of abmilp head")
    parser.add_argument("--abmilp_cond", type=str, choices=["none", "pe"],
                        help="what to condition abmilp with?")
    parser.add_argument("--abmilp_content", type=str, choices=["all", "patch"], default="all")
    parser.add_argument("--suffix", type=str, default="")
    # EP
    parser.add_argument("--ep_queries", type=int, default=32, help="number of EfficientProbing queries")
    parser.add_argument("--d_out", type=int, default=1, help="Denominator of classifier dimensionality")
    # Other poolings
    parser.add_argument("--num_heads", type=int, default=16, help="number of other pooling methods heads")

    # Dataset parameters
    parser.add_argument('--dataset_name', default='imagenet1k', type=str,
                        help='dataset name')
    parser.add_argument('--data_path', default=Path('/datasets01/imagenet_full_size/061417/'), type=Path,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--train_aug', default='default', type=str, choices=['default', 'aimv2'],
                        help='Augmentation setup for training')

    # Training parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--finetune', default='',
                    help='finetune from checkpoint')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--auto_resume', action='store_true', default=False,
                        help='If set, look for the newest checkpoint-*.pth inside '
                        '--output_dir and resume from it unless --resume is given.')
    parser.add_argument('--finetuning', action='store_true', default=False,
                        help='Unfreeze the backbone and perform fine-tuning instead of probing '
                             '(set this to True for full fine‑tuning)')
    
    # Perform kNN evaluation only
    parser.add_argument('--knn_eval', action='store_true',
                        help='Perform kNN evaluation only')
    parser.add_argument('--T', type=float, default=0.07,
                        help='Temperature for kNN evaluation. We recommend starting with the default value 0.07 and increase slightly up to 0.1-0.2 for the openclip models.')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--dataloader_affinity_hack", "-dlah",
                        action='store_true',
                        help="See: https://github.com/pytorch/pytorch/issues/101850#issuecomment-1717363898")
    parser.add_argument("--amp", default="float16", choices=list(AMP_PRECISIONS.keys()), type=str)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    log_file_path = os.path.join(args.output_dir, "training_log.txt")
    if misc.is_main_process():
        with open(log_file_path, "w") as log_file:
            log_file.write("Training Log\n")
            log_file.write(f"Model: {args.model}\n")
            log_file.write(f"Model Details: {args.finetune}\n")
            log_file.write(f"Dataset: {args.dataset_name}\n")
            log_file.write(f"Representation: {args.cls_features}\n")
            log_file.write(f"Batch size per GPU: {args.batch_size}\n")
            log_file.write(f"Base learning rate: {args.blr}\n")

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.openclip:
        _, transform_train, transform_val = open_clip.create_model_and_transforms(args.model, pretrained=args.openclip_pretrain)
    else:
        # Choose between weak or stronger augmentation
        if args.train_aug == 'default':
            transform_train = transforms.Compose([
                    RandomResizedCrop(224, interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif args.train_aug == 'aimv2':
            transform_train = transforms.Compose([
                    RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(0.3),
                    AutoAugment(policy=AutoAugmentPolicy.IMAGENET),  # corresponds to 'rand-m9-mstd0.5-inc1'
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        transform_val = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    if args.knn_eval:
        transform_train = transform_val

    if args.dataset_name == 'imagenet1k':
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    elif args.dataset_name == 'places365':
        dataset_train = datasets.Places365(root=args.data_path, split='train-standard', small=True, download=False, transform=transform_train)
        dataset_val = datasets.Places365(root=args.data_path, split='val', small=True, download=False, transform=transform_val)
    elif args.dataset_name == 'CIFAR100':
        dataset_train = datasets.CIFAR100(root=args.data_path, train=True, download=False, transform=transform_train)
        dataset_val = datasets.CIFAR100(root=args.data_path, train=False, download=False, transform=transform_val)
    elif args.dataset_name == 'StanfordCars':
        dataset_train = datasets.StanfordCars(root=args.data_path, split='train', download=False, transform=transform_train)
        dataset_val = datasets.StanfordCars(root=args.data_path, split='test', download=False, transform=transform_val)
    elif args.dataset_name == 'Food101':
        dataset_train = datasets.Food101(root=args.data_path, split='train', download=False, transform=transform_train)
        dataset_val = datasets.Food101(root=args.data_path, split='test', download=False, transform=transform_val)
    elif args.dataset_name == 'FGVCAircraft':
        dataset_train = datasets.FGVCAircraft(root=args.data_path, split='train', download=False, transform=transform_train)
        dataset_val = datasets.FGVCAircraft(root=args.data_path, split='val', download=False, transform=transform_val)
    elif args.dataset_name == 'SUN397':
        dataset_train = SUN397(root=args.data_path, split='train', download=False, transform=transform_train)
        dataset_val = SUN397(root=args.data_path, split='test', download=False, transform=transform_val)
    elif args.dataset_name == 'DTD':
        dataset_train = datasets.DTD(root=args.data_path, split='train', download=False, transform=transform_train)
        dataset_val = datasets.DTD(root=args.data_path, split='val', download=False, transform=transform_val)
    elif args.dataset_name == 'OxfordIIITPet':
        dataset_train = datasets.OxfordIIITPet(root=args.data_path, split='trainval', download=False, transform=transform_train)
        dataset_val = datasets.OxfordIIITPet(root=args.data_path, split='test', download=False, transform=transform_val)
    elif args.dataset_name == 'CUB200':
        dataset_train = CUB200(root=args.data_path, split='train', transform=transform_train)
        dataset_val = CUB200(root=args.data_path, split='test', transform=transform_val)
    elif args.dataset_name == 'stl10':
        dataset_train = datasets.STL10(args.data_path, split="train", transform=transform_train, download=True)
        dataset_val = datasets.STL10(args.data_path, split='test', transform=transform_val, download=True)
    else:
        raise ValueError(f'Unsupported dataset "{args.dataset_name}"')
    print(dataset_train)
    print(dataset_val)


    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    args.eff_batch_size = eff_batch_size

    if global_rank == 0 and args.output_dir is not None and not args.eval and not args.knn_eval:
        misc.maybe_setup_wandb(
            args.output_dir, args=args,
            job_type="linprobe_v1", run_name_suffix=args.suffix
        )
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(os.cpu_count()))

    if args.knn_eval:
        drop_last = False
    else:
        drop_last = True

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn if args.dataloader_affinity_hack else None
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn if args.dataloader_affinity_hack else None
    )

    if args.model.startswith("capi"):    
        capi_backbone = torch.hub.load('facebookresearch/capi:main', args.model)
        model = models_capi.CapiWrapper(
            capi_model=capi_backbone,
            num_classes=args.nb_classes,
            features=args.cls_features
        )
    elif args.model.startswith("dinov2"):
        dinov2_backbone = torch.hub.load('facebookresearch/dinov2', args.model)
        model = models_more.DinoWrapper(
            dino_model=dinov2_backbone, 
            num_classes=args.nb_classes,
            features=args.cls_features
        )
    elif args.openclip:
        backbone, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.openclip_pretrain)
        vision_encoder = backbone.visual
        model = models_more.CLIPWrapper(
            clip_model=vision_encoder,
            num_classes=args.nb_classes,
            features=args.cls_features
        )
    elif args.simmim:
        model = models_simmim.__dict__[args.model](
            checkpoint_path=args.finetune
        )
    else:
        cls_kwargs = dict()
        if "huge" in args.model:
            cls_kwargs["class_token"] = not args.no_cls_token
        model: models_vit.VisionTransformer = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            **cls_kwargs
        )

    if args.finetune and not args.eval and not args.knn_eval and not args.simmim and not args.model.startswith(("capi", "dinov2")):
        if Path(args.finetune).exists():
            print("Interpreting", args.finetune, "as path")
            checkpoint_model = torch.load(args.finetune, map_location='cpu')[args.checkpoint_key]
        else:
            print("Interpreting", args.finetune, "as timm model")
            from timm.models.vision_transformer import _create_vision_transformer

            model_to_kwargs = {
                "vit_tiny_patch16": dict(patch_size=16, embed_dim=192, depth=12, num_heads=12),
                "vit_small_patch16": dict(patch_size=16, embed_dim=384, depth=12, num_heads=12),
                "vit_base_patch16": dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
                "vit_large_patch16": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
                "vit_huge_patch14": dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
            }
            model_kwargs = model_to_kwargs[args.model]
            checkpoint_model = _create_vision_transformer(args.finetune, pretrained=True, **model_kwargs).state_dict()

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        try:
            interpolate_pos_embed(model, checkpoint_model)
        except Exception as e:
            print("couldn't interpolate bc of", e)
            print("Is [cls] switched off?", args.no_cls_token)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        assert all([
            k.startswith("head") or k.startswith("oracle") or k.startswith("fc")
            for k in msg.missing_keys
        ]), sorted(msg.missing_keys)

    if args.cls_features == "abmilp" or args.cls_features == "abmilp_all":
        abmilp = ABMILPHead(
                dim=model.head.in_features,
                self_attention_apply_to=args.abmilp_sa,
                activation=args.abmilp_act,
                depth=args.abmilp_depth,
                cond=args.abmilp_cond,
                content=args.abmilp_content,
                num_patches=model.patch_embed.num_patches,

            )
        model.head = torch.nn.Sequential(
            abmilp,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "simpool" or args.cls_features == "simpool_all":
        # Optionally expose more SimPool-related hyperparams as CLI args
        simpool = SimPool(
            dim=model.head.in_features,
            num_heads=1,
            qkv_bias=False,
            qk_scale=None,
            gamma=None,
            use_beta=False
        )
        # Now wrap it exactly like abmilp
        model.head = torch.nn.Sequential(
            simpool,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "esimpool" or args.cls_features == "esimpool_all":
        simpool_nolinears = SimPool_nolinears(
            dim=model.head.in_features,
            num_heads=12,
            qk_scale=None,
            gamma=None,
            use_beta=False
        )
        model.head = torch.nn.Sequential(
            simpool_nolinears,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "clip" or args.cls_features == "clip_all":
        if args.model == 'capi_vitl14_in1k':
            feat_size = 16
        else:
            feat_size = 14
        clip = AttentionPool2d(
            in_features=model.head.in_features,
            feat_size=feat_size
        )
        model.head = torch.nn.Sequential(
            clip,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "siglip" or args.cls_features == "siglip_all":
        siglip = AttentionPoolLatent(in_features=model.head.in_features)
        model.head = torch.nn.Sequential(
            siglip,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "aim" or args.cls_features == "aim_all":
        aim = AttentionPoolingClassifier(dim=model.head.in_features, num_heads=args.num_heads)
        model.head = torch.nn.Sequential(
            aim,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "ep" or args.cls_features == "ep_all":
        ep = EfficientProbing(dim=model.head.in_features, num_queries=args.ep_queries, d_out=args.d_out)
        new_classifier = torch.nn.Linear(model.head.in_features // args.d_out, args.nb_classes, bias=True)
        model.head = torch.nn.Sequential(
            ep,
            torch.nn.BatchNorm1d(model.head.in_features // args.d_out, affine=False, eps=1e-6),
            new_classifier
        )
    elif args.cls_features == "cbam" or args.cls_features == "cbam_all":
        cbam = CbamPooling(
            channels=model.head.in_features,
            spatial_kernel_size=7
        )
        model.head = torch.nn.Sequential(
            cbam,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "coca" or args.cls_features == "coca_all":
        coca = CocaPooling(dim=model.head.in_features)
        model.head = torch.nn.Sequential(
            coca,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "cait" or args.cls_features == "cait_all":
        cait = CAPooling(embed_dim=model.head.in_features)
        model.head = torch.nn.Sequential(
            cait,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "dinovit" or args.cls_features == "dinovit_all":
        dinovit_block = DinoViTBlockPooling(d_model=model.head.in_features)
        model.head = torch.nn.Sequential(
            dinovit_block,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "jepa" or args.cls_features == "jepa_all":
        jepa = AttentivePooler(embed_dim=model.head.in_features, num_heads=args.num_heads)
        model.head = torch.nn.Sequential(
            jepa,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "dolg" or args.cls_features == "dolg_all":
        dolg = SpatialAttention2d(
            in_c=model.head.in_features,
            s3_dim=model.head.in_features,
            with_aspp=False
        )
        model.head = torch.nn.Sequential(
            dolg,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    elif args.cls_features == "cae" or args.cls_features == "cae_all":
        cae_att = CAEAttentiveBlock(dim=model.head.in_features)
        model.head = torch.nn.Sequential(
            cae_att,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    else:
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    
    if args.finetuning:
        # unfreeze all
        for _, p in model.named_parameters():
            p.requires_grad = True
    else:
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    # Log to file
    if misc.is_main_process():
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Effective batch size: {eff_batch_size}\n")
            log_file.write(f"Trainable Parameters: {n_parameters:,}\n")
            log_file.write("Epoch, Train Loss, Train Acc1, Val Loss, Val Acc1, Val Acc5\n")

    # NOTE: Added extra computation info to log file
    # Measure FLOPS for a 3x224x224 image
    '''
    model.eval()
    flops = FlopCountAnalysis(model, torch.randn(8, 3, 224, 224).to(device))
    model.train()
    flops_count = flops.total()  # Total FLOPS
    flops_count_gflops = flops_count / 1e9  # Convert to GFLOPS
    # Measure throughput during evaluation on 10 batches
    torch.cuda.synchronize()
    start_t = time.time()
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader_val):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            if i == 10:  # Evaluate throughput on a subset for consistency
                break
    end_t = time.time()
    throughput = (10 * args.batch_size) / (end_t - start_t)  # Images per second
    
    # Log to file
    if misc.is_main_process():
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Effective batch size: {eff_batch_size}\n")
            log_file.write(f"Trainable Parameters: {n_parameters:,}\n")
            log_file.write(f"Model FLOPS: {flops_count_gflops:.2f} GFLOPS\n")
            log_file.write(f"Throughput (10 batches): {throughput:.2f} images/sec\n")
            log_file.write("Epoch, Train Loss, Train Acc1, Val Loss, Val Acc1, Val Acc5\n")
    '''
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = (model_without_ddp.parameters()
                    if args.finetuning else model_without_ddp.head.parameters())

    if args.optimizer == "lars":
        optimizer = LARS(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    try:
        misc.load_model(args=args,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        strict=True)
    except RuntimeError as err:
        print('[resume] strict load failed, falling back to strict=False '
              '(checkpoint probably contains only the head) – '
              'backbone params will stay as loaded from --finetune.')
        misc.load_model(args=args,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        strict=False)

    if args.knn_eval:
        train_stats = extract_features(data_loader_train, model, device, return_targets_and_preds=True)
        test_stats = extract_features(data_loader_val, model, device, return_targets_and_preds=True)
        print(f"Train features shape: {train_stats['features'].shape}")
        print(f"Train targets shape: {train_stats['targets'].shape}")
        print(f"Test features shape: {test_stats['features'].shape}")
        print(f"Test targets shape: {test_stats['targets'].shape}")

        print("Features are ready!\nStart the k-NN classification.")
        train_features = train_stats['features'].cuda()
        test_features = test_stats['features'].cuda()
        train_labels = train_stats['targets'].cuda()
        test_labels = test_stats['targets'].cuda()

        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

        for k in [5,10,15,20,50,100,200]:
            top1, top5 = knn_classifier(train_features, train_labels,
                test_features, test_labels, k, T=args.T)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
        exit(0)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        test_stats = evaluate(data_loader_val, model, device, cls_features=args.cls_features, return_block=args.return_block)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir:
            if args.finetuning:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=log_stats, include_epoch_in_filename=False)
            else:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp.head, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=log_stats, include_epoch_in_filename=False)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if misc.is_main_process():
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{epoch}, {train_stats['loss']:.4f}, {train_stats['acc1']:.2f}, "
                            f"{test_stats['loss']:.4f}, {test_stats['acc1']:.2f}, {test_stats['acc5']:.2f}\n")

        if log_writer is not None:
            log_writer.add_scalar(f'test_v1_{args.cls_features}/train_acc1', train_stats['acc1'], epoch)
            log_writer.add_scalar(f'test_v1_{args.cls_features}/train_loss', train_stats['loss'], epoch)
            log_writer.add_scalar(f'test_v1_{args.cls_features}/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar(f'test_v1_{args.cls_features}/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar(f'test_v1_{args.cls_features}/test_loss', test_stats['loss'], epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if misc.is_main_process():
        with open(log_file_path, "a") as log_file:
            log_file.write("\nFinal Results:\n")
            log_file.write(f"Max Accuracy: {max_accuracy:.2f}%\n")
            log_file.write(f"Final Val Acc1: {test_stats['acc1']:.2f}%\n")
            log_file.write(f"Final Val Acc5: {test_stats['acc5']:.2f}%\n")
            log_file.write(f"Total Training Time: {total_time_str}\n")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.auto_resume and args.resume == '' and args.output_dir:
        out_dir = Path(args.output_dir)
        if out_dir.is_dir():
            # look for files like checkpoint‑12.pth, checkpoint‑epoch12.pth, etc.
            ckpts = sorted(out_dir.glob('checkpoint*.pth'))
            if ckpts:
                args.resume = str(ckpts[-1])          # newest by name
                print(f'[auto‑resume] Will load {args.resume}')
            else:
                print('[auto‑resume] No checkpoint found – starting fresh')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
