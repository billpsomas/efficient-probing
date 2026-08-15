
import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

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

import open_clip

# assert timm.__version__ == "0.3.2" # version check
# from timm.models.layers import trunc_normal_

import util.misc as misc

from backbones import build_backbone
from probe_heads import build_probe_head

from models_vit import CLS_FT_CHOICES
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler, AMP_PRECISIONS
from util.lars import LARS
from util.crop import RandomResizedCrop

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
    parser.add_argument('--dinov3_weights', type=str, default=None, metavar='DINOV3_WEIGHTS',
                        help='Path to (or URL of) DINOv3 weights; required for --model dinov3_*')
    parser.add_argument("--franca_img_size", type=int, default=518,
                        help="Grid the Franca checkpoint was trained at; the LAION ViT-L "
                             "weights are 518 while the hub builds 224 by default.")
    parser.add_argument("--franca_weights", type=str, default=None,
                        help='Which Franca weights to pull, e.g. "LAION". Default (None) uses '
                             'the hub default, which is In21K -- unavailable for ViT-L.')
    parser.add_argument("--use_rasa_head", action="store_true", default=False,
                        help="Use debiased patch tokens from the RASA head (Franca only)")
    parser.add_argument("--dit_image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--dit_ckpt", type=str, default=None,
                        help="Optional DiT checkpoint (default: auto-download DiT-XL/2)")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--timm", action="store_true", default=False,
                        help="Load --model as a plain timm backbone (SAM, Hiera, ConvNeXt, ...)")
    parser.add_argument("--radio", action="store_true", default=False,
                        help="Load --model from the NVlabs/RADIO torch.hub repo")
    parser.add_argument("--input_size", default=224, type=int,
                        help="Train and eval resolution. Applies to every backbone except "
                             "--openclip, where open_clip supplies its own transforms. "
                             "DiT/SiT take their latent grid from --dit_image_size instead.")

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
    # Early stopping — stop once validation accuracy plateaus, instead of always
    # running the full --epochs. Saves a lot of compute on the large encoders.
    parser.add_argument('--early_stop', action='store_true', default=False,
                        help='Stop when val acc1 stops improving (see the three flags below)')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Stop after this many consecutive epochs without a gain > --early_stop_min_delta')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.1,
                        help='Improvement in val acc1 (percentage points) that counts as progress')
    parser.add_argument('--early_stop_min_epochs', type=int, default=15,
                        help='Never stop before this epoch, so the LR warmup is always covered')

    parser.add_argument('--knn_eval', action='store_true',
                        help='Perform kNN evaluation only')
    parser.add_argument('--T_sweep', type=str, default="",
                        help='Comma-separated temperatures to sweep in one pass, e.g. '
                             '"0.07,0.1,0.2". Features are extracted once and reused.')
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


# name -> (constructor, train kwargs, val kwargs). Root is --data_path unless the
# entry names a subdirectory. The per-dataset asymmetries here are deliberate and
# match what these benchmarks ship: FGVCAircraft and DTD evaluate on their 'val'
# split rather than 'test', CUB200 has no download support, and STL10 is the only
# one fetched on demand.
DATASET_SPECS = {
    "imagenet1k":    (datasets.ImageFolder,   {"subdir": "train"}, {"subdir": "val"}),
    "places365":     (datasets.Places365,     {"split": "train-standard", "small": True, "download": False},
                                              {"split": "val", "small": True, "download": False}),
    "CIFAR100":      (datasets.CIFAR100,      {"train": True, "download": False},
                                              {"train": False, "download": False}),
    "StanfordCars":  (datasets.StanfordCars,  {"split": "train", "download": False},
                                              {"split": "test", "download": False}),
    "Food101":       (datasets.Food101,       {"split": "train", "download": False},
                                              {"split": "test", "download": False}),
    "FGVCAircraft":  (datasets.FGVCAircraft,  {"split": "train", "download": False},
                                              {"split": "val", "download": False}),
    "SUN397":        (SUN397,                 {"split": "train", "download": False},
                                              {"split": "test", "download": False}),
    "DTD":           (datasets.DTD,           {"split": "train", "download": False},
                                              {"split": "val", "download": False}),
    "OxfordIIITPet": (datasets.OxfordIIITPet, {"split": "trainval", "download": False},
                                              {"split": "test", "download": False}),
    "CUB200":        (CUB200,                 {"split": "train"}, {"split": "test"}),
    "stl10":         (datasets.STL10,         {"split": "train", "download": True},
                                              {"split": "test", "download": True}),
}


def build_datasets(args, transform_train, transform_val):
    if args.dataset_name not in DATASET_SPECS:
        raise ValueError(f'Unsupported dataset "{args.dataset_name}"')
    ctor, train_kwargs, val_kwargs = DATASET_SPECS[args.dataset_name]

    def build(kwargs, transform):
        kwargs = dict(kwargs)
        subdir = kwargs.pop("subdir", None)
        root = os.path.join(args.data_path, subdir) if subdir else args.data_path
        return ctor(root=root, transform=transform, **kwargs)

    return build(train_kwargs, transform_train), build(val_kwargs, transform_val)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(args):
    """Train and eval transforms.

    Under --openclip the transforms come from open_clip so each checkpoint gets the
    normalisation it was trained with; every other backbone is fed ImageNet
    statistics regardless of what it was pretrained on, and --input_size is ignored
    in the open_clip case because open_clip decides resolution itself.

    Called after the seed is set and before the datasets are built, and it must stay
    there: the --openclip branch constructs (and throws away) a full CLIP model, so
    moving this call would change every downstream random draw.
    """
    if args.openclip:
        _, transform_train, transform_val = open_clip.create_model_and_transforms(
            args.model, pretrained=args.openclip_pretrain)
        return transform_train, transform_val

    if args.train_aug == 'default':
        transform_train = transforms.Compose([
            RandomResizedCrop(args.input_size, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    elif args.train_aug == 'aimv2':
        transform_train = transforms.Compose([
            RandomResizedCrop(args.input_size, scale=(0.08, 1.0), ratio=(0.75, 1.33),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.3),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),  # corresponds to 'rand-m9-mstd0.5-inc1'
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    transform_val = transforms.Compose([
        transforms.Resize(int(args.input_size * 256 / 224), interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform_train, transform_val


def build_samplers(args, dataset_train, dataset_val):
    if not args.distributed:
        return (0,
                torch.utils.data.RandomSampler(dataset_train),
                torch.utils.data.SequentialSampler(dataset_val))

    num_tasks, global_rank = misc.get_world_size(), misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
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
    return global_rank, sampler_train, sampler_val


def build_dataloaders(args, dataset_train, dataset_val, sampler_train, sampler_val):
    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(os.cpu_count()))

    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        worker_init_fn=worker_init_fn if args.dataloader_affinity_hack else None,
    )
    # k-NN featurises every training image exactly once, so it must not drop the
    # ragged last batch the way training does.
    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, drop_last=not args.knn_eval, **common)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, drop_last=False, **common)
    return loader_train, loader_val


def load_finetune_checkpoint(model, args):
    """Load --finetune weights into the frozen encoder.

    Must run BEFORE build_probe_head: the checkpoint's classifier keys are
    `head.weight`/`head.bias`, and once the head is a Sequential the same weights
    would be looked for under `head.2.weight` and silently not load.

    Skipped for --eval, which restores a fully trained model through --resume, and
    for the loaders that take their own weights (--simmim, capi, dinov2). k-NN is
    NOT skipped: it has no head to restore, so without this it would run on a
    randomly initialised backbone and score chance (~1% top-1 on ImageNet).
    """
    if not args.finetune or args.eval or args.simmim or args.model.startswith(("capi", "dinov2")):
        return

    if Path(args.finetune).exists():
        print("Interpreting", args.finetune, "as path")
        # weights_only=False: MAE-style checkpoints carry an argparse Namespace
        # alongside the tensors, which torch>=2.6 refuses to unpickle by default.
        # Only load --finetune checkpoints you trust.
        checkpoint_model = torch.load(args.finetune, map_location='cpu',
                                      weights_only=False)[args.checkpoint_key]
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
        checkpoint_model = _create_vision_transformer(
            args.finetune, pretrained=True, **model_to_kwargs[args.model]).state_dict()

    # Always drop the pretrained classifier: we are fitting a probe, so its weights
    # are never wanted. The old guard only dropped it on a shape mismatch, which
    # meant a checkpoint whose head happened to be 1000-way was loaded straight into
    # the probe -- and MaskFeat ViT-L's 1000-way head is entirely NaN, so training
    # died with "Loss is nan" at epoch 0 while its ViT-B sibling was fine.
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    # Some converted checkpoints carry NaN in tensors their pre-training never
    # trained. Refuse them rather than letting a NaN reach the optimiser.
    bad = [k for k, v in checkpoint_model.items()
           if isinstance(v, torch.Tensor) and torch.isnan(v).any()]
    if bad:
        print(f"Dropping {len(bad)} NaN tensor(s) from the checkpoint: {bad[:6]}")
        for k in bad:
            del checkpoint_model[k]
    dropped_on_purpose = set(bad) | {'head.weight', 'head.bias'}

    try:
        interpolate_pos_embed(model, checkpoint_model)
    except Exception as e:
        print("couldn't interpolate bc of", e)
        print("Is [cls] switched off?", args.no_cls_token)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # A key may be missing either because it is one the probe supplies itself
    # (head/oracle/fc) or because we deliberately removed it above -- the NaN
    # tensors and the pretrained classifier. Anything else missing is a real
    # mismatch and should still stop the run.
    unexplained = sorted(k for k in msg.missing_keys
                         if not (k.startswith(("head", "oracle", "fc")) or k in dropped_on_purpose))
    assert not unexplained, unexplained


def set_trainable(model, args):
    """Freeze the encoder and train the probe only, unless --finetuning."""
    trainable = args.finetuning
    for _, p in model.named_parameters():
        p.requires_grad = trainable
    if not trainable:
        for _, p in model.head.named_parameters():
            p.requires_grad = True


def build_optimizer(model_without_ddp, args):
    param_groups = (model_without_ddp.parameters()
                    if args.finetuning else model_without_ddp.head.parameters())
    optimizers = {"lars": LARS, "adamw": AdamW}
    make = optimizers.get(args.optimizer, SGD)
    return make(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def run_knn_eval(args, model, model_without_ddp, device, data_loader_train, data_loader_val):
    """Training-free k-NN on the same representation the linear probe would use.

    Feature extraction is the whole cost here (one pass over train+val); the search
    itself is seconds. So both k and the temperature are swept in memory rather than
    re-extracting per temperature, which used to triple the bill for a difference in
    the third decimal.

    Reports two feature variants where the backbone allows it: `raw`, and
    `final_norm` with the encoder's final LayerNorm applied. Which one is comparable
    to published numbers depends on the encoder, so both are printed rather than one
    being chosen here.
    """
    train_stats = extract_features(data_loader_train, model, device,
                                   return_targets_and_preds=True,
                                   cls_features=args.cls_features)
    test_stats = extract_features(data_loader_val, model, device,
                                  return_targets_and_preds=True,
                                  cls_features=args.cls_features)
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

    variants = [("raw", train_features, test_features)]
    final_norm = getattr(model_without_ddp, "norm", None)
    if final_norm is not None and hasattr(final_norm, "weight") \
            and final_norm.weight.shape[0] == train_features.shape[1]:
        with torch.no_grad():
            fn = final_norm.to(train_features.device).float()
            tr = nn.functional.normalize(fn(train_features), dim=1, p=2)
            te = nn.functional.normalize(fn(test_features), dim=1, p=2)
        variants.append(("final_norm", tr, te))
        print("[knn] reporting both raw and final-LayerNorm features")
    else:
        print("[knn] no usable final LayerNorm on this backbone; raw features only")

    temps = [float(t) for t in args.T_sweep.split(",")] if args.T_sweep else [args.T]
    for vname, trf, tef in variants:
      print(f"=== features={vname} ===")
      for T in temps:
        print(f"=== T={T} ===")
        for k in [5,10,15,20,50,100,200]:
            top1, top5 = knn_classifier(trf, train_labels, tef, test_labels, k, T=T)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")


def write_log_header(log_file_path, args):
    """Open training_log.txt and record the run's settings.

    Never truncates an existing log when resuming: args.resume is already resolved
    by --auto_resume at this point, and a resume that later fails would otherwise
    destroy the epoch history of the run it is continuing.
    """
    if not misc.is_main_process():
        return
    resuming = bool(args.resume) and os.path.exists(log_file_path)
    with open(log_file_path, "a" if resuming else "w") as log_file:
        if resuming:
            log_file.write(f"# resumed from {args.resume}\n")
        else:
            log_file.write("Training Log\n")
            log_file.write(f"Model: {args.model}\n")
            log_file.write(f"Model Details: {args.finetune}\n")
            log_file.write(f"Dataset: {args.dataset_name}\n")
            log_file.write(f"Representation: {args.cls_features}\n")
            log_file.write(f"Batch size per GPU: {args.batch_size}\n")
            log_file.write(f"Base learning rate: {args.blr}\n")


def main(args):
    """Probe a frozen encoder: build it, attach a head, train the head, report.

    The order of the steps below is load-bearing in three places, and getting any
    of them wrong fails silently rather than loudly:

      * load_finetune_checkpoint BEFORE build_probe_head. The checkpoint's
        classifier keys are `head.weight`/`head.bias`; once the head is a
        Sequential they would be `head.2.weight` and simply would not match.
      * misc.load_model (--resume) AFTER build_probe_head, for the mirror reason:
        resume checkpoints hold the Sequential's keys.
      * everything that constructs a module AFTER the seed is set. The head's
        initialisation depends on how many random draws precede it, and
        build_transforms is one of the culprits -- under --openclip it builds and
        discards a whole CLIP model.
    """
    misc.init_distributed_mode(args)

    log_file_path = os.path.join(args.output_dir, "training_log.txt")
    write_log_header(log_file_path, args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transform_train, transform_val = build_transforms(args)

    if args.knn_eval:
        transform_train = transform_val

    dataset_train, dataset_val = build_datasets(args, transform_train, transform_val)
    print(dataset_train)
    print(dataset_val)

    global_rank, sampler_train, sampler_val = build_samplers(args, dataset_train, dataset_val)

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

    data_loader_train, data_loader_val = build_dataloaders(
        args, dataset_train, dataset_val, sampler_train, sampler_val)

    model = build_backbone(args, device)

    load_finetune_checkpoint(model, args)

    build_probe_head(model, args)

    set_trainable(model, args)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # Log to file
    if misc.is_main_process():
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Effective batch size: {eff_batch_size}\n")
            log_file.write(f"Trainable Parameters: {n_parameters:,}\n")
            log_file.write("Epoch, Train Loss, Train Acc1, Val Loss, Val Acc1, Val Acc5\n")

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = build_optimizer(model_without_ddp, args)
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
    except RuntimeError:
        print('[resume] strict load failed, falling back to strict=False '
              '(checkpoint probably contains only the head) – '
              'backbone params will stay as loaded from --finetune.')
        misc.load_model(args=args,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        strict=False)

    if args.knn_eval:
        run_knn_eval(args, model, model_without_ddp, device,
                     data_loader_train, data_loader_val)
        exit(0)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    es_best, es_stale = -1.0, 0

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
                model_without_ddp._ep_saved_module = 'full'
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=log_stats, include_epoch_in_filename=False)
            else:
                model_without_ddp.head._ep_saved_module = 'head'
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

        # ---- early stopping on a validation plateau -------------------------
        # test_stats comes out of evaluate(), which all-reduces across ranks, so
        # every rank sees the same number and they all break on the same epoch.
        if args.early_stop:
            if test_stats["acc1"] > es_best + args.early_stop_min_delta:
                es_best, es_stale = test_stats["acc1"], 0
            else:
                es_stale += 1
            if (epoch + 1) >= args.early_stop_min_epochs and es_stale >= args.early_stop_patience:
                msg = (f"[early-stop] no gain > {args.early_stop_min_delta} pts for "
                       f"{es_stale} epochs (best {es_best:.2f}%); stopping at epoch {epoch} "
                       f"of {args.epochs}")
                print(msg)
                if misc.is_main_process():
                    with open(log_file_path, "a") as log_file:
                        log_file.write(msg + "\n")
                break

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
