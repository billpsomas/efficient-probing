import os
from typing import Type

import PIL
from PIL import Image

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset_v2(args, is_pretrain: bool):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    if args.dino_aug:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.4, 1.), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    transform_val = transforms.Compose([
        transforms.Resize(int(args.input_size * 16/14), interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ds_name = args.data_path.name

    if "cifar" in args.dataset_name:
        assert args.input_size == 32
        CIFAR_DS: Type[datasets.CIFAR10] = datasets.CIFAR10 if args.dataset_name == 'cifar10' else datasets.CIFAR100
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        dataset_train = CIFAR_DS(args.data_path, train=True, download=True, transform=trans)
        dataset_val = CIFAR_DS(args.data_path, train=False, download=True, transform=trans)

    elif "stl10" in args.dataset_name:
        dataset_train =  datasets.STL10(
            args.data_path,
            split=("train+unlabeled" if is_pretrain else "train"),
            transform=transform_train, download=True
        )
        dataset_val = datasets.STL10(args.data_path, split='test', transform=transform_val, download=True)
    else:
        dataset_train = datasets.ImageFolder(args.data_path / 'train', transform=transform_train)
        dataset_val = datasets.ImageFolder(args.data_path / 'val', transform=transform_val)

    return dataset_train, dataset_val

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
