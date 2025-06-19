
import math
import sys
from typing import Iterable, Optional, Union

import numpy as np
import torch
from einops import rearrange

from timm.data import Mixup
from timm.utils import accuracy
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import util.misc as misc
import util.lr_sched as lr_sched
from util.misc import AMP_PRECISIONS
from models_simmim import VisionTransformerSimMIM
from models_vit import VisionTransformer
from models_mae import MaskedAutoencoderViT

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in tqdm(enumerate(metric_logger.log_every(data_loader, print_freq, header))):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(
                enabled=args.amp != "none",
                dtype=AMP_PRECISIONS[args.amp]
        ):
            model_wo_ddp = model if not isinstance(model, DistributedDataParallel) else model.module
            if isinstance(model_wo_ddp, (VisionTransformer, VisionTransformerSimMIM)):
                outputs = model(samples, return_features=args.cls_features, return_block=args.return_block)
            else:
                outputs = model(samples)

            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            metric_logger.update(acc1=acc1.item(), acc5=acc5.item())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        data_loader,
        model: VisionTransformer,
        device, *,
        return_targets_and_preds: bool = False, cls_features: str = "cls",
        return_block: Optional[int] = None
):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    targets = []
    preds = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            model_wo_ddp = model if not isinstance(model, DistributedDataParallel) else model.module
            if isinstance(model_wo_ddp, MaskedAutoencoderViT):
                assert return_block is None, f"{return_block=} not used"
                _, _, _, (_, output, _, _, _) = model.forward(images, cls_features)
            elif isinstance(model_wo_ddp, (VisionTransformer, VisionTransformerSimMIM)):
                output = model.forward(images, return_features=cls_features, return_block=return_block)
            else:
                assert return_block is None, f"{return_block=} not used"
                output = model.forward(images)


            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        pred = output.argmax(dim=1).detach().cpu()
        targets.append(target.cpu())
        preds.append(pred.cpu())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if return_targets_and_preds:
        stats["targets"] = torch.cat(targets)
        stats["preds"] = torch.cat(preds)

    return stats

@torch.no_grad()
def extract_features(
        data_loader,
        model: VisionTransformer,
        device, *,
        return_targets_and_preds: bool = False, cls_features: str = "cls",
        return_block: Optional[int] = None
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    targets = []
    features = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            model_wo_ddp = model if not isinstance(model, DistributedDataParallel) else model.module
            if isinstance(model_wo_ddp, MaskedAutoencoderViT):
                assert return_block is None, f"{return_block=} not used"
                _, _, _, (_, output, _, _, _) = model.forward(images, cls_features)
            elif isinstance(model_wo_ddp, (VisionTransformer, VisionTransformerSimMIM)):
                _, output_feat = model.forward(images, return_features=cls_features, return_block=return_block, return_backbone_features=True)
            else:
                assert return_block is None, f"{return_block=} not used"
                _, output_feat = model.forward(images, return_backbone_features=True)

        targets.append(target.cpu())
        features.append(output_feat.cpu())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if return_targets_and_preds:
        stats["targets"] = torch.cat(targets)
        stats["features"] = torch.cat(features)

    return stats

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, use_cuda=True, num_classes=1000, num_chunks=500):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images = test_labels.shape[0]
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes)
    if use_cuda:
        retrieval_one_hot = retrieval_one_hot.cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5
