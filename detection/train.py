import argparse
import gc
import itertools
import os
import shutil

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.metrics import Mean, Last
from all_the_tools.torch.utils import Saver, one_hot, seed_torch
from all_the_tools.transforms import ApplyTo
from all_the_tools.utils import seed_python
from tensorboardX import SummaryWriter
from tqdm import tqdm

from detection.anchors import build_anchors_maps, compute_anchor
from detection.box_coding import decode_boxes
from detection.config import build_default_config
from detection.datasets.coco import Dataset as CocoDataset
from detection.datasets.wider import Dataset as WiderDataset
from detection.model import RetinaNet
from detection.transform import Resize, BuildLabels, RandomCrop, RandomFlipLeftRight, denormalize
from detection.utils import logit, draw_boxes

# TODO: visualization scores sigmoid
# TODO: move logits slicing to helpers


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# TODO: check all usages
def encode_class_ids(input):
    return one_hot(input + 1, Dataset.num_classes + 2)[:, 2:]


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/detection')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--restore-path', type=str)
parser.add_argument('--workers', type=int, default=os.cpu_count())
args = parser.parse_args()
config = build_default_config()
config.merge_from_file(args.config_path)
config.freeze()
os.makedirs(args.experiment_path, exist_ok=True)
shutil.copy(args.config_path, args.experiment_path)

if config.dataset == 'coco':
    Dataset = CocoDataset
elif config.dataset == 'wider':
    Dataset = WiderDataset
else:
    raise AssertionError('invalid config.dataset {}'.format(config.dataset))

ANCHOR_TYPES = list(itertools.product(config.anchors.ratios, config.anchors.scales))
ANCHORS = [
    [compute_anchor(size, ratio, scale) for ratio, scale in ANCHOR_TYPES]
    if size is not None else None
    for size in config.anchors.sizes
]
anchor_maps = build_anchors_maps((config.crop_size, config.crop_size), ANCHORS).to(DEVICE)

train_transform = T.Compose([
    Resize(config.resize_size),
    RandomCrop(config.crop_size),
    RandomFlipLeftRight(),
    ApplyTo('image', T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])),
    BuildLabels(ANCHORS, min_iou=config.anchors.min_iou, max_iou=config.anchors.max_iou),
])
eval_transform = T.Compose([
    Resize(config.resize_size),
    RandomCrop(config.crop_size),
    ApplyTo('image', T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])),
    BuildLabels(ANCHORS, min_iou=config.anchors.min_iou, max_iou=config.anchors.max_iou),
])


def worker_init_fn(_):
    seed_python(torch.initial_seed() % 2**32)


def focal_loss(input, target, gamma=2., alpha=0.25):
    norm = (target > 0).sum()
    assert norm > 0

    target = encode_class_ids(target)

    prob = input.sigmoid()
    prob_true = prob * target + (1 - prob) * (1 - target)
    alpha = alpha * target + (1 - alpha) * (1 - target)
    weight = alpha * (1 - prob_true)**gamma

    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='none')
    loss = (weight * loss).sum() / norm

    return loss


def smooth_l1_loss(input, target):
    assert input.numel() == target.numel()
    assert input.numel() > 0

    loss = F.smooth_l1_loss(input=input, target=target, reduction='mean')

    return loss


# TODO: check loss
def compute_loss(input, target):
    input_class, input_regr = input
    target_class, target_regr = target

    # classification loss
    class_mask = target_class != -1
    class_loss = focal_loss(input=input_class[class_mask], target=target_class[class_mask])

    # regression loss
    regr_mask = target_class > 0
    regr_loss = smooth_l1_loss(input=input_regr[regr_mask], target=target_regr[regr_mask])

    assert class_loss.dim() == regr_loss.dim() == 0
    loss = class_loss + regr_loss

    return loss


def build_optimizer(parameters, config):
    if config.opt.type == 'sgd':
        return torch.optim.SGD(
            parameters,
            config.opt.learning_rate,
            momentum=config.opt.sgd.momentum,
            weight_decay=config.opt.weight_decay,
            nesterov=True)
    else:
        raise AssertionError('invalid config.opt.type {}'.format(config.opt.type))


def build_scheduler(optimizer, config, start_epoch):
    # FIXME:
    if config.sched.type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            config.epochs * config.train_steps,
            last_epoch=start_epoch * config.train_steps - 1)
    else:
        raise AssertionError('invalid config.sched.type {}'.format(config.sched.type))


def train_epoch(model, optimizer, scheduler, data_loader, class_names, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'train'))

    metrics = {
        'loss': Mean(),
        'learning_rate': Last(),
    }

    model.train()
    optimizer.zero_grad()
    for i, (images, maps) in enumerate(tqdm(data_loader, desc='epoch {} train'.format(epoch))):
        images, maps = images.to(DEVICE), [m.to(DEVICE) for m in maps]
        output = model(images)

        loss = compute_loss(input=output, target=maps)
        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['learning_rate'].update(np.squeeze(scheduler.get_lr()))

        (loss.mean() / config.opt.acc_steps).backward()

        if i % config.opt.acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        if i >= config.train_steps:
            break

    with torch.no_grad():
        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        print('[EPOCH {}][TRAIN] {}'.format(epoch, ', '.join('{}: {:.8f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)

        dets_true = [
            decode_boxes((logit(encode_class_ids(c)), r), anchor_maps)
            for c, r in zip(*maps)]
        images_true = [
            draw_boxes(denormalize(i, mean=MEAN, std=STD), d, class_names)
            for i, d in zip(images, dets_true)]
        dets_pred = [
            decode_boxes((c, r), anchor_maps)
            for c, r in zip(*output)]
        images_pred = [
            draw_boxes(denormalize(i, mean=MEAN, std=STD), d, class_names)
            for i, d in zip(images, dets_pred)]

        writer.add_image(
            'images_true', torchvision.utils.make_grid(images_true, nrow=4, normalize=True), global_step=epoch)
        writer.add_image(
            'images_pred', torchvision.utils.make_grid(images_pred, nrow=4, normalize=True), global_step=epoch)


def eval_epoch(model, data_loader, class_names, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))

    metrics = {
        'loss': Mean(),
    }

    model.eval()
    with torch.no_grad():
        for images, maps in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, maps = images.to(DEVICE), [m.to(DEVICE) for m in maps]
            output = model(images)

            loss = compute_loss(input=output, target=maps)
            metrics['loss'].update(loss.data.cpu().numpy())

        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        print('[EPOCH {}][EVAL] {}'.format(epoch, ', '.join('{}: {:.8f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)

        dets_true = [
            decode_boxes((logit(encode_class_ids(c)), r), anchor_maps)
            for c, r in zip(*maps)]
        images_true = [
            draw_boxes(denormalize(i, mean=MEAN, std=STD), d, class_names)
            for i, d in zip(images, dets_true)]
        dets_pred = [
            decode_boxes((c, r), anchor_maps)
            for c, r in zip(*output)]
        images_pred = [
            draw_boxes(denormalize(i, mean=MEAN, std=STD), d, class_names)
            for i, d in zip(images, dets_pred)]

        writer.add_image(
            'images_true', torchvision.utils.make_grid(images_true, nrow=4, normalize=True), global_step=epoch)
        writer.add_image(
            'images_pred', torchvision.utils.make_grid(images_pred, nrow=4, normalize=True), global_step=epoch)


def collate_cat_fn(batch):
    class_ids, boxes = zip(*batch)
    image_ids = [torch.full_like(c, i) for i, c in enumerate(class_ids)]

    class_ids = torch.cat(class_ids, 0)
    boxes = torch.cat(boxes, 0)
    # masks = torch.cat(masks, 0)
    image_ids = torch.cat(image_ids, 0)

    return class_ids, boxes, image_ids


def collate_fn(batch):
    images, maps = zip(*batch)

    images = torch.utils.data.dataloader.default_collate(images)
    maps = torch.utils.data.dataloader.default_collate(maps)

    return images, maps


def train():
    train_dataset = Dataset(args.dataset_path, subset='train', transform=train_transform)
    class_names = train_dataset.class_names
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)

    eval_dataset = Dataset(args.dataset_path, subset='eval', transform=eval_transform)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        drop_last=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)

    model = RetinaNet(Dataset.num_classes, len(ANCHOR_TYPES), anchor_levels=[a is not None for a in ANCHORS])
    model = model.to(DEVICE)

    optimizer = build_optimizer(model.parameters(), config)

    saver = Saver({'model': model, 'optimizer': optimizer})
    start_epoch = 0
    if args.restore_path is not None:
        saver.load(args.restore_path, keys=['model'])
    if os.path.exists(os.path.join(args.experiment_path, 'checkpoint.pth')):
        start_epoch = saver.load(os.path.join(args.experiment_path, 'checkpoint.pth'))

    scheduler = build_scheduler(optimizer, config, start_epoch)

    for epoch in range(start_epoch, config.epochs):
        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            class_names=class_names,
            epoch=epoch)
        gc.collect()
        eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            class_names=class_names,
            epoch=epoch)
        gc.collect()

        saver.save(os.path.join(args.experiment_path, 'checkpoint.pth'), epoch=epoch + 1)


def main():
    seed_python(config.seed)
    seed_torch(config.seed)
    train()


if __name__ == '__main__':
    main()
