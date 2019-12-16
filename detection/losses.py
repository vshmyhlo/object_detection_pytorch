import torch
from torch.nn import functional as F

from detection.box_utils import boxes_center, boxes_tl_br, boxes_area, boxes_intersection


def smooth_l1_loss(input, target):
    assert input.numel() == target.numel()
    assert input.numel() > 0

    loss = F.smooth_l1_loss(input=input, target=target, reduction='none')

    return loss


def boxes_iou_loss(input, target):
    intersection = boxes_intersection(input, target)
    union = boxes_area(input) + boxes_area(target) - intersection
    iou = intersection / union
    loss = 1 - iou

    return loss


def boxes_distance_iou_loss(input, target):
    input_tl, input_br = boxes_tl_br(input)
    target_tl, target_br = boxes_tl_br(target)

    intersection = boxes_intersection(input, target)
    union = boxes_area(input) + boxes_area(target) - intersection
    iou = intersection / union

    inner_dist = torch.norm(boxes_center(input) - boxes_center(target), 2, -1)**2

    outer_tl = torch.min(input_tl, target_tl)
    outer_br = torch.max(input_br, target_br)
    outer_size = torch.clamp(outer_br - outer_tl, min=0)
    outer_dist = torch.norm(outer_size, 2, -1)**2

    dist = inner_dist / outer_dist

    loss = 1 - iou + dist

    return loss
