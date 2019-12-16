from torch.nn import functional as F

from detection.box_utils import boxes_area, boxes_intersection


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
