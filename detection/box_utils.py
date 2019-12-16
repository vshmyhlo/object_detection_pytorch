import torch
import torchvision


def boxes_tl_br(boxes):
    return torch.split(boxes, 2, -1)


def boxes_center(boxes):
    tl, br = boxes_tl_br(boxes)
    return (tl + br) / 2


def boxes_hw(boxes):
    tl, br = boxes_tl_br(boxes)
    return br - tl


def boxes_pairwise_iou(a, b):
    iou = torchvision.ops.box_iou(a, b)

    return iou


def boxes_area(boxes):
    hw = boxes_hw(boxes)
    area = torch.prod(hw, -1)

    return area


# TODO: test
def boxes_intersection(a, b):
    a_tl, a_br = boxes_tl_br(a)
    b_tl, b_br = boxes_tl_br(b)

    inner_tl = torch.max(a_tl, b_tl)
    inner_br = torch.min(a_br, b_br)
    inner_size = torch.clamp(inner_br - inner_tl, min=0)
    intersection = torch.prod(inner_size, -1)

    return intersection


# TODO: test
def boxes_iou(input, target):
    intersection = boxes_intersection(input, target)
    union = boxes_area(input) + boxes_area(target) - intersection
    iou = intersection / union

    return iou
