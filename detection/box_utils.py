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


def boxes_iou(a, b):
    iou = torchvision.ops.box_iou(a, b)

    return iou


def boxes_area(boxes):
    hw = boxes_hw(boxes)
    area = torch.prod(hw, -1)

    return area
