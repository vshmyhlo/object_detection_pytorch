import torch


def boxes_iou(input, target):
    intersection_tl = torch.max(input[..., :2], target[..., :2])
    intersection_br = torch.min(input[..., 2:], target[..., 2:])
    intersection_size = torch.clamp(intersection_br - intersection_tl, min=0)
    intersection = torch.prod(intersection_size, -1)

    input_area = torch.prod(input[..., 2:] - input[..., :2], -1)
    target_area = torch.prod(target[..., 2:] - target[..., :2], -1)

    union = input_area + target_area - intersection
    iou = intersection / union

    return iou
