import torch

from detection.box_coding import boxes_to_shifts_scales, shifts_scales_to_boxes


def test_conversion():
    anchors = torch.empty(10, 4).uniform_() * 10
    anchors[:, 2:] += anchors[:, :2]

    boxes = torch.empty(10, 4).uniform_() * 10
    boxes[:, 2:] += boxes[:, :2]

    assert torch.allclose(
        boxes,
        shifts_scales_to_boxes(boxes_to_shifts_scales(boxes, anchors), anchors))

    shifts_scales = torch.empty(10, 4).normal_()

    assert torch.allclose(
        shifts_scales,
        boxes_to_shifts_scales(shifts_scales_to_boxes(shifts_scales, anchors), anchors),
        atol=1e-7)
