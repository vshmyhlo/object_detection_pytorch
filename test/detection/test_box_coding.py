import torch
from all_the_tools.torch.utils import one_hot

from detection.box_coding import boxes_to_shifts_scales, shifts_scales_to_boxes, encode_boxes, decode_boxes


def test_conversion():
    anchors = torch.empty(10, 4).uniform_() * 10
    anchors[:, 2:] += anchors[:, :2]

    boxes = torch.empty(10, 4).uniform_() * 10
    boxes[:, 2:] += boxes[:, :2]

    assert torch.allclose(
        boxes,
        shifts_scales_to_boxes(boxes_to_shifts_scales(boxes, anchors), anchors),
        atol=1e-6)

    shifts_scales = torch.empty(10, 4).normal_()

    assert torch.allclose(
        shifts_scales,
        boxes_to_shifts_scales(shifts_scales_to_boxes(shifts_scales, anchors), anchors),
        atol=1e-6)


def test_coding():
    boxes = torch.tensor([
        [0, 0, 10, 10],
    ], dtype=torch.float)
    class_ids = torch.randint(0, 10, (boxes.size(0),))
    anchors = torch.tensor([
        [1, 1, 9, 9],
        [10, 10, 20, 20],
    ], dtype=torch.float)

    expected = class_ids, boxes

    class_output, loc_output = encode_boxes(expected, anchors, min_iou=0.4, max_iou=0.5)
    class_output = one_hot(class_output + 1, 10 + 2)[:, 2:]

    actual = decode_boxes((class_output, loc_output))

    for e, a in zip(expected, actual):
        assert torch.equal(e, a)
