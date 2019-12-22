import torch

from detection.box_coding import boxes_to_shifts_scales, shifts_scales_to_boxes, encode_boxes, decode_boxes
from detection.utils import foreground_binary_coding, Detections


def test_conversion():
    anchors = torch.randint(0, 100, size=(10, 4), dtype=torch.float)
    anchors[:, 2:] += anchors[:, :2]

    boxes = torch.randint(0, 100, size=(10, 4), dtype=torch.float)
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
    anchors = torch.tensor([
        [1, 1, 9, 9],
        [10, 10, 20, 20],
    ], dtype=torch.float)

    expected = Detections(
        class_ids=torch.randint(0, 10, (1,)),
        boxes=torch.tensor([
            [0, 0, 10, 10],
        ], dtype=torch.float),
        scores=None)
   
    class_output, loc_output = encode_boxes(expected, anchors, min_iou=0.4, max_iou=0.5)
    class_output = foreground_binary_coding(class_output, 10)
    actual = decode_boxes((class_output, loc_output))

    assert torch.equal(expected.boxes, actual.boxes)
    assert torch.equal(expected.class_ids, actual.class_ids)
