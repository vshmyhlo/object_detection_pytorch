import torch

from detection.model import ROIAlign
from detection.todo import compute_level_ids, STRIDES


def test_copmute_level_ids():
    boxes = torch.tensor([
        [0, 0, 16, 16],
        [0, 0, 32, 32],
        [0, 0, 64, 64],
        [0, 0, 128, 128],
        [0, 0, 256, 256],
        [0, 0, 512, 512],
        [0, 0, 1024, 1024],
    ], dtype=torch.float)

    level_ids = compute_level_ids(boxes)

    assert torch.equal(level_ids, torch.tensor([2, 2, 2, 3, 4, 5, 5], dtype=torch.long))


def test_roi_align():
    model = ROIAlign((7, 7))

    fpn_output = [torch.zeros(3, 2, 256 // s, 224 // s) for s in STRIDES]
    for i, x in enumerate(fpn_output):
        x[:, 0] = i
        for j in range(x.size(0)):
            x[j, 1] = j

    boxes = torch.tensor([
        [128, 128, 32, 32],
        [128, 128, 512, 512],
    ], dtype=torch.float)
    class_ids = torch.tensor([2, 0], dtype=torch.long)

    actual = model(fpn_output, boxes, class_ids)

    expected = torch.zeros(2, 2, 7, 7)
    expected[0, 0] = 2
    expected[1, 0] = 5
    expected[0, 1] = 2
    expected[1, 1] = 0

    # print(actual.size())
    # print(expected.size())

    assert torch.equal(actual, expected)


test_copmute_level_ids()
test_roi_align()
