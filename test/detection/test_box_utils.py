import torch

from detection.box_utils import (
    boxes_area,
    boxes_center,
    boxes_clip,
    boxes_pairwise_iou,
    boxes_size,
)


def test_boxes_center():
    boxes = torch.tensor(
        [
            [0.1, 0.2, 0.2, 0.4],
            [10, 20, 20, 40],
            [0, 0, 5, 10],
        ],
        dtype=torch.float,
    )

    actual = boxes_center(boxes)

    expected = torch.tensor(
        [
            [0.15, 0.3],
            [15, 30],
            [2.5, 5],
        ],
        dtype=torch.float,
    )

    assert torch.allclose(actual, expected)


def test_boxes_size():
    boxes = torch.tensor(
        [
            [0.1, 0.2, 0.2, 0.4],
            [10, 20, 20, 40],
            [0, 0, 5, 10],
        ],
        dtype=torch.float,
    )

    actual = boxes_size(boxes)

    expected = torch.tensor(
        [
            [0.1, 0.2],
            [10, 20],
            [5, 10],
        ],
        dtype=torch.float,
    )

    assert torch.allclose(actual, expected)


def test_boxes_iou():
    a = torch.tensor(
        [
            [15, 20, 25, 40],
            [30, 20, 40, 40],
        ],
        dtype=torch.float,
    )
    b = torch.tensor(
        [
            [10, 20, 20, 40],
        ],
        dtype=torch.float,
    )

    actual = boxes_pairwise_iou(a, b)
    expected = torch.tensor(
        [
            [1 / 3],
            [0.0],
        ],
        dtype=torch.float,
    )

    assert torch.allclose(actual, expected)


def test_boxes_area():
    boxes = torch.tensor(
        [
            [0.1, 0.2, 0.2, 0.4],
            [10, 20, 20, 40],
            [0, 0, 5, 10],
        ],
        dtype=torch.float,
    )

    actual = boxes_area(boxes)
    expected = torch.tensor(
        [
            0.02,
            200,
            50,
        ],
        dtype=torch.float,
    )

    assert torch.allclose(actual, expected)


def test_boxes_clip():
    boxes = torch.tensor(
        [
            [-10, 5, 20, 30],
            [10, 10, 200, 300],
        ]
    )

    actual = boxes_clip(boxes, (100, 150))
    expected = torch.tensor(
        [
            [0, 5, 20, 30],
            [10, 10, 100, 150],
        ]
    )

    assert torch.equal(actual, expected)
