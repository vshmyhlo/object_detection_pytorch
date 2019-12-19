import torch

from detection.anchor_utils import arrange_anchor_on_grid, arrange_anchors_on_grid


def test_arange_anchors_on_grid():
    anchors = [
        None,
        None,
        None,
        None,
        [torch.tensor((16, 16)), torch.tensor((20, 20))],
    ]

    actual = arrange_anchors_on_grid(
        torch.tensor((32, 32)),
        anchors)

    expected = torch.tensor([
        [0, 0, 16, 16],
        [-2, -2, 18, 18],
        [0, 16, 16, 32],
        [-2, 14, 18, 34],
        [16, 0, 32, 16],
        [14, -2, 34, 18],
        [16, 16, 32, 32],
        [14, 14, 34, 34],
    ], dtype=torch.float)

    assert torch.equal(actual, expected)


def test_arange_anchor_on_grid():
    anchor = torch.tensor((20, 20))

    actual = arrange_anchor_on_grid(
        torch.tensor((32, 32)),
        torch.tensor((2, 2)),
        anchor)

    expected = torch.tensor([
        [-2, -2, 18, 18],
        [-2, 14, 18, 34],
        [14, -2, 34, 18],
        [14, 14, 34, 34],
    ], dtype=torch.float).t().view(4, 2, 2)

    assert torch.equal(actual, expected)
