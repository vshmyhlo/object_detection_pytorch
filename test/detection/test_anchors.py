import torch

from detection.anchors import arrange_anchor_on_grid


def test_arange_anchor_on_grid():
    actual = arrange_anchor_on_grid((60, 60), (2, 2), (40, 40))
    expected = torch.tensor([
        [-5, -5, 35, 35],
        [-5, 25, 35, 65],
        [25, -5, 65, 35],
        [25, 25, 65, 65],
    ], dtype=torch.float)

    assert torch.equal(actual, expected)
