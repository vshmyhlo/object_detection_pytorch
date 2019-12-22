import torch

from detection.map import match_to_precision_recall, pr_auc


def test_pr_auc():
    pr = torch.tensor([
        [0.5, 0.25],
        [1, 0.75],
    ])

    auc = pr_auc(pr)

    assert auc == 0.375
   

def test_match_to_precision_recall():
    match = torch.tensor([
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
    ])

    actual = match_to_precision_recall(match, num_true=5)

    expected = torch.tensor([
        [1., 0.2],
        [1., 0.4],
        [0.67, 0.4],
        [0.5, 0.4],
        [0.4, 0.4],
        [0.5, 0.6],
        [0.57, 0.8],
        [0.5, 0.8],
        [0.44, 0.8],
        [0.5, 1.],
    ])

    assert torch.allclose(actual, expected, atol=1e-2)
