import numpy as np
import pytest
import torch
from PIL import Image

from detection.transform import crop, flip_left_right, resize


@pytest.fixture()
def input():
    return {
        'image': Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)),
        'class_ids': torch.tensor([0], dtype=torch.long),
        'boxes': torch.tensor([
            [10, 20, 20, 40],
        ], dtype=torch.float),
    }


def test_flip_left_right(input):
    actual = flip_left_right(input)

    expected = {
        'class_ids': torch.tensor([0], dtype=torch.long),
        'boxes': torch.tensor([
            [10, 160, 20, 180],
        ], dtype=torch.float),
    }

    assert torch.allclose(actual['class_ids'], expected['class_ids'])
    assert torch.allclose(actual['boxes'], expected['boxes'])


def test_resize(input):
    actual = resize(input, 50)
    expected = {
        'class_ids': torch.tensor([0], dtype=torch.long),
        'boxes': torch.tensor([
            [5, 10, 10, 20],
        ], dtype=torch.float),
    }

    assert actual['image'].size == (100, 50)
    assert torch.allclose(actual['class_ids'], expected['class_ids'])
    assert torch.allclose(actual['boxes'], expected['boxes'])


def test_crop(input):
    actual = crop(input, (5, 30), (20, 20), min_size=0)

    expected = {
        'class_ids': torch.tensor([0], dtype=torch.long),
        'boxes': torch.tensor([
            [5, 0, 15, 10],
        ], dtype=torch.float),
    }

    assert actual['image'].size == (20, 20)
    assert torch.allclose(actual['class_ids'], expected['class_ids'])
    assert torch.allclose(actual['boxes'], expected['boxes'])
