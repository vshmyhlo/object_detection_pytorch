import math

import torch


# TODO: test
# TODO: check that network predicts same order of boxes
# TODO: rename all anchors maps to anchors
def arrange_anchors_on_grid(image_size, anchor_levels):
    image_size = torch.tensor(image_size)
    map_size = torch.tensor(image_size)

    anchor_maps = []
    for anchors in anchor_levels:
        if anchors is not None:
            for anchor in anchors:
                anchor_map = arrange_anchor_on_grid(image_size, map_size, anchor)
                anchor_maps.append(anchor_map)

        map_size = torch.ceil(map_size.float() / 2).long()

    anchor_maps = torch.cat(anchor_maps, 0)

    return anchor_maps


# TODO: use helpers
def arrange_anchor_on_grid(image_size, map_size, anchor):
    image_size = torch.tensor(image_size)
    map_size = torch.tensor(map_size)
    anchor = torch.tensor(anchor)

    cell_size = image_size / map_size

    y = torch.linspace(cell_size[0] / 2, image_size[0] - cell_size[0] / 2, map_size[0])
    x = torch.linspace(cell_size[1] / 2, image_size[1] - cell_size[1] / 2, map_size[1])
    y, x = torch.meshgrid(y, x)
    yx = torch.stack([y, x], -1)
    del y, x

    tl = yx - anchor / 2
    br = yx + anchor / 2

    anchor_map = torch.cat([tl, br], -1)
    anchor_map = anchor_map.view(anchor_map.size(0) * anchor_map.size(1), anchor_map.size(2))

    return anchor_map


# TODO: torch
def compute_anchor(size, ratio, scale):
    h = math.sqrt(size**2 / ratio) * scale
    w = h * ratio

    return h, w
