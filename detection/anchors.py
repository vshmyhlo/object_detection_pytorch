import math

import torch


# TODO: check that network predicts same order of boxes
# TODO: rename all anchors maps to anchors
def arrange_anchors_on_grid(image_size, anchor_levels):
    h, w = image_size

    anchor_maps = []
    for anchors in anchor_levels:
        if anchors is not None:
            for anchor in anchors:
                anchor_map = arrange_anchor_on_grid(image_size, (h, w), anchor)
                anchor_maps.append(anchor_map)

        h, w = math.ceil(h / 2), math.ceil(w / 2)

    anchor_maps = torch.cat(anchor_maps, 0)

    return anchor_maps


# TODO: use helpers
def arrange_anchor_on_grid(image_size, map_size, anchor):
    cell_size = (image_size[0] / map_size[0], image_size[1] / map_size[1])

    y = torch.linspace(cell_size[0] / 2, image_size[0] - cell_size[0] / 2, map_size[0])
    x = torch.linspace(cell_size[1] / 2, image_size[1] - cell_size[1] / 2, map_size[1])
    y, x = torch.meshgrid(y, x)
    yx = torch.stack([y, x], -1)
    del y, x

    anchor = torch.tensor(anchor)
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
