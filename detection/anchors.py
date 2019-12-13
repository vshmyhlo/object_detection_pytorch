import math

import torch


def build_anchors_maps(image_size, anchor_levels):
    h, w = image_size

    anchor_maps = []
    for anchors in anchor_levels:
        if anchors is not None:
            for anchor in anchors:
                anchor_map = build_anchor_map(image_size, (h, w), anchor)
                anchor_maps.append(anchor_map)

        h, w = math.ceil(h / 2), math.ceil(w / 2)

    anchor_maps = torch.cat(anchor_maps, 0)

    return anchor_maps


def build_anchor_map(image_size, map_size, anchor):
    cell_size = (image_size[0] / map_size[0], image_size[1] / map_size[1])

    y = torch.linspace(cell_size[0] / 2, image_size[0] - cell_size[0] / 2, map_size[0])
    x = torch.linspace(cell_size[1] / 2, image_size[1] - cell_size[1] / 2, map_size[1])

    y, x = torch.meshgrid(y, x)

    t = y - anchor[0] / 2
    l = x - anchor[1] / 2
    b = y + anchor[0] / 2
    r = x + anchor[1] / 2

    anchor_map = torch.stack([t, l, b, r], 2)
    anchor_map = anchor_map.view(anchor_map.size(0) * anchor_map.size(1), anchor_map.size(2))

    return anchor_map


def compute_anchor(size, ratio, scale):
    h = math.sqrt(size**2 / ratio) * scale
    w = h * ratio

    return h, w
