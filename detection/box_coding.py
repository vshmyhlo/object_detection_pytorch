import torch
import torchvision

from detection.box_utils import boxes_pairwise_iou, boxes_center, boxes_hw


def boxes_to_shifts_scales(boxes, anchors):
    shifts = (boxes_center(boxes) - boxes_center(anchors)) / boxes_hw(anchors)
    scales = boxes_hw(boxes) / boxes_hw(anchors)
    shifts_scales = torch.cat([shifts, scales.log()], -1)

    return shifts_scales


def shifts_scales_to_boxes(shifts_scales, anchors):
    shifts, scales = torch.split(shifts_scales, 2, -1)
    centers = shifts * boxes_hw(anchors) + boxes_center(anchors)
    sizes = scales.exp() * boxes_hw(anchors)
    boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], -1)

    return boxes


# TODO: rename encode/decode with anchors
def encode_boxes(input, anchors, min_iou, max_iou):
    class_ids, boxes = input

    if boxes.size(0) == 0:
        class_output = torch.zeros(anchors.size(0), dtype=torch.long)
        loc_output = torch.zeros(anchors.size(0), 4, dtype=torch.float)

        return class_output, loc_output

    ious = boxes_pairwise_iou(boxes, anchors)
    iou_values, iou_indices = ious.max(0)

    # build class_output
    class_output = class_ids[iou_indices] + 1
    class_output[iou_values < min_iou] = 0
    class_output[(min_iou <= iou_values) & (iou_values <= max_iou)] = -1

    # build loc_output
    boxes = boxes[iou_indices]
    loc_output = boxes

    return class_output, loc_output


def decode_boxes(input):
    class_output, loc_output = input

    scores, class_ids = class_output.max(1)
    fg = scores > 0.

    boxes = loc_output
    boxes = boxes[fg]
    class_ids = class_ids[fg]
    scores = scores[fg]

    keep = torchvision.ops.nms(boxes, scores, 0.5)
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]

    return class_ids, boxes, scores
