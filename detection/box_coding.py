import torch
import torchvision

from detection.box_utils import boxes_iou, boxes_center, boxes_size


def boxes_to_shifts_scales(boxes, anchors):
    shifts = (boxes_center(boxes) - boxes_center(anchors)) / boxes_size(anchors)
    scales = boxes_size(boxes) / boxes_size(anchors)
    shifts_scales = torch.cat([shifts, scales.log()], 1)

    return shifts_scales


def shifts_scales_to_boxes(shifts_scales, anchors):
    shifts, scales = torch.split(shifts_scales, 2, 1)
    centers = shifts * boxes_size(anchors) + boxes_center(anchors)
    sizes = scales.exp() * boxes_size(anchors)
    boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], 1)

    return boxes


def encode_boxes(input, anchors, min_iou, max_iou):
    class_ids, boxes = input

    if boxes.size(0) == 0:
        class_output = torch.zeros(anchors.size(0), dtype=torch.long)
        regr_output = torch.zeros(anchors.size(0), 4, dtype=torch.float)

        return class_output, regr_output

    ious = boxes_iou(boxes, anchors)
    iou_values, iou_indices = ious.max(0)

    # build class_output
    class_output = class_ids[iou_indices] + 1
    class_output[iou_values < min_iou] = 0
    class_output[(min_iou <= iou_values) & (iou_values <= max_iou)] = -1

    # build regr_output
    boxes = boxes[iou_indices]

    regr_output = boxes_to_shifts_scales(boxes, anchors)

    return class_output, regr_output


def decode_boxes(input, anchors):
    class_output, regr_output = input

    scores, class_ids = class_output.max(1)
    fg = scores > 0.

    boxes = shifts_scales_to_boxes(regr_output, anchors)

    boxes = boxes[fg]
    class_ids = class_ids[fg]
    scores = scores[fg]

    keep = torchvision.ops.nms(boxes, scores, 0.5)
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]

    return class_ids, boxes, scores
