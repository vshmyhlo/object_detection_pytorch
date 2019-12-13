from itertools import islice

import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw


class DataLoaderSlice(object):
    def __init__(self, data_loader, max_size):
        self.data_loader = data_loader
        self.max_size = max_size

    def __len__(self):
        return min(len(self.data_loader), self.max_size)

    def __iter__(self):
        return islice(self.data_loader, self.max_size)


def logit(input):
    return torch.log(input / (1 - input))


def draw_boxes(image, detections, class_names, line_width=2, shade=True):
    font = ImageFont.truetype('./data/Droid+Sans+Mono+Awesome.ttf', size=14)

    class_ids, boxes, scores = detections

    boxes = boxes.round().long()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image.size(1))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image.size(2))

    scores = scores.sigmoid()  # TODO: fixme

    device = image.device
    image = image.permute(1, 2, 0).data.cpu().numpy()

    if shade:
        mask = np.zeros_like(image, dtype=np.bool)
        for t, l, b, r in boxes.data.cpu().numpy():
            mask[t:b, l:r] = True
        image = np.where(mask, image, image * 0.5)

    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for c, (t, l, b, r), s in zip(class_ids.data.cpu().numpy(), boxes.data.cpu().numpy(), scores.data.cpu().numpy()):
        if len(class_names) > 1:
            colors = np.random.RandomState(42).uniform(85, 255, size=(len(class_names), 3)).round().astype(np.uint8)
            color = tuple(colors[c])
            text = '{}: {:.2f}'.format(class_names[c], s)
            size = draw.textsize(text, font=font)
            draw.rectangle(((l, t - size[1]), (l + size[0] + line_width * 2, t)), fill=color)
            draw.text((l + line_width, t - size[1]), text, font=font, fill=(0, 0, 0))
        else:
            color = (s - 0.5) / 0.5
            color = color * np.array([255, 85, 85]) + (1 - color) * np.array([85, 85, 255])
            color = tuple(color.round().astype(np.uint8))
        draw.rectangle(((l, t), (r, b)), outline=color, width=line_width)

    image = np.array(image) / 255
    image = torch.tensor(image).permute(2, 0, 1).to(device)

    return image
