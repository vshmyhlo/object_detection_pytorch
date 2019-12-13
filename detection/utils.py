import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw


def logit(input):
    return torch.log(input / (1 - input))


def draw_boxes(image, detections, class_names, line_width=2, shade=True):
    font = ImageFont.truetype('./imet/Droid+Sans+Mono+Awesome.ttf', size=14)

    class_ids, boxes, scores = detections
    scores = scores.sigmoid()  # TODO: fixme

    device = image.device
    image = image.permute(1, 2, 0).data.cpu().numpy()

    if shade:
        mask = np.zeros_like(image, dtype=np.bool)
        boxes_clipped = boxes.clone()
        boxes_clipped[:, [0, 2]] = boxes_clipped[:, [0, 2]].clamp(0, mask.shape[0])
        boxes_clipped[:, [1, 3]] = boxes_clipped[:, [1, 3]].clamp(0, mask.shape[1])
        for t, l, b, r in boxes_clipped.data.cpu().numpy().round().astype(np.int32):
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
