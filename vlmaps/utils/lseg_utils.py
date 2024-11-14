import math

import numpy as np
import cv2
import torch

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from vlmaps.utils.mapping_utils import *

from vlmaps.lseg.modules.models.lseg_net import LSegEncNet
from vlmaps.lseg.additional_utils.models import resize_image, pad_image, crop_image

def get_lseg_feat(
    model: LSegEncNet,
    image: np.array,
    labels,
    transform,
    device,
    crop_size=480,
    base_size=520,
    norm_mean=[0.5, 0.5, 0.5],
    norm_std=[0.5, 0.5, 0.5],
    vis=False,
):
    # 复制图像以便后续可视化
    vis_image = image.copy()

    # 对图像进行预处理和转换
    image = transform(image).unsqueeze(0).to(device)
    img = image[0].permute(1, 2, 0)
    img = img * 0.5 + 0.5

    # 获取图像尺寸和步长
    batch, _, h, w = image.size()
    stride_rate = 2.0 / 3.0
    stride = int(crop_size * stride_rate)

    # 设置长边尺寸
    # long_size = int(math.ceil(base_size * scale))
    long_size = base_size

    # 根据图像长宽比计算短边尺寸
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    # 调整图像尺寸
    cur_img = resize_image(image, height, width, **{"mode": "bilinear", "align_corners": True})

    # 处理图像尺寸小于等于裁剪尺寸的情况
    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        print(pad_img.shape)
        with torch.no_grad():
            # 获取模型输出
            # outputs = model(pad_img)
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)

    else:
        # 处理图像尺寸大于裁剪尺寸的情况
        if short_size < crop_size:
            # 如有需要则进行填充
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
        _, _, ph, pw = pad_img.shape  # .size()
        assert ph >= height and pw >= width
        h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                # 初始化输出和逻辑输出
                outputs = image.new().resize_(batch, model.out_c, ph, pw).zero_().to(device)
                logits_outputs = image.new().resize_(batch, len(labels), ph, pw).zero_().to(device)
                count_norm = image.new().resize_(batch, 1, ph, pw).zero_().to(device)
            # 网格评估
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + crop_size, ph)
                    w1 = min(w0 + crop_size, pw)
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    # 如有需要则进行填充
                    pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                    with torch.no_grad():
                        # 获取裁剪图像的模型输出
                        # output = model(pad_crop_img)
                        output, logits = model(pad_crop_img, labels)
                    cropped = crop_image(output, 0, h1 - h0, 0, w1 - w0)
                    cropped_logits = crop_image(logits, 0, h1 - h0, 0, w1 - w0)
                    outputs[:, :, h0:h1, w0:w1] += cropped
                    logits_outputs[:, :, h0:h1, w0:w1] += cropped_logits
                    count_norm[:, :, h0:h1, w0:w1] += 1
            assert (count_norm == 0).sum() == 0
            outputs = outputs / count_norm
            logits_outputs = logits_outputs / count_norm
            outputs = outputs[:, :, :height, :width]
            logits_outputs = logits_outputs[:, :, :height, :width]

    # 将输出转换为NumPy数组
    outputs = outputs.cpu().numpy()  # B, D, H, W
    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
    pred = predicts[0]

    if vis:
        # 获取新的调色板和掩码
        new_palette = get_new_pallete(len(labels))
        mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
        seg = mask.convert("RGBA")

        # 创建图形和子图
        fig, axs = plt.subplots(2, 1, figsize=(12, 12),
                                 gridspec_kw={'hspace': 0.1},  # 减少垂直间距
                                 subplot_kw=dict(xticks=[], yticks=[]))
        # 展示原始图像
        axs[1].imshow(vis_image)
        axs[1].set_title('original image')
        axs[1].axis('off')  # 关闭坐标轴

        # 展示分割图像
        axs[0].imshow(seg)
        axs[0].set_title('split image')
        axs[0].axis('off')  # 关闭坐标轴

        # 创建图例颜色
        rgb_colors = [(new_palette[i] / 255.0, new_palette[i + 1] / 255.0, new_palette[i + 2] / 255.0) for i in range(0, len(new_palette) - 2, 3)]

        # 添加图例
        legend_elements = [Patch(facecolor=rgb_colors[label_id], label=label) for label_id, label in enumerate(labels)]
        axs[0].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 13})

        # 调整布局并显示图像
        plt.tight_layout()
        plt.show()

    return outputs
