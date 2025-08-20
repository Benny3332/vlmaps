import math

import numpy as np
import cv2
import torch
import os
from scipy.ndimage import label
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm

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
    is_save=False,
    save_path=None,
    file_name=None
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
    # stride_rate = 0.5
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
        num_labels = len(labels)
        new_palette, rgb_colors = generate_palette(num_labels)
        visual_semantic_picture(labels, vis_image, pred, new_palette, rgb_colors)

    if is_save and save_path and file_name:
        num_labels = len(labels)
        new_palette, rgb_colors = generate_palette(num_labels)
        save_semantic_picture(labels, vis_image, pred, new_palette, rgb_colors, save_path, file_name)
    return outputs

def generate_palette(num_labels):
    if num_labels <= 20:
        cmap = cm.get_cmap('tab20', num_labels)
    elif num_labels <= 30:
        cmap = cm.get_cmap('tab20', 20)
    else:
        cmap = cm.get_cmap('hsv', num_labels)

    new_palette = []
    rgb_colors = []
    for i in range(num_labels):
        color_idx = i % min(20, num_labels) if num_labels > 20 else i
        r, g, b, _ = [int(x * 255) for x in cmap(color_idx)]
        new_palette.extend([r, g, b])
        rgb_colors.append((r / 255.0, g / 255.0, b / 255.0))

    while len(new_palette) < 256 * 3:
        new_palette.append(0)

    return new_palette, rgb_colors

def save_semantic_picture(labels, vis_image, pred, new_palette, rgb_colors, save_path, file_name):
    num_labels = len(labels)
    mask, _ = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
    seg = mask.convert("RGBA")

    fig_height = max(12, 8 + num_labels * 0.3)
    fig, axs = plt.subplots(2, 1, figsize=(14, fig_height),
                           gridspec_kw={'hspace': 0.15, 'height_ratios': [3, 2]},
                           subplot_kw=dict(xticks=[], yticks=[]))

    axs[1].imshow(vis_image)
    axs[1].set_title('Original Image', fontsize=14, pad=10)
    axs[1].axis('off')

    ax0 = axs[0]
    ax0.imshow(seg)
    ax0.set_title('Segmentation Result', fontsize=14, pad=10)
    ax0.axis('off')

    legend_elements = [Patch(facecolor=rgb_colors[i], label=f"{i}: {labels[i]}") for i in range(num_labels)]
    ncol = min(3, max(1, num_labels // 8 + 1))

    legend = ax0.legend(handles=legend_elements,
                       loc='center left',
                       bbox_to_anchor=(1.02, 0.5),
                       fontsize=9,
                       title="Labels",
                       title_fontsize=11,
                       frameon=True,
                       ncol=ncol,
                       columnspacing=1.0,
                       handletextpad=0.5,
                       handlelength=1.0,
                       borderaxespad=0.5)

    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('gray')

    for label_id in range(num_labels):
        mask_region = (pred == label_id).astype(np.uint8)
        if mask_region.sum() == 0:
            continue
        # Find connected components
        labeled_array, num_features = label(mask_region)
        for region_id in range(1, num_features + 1):
            region = (labeled_array == region_id).astype(np.uint8)
            if region.sum() < 100:  # Ignore small regions (noise)
                continue
            coords = np.where(region > 0)
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            ax0.text(cx, cy, str(label_id),
                     color='white',
                     fontsize=11,
                     ha='center',
                     va='center',
                     weight='bold',
                     bbox=dict(boxstyle="circle,pad=0.2", facecolor='black', alpha=0.6))

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    save_file = os.path.join(save_path, file_name)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close(fig)


def visual_semantic_picture(labels, vis_image, pred, new_palette, rgb_colors):
    num_labels = len(labels)
    mask, _ = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
    seg = mask.convert("RGBA")

    fig_height = max(12, 8 + num_labels * 0.3)
    fig, axs = plt.subplots(2, 1, figsize=(14, fig_height),
                           gridspec_kw={'hspace': 0.15, 'height_ratios': [3, 2]},
                           subplot_kw=dict(xticks=[], yticks=[]))

    axs[1].imshow(vis_image)
    axs[1].set_title('Original Image', fontsize=14, pad=10)
    axs[1].axis('off')

    ax0 = axs[0]
    ax0.imshow(seg)
    ax0.set_title('Segmentation Result', fontsize=14, pad=10)
    ax0.axis('off')

    legend_elements = [Patch(facecolor=rgb_colors[i], label=f"{i}: {labels[i]}") for i in range(num_labels)]
    ncol = min(3, max(1, num_labels // 8 + 1))

    legend = ax0.legend(handles=legend_elements,
                       loc='center left',
                       bbox_to_anchor=(1.02, 0.5),
                       fontsize=9,
                       title="Labels",
                       title_fontsize=11,
                       frameon=True,
                       ncol=ncol,
                       columnspacing=1.0,
                       handletextpad=0.5,
                       handlelength=1.0,
                       borderaxespad=0.5)

    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('gray')

    for label_id in range(num_labels):
        mask_region = (pred == label_id).astype(np.uint8)
        if mask_region.sum() == 0:
            continue
        # Find connected components
        labeled_array, num_features = label(mask_region)
        for region_id in range(1, num_features + 1):
            region = (labeled_array == region_id).astype(np.uint8)
            if region.sum() < 100:  # Ignore small regions (noise)
                continue
            coords = np.where(region > 0)
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            ax0.text(cx, cy, str(label_id),
                     color='white',
                     fontsize=11,
                     ha='center',
                     va='center',
                     weight='bold',
                     bbox=dict(boxstyle="circle,pad=0.2", facecolor='black', alpha=0.6))

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
