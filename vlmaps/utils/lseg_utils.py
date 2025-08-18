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
        visual_semantic_picture(labels, vis_image, pred)

    return outputs

def visual_semantic_picture(labels, vis_image, pred):
    # 获取新的调色板和掩码
    from matplotlib import cm
    num_labels = len(labels)
        
        # 根据标签数量选择合适的调色板
    if num_labels <= 20:
        cmap = cm.get_cmap('tab20', num_labels)
    elif num_labels <= 30:
        cmap = cm.get_cmap('tab20', 20)  # 重复使用
            # 补充额外颜色
    else:
        cmap = cm.get_cmap('hsv', num_labels)  # 使用hsv提供更多颜色
            
        # 转换为 0-255 的整数列表格式
    new_palette = []
    for i in range(num_labels):
            # 处理颜色数量不足的情况
        color_idx = i % min(20, num_labels) if num_labels > 20 else i
        r, g, b, _ = [int(x * 255) for x in cmap(color_idx)]
        new_palette.extend([r, g, b])

        # 如果 palette 长度不足 256*3，需要补全（PIL 要求）
    while len(new_palette) < 256 * 3:
        new_palette.append(0)

    mask, _ = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
    seg = mask.convert("RGBA")

        # 创建图形和子图 - 增加图形高度以适应多列图例
    fig_height = max(12, 8 + num_labels * 0.3)  # 根据标签数量动态调整高度
    fig, axs = plt.subplots(2, 1, figsize=(14, fig_height),
                                 gridspec_kw={'hspace': 0.15, 'height_ratios': [3, 2]},
                                 subplot_kw=dict(xticks=[], yticks=[]))

        # 显示原始图像
    axs[1].imshow(vis_image)
    axs[1].set_title('Original Image', fontsize=14, pad=10)
    axs[1].axis('off')

        # 显示分割图像
    ax0 = axs[0]
    ax0.imshow(seg)
    ax0.set_title('Segmentation Result', fontsize=14, pad=10)
    ax0.axis('off')

        # 创建图例颜色（用于 matplotlib 显示）
    rgb_colors = []
    for i in range(num_labels):
        color_idx = i % min(20, num_labels) if num_labels > 20 else i
        r, g, b, _ = cmap(color_idx)
        rgb_colors.append((r, g, b))

        # 添加图例并编号 - 支持多列显示
    legend_elements = [Patch(facecolor=rgb_colors[i], label=f"{i}: {labels[i]}") for i in range(num_labels)]
        
        # 根据标签数量动态调整列数
    ncol = min(3, max(1, num_labels // 8 + 1))  # 最多3列，每列约8个标签
        
        # 计算图例位置和大小
    legend = ax0.legend(handles=legend_elements,
                           loc='center left',
                           bbox_to_anchor=(1.02, 0.5),
                           fontsize=9,
                           title="Labels",
                           title_fontsize=11,
                           frameon=True,
                           ncol=ncol,  # 多列显示
                           columnspacing=1.0,
                           handletextpad=0.5,
                           handlelength=1.0,
                           borderaxespad=0.5)

        # 设置图例背景和边框
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('gray')

        # 在每个mask区域中心添加编号
    for label_id in range(num_labels):
            # 提取当前label对应的mask区域
        mask_region = (pred == label_id).astype(np.uint8)
        if mask_region.sum() == 0:
            continue  # 没有该类别的mask

            # 计算质心坐标
        coords = np.where(mask_region > 0)
        cy, cx = int(coords[0].mean()), int(coords[1].mean())

            # 添加文本标注
        ax0.text(cx, cy, str(label_id),
                     color='white',
                     fontsize=11,
                     ha='center',
                     va='center',
                     weight='bold',
                     bbox=dict(boxstyle="circle,pad=0.2", facecolor='black', alpha=0.6))

        # 调整布局防止图例被裁剪
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # 留出右边空间给图例
    plt.show()
