
import os
import cv2
import torch
import numpy as np


def save_tensor_image(tensor_C_HW, save_path, filename):
    """
    tensor_C_HW: torch.Tensor，形状为 (C, H, W)，数值类型float，数值无特定范围
    save_path: str，目标文件夹路径
    filename: str，保存文件名，带扩展名，如 "image.png"

    功能：
    - 归一化tensor数据到0-255
    - 若C==3，转换为BGR格式保存彩色图像
    - 若C==1，保存为灰度图
    - 其它通道数逐通道保存伪彩色单通道图像
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tensor = tensor_C_HW.cpu().clone()
    C, H, W = tensor.shape

    # 归一化
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max - t_min > 1e-5:
        tensor = (tensor - t_min) / (t_max - t_min)
    else:
        tensor = torch.zeros_like(tensor)

    # 转成numpy数组，形状 (C,H,W) -> (H,W,C)
    np_img = tensor.numpy().transpose(1, 2, 0)  # (H,W,C)
    np_img = (np_img * 255).astype(np.uint8)

    if C == 3:
        # 假设是RGB，转BGR保存（OpenCV默认BGR）
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, filename), np_img)

    elif C == 1:
        # 单通道灰度图
        cv2.imwrite(os.path.join(save_path, filename), np_img[:, :, 0])

    else:
        # 多通道非3，逐通道保存伪彩色图
        # 利用OpenCV去给每个通道应用jet彩色映射
        for i in range(C):
            single_channel = np_img[:, :, i]
            colored = cv2.applyColorMap(single_channel, cv2.COLORMAP_JET)
            save_name = f"{os.path.splitext(filename)[0]}_ch{i}.png"
            cv2.imwrite(os.path.join(save_path, save_name), colored)


if __name__ == "__main__":
    # 示例用法
    dummy_tensor = torch.rand(4, 100, 100)  # 4通道示例
    save_tensor_image(dummy_tensor, "./Output_MidFeat", "test_image.png")