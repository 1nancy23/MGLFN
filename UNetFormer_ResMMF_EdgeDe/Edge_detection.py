import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class EdgeDetectionModule(nn.Module):
    """
    边缘检测模块，支持多种边缘检测算法
    输入: batch*3*512*512
    输出: batch*1*512*512
    """

    def __init__(self, method='sobel', threshold_low=50, threshold_high=150):
        super(EdgeDetectionModule, self).__init__()
        self.method = method
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

        # 定义各种卷积核
        self.register_sobel_kernels()
        self.register_laplacian_kernel()
        self.register_prewitt_kernels()
        self.register_roberts_kernels()

    def register_sobel_kernels(self):
        """注册Sobel算子"""
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).cuda()
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).cuda()

        # 扩展维度以适应卷积操作 [out_channels, in_channels, H, W]
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))

    def register_laplacian_kernel(self):
        """注册拉普拉斯算子"""
        laplacian = torch.tensor([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], dtype=torch.float32).cuda()
        self.register_buffer('laplacian', laplacian.unsqueeze(0).unsqueeze(0))

    def register_prewitt_kernels(self):
        """注册Prewitt算子"""
        prewitt_x = torch.tensor([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]], dtype=torch.float32)
        prewitt_y = torch.tensor([[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]], dtype=torch.float32)

        self.register_buffer('prewitt_x', prewitt_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('prewitt_y', prewitt_y.unsqueeze(0).unsqueeze(0))

    def register_roberts_kernels(self):
        """注册Roberts算子"""
        roberts_x = torch.tensor([[1, 0],
                                  [0, -1]], dtype=torch.float32)
        roberts_y = torch.tensor([[0, 1],
                                  [-1, 0]], dtype=torch.float32)

        self.register_buffer('roberts_x', roberts_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('roberts_y', roberts_y.unsqueeze(0).unsqueeze(0))

    def rgb_to_grayscale(self, x):
        """将RGB图像转换为灰度图像"""
        # 使用标准的RGB到灰度转换权重
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
        weights = weights.view(1, 3, 1, 1)
        gray = torch.sum(x * weights, dim=1, keepdim=True)
        return gray

    def sobel_edge_detection(self, x):
        """Sobel边缘检测"""
        gray = self.rgb_to_grayscale(x)
        # print(gray.device,self.sobel_x.device)
        # 应用Sobel算子
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)

        # 计算梯度幅值
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        return edge_magnitude

    def laplacian_edge_detection(self, x):
        """拉普拉斯边缘检测"""
        gray = self.rgb_to_grayscale(x)

        # 应用拉普拉斯算子
        edges = F.conv2d(gray, self.laplacian, padding=1)
        edges = torch.abs(edges)

        return edges

    def prewitt_edge_detection(self, x):
        """Prewitt边缘检测"""
        gray = self.rgb_to_grayscale(x)

        # 应用Prewitt算子
        edge_x = F.conv2d(gray, self.prewitt_x, padding=1)
        edge_y = F.conv2d(gray, self.prewitt_y, padding=1)

        # 计算梯度幅值
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        return edge_magnitude

    def roberts_edge_detection(self, x):
        """Roberts边缘检测"""
        gray = self.rgb_to_grayscale(x)

        # Roberts算子使用2x2卷积核，所以不需要padding
        edge_x = F.conv2d(gray, self.roberts_x)
        edge_y = F.conv2d(gray, self.roberts_y)

        # 计算梯度幅值
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        # 由于Roberts算子会减少图像尺寸，需要插值回原尺寸
        edge_magnitude = F.interpolate(edge_magnitude, size=(512, 512), mode='bilinear', align_corners=False)

        return edge_magnitude

    def canny_edge_detection(self, x):
        """Canny边缘检测（使用OpenCV实现）"""
        batch_size = x.shape[0]
        edges_list = []

        # 将tensor转换为numpy进行处理
        x_np = x.detach().cpu().numpy()

        for i in range(batch_size):
            # 转换为uint8格式
            img = (x_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)

            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # 应用Canny边缘检测
            edges = cv2.Canny(gray, self.threshold_low, self.threshold_high)

            # 转换回tensor格式
            edges_tensor = torch.from_numpy(edges).float() / 255.0
            edges_list.append(edges_tensor.unsqueeze(0))

        # 合并batch
        edges_batch = torch.stack(edges_list, dim=0).to(x.device)

        return edges_batch

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入tensor，形状为 [batch, 3, 512, 512]
        Returns:
            edges: 边缘检测结果，形状为 [batch, 1, 512, 512]
        """
        if self.method == 'sobel':
            edges = self.sobel_edge_detection(x)
        elif self.method == 'laplacian':
            edges = self.laplacian_edge_detection(x)
        elif self.method == 'prewitt':
            edges = self.prewitt_edge_detection(x)
        elif self.method == 'roberts':
            edges = self.roberts_edge_detection(x)
        elif self.method == 'canny':
            edges = self.canny_edge_detection(x)
        else:
            raise ValueError(f"Unsupported edge detection method: {self.method}")

        # 归一化到[0, 1]范围
        edges = torch.clamp(edges, 0, 1)

        return edges


# 高级边缘检测模块（可学习参数）
class LearnableEdgeDetection(nn.Module):
    """
    可学习的边缘检测模块
    """

    def __init__(self, in_channels=3, out_channels=1):
        super(LearnableEdgeDetection, self).__init__()

        # 可学习的边缘检测卷积层
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            # nn.Sigmoid()  # 输出范围[0, 1]
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入tensor，形状为 [batch, 3, 512, 512]
        Returns:
            edges: 边缘检测结果，形状为 [batch, 1, 512, 512]
        """
        return self.edge_conv(x)


# 混合边缘检测模块
class HybridEdgeDetection(nn.Module):
    """
    混合边缘检测模块，结合传统方法和深度学习方法
    """

    def __init__(self, methods=['sobel', 'laplacian'], weights=None):
        super(HybridEdgeDetection, self).__init__()

        self.methods = methods
        self.edge_detectors = nn.ModuleDict()

        # 创建各种边缘检测器
        for method in methods:
            self.edge_detectors[method] = EdgeDetectionModule(method=method)

        # 设置权重
        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(methods)) / len(methods))
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

        # 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(len(methods), 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入tensor，形状为 [batch, 3, 512, 512]
        Returns:
            edges: 边缘检测结果，形状为 [batch, 1, 512, 512]
        """
        edge_maps = []
        x_t = x[0].permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite(f"./MideView/Ori.png",x_t*225.0)
        # 应用各种边缘检测方法
        for i, method in enumerate(self.methods):
            edge_map = self.edge_detectors[method](x)
            edge_map_t = edge_map[0,0].detach().cpu().numpy()
            # cv2.imshow("ccc",edge_map_t)
            # cv2.imwrite(f"./MideView/No.{i}.png",edge_map_t*225.0)
            # cv2.waitKey()
            # cv2.close()
            edge_maps.append(edge_map * self.weights[i])
        # plt.imshow(edge_map_t)
        # plt.show()
        # 连接所有边缘图
        combined_edges = torch.cat(edge_maps, dim=1)

        # 通过融合层得到最终结果
        final_edges = self.fusion_conv(combined_edges)
        final_edges_t = final_edges[0,0].detach().cpu().numpy()
        # cv2.imwrite(f"./MideView/After_{time.time()}.png",final_edges_t*225.0)
        return final_edges


# 使用示例
if __name__ == "__main__":
    # 创建示例输入
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 128, 128)

    print("输入形状:", input_tensor.shape)

    # 1. 传统边缘检测方法
    print("\n=== 传统边缘检测方法 ===")
    methods = ['sobel', 'laplacian', 'prewitt', 'roberts']

    for method in methods:
        edge_detector = EdgeDetectionModule(method=method)
        edges = edge_detector(input_tensor)
        print(f"{method.capitalize()} 边缘检测输出形状:", edges.shape)

    # 2. Canny边缘检测（需要安装opencv-python）
    try:
        canny_detector = EdgeDetectionModule(method='canny', threshold_low=50, threshold_high=150)
        canny_edges = canny_detector(input_tensor)
        print("Canny 边缘检测输出形状:", canny_edges.shape)
    except Exception as e:
        print("Canny边缘检测需要安装opencv-python:", str(e))

    # 3. 可学习边缘检测
    print("\n=== 可学习边缘检测 ===")
    learnable_detector = LearnableEdgeDetection()
    learnable_edges = learnable_detector(input_tensor)
    print("可学习边缘检测输出形状:", learnable_edges.shape)

    # 4. 混合边缘检测
    print("\n=== 混合边缘检测 ===")
    hybrid_detector = HybridEdgeDetection(methods=['sobel', 'laplacian'])
    hybrid_edges = hybrid_detector(input_tensor)
    print(hybrid_edges)
    print("混合边缘检测输出形状:", hybrid_edges.shape)


    # 计算参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f"\n可学习边缘检测参数量: {count_parameters(learnable_detector):,}")
    print(f"混合边缘检测参数量: {count_parameters(hybrid_detector):,}")