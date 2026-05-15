import torch

import SEKLoss
from UNetFormer_ResMMF_EdgeDe.Edge_detection import *
from UNetFormer_ResMMF_EdgeDe.Network import UNetFormer
import sys
sys.path.append("D:\ZJF\种植作物分类\ZhongzhiCodes\XiuGai1")
from sklearn.metrics import precision_recall_curve, roc_curve, auc,confusion_matrix
from torch.nn import CrossEntropyLoss
from scipy.ndimage import distance_transform_edt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import seaborn as sns

# 导入半精度训练相关模块
from torch.cuda.amp import GradScaler, autocast

# 导入之前定义的数据加载相关代码
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from Datain import *
from XiuGai1.HSUNet import HSUNet
# 设置全局字体大小和样式
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Dice损失函数实现
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6, ignore_index=255):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#
#     def forward(self, inputs, targets):
#         """
#         计算Dice损失
#         Args:
#             inputs: 模型输出 [B, C, H, W]
#             targets: 真实标签 [B, H, W]
#         """
#         # 将输入转换为概率
#         inputs = torch.softmax(inputs, dim=1)
#
#         # 创建有效掩码（排除ignore_index）
#         valid_mask = (targets != self.ignore_index)
#
#         # 将targets转换为one-hot编码
#         num_classes = inputs.shape[1]
#         targets_one_hot = torch.zeros_like(inputs)
#
#         # 只对有效像素进行one-hot编码
#         valid_targets = targets * valid_mask.long()
#         targets_one_hot.scatter_(1, valid_targets.unsqueeze(1), 1)
#
#         # 应用有效掩码
#         valid_mask = valid_mask.unsqueeze(1).float()
#         inputs = inputs * valid_mask
#         targets_one_hot = targets_one_hot * valid_mask
#
#         # 计算每个类别的Dice系数
#         intersection = torch.sum(inputs * targets_one_hot, dim=(2, 3))
#         union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
#
#         dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
#
#         # 返回Dice损失（1 - Dice系数的平均值）
#         return 1.0 - dice.mean()


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, ignore_index=255):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
#
#     def forward(self, inputs, targets):
#         ce_loss = self.ce_loss(inputs, targets)
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#         return focal_loss.mean()


# class CombinedLoss(nn.Module):
#     def __init__(self, dice_weight=0.7, focal_weight=0.4, smooth=1e-6, ignore_index=255):
#         super(CombinedLoss, self).__init__()
#         self.dice_weight = dice_weight
#         self.focal_weight = focal_weight
#         self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)
#         self.focal_loss = FocalLoss(ignore_index=ignore_index)
#
#     def forward(self, inputs, targets):
#         dice_loss = self.dice_loss(inputs, targets)
#         focal_loss = self.focal_loss(inputs, targets)
#
#         combined_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
#         return combined_loss, dice_loss, focal_loss


# class MultiOutputDiceLoss(nn.Module):
#     def __init__(self, aux_weight=0.3, dice_weight=0.6, focal_weight=0.4, ignore_index=255, use_combined=True):
#         super(MultiOutputDiceLoss, self).__init__()
#         self.aux_weight = aux_weight
#         self.use_combined = use_combined
#
#         if use_combined:
#             self.main_criterion = CombinedLoss(
#                 dice_weight=dice_weight,
#                 focal_weight=focal_weight,
#                 ignore_index=ignore_index
#             )
#             self.aux_criterion = CombinedLoss(
#                 dice_weight=dice_weight,
#                 focal_weight=focal_weight,
#                 ignore_index=ignore_index
#             )
#         else:
#             # 纯Dice损失
#             self.main_criterion = DiceLoss(ignore_index=ignore_index)
#             self.aux_criterion = DiceLoss(ignore_index=ignore_index)
#
#     def forward(self, outputs, targets):
#         """
#         处理模型的多输出结构
#         Args:
#             outputs: tuple (main_output, aux_output) 或 单个输出
#             targets: 真实标签
#         """
#         if isinstance(outputs, tuple):
#             main_output, aux_output = outputs
#
#             if self.use_combined:
#                 # 使用组合损失
#                 main_loss, main_dice, main_focal = self.main_criterion(main_output, targets)
#                 aux_loss, aux_dice, aux_focal = self.aux_criterion(aux_output, targets)
#                 total_loss = main_loss + self.aux_weight * aux_loss
#                 return total_loss, main_loss, aux_loss, main_dice, main_focal, aux_dice, aux_focal
#             else:
#                 # 使用纯Dice损失
#                 main_loss = self.main_criterion(main_output, targets)
#                 aux_loss = self.aux_criterion(aux_output, targets)
#                 total_loss = main_loss + self.aux_weight * aux_loss
#                 return total_loss, main_loss, aux_loss
#         else:
#             # 如果只有单一输出
#             if self.use_combined:
#                 main_loss, main_dice, main_focal = self.main_criterion(outputs, targets)
#                 return (main_loss, main_loss, torch.tensor(0.0),
#                         main_dice, main_focal, torch.tensor(0.0), torch.tensor(0.0))
#             else:
#                 main_loss = self.main_criterion(outputs, targets)
#                 return main_loss, main_loss, torch.tensor(0.0)
class DiceLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.Soft=nn.Softmax(dim=1)
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) 预测logits
            target: (B, H, W) 真实标签
        """
        num_classes = pred.shape[1]
        pred = self.Soft(pred)

        # 创建有效mask
        valid_mask = (target != self.ignore_index)

        dice_loss = 0.0
        for c in range(num_classes):
            pred_c = pred[:, c, :, :]
            target_c = (target == c).float()

            # 应用有效mask
            pred_c = pred_c * valid_mask.float()
            target_c = target_c * valid_mask.float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice)

        return dice_loss / num_classes


class mIoULoss(nn.Module):
    """mIoU Loss (基于Soft IoU)"""

    def __init__(self, smooth=1.0, ignore_index=255):
        super(mIoULoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.Soft=nn.Softmax(dim=1)
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) 预测logits
            target: (B, H, W) 真实标签
        """
        num_classes = pred.shape[1]
        pred = self.Soft(pred)

        # 创建有效mask
        valid_mask = (target != self.ignore_index)

        iou_loss = 0.0
        for c in range(num_classes):
            pred_c = pred[:, c, :, :]
            target_c = (target == c).float()

            # 应用有效mask
            pred_c = pred_c * valid_mask.float()
            target_c = target_c * valid_mask.float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection

            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_loss += (1 - iou)

        return iou_loss / num_classes


class DistanceMapCrossEntropyLoss(nn.Module):
    """基于距离图的加权交叉熵损失"""

    def __init__(self, alpha=1.0, ignore_index=255):
        super(DistanceMapCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def compute_distance_map(self, target):
        """
        计算距离图：离边界越近权重越大
        Args:
            target: (B, H, W) 标签图
        Returns:
            distance_map: (B, H, W) 距离权重图
        """
        batch_size = target.shape[0]
        distance_maps = []

        for b in range(batch_size):
            target_np = target[b].cpu().numpy()

            # 计算边界
            edges = np.zeros_like(target_np, dtype=np.float32)
            for c in range(target_np.min(), target_np.max() + 1):
                if c == self.ignore_index:
                    continue
                mask = (target_np == c).astype(np.uint8)
                # 使用形态学操作检测边界
                from scipy.ndimage import binary_erosion
                eroded = binary_erosion(mask, iterations=1)
                edge = mask - eroded
                edges += edge

            # 计算到边界的距离
            if edges.sum() > 0:
                dist_map = distance_transform_edt(1 - edges)
                # 归一化并反转：边界处权重大
                dist_map = 1.0 / (1.0 + dist_map * 0.1)
            else:
                dist_map = np.ones_like(target_np, dtype=np.float32)

            distance_maps.append(torch.from_numpy(dist_map))

        distance_maps = torch.stack(distance_maps).to(target.device)
        return distance_maps

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) 预测logits
            target: (B, H, W) 真实标签
        """
        # 计算距离权重图
        distance_map = self.compute_distance_map(target)

        # 计算交叉熵损失（逐像素）
        ce_loss = self.ce_loss(pred, target)

        # 应用距离权重
        weighted_loss = ce_loss * distance_map

        # 只对有效像素求平均
        valid_mask = (target != self.ignore_index)
        loss = weighted_loss[valid_mask].mean()

        return loss * self.alpha


class CombinedLoss(nn.Module):
    """组合损失：Dice + mIoU + Distance-based CE"""

    def __init__(self, dice_weight=0.7, miou_weight=0.3, dce_weight=0.3,
                 dce_alpha=1.0, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.miou_weight = miou_weight
        self.dce_weight = dce_weight

        self.dice_loss = CrossEntropyLoss(label_smoothing=0.1)
        self.miou_loss = mIoULoss(ignore_index=ignore_index)
        self.dce_loss = DistanceMapCrossEntropyLoss(alpha=dce_alpha, ignore_index=ignore_index)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) 预测logits
            target: (B, H, W) 真实标签
        Returns:
            total_loss, dice_loss, miou_loss, dce_loss
        """
        dice = self.dice_loss(pred, target)
        miou = self.miou_loss(pred, target)
        dce = self.dce_loss(pred, target)

        total_loss = (self.dice_weight * dice +
                      self.miou_weight * miou +
                      self.dce_weight * dce)

        return total_loss, dice, miou, dce


class MultiOutputDiceLoss(nn.Module):
    """多输出损失函数（修改版）"""

    def __init__(self, aux_weight=0.3, dice_weight=0.4, miou_weight=0.3,
                 dce_weight=0.3, dce_alpha=1.0, ignore_index=255, use_combined=True):
        super(MultiOutputDiceLoss, self).__init__()
        self.aux_weight = aux_weight
        self.use_combined = use_combined

        if use_combined:
            self.main_criterion = CombinedLoss(
                dice_weight=dice_weight,
                miou_weight=miou_weight,
                dce_weight=dce_weight,
                dce_alpha=dce_alpha,
                ignore_index=ignore_index
            )
            self.aux_criterion = CombinedLoss(
                dice_weight=dice_weight,
                miou_weight=miou_weight,
                dce_weight=dce_weight,
                dce_alpha=dce_alpha,
                ignore_index=ignore_index
            )
        else:
            # 纯Dice损失
            self.main_criterion = DiceLoss(ignore_index=ignore_index)
            self.aux_criterion = DiceLoss(ignore_index=ignore_index)

    def forward(self, outputs, targets):
        """
        处理模型的多输出结构
        Args:
            outputs: tuple (main_output, aux_output) 或 单个输出
            targets: 真实标签
        Returns:
            use_combined=True: total_loss, main_loss, aux_loss, main_dice, main_miou, main_dce, aux_dice, aux_miou, aux_dce
            use_combined=False: total_loss, main_loss, aux_loss
        """
        if isinstance(outputs, tuple):
            main_output, aux_output = outputs

            if self.use_combined:
                # 使用组合损失
                main_loss, main_dice, main_miou, main_dce = self.main_criterion(main_output, targets)
                aux_loss, aux_dice, aux_miou, aux_dce = self.aux_criterion(aux_output, targets)
                total_loss = main_loss + self.aux_weight * aux_loss
                return (total_loss, main_loss, aux_loss,
                        main_dice, main_miou, main_dce,
                        aux_dice, aux_miou, aux_dce)
            else:
                # 使用纯Dice损失
                main_loss = self.main_criterion(main_output, targets)
                aux_loss = self.aux_criterion(aux_output, targets)
                total_loss = main_loss + self.aux_weight * aux_loss
                return total_loss, main_loss, aux_loss
        else:
            # 如果只有单一输出
            if self.use_combined:
                main_loss, main_dice, main_miou, main_dce = self.main_criterion(outputs, targets)
                return (main_loss, main_loss, torch.tensor(0.0, device=outputs.device),
                        main_dice, main_miou, main_dce,
                        torch.tensor(0.0, device=outputs.device),
                        torch.tensor(0.0, device=outputs.device),
                        torch.tensor(0.0, device=outputs.device))
            else:
                main_loss = self.main_criterion(outputs, targets)
                return main_loss, main_loss, torch.tensor(0.0, device=outputs.device)


class SegmentationMetrics:
    """
    分割评估指标类 - 完全修正版本
    支持: Pixel Accuracy, Mean Accuracy, IoU, Mean IoU, F1-Score等
    """

    def __init__(self, num_classes=9, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        """
        更新混淆矩阵
        Args:
            pred: [B, H, W] 或 [B, C, H, W]，预测结果
            target: [B, H, W]，真实标签
        """
        # 处理不同的输入格式
        if pred.dim() == 4:  # [B, C, H, W]
            pred = torch.argmax(pred, dim=1)

        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # 掩码过滤
        mask = (target >= 0) & (target < self.num_classes) & \
               (pred >= 0) & (pred < self.num_classes)
        pred = pred[mask]
        target = target[mask]

        if len(pred) > 0:
            # 正确使用confusion_matrix
            # confusion_matrix[i, j] = 真实为i，预测为j的像素数
            cm = confusion_matrix(target, pred, labels=range(self.num_classes))
            self.confusion_matrix += cm

    def get_metrics(self):
        """
        计算所有评估指标
        Returns:
            dict: 包含所有评估指标
        """
        eps = 1e-8

        # ============ Pixel Accuracy (PA) ============
        # PA = (TP1 + TP2 + ... + TPn) / Total
        pa = np.diag(self.confusion_matrix).sum() / \
             (self.confusion_matrix.sum() + eps)

        # ============ Mean Accuracy (MA) ============
        # 每个类别的准确率平均值
        # Class Accuracy_i = TP_i / (TP_i + FN_i) = diagonal[i] / row_sum[i]
        class_acc = np.diag(self.confusion_matrix) / \
                    (self.confusion_matrix.sum(axis=1) + eps)
        ma = np.nanmean(class_acc)

        # ============ IoU (Intersection over Union) ============
        # 正确的IoU计算方式
        intersection = np.diag(self.confusion_matrix)  # TP

        # Union = TP + FP + FN
        # FP_i = sum of column i except diagonal = predicted as i but not i
        # FN_i = sum of row i except diagonal = actually i but not predicted as i
        fp = self.confusion_matrix.sum(axis=0) - intersection
        fn = self.confusion_matrix.sum(axis=1) - intersection

        union = intersection + fp + fn
        iou = intersection / (union + eps)
        miou = np.nanmean(iou)

        # ============ Precision 和 Recall ============
        # Precision_i = TP_i / (TP_i + FP_i)
        # Recall_i = TP_i / (TP_i + FN_i)
        precision = intersection / (intersection + fp + eps)
        recall = intersection / (intersection + fn + eps)

        # ============ F1-Score ============
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        mf1 = np.nanmean(f1_score)

        # ============ 构建返回字典 ============
        metrics = {
            'pixel_accuracy': pa,
            'mean_accuracy': ma,
            'class_accuracy': class_acc,
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'mean_iou': miou,
            'f1_score': f1_score,
            'mean_f1': mf1,
            'confusion_matrix': self.confusion_matrix
        }

        return metrics

    def print_metrics(self):
        """打印格式化的评估指标"""
        metrics = self.get_metrics()

        print("=" * 80)
        print("SEGMENTATION EVALUATION METRICS")
        print("=" * 80)
        print(f"\nPixel Accuracy (PA):     {metrics['pixel_accuracy']:.4f}")
        print(f"Mean Accuracy (MA):      {metrics['mean_accuracy']:.4f}")
        print(f"Mean IoU (mIoU):         {metrics['mean_iou']:.4f}")
        print(f"Mean F1-Score (mF1):     {metrics['mean_f1']:.4f}")

        print("\n" + "-" * 80)
        print("Per-Class Metrics:")
        print("-" * 80)
        print(f"{'Class':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'IoU':<12} {'F1':<12}")
        print("-" * 80)

        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
            print(f"{class_name:<20} {metrics['class_accuracy'][i]:<12.4f} "
                  f"{metrics['precision'][i]:<12.4f} {metrics['recall'][i]:<12.4f} "
                  f"{metrics['iou'][i]:<12.4f} {metrics['f1_score'][i]:<12.4f}")

        print("=" * 80)

        return metrics

    def plot_confusion_matrix(self, normalize=True, save_path=None):
        """可视化混淆矩阵"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = self.confusion_matrix.copy()

        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else '.0f',
                    cmap='Blues', xticklabels=self.class_names,
                    yticklabels=self.class_names, cbar_kws={'label': 'Count'})
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def create_subset_dataloader(image_dir, label_dir, batch_size, subset_ratio=0.4, shuffle=True,
                             num_workers=4, pin_memory=True, seed=None):
    """
    创建数据集子集的DataLoader
    Args:
        image_dir: 图像目录
        label_dir: 标签目录
        batch_size: 批次大小
        subset_ratio: 子集比例（0.4表示使用40%的数据）
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否启用pin_memory
        seed: 随机种子，用于控制子集选择的随机性
    """
    # 创建完整数据集
    full_dataset = SegmentationDataset(image_dir, label_dir)

    # 计算子集大小
    full_size = len(full_dataset)
    subset_size = int(full_size * subset_ratio)

    # 设置随机种子（如果提供）
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 随机选择索引
    indices = list(range(full_size))
    if shuffle:
        random.shuffle(indices)
    subset_indices = indices[:subset_size]

    # 创建子集
    subset_dataset = Subset(full_dataset, subset_indices)

    # 创建DataLoader
    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print(f"创建子集数据加载器: {subset_size}/{full_size} ({subset_ratio * 100:.1f}%)")

    return dataloader



edge_detector = EdgeDetectionModule(method='sobel')
DownSampler=nn.AvgPool2d(stride=4,kernel_size=4)
from ExtendClass import map_252_to_9
def train_epoch(model, train_loader, criterion, edge_loss, sekloss, optimizer, device, scaler,
                train_subset_ratio=0.4, use_combined_loss=True, epoch=0):
    """
    支持半精度训练和部分数据训练的训练函数
    Args:
        train_subset_ratio: 每个epoch使用的训练数据比例
        epoch: 当前epoch，用于生成不同的随机种子
    """
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    total_dice_loss = 0
    total_focal_loss = 0

    # 创建当前epoch的训练子集
    # 每个epoch使用不同的随机种子，确保每次选择的数据不同
    current_seed = epoch * 42  # 使用epoch相关的种子

    # 如果train_loader是完整的数据集，我们需要重新创建子集
    if hasattr(train_loader.dataset, 'dataset'):
        # 如果已经是Subset，获取原始数据集
        original_dataset = train_loader.dataset.dataset
    else:
        original_dataset = train_loader.dataset

    # 创建当前epoch的随机子集
    full_size = len(original_dataset)
    subset_size = int(full_size * train_subset_ratio)

    # 设置随机种子
    random.seed(current_seed)
    np.random.seed(current_seed)

    # 随机选择索引
    indices = list(range(full_size))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]

    # 创建子集
    subset_dataset = Subset(original_dataset, subset_indices)

    # 创建当前epoch的DataLoader
    epoch_loader = DataLoader(
        subset_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=getattr(train_loader, 'pin_memory', False),
        drop_last=False
    )

    num_batches = len(epoch_loader)
    progress_bar = tqdm(epoch_loader, desc=f"Training (FP16, {train_subset_ratio * 100:.0f}% data)")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        # print("---------------------")
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        labelsx=torch.unsqueeze(labels,dim=1).float()
        labels_edge=edge_detector(torch.cat((labelsx,labelsx,labelsx),dim=1))
        labels_edge=DownSampler(labels_edge)
        labels_edge = (labels_edge > 0.0).long()

        # print(labels_edge)
        # Temp=labels_edge.cpu().detach().numpy()
        # plt.figure(figsize=(10,5))
        # plt.imshow(Temp[0])
        # plt.show()

        optimizer.zero_grad()

        # 使用autocast进行前向传播
        with autocast():
            outputs,feat_edge = model(images)
            # outputs = model(images)
            # outputs = (map_252_to_9(outputs[0]),map_252_to_9(outputs[1]))
            loss_outputs = criterion(outputs, labels)
            # print('损失函数个数',len(loss_outputs))
            # print(feat_edge)
            # print(feat_edge)
            loss_edge = edge_loss(feat_edge,labels_edge.long())
            loss_sek = sekloss(outputs[0], labels)
        if use_combined_loss and len(loss_outputs) == 7:
            loss, main_loss, aux_loss, main_dice, main_focal, aux_dice, aux_focal = loss_outputs
            total_dice_loss += (main_dice.item() + aux_dice.item()) / 2
            total_focal_loss += (main_focal.item() + aux_focal.item()) / 2
        else:
            # print("-----------")
            loss, main_loss, aux_loss = loss_outputs[:3]
        # print(loss_edge,"-----")
        loss+=loss_edge
        loss+=loss_sek
        # 使用scaler进行反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()

        # 更新进度条
        if use_combined_loss and len(loss_outputs) == 7:
            progress_bar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'Main': f'{main_loss.item():.4f}',
                'Dice': f'{main_dice.item():.4f}',
                'Focal': f'{main_focal.item():.4f}',
                'Samples': f'{subset_size}/{full_size}'
            })
        else:
            progress_bar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'Main': f'{main_loss.item():.4f}',
                'Sek': f'{loss_sek.item():.4f}',
                'Edge': f'{loss_edge.item():.4f}',
                'Aux': f'{aux_loss.item():.4f}',
                'Samples': f'{subset_size}/{full_size}'
            })

    result = {
        'total_loss': total_loss / num_batches,
        'main_loss': total_main_loss / num_batches,
        'aux_loss': total_aux_loss / num_batches,
        'samples_used': subset_size,
        'total_samples': full_size
    }

    if use_combined_loss and total_dice_loss > 0:
        result['dice_loss'] = total_dice_loss / num_batches
        result['focal_loss'] = total_focal_loss / num_batches

    return result


def evaluate_model(model, val_loader, criterion, device, num_classes=9, use_combined_loss=True):
    """
    支持半精度的评估函数
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes)

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating (FP16)")

        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # 使用autocast进行前向传播
            with autocast():
                outputs = model(images)

            # 获取主输出的预测结果用于评估
            if isinstance(outputs, tuple):
                main_output, _ = outputs
                predictions = torch.argmax(main_output, dim=1)
            else:
                predictions = torch.argmax(outputs, dim=1)

            metrics.update(predictions, labels)

    eval_metrics = metrics.get_metrics()
    return eval_metrics


def save_checkpoint(model, optimizer, scaler, epoch, train_losses, val_losses, metrics, save_dir):
    """
    修改的保存函数，包含scaler状态
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics
    }

    save_path = os.path.join(save_dir, f'model_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, save_path)
    print(f"模型已保存到: {save_path}")


# def plot_confusion_matrix(cm, class_names, save_path=None):
#     """
#     绘制混淆矩阵（高清版）
#     """
#     plt.figure(figsize=(14, 12), dpi=300)
#
#     # 使用更大的字体绘制热力图
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names,
#                 annot_kws={'size': 24},  # 数字标注字体大小
#                 cbar_kws={'label': 'Count'})
#
#     plt.title('Confusion Matrix', fontsize=26, pad=25, fontweight='bold')
#     plt.ylabel('True Label', fontsize=20, fontweight='bold')
#     plt.xlabel('Predicted Label', fontsize=20, fontweight='bold')
#
#     # 调整刻度标签
#     plt.xticks(rotation=45, ha='right', fontsize=16)
#     plt.yticks(rotation=0, fontsize=16)
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#         print(f"混淆矩阵已保存到: {save_path}")
#     plt.show()
#     plt.close()
from torch.nn import SmoothL1Loss

def train_model(model, train_loader, val_loader, num_epochs, device, save_dir='checkpoints',
                train_subset_ratio=0.4):
    """
    支持半精度训练和部分数据训练的主训练函数
    Args:
        train_subset_ratio: 每个epoch使用的训练数据比例
    """
    # 使用基于Dice的多输出损失函数
    criterion = MultiOutputDiceLoss(aux_weight=0.1,
        dice_weight=0.6,
        miou_weight=1.2,
        dce_weight=0.6,
        dce_alpha=1.0,
        use_combined=True)
    use_combined_loss = True
    edge_loss=SmoothL1Loss(size_average="mean")
    Sekloss=SEKLoss.SeKLoss(num_classes=9)
    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=0.00004, weight_decay=0.12)

    # 初始化GradScaler用于混合精度训练
    scaler = GradScaler()

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=6)

    # 训练历史记录
    train_history = {
        'total_loss': [], 'main_loss': [], 'aux_loss': [], 'dice_loss': [], 'focal_loss': []
    }
    val_history = {
        'total_loss': [], 'main_loss': [], 'aux_loss': [], 'dice_loss': [], 'focal_loss': []
    }
    val_metrics_history = []

    best_miou = 0.0

    print("开始半精度训练 - 使用基于Dice的损失函数...")
    print(f"设备: {device}")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"每epoch训练数据比例: {train_subset_ratio * 100:.1f}%")
    print(f"损失函数: {'Dice+Focal组合损失' if use_combined_loss else '纯Dice损失'}")
    print(f"训练模式: 半精度 (FP16)")
    print("-" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Scaler Scale: {scaler.get_scale():.1f}")

        # 使用半精度训练和部分数据
        train_losses = train_epoch(
            model, train_loader, criterion, edge_loss, Sekloss, optimizer, device, scaler,
            train_subset_ratio, use_combined_loss, epoch
        )

        # 记录训练损失
        for key in train_losses.keys():
            if key in train_history:
                train_history[key].append(train_losses[key])

        # 每5个epoch进行评估
        if (epoch) % 3 == 0:
            print("\n进行模型评估...")
            eval_metrics = test_model(model, val_loader, device)

            print(f"像素准确率: {eval_metrics['pixel_accuracy']:.4f}")
            print(f"平均准确率: {eval_metrics['mean_accuracy']:.4f}")
            print(f"平均IoU: {eval_metrics['mean_iou']:.4f}")

            # 打印每类IoU
            print("\n各类别IoU:")
            for i, iou in enumerate(eval_metrics['iou']):
                print(f"类别 {i}: {iou:.4f}")

            # 保存模型
            is_best = eval_metrics['mean_iou'] > best_miou
            if is_best:
                best_miou = eval_metrics['mean_iou']
                print(f"新的最佳模型! mIoU: {best_miou:.4f}")

            save_checkpoint(model, optimizer, scaler, epoch + 1, train_losses, 1.0,
                            eval_metrics, save_dir)

            # 保存最佳模型
            if is_best:
                best_path = os.path.join(save_dir, 'best_model.pt')
                checkpoint = torch.load(os.path.join(save_dir, f'model_epoch_{epoch + 1:03d}.pt'))
                torch.save(checkpoint, best_path)
                print(f"最佳模型已保存到: {best_path}")

        # 更新学习率调度器
        scheduler.step()
        epoch_time = time.time() - start_time

        print(f"训练总损失: {train_losses['total_loss']:.4f}")
        print(f"训练主损失: {train_losses['main_loss']:.4f}")
        print(f"训练辅助损失: {train_losses['aux_loss']:.4f}")
        print(f"使用样本数: {train_losses['samples_used']}/{train_losses['total_samples']}")

        if 'dice_loss' in train_losses:
            print(f"训练Dice损失: {train_losses['dice_loss']:.4f}")
            print(f"训练Focal损失: {train_losses['focal_loss']:.4f}")
        print(f"训练时间: {epoch_time:.2f}秒")
        print("-" * 60)

    return train_history, val_history, val_metrics_history


def plot_confusion_matrix(confusion_matrix, class_names, save_path='confusion_matrix.png'):
    cm_normalized = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    plt.title('Normalized Confusion Matrix (Proportion)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")


def plot_pr_curves(all_labels, all_probs, num_classes, class_names, save_path='pr_curves.png'):
    """
    绘制PR曲线（高清版）
    """
    # 定义颜色和样式
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
              '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']

    # 创建更大的图形
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

    for i in range(num_classes):
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]

        n_positive = np.sum(binary_labels)
        if n_positive == 0:
            print(f"警告: {class_names[i]} 没有正样本，跳过")
            continue

        # 验证概率范围
        print(f"{class_names[i]}: 正样本={n_positive}, "
              f"概率范围=[{class_probs.min():.4f}, {class_probs.max():.4f}], "
              f"均值={class_probs.mean():.4f}")

        precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
        pr_auc = auc(recall, precision)

        # 使用更粗的线条和不同的颜色、样式
        ax.plot(recall, precision, lw=3.5, linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                label=f'{class_names[i]} (AUC={pr_auc:.3f})',
                marker='o', markersize=4, markevery=max(1, len(recall) // 15),
                alpha=0.85)

    ax.set_xlabel('Recall', fontsize=26, fontweight='bold', labelpad=15)
    ax.set_ylabel('Precision', fontsize=26, fontweight='bold', labelpad=15)
    ax.set_title('Precision-Recall Curves', fontsize=32, pad=30, fontweight='bold')

    # 调整图例
    ax.legend(loc='best', fontsize=22, frameon=True, shadow=True,
              fancybox=True, framealpha=0.95, edgecolor='black',
              borderpad=1, labelspacing=1.2)

    # 网格线
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.5, color='gray')

    # 坐标轴范围和刻度
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=22, width=2.5, length=8)

    # 加粗坐标轴边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"PR曲线已保存至: {save_path}")


def plot_roc_curves(all_labels, all_probs, num_classes, class_names, save_path='roc_curves.png'):
    """
    绘制ROC曲线（高清版）
    """
    # 定义颜色和样式
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
              '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']

    # 创建更大的图形
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

    for i in range(num_classes):
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]

        if np.sum(binary_labels) == 0:
            continue

        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)

        # 使用更粗的线条和不同的颜色、样式
        ax.plot(fpr, tpr, lw=3.5, linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                label=f'{class_names[i]} (AUC={roc_auc:.3f})',
                marker='s', markersize=4, markevery=max(1, len(fpr) // 15),
                alpha=0.85)

    # 绘制随机分类器基线（虚线，更粗，更明显）
    ax.plot([0, 1], [0, 1], 'k--', lw=3.5, label='Random Classifier (AUC=0.500)',
            alpha=0.6, marker='d', markersize=3, markevery=10)

    ax.set_xlabel('False Positive Rate', fontsize=26, fontweight='bold', labelpad=15)
    ax.set_ylabel('True Positive Rate', fontsize=26, fontweight='bold', labelpad=15)
    ax.set_title('ROC Curves', fontsize=32, pad=30, fontweight='bold')

    # 调整图例
    ax.legend(loc='lower right', fontsize=22, frameon=True, shadow=True,
              fancybox=True, framealpha=0.95, edgecolor='black',
              borderpad=1, labelspacing=1.2)

    # 网格线
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.5, color='gray')

    # 坐标轴范围和刻度
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=22, width=2.5, length=8)

    # 加粗坐标轴边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"ROC曲线已保存至: {save_path}")


def test_model_with_visualization(model, test_loader, device, checkpoint_path=None,
                                  save_dir='./results', sample_ratio=0.1, max_samples=5000000):
    import os
    os.makedirs(save_dir, exist_ok=True)

    if checkpoint_path:
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        print(f"模型加载完成 (Epoch {checkpoint['epoch']})")

    model.eval()
    num_classes = 9
    class_names = ["背景","耕地粮菜","花椒油茶","经果林","花卉园圃","草地","林地","人造地表","水体"]

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    sampled_probs = []
    sampled_labels = []
    sampled_predictions = []  # 添加预测结果用于验证
    total_pixels = 0
    sampled_pixels = 0

    print(f"开始测试并收集数据 (采样比例: {sample_ratio * 100:.1f}%, 最大样本数: {max_samples})...")

    np.random.seed(42)

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)

            if isinstance(outputs, tuple):
                main_output, _ = outputs
            else:
                main_output = outputs

            # 关键：先计算softmax概率，再argmax
            probs = torch.softmax(main_output, dim=1)
            predictions = torch.argmax(probs, dim=1)

            # 转移到CPU并转换为numpy
            predictions_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()

            batch_size = labels_np.shape[0]

            for b in range(batch_size):
                pred_flat = predictions_np[b].flatten()
                label_flat = labels_np[b].flatten()
                prob_flat = probs_np[b].transpose(1, 2, 0).reshape(-1, num_classes)  # 确保形状正确

                n_pixels = len(label_flat)
                total_pixels += n_pixels

                # 更新混淆矩阵
                np.add.at(confusion_matrix, (label_flat, pred_flat), 1)

                # 采样
                if sampled_pixels < max_samples:
                    sample_size = min(int(n_pixels * sample_ratio), max_samples - sampled_pixels)
                    if sample_size > 0:
                        indices = np.random.choice(n_pixels, size=sample_size, replace=False)

                        sampled_probs.append(prob_flat[indices])
                        sampled_labels.append(label_flat[indices])
                        sampled_predictions.append(pred_flat[indices])
                        sampled_pixels += sample_size

            progress_bar.set_postfix({
                'total': f'{total_pixels:,}',
                'sampled': f'{sampled_pixels:,}'
            })

    # 合并数据
    if sampled_probs:
        sampled_probs = np.vstack(sampled_probs)
        sampled_labels = np.concatenate(sampled_labels)
        sampled_predictions = np.concatenate(sampled_predictions)

        print(f"\n已采样 {len(sampled_labels):,} 个像素")
        print(f"标签范围: [{sampled_labels.min()}, {sampled_labels.max()}]")
        print(f"概率形状: {sampled_probs.shape}")
        print(f"概率总和检查 (应接近1.0): {sampled_probs.sum(axis=1).mean():.4f}")

        # 验证：检查预测是否与最大概率对应
        pred_from_probs = np.argmax(sampled_probs, axis=1)
        match_rate = np.mean(pred_from_probs == sampled_predictions)
        print(f"预测一致性检查: {match_rate:.4f} (应为1.0)")

        # 打印各类别分布
        print("\n各类别采样分布:")
        for i in range(num_classes):
            n_samples = np.sum(sampled_labels == i)
            avg_prob = sampled_probs[sampled_labels == i, i].mean() if n_samples > 0 else 0
            print(f"  {class_names[i]}: {n_samples:,} 样本, 平均概率={avg_prob:.4f}")
    else:
        print("\n警告: 没有采样到数据")
        return None

    print("\n生成可视化结果...")

    plot_confusion_matrix(
        confusion_matrix,
        class_names,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )

    if len(sampled_probs) > 0:
        plot_pr_curves(
            sampled_labels,
            sampled_probs,
            num_classes,
            class_names,
            save_path=os.path.join(save_dir, 'pr_curves.png')
        )

        plot_roc_curves(
            sampled_labels,
            sampled_probs,
            num_classes,
            class_names,
            save_path=os.path.join(save_dir, 'roc_curves.png')
        )

    # 计算指标
    print("\n测试结果:")
    pixel_accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    print(f"像素准确率: {pixel_accuracy:.4f}")

    class_accuracy = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-10)
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - intersection
    iou = intersection / (union + 1e-10)

    print(f"平均准确率: {np.mean(class_accuracy):.4f}")
    print(f"平均IoU: {np.mean(iou):.4f}")

    print("\n各类别详细结果:")
    for i, (acc, iou_val) in enumerate(zip(class_accuracy, iou)):
        print(f"{class_names[i]}: 准确率={acc:.4f}, IoU={iou_val:.4f}")

    print(f"\n所有可视化结果已保存至: {save_dir}")

    return {
        'pixel_accuracy': pixel_accuracy,
        'mean_accuracy': np.mean(class_accuracy),
        'mean_iou': np.mean(iou),
        'class_accuracy': class_accuracy,
        'iou': iou,
        'confusion_matrix': confusion_matrix
    }

import torch.nn.functional as F
def calculate_dice_coefficient(predictions, labels, num_classes=9):
    """
    计算每个类别的Dice系数
    """
    dice_scores = []
    smooth = 1e-6

    for cls in range(num_classes):
        pred_cls = (predictions == cls).float()
        label_cls = (labels == cls).float()

        intersection = (pred_cls * label_cls).sum()
        union = pred_cls.sum() + label_cls.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    mean_dice = sum(dice_scores) / len(dice_scores)
    return dice_scores, mean_dice


def calculate_ssim(predictions, labels, num_classes=9, window_size=11):
    """
    计算每个类别的SSIM
    """

    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, -1) * g.view(1, -1, 1)

    window = gaussian_window(window_size).to(predictions.device)
    window = window.unsqueeze(0)  # [1, 1, window_size, window_size]

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_scores = []

    for cls in range(num_classes):
        pred_cls = (predictions == cls).float().unsqueeze(1)  # [B, 1, H, W]
        label_cls = (labels == cls).float().unsqueeze(1)

        mu1 = F.conv2d(pred_cls, window, padding=window_size // 2, stride=1)
        mu2 = F.conv2d(label_cls, window, padding=window_size // 2, stride=1)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred_cls * pred_cls, window, padding=window_size // 2, stride=1) - mu1_sq
        sigma2_sq = F.conv2d(label_cls * label_cls, window, padding=window_size // 2, stride=1) - mu2_sq
        sigma12 = F.conv2d(pred_cls * label_cls, window, padding=window_size // 2, stride=1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        ssim_scores.append(ssim_map.mean().item())

    mean_ssim = sum(ssim_scores) / len(ssim_scores)
    return ssim_scores, mean_ssim
from skimage.segmentation import slic
from scipy.stats import mode
def postprocess_mask_advanced(pred_mask: np.ndarray, image: np.ndarray, num_classes=9,
                              morph_kernel_size=5,
                              slic_segments=1000,
                              slic_compactness=15,
                              boundary_mag_threshold=15) -> np.ndarray:
    """
    高级分割后处理函数，结合形态学、超像素和边界梯度优化，提升分割边界精度和结构连贯性。

    参数：
    - pred_mask: (H, W)网络预测的整数类别掩码
    - image: (H, W, 3) 原始RGB uint8图像
    - num_classes:类别数
    - morph_kernel_size: 形态学操作卷积核大小
    - slic_segments: 超像素分割数，影响细化粒度
    - slic_compactness: 超像素紧凑度参数
    - boundary_mag_threshold: 梯度幅度阈值，用于边界精细修正

    返回：
    - refined_mask: (H, W) 后处理后的类别掩码
    """
    # 一、形态学清理 (开运算->闭运算)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

    clean_mask = np.zeros_like(pred_mask, dtype=np.uint8)

    for cls in range(num_classes):
        single_class = (pred_mask == cls).astype(np.uint8) * 255
        cleaned = cv2.morphologyEx(single_class, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        clean_mask[cleaned == 255] = cls

    # 二、超像素一致性修正
    segments = slic(image, n_segments=slic_segments, compactness=slic_compactness, start_label=0)

    refined_mask = clean_mask.copy()
    for seg_val in np.unique(segments):
        seg_pixels = refined_mask[segments == seg_val]
        if len(seg_pixels) == 0:
            continue
        major_class = mode(seg_pixels)[0][0]
        refined_mask[segments == seg_val] = major_class

    # 三、基于图像梯度的边界细化
    # 计算原图梯度幅度（灰度图）
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # 找到掩码边界像素
    edges = cv2.Canny(refined_mask.astype(np.uint8), 100, 200)

    h, w = refined_mask.shape

    # 对边界周围一定像素范围内进行梯度增强修正
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    ys, xs = np.where(dilated_edges > 0)
    patch_radius = 3

    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - patch_radius), min(h, y + patch_radius + 1)
        x0, x1 = max(0, x - patch_radius), min(w, x + patch_radius + 1)

        # 梯度大且邻域内多类不一致表明边界可能定位不准
        local_grad = grad_magnitude[y0:y1, x0:x1]
        local_mask = refined_mask[y0:y1, x0:x1]

        # 如果梯度强且邻域内类多样性高，尝试边界微调
        if np.max(local_grad) > boundary_mag_threshold and len(np.unique(local_mask)) > 1:
            # 使用原图像素颜色距离定位更合理类别
            # 计算邻域内每类平均颜色，当前像素颜色
            pixel_color = image[y, x, :].astype(np.float32)
            classes = np.unique(local_mask)
            min_dist = float('inf')
            best_cls = refined_mask[y, x]
            for c in classes:
                cls_mask = (local_mask == c)
                if np.sum(cls_mask) == 0:
                    continue
                avg_color = image[y0:y1, x0:x1][cls_mask].mean(axis=0)
                dist = np.linalg.norm(pixel_color - avg_color)
                if dist < min_dist:
                    min_dist = dist
                    best_cls = c
            refined_mask[y, x] = best_cls

    return refined_mask


def postprocess_single_mask_simple(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    简易后处理：形态学操作 + 边缘检测辅助细化
    Args:
        mask: (H, W) 整数类别掩码
        image: (H, W, 3) uint8原图，用于边缘检测参考

    Returns:
        后处理后的掩码 (H, W)
    """

    mask_uint8 = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 1. 闭操作填补小孔洞
    mask_closed = np.zeros_like(mask_uint8)
    for cls in range(1, 9):  # 除背景外类别
        cls_mask = (mask_uint8 == cls).astype(np.uint8) * 255
        closed = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel)
        mask_closed[closed == 255] = cls

    # 2. 腐蚀去噪声（小碎块）
    mask_erode = np.zeros_like(mask_closed)
    for cls in range(1,9):
        cls_mask = (mask_closed == cls).astype(np.uint8) * 255
        eroded = cv2.erode(cls_mask, kernel, iterations=1)
        mask_erode[eroded == 255] = cls

    # 3. 用Canny边缘检测找原图边缘，辅助修正分割边界
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 4. 对分割边界邻域进行微调：比如将边缘附近的像素重新归类为邻近主区域
    # 这里示范简单的边缘膨胀调整
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # 把边缘膨胀区内的掩码像素修改为最频繁的类别（局部众数）
    h, w = mask_erode.shape
    mask_refined = mask_erode.copy()

    # 遍历边缘膨胀区域像素
    ys, xs = np.where(edges_dilated > 0)
    for y, x in zip(ys, xs):
        # 在3x3邻域选众数类别
        y0, y1 = max(0, y-1), min(h, y+2)
        x0, x1 = max(0, x-1), min(w, x+2)
        patch = mask_erode[y0:y1, x0:x1]
        # 计算众数
        unique, counts = np.unique(patch, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        mask_refined[y, x] = majority_class

    return mask_refined

def test_model(model, test_loader, device, checkpoint_path=None):
    """
    支持半精度的测试函数，包含Dice和SSIM计算，并集成后处理增强分割结果
    """

    if checkpoint_path:
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"模型加载完成 (Epoch {checkpoint['epoch']})")

    model.eval()
    metrics = SegmentationMetrics(num_classes=9)

    all_dice_scores = []
    all_ssim_scores = []

    print("开始测试 (FP16)...")
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing (FP16) with Postprocessing")

        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)

            if isinstance(outputs, tuple):
                main_output, _ = outputs
                predictions = torch.argmax(main_output, dim=1)
            else:
                predictions = torch.argmax(outputs, dim=1)

            # -------- 集成后处理部分开始 --------
            # Convert predictions and images to numpy for postprocessing
            preds_np = predictions.cpu().numpy()

            # 将 images 反归一化到[0,255]且转换为uint8格式
            # 这里根据你的预处理细节调整，如果输入是归一化到[0,1]，乘255即可，若有均值方差还需反归一化
            images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            postprocessed_preds = []
            for pred_mask, img in zip(preds_np, images_np):
                pp_mask = postprocess_mask_advanced(pred_mask, img)
                postprocessed_preds.append(pp_mask)

            # 转回tensor并放回device
            postprocessed_preds = torch.tensor(postprocessed_preds, dtype=torch.long, device=device)

            # 更新度量计算，使用后处理后的预测结果
            metrics.update(postprocessed_preds, labels)

            # ---------- 计算Dice和SSIM也用后处理后的结果---------
            dice_scores, _ = calculate_dice_coefficient(postprocessed_preds, labels, num_classes=9)
            ssim_scores, _ = calculate_ssim(postprocessed_preds, labels, num_classes=9)

            all_dice_scores.append(dice_scores)
            all_ssim_scores.append(ssim_scores)

    test_metrics = metrics.get_metrics()

    mean_dice_per_class = [sum(scores[i] for scores in all_dice_scores) / len(all_dice_scores)
                           for i in range(9)]
    mean_ssim_per_class = [sum(scores[i] for scores in all_ssim_scores) / len(all_ssim_scores)
                           for i in range(9)]

    overall_dice = sum(mean_dice_per_class) / len(mean_dice_per_class)
    overall_ssim = sum(mean_ssim_per_class) / len(mean_ssim_per_class)

    print("\n测试结果:")
    print(f"像素准确率: {test_metrics['pixel_accuracy']:.4f}")
    print(f"平均准确率: {test_metrics['mean_accuracy']:.4f}")
    print(f"平均IoU: {test_metrics['mean_iou']:.4f}")
    print(f"平均Dice: {overall_dice:.4f}")
    print(f"平均SSIM: {overall_ssim:.4f}")

    print("\n各类别详细结果:")
    class_names = ["背景", "耕地粮菜", "花椒油茶", "经果林", "花卉园圃", "草地", "林地", "人造地表", "水体"]
    for i, (acc, iou, dice, ssim) in enumerate(zip(
            test_metrics['class_accuracy'],
            test_metrics['iou'],
            mean_dice_per_class,
            mean_ssim_per_class
    )):
        print(f"{class_names[i]}: 准确率={acc:.4f}, IoU={iou:.4f}, Dice={dice:.4f}, SSIM={ssim:.4f}")

    test_metrics['dice_scores'] = mean_dice_per_class
    test_metrics['ssim_scores'] = mean_ssim_per_class
    test_metrics['mean_dice'] = overall_dice
    test_metrics['mean_ssim'] = overall_ssim

    return test_metrics

def main():
    config = {
        'train_image_dir': 'D:/ZJF/种植作物分类/DataSplit/train/images',
        'train_label_dir': 'D:/ZJF/种植作物分类/DataSplit/train/labels',
        'val_image_dir': 'D:/ZJF/种植作物分类/DataSplit/train/images',
        'val_label_dir': 'D:/ZJF/种植作物分类/Datas/GT',
        'test_image_dir': 'D:/ZJF/种植作物分类/DataSplit/test/images',
        'test_label_dir': 'D:/ZJF/种植作物分类/DataSplit/test/labels',
        'batch_size':14,  # 可以适当增加batch size，因为半精度占用内存更少
        'num_epochs': 160,
        'num_workers': 1,
        'train_subset_ratio': 0.7,  # 每个epoch使用40%的训练数据
        'save_dir': 'UMRRF_9Class'  # 标识半精度训练和40%数据
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 检查是否支持半精度训练
    if device.type == 'cuda':
        print(f"CUDA Capability: {torch.cuda.get_device_capability(device)}")
        print("支持半精度训练")
    else:
        print("警告: 非CUDA设备，半精度训练可能不会带来性能提升")

    # 创建数据加载器 - 启用pin_memory加速数据传输
    print("创建数据加载器...")
    train_loader = create_dataloader(
        config['train_image_dir'], config['train_label_dir'],
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'],
        # apply_mask=True,  # 启用掩码lr
        # mask_area_ratio_min=0.02,  # 最小10%面积
        # mask_area_ratio_max=0.05  # 最大40%面积
    )

    # val_loader = create_dataloader(
    #     config['val_image_dir'], config['val_label_dir'],
    #     batch_size=config['batch_size'], shuffle=False,
    #     num_workers=config['num_workers']
    # )

    test_loader = create_dataloader(
        config['test_image_dir'], config['test_label_dir'],
        batch_size=84, shuffle=True,
        num_workers=config['num_workers']
    )

    # 创建UNetFormer模型
    # try:
    from ExtendClass import load_pretrained_partial
    model = UNetFormer(num_classes=9,dropout=0.1)
    # print(model)
#     model  = DynamicDictionaryLearning(
#     model="swinv2_small",
#     token_length=16,
#     l=1,
#     num_classes=9,
#     has_conv=True,  # 确保输出通道是类别数
#     use_aux=True,
#     has_contrastive_loss=False
# ).to(device)
#     load_pretrained_partial(model,'./UNetFormer_ResMMFormer/best_model.pt')
    # 如果要从之前的检查点继续训练，取消下面的注释
    checkpoint_path = './UMRRF_9Class/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("已加载预训练模型")

    # except Exception as e:
    #     print("------")
    #     print(f"模型创建失败: {e}")
    #     return

    model = model.to(device)

    # 启用半精度训练的模型优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # 加速卷积计算
        torch.backends.cudnn.enabled = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 开始半精度训练，每个epoch使用40%的数据
    # train_history, val_history, val_metrics_history = train_model(
    #     model, train_loader, test_loader,
    #     config['num_epochs'], device, config['save_dir'],
    #     train_subset_ratio=config['train_subset_ratio']
    # )

    # 测试模型
    best_model_path = os.path.join(config['save_dir'], 'best_model.pt')
    if os.path.exists(best_model_path):
        test_metrics = test_model(model, test_loader, device, best_model_path)
        print("\n最终测试结果:")
        print(test_metrics)
    else:
        test_metrics = test_model(model, train_loader, device)
        print("\n最终测试结果:")
        print(test_metrics)
    # results = test_model_with_visualization(
    #     model=model,
    #     test_loader=test_loader,
    #     device=device,
    #     checkpoint_path='UNetFormer_ResMMFormer_Fix_1110/best_model.pt',
    #     save_dir='./test_results/Gai1'
    # )
    # print("半精度训练完成!")


if __name__ == "__main__":
    main()
    # A=torchvision.models.segmentation.FCN()