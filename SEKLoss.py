import torch
import torch.nn.functional as F

class SeKLoss(torch.nn.Module):
    def __init__(self, num_classes=9, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (N, C, H, W) raw output scores
        targets: (N, H, W) ground-truth labels, int64 in [0, num_classes-1]
        """

        N, C, H, W = logits.shape
        assert C == self.num_classes, f"Expected {self.num_classes} classes, got {C}"

        # 计算softmax概率
        probs = F.softmax(logits, dim=1)  # (N,C,H,W)

        # 预测类别
        preds = torch.argmax(probs, dim=1)  # (N,H,W)

        # 计算混淆矩阵 Q: shape (C, C)
        # Q[i, j]: predicted class i, true class j 的像素数
        Q = torch.zeros((C, C), device=logits.device, dtype=torch.float32)

        # 将预测和标签展平
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        for i in range(C):
            for j in range(C):
                mask = (preds_flat == i) & (targets_flat == j)
                Q[i, j] = mask.sum()

        # 提取 q11（类别0的对角），以及其他元素
        q11 = Q[0, 0]
        total = Q.sum()

        # 避免类别不平衡影响，排除无变化类0
        denom = total - q11 + self.eps

        # 计算 rho_hat = sum_{i=1}^{C-1} q_{ii} / denom
        rho_hat = Q[1:, 1:].diagonal().sum() / denom

        # 计算 eta_hat = sum_{j=0}^{C-1} (sum_{i=0}^{C-1} q_{ij})(sum_{i=0}^{C-1} q_{ji}) / denom^2
        col_sum = Q.sum(dim=0)  # 真值每类像素数
        row_sum = Q.sum(dim=1)  # 预测每类像素数
        eta_hat = ((col_sum * row_sum).sum()) / (denom * denom)

        # 计算 IoU_2, 为“变化类”即类别1~C-1的交并比平均，这里近似用整体变化类IoU
        # 对于二分类IoU2计算，形如 Eq (12):
        # IoU2 = q22/(sum of predicted and GT for class 2 - q22),
        # 但多类时，论文中定义类似，简化为：（这里基于无变化类排除）
        intersection = Q[1:, 1:].diagonal().sum()
        union = denom  # 按论文公式，即总变化像素 - 无变化像素
        IoU2 = intersection / (union + self.eps)

        # SeK 公式
        # SeK = exp(IoU2 - 1) * (rho_hat - eta_hat) / (1 - eta_hat)
        numerator = (rho_hat - eta_hat)
        denominator = (1 - eta_hat + self.eps)
        SeK = torch.exp(IoU2 - 1) * (numerator / denominator)

        # 处理负值（极端情况）
        SeK = torch.clamp(SeK, min=0.0)

        # 损失: -log(SeK + eps)
        loss = -torch.log(SeK + self.eps)

        return loss

# 假设模型输出和标签
logits = torch.randn(4, 9, 256, 256, requires_grad=True)  # batch 4, 9 classes
targets = torch.randint(0, 9, (4, 256, 256))

sek_loss = SeKLoss(num_classes=9)
loss = sek_loss(logits, targets)

# loss.backward()
print("SeK Loss:", loss.item())