import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple


class SpatialAttention(nn.Module):
    """空间注意力模块 - 修复版本"""

    def __init__(self, rgb_channels: int, dsm_channels: int):
        super(SpatialAttention, self).__init__()
        # 修复：使用实际的通道数
        total_channels = rgb_channels + dsm_channels
        self.conv1 = nn.Conv2d(total_channels, max(rgb_channels, dsm_channels), kernel_size=1)
        self.conv2 = nn.Conv2d(max(rgb_channels, dsm_channels), 1, kernel_size=1)  # 输出单通道注意力图
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_feat: torch.Tensor, dsm_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 连接RGB和DSM特征
        concat_feat = torch.cat([rgb_feat, dsm_feat], dim=1)
        # 生成单通道注意力图
        attention_map = self.sigmoid(self.conv2(F.relu(self.conv1(concat_feat))))

        # 应用注意力权重 - 单通道注意力图可以广播到任意通道数
        rgb_weighted = attention_map * rgb_feat + rgb_feat
        dsm_weighted = attention_map * dsm_feat + dsm_feat

        return rgb_weighted, dsm_weighted


class CrossModalAttention(nn.Module):
    """跨模态注意力模块"""

    def __init__(self, rgb_channels: int, dsm_channels: int, num_heads: int = 8):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.rgb_channels = rgb_channels
        self.dsm_channels = dsm_channels

        # Query生成 - 来自融合特征
        self.q_proj = nn.Linear(rgb_channels + dsm_channels, rgb_channels + dsm_channels)

        # Key和Value生成 - 分别来自RGB和DSM
        self.k_proj_rgb = nn.Linear(rgb_channels, rgb_channels)
        self.v_proj_rgb = nn.Linear(rgb_channels, rgb_channels)
        self.k_proj_dsm = nn.Linear(dsm_channels, dsm_channels)
        self.v_proj_dsm = nn.Linear(dsm_channels, dsm_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, rgb_feat: torch.Tensor, dsm_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C_rgb, H, W = rgb_feat.shape
        _, C_dsm, _, _ = dsm_feat.shape

        # 生成Query - 来自融合特征的平均池化
        fused_feat = torch.cat([rgb_feat, dsm_feat], dim=1)  # [B, C_rgb+C_dsm, H, W]
        fused_pooled = self.avgpool(fused_feat)  # [B, C_rgb+C_dsm, 7, 7]
        fused_pooled = fused_pooled.flatten(2).transpose(1, 2)  # [B, 49, C_rgb+C_dsm]
        Q = self.q_proj(fused_pooled)  # [B, 49, C_rgb+C_dsm]

        # RGB分支的K, V
        rgb_flat = rgb_feat.flatten(2).transpose(1, 2)  # [B, H*W, C_rgb]
        K_rgb = self.k_proj_rgb(rgb_flat)  # [B, H*W, C_rgb]
        V_rgb = self.v_proj_rgb(rgb_flat)  # [B, H*W, C_rgb]

        # DSM分支的K, V
        dsm_flat = dsm_feat.flatten(2).transpose(1, 2)  # [B, H*W, C_dsm]
        K_dsm = self.k_proj_dsm(dsm_flat)  # [B, H*W, C_dsm]
        V_dsm = self.v_proj_dsm(dsm_flat)  # [B, H*W, C_dsm]

        # 跨模态注意力计算
        # RGB -> DSM融合
        Q_rgb = Q[:, :, :C_rgb]  # [B, 49, C_rgb]
        attn_rgb_dsm = torch.matmul(Q_rgb, K_dsm.transpose(-2, -1)) / (C_dsm ** 0.5)
        attn_rgb_dsm = F.softmax(attn_rgb_dsm, dim=-1)  # [B, 49, H*W]
        fusion_rgb = torch.matmul(attn_rgb_dsm, V_dsm)  # [B, 49, C_dsm]

        # DSM -> RGB融合
        Q_dsm = Q[:, :, C_rgb:]  # [B, 49, C_dsm]
        attn_dsm_rgb = torch.matmul(Q_dsm, K_rgb.transpose(-2, -1)) / (C_rgb ** 0.5)
        attn_dsm_rgb = F.softmax(attn_dsm_rgb, dim=-1)  # [B, 49, H*W]
        fusion_dsm = torch.matmul(attn_dsm_rgb, V_rgb)  # [B, 49, C_rgb]

        # 重塑并上采样到原始尺寸
        fusion_rgb = fusion_rgb.transpose(1, 2).reshape(B, C_dsm, 7, 7)
        fusion_dsm = fusion_dsm.transpose(1, 2).reshape(B, C_rgb, 7, 7)

        fusion_rgb = F.interpolate(fusion_rgb, size=(H, W), mode='bilinear', align_corners=False)
        fusion_dsm = F.interpolate(fusion_dsm, size=(H, W), mode='bilinear', align_corners=False)

        return fusion_rgb, fusion_dsm


class SelfEnhancement(nn.Module):
    """自增强模块"""

    def __init__(self, channels: int):
        super(SelfEnhancement, self).__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 线性投影
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_proj = self.linear1(x_flat)  # [B, H*W, C]
        x_proj = x_proj.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

        # 深度卷积生成注意力权重
        attn_weight = self.dwconv(x_proj)

        # 另一个线性投影
        x_value = self.linear2(x_flat)  # [B, H*W, C]
        x_value = x_value.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

        # 元素级乘法
        output = attn_weight * x_value

        return output


class MMformerLayer(nn.Module):
    """MMformer层"""

    def __init__(self, rgb_channels: int, dsm_channels: int, num_heads: int = 8, use_cross_modal: bool = True):
        super(MMformerLayer, self).__init__()
        self.use_cross_modal = use_cross_modal

        self.ln_rgb = nn.LayerNorm(rgb_channels)
        self.ln_dsm = nn.LayerNorm(dsm_channels)

        if use_cross_modal:
            self.cross_modal_attn = CrossModalAttention(rgb_channels, dsm_channels, num_heads)

        self.self_enhance_rgb = SelfEnhancement(rgb_channels)
        self.self_enhance_dsm = SelfEnhancement(dsm_channels)

        # MLP
        self.mlp_rgb = nn.Sequential(
            nn.Linear(rgb_channels, rgb_channels * 4),
            nn.GELU(),
            nn.Linear(rgb_channels * 4, rgb_channels)
        )

        self.mlp_dsm = nn.Sequential(
            nn.Linear(dsm_channels, dsm_channels * 4),
            nn.GELU(),
            nn.Linear(dsm_channels * 4, dsm_channels)
        )

        if use_cross_modal:
            # 融合后的特征投影 - 修复通道数计算
            # RGB分支: rgb_self + fuse_token + rgb_fusion + dsm_self
            # 其中 fuse_token = rgb_self + dsm_fusion (通道数为rgb_channels)
            self.fusion_proj_rgb = nn.Linear(rgb_channels * 3 + dsm_channels, rgb_channels)
            # DSM分支: dsm_self + fuse_token + dsm_fusion + rgb_self
            # 其中 fuse_token = rgb_self + dsm_fusion (通道数为rgb_channels)
            self.fusion_proj_dsm = nn.Linear(dsm_channels * 2 + rgb_channels * 2, dsm_channels)
        else:
            self.fusion_proj_rgb = nn.Linear(rgb_channels + dsm_channels, rgb_channels)
            self.fusion_proj_dsm = nn.Linear(dsm_channels + rgb_channels, dsm_channels)

    def forward(self, rgb_feat: torch.Tensor, dsm_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C_rgb, H, W = rgb_feat.shape
        _, C_dsm, _, _ = dsm_feat.shape

        # Layer Normalization
        rgb_norm = self.apply_layer_norm(rgb_feat, self.ln_rgb)
        dsm_norm = self.apply_layer_norm(dsm_feat, self.ln_dsm)

        # 自增强
        rgb_self = self.self_enhance_rgb(rgb_norm)
        dsm_self = self.self_enhance_dsm(dsm_norm)

        if self.use_cross_modal:
            # 跨模态注意力
            rgb_fusion, dsm_fusion = self.cross_modal_attn(rgb_norm, dsm_norm)

            # Token加法融合 - 确保通道数匹配
            # 将dsm_fusion投影到rgb_channels维度
            dsm_fusion_proj = F.interpolate(dsm_fusion, size=(H, W), mode='bilinear', align_corners=False)
            if dsm_fusion_proj.shape[1] != rgb_self.shape[1]:
                # 如果通道数不匹配，使用1x1卷积调整
                if not hasattr(self, 'dsm_to_rgb_proj'):
                    self.dsm_to_rgb_proj = nn.Conv2d(dsm_fusion_proj.shape[1], rgb_self.shape[1], 1).to(
                        dsm_fusion_proj.device)
                dsm_fusion_proj = self.dsm_to_rgb_proj(dsm_fusion_proj)

            fuse_token = rgb_self + dsm_fusion_proj

            # 连接所有特征
            rgb_concat = torch.cat([rgb_self, fuse_token, rgb_fusion, dsm_self], dim=1)
            dsm_concat = torch.cat([dsm_self, fuse_token, dsm_fusion, rgb_self], dim=1)
        else:
            # 仅连接自增强特征
            rgb_concat = torch.cat([rgb_self, dsm_self], dim=1)
            dsm_concat = torch.cat([dsm_self, rgb_self], dim=1)

        # 特征投影
        rgb_fused = self.apply_linear_proj(rgb_concat, self.fusion_proj_rgb)
        dsm_fused = self.apply_linear_proj(dsm_concat, self.fusion_proj_dsm)

        # 残差连接
        rgb_out = rgb_fused + rgb_feat
        dsm_out = dsm_fused + dsm_feat

        # MLP
        rgb_mlp = self.apply_mlp(rgb_out, self.mlp_rgb)
        dsm_mlp = self.apply_mlp(dsm_out, self.mlp_dsm)

        # 最终残差连接
        rgb_final = rgb_mlp + rgb_out
        dsm_final = dsm_mlp + dsm_out

        return rgb_final, dsm_final

    def apply_layer_norm(self, x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_norm = ln(x_flat)
        return x_norm.transpose(1, 2).reshape(B, C, H, W)

    def apply_linear_proj(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_proj = proj(x_flat)
        return x_proj.transpose(1, 2).reshape(B, x_proj.shape[-1], H, W)

    def apply_mlp(self, x: torch.Tensor, mlp: nn.Sequential) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_mlp = mlp(x_flat)
        return x_mlp.transpose(1, 2).reshape(B, C, H, W)


class MMformer(nn.Module):
    """MMformer模块"""

    def __init__(self, rgb_channels: int, dsm_channels: int, num_layers: int, num_heads: int = 8, level: int = 1):
        super(MMformer, self).__init__()
        self.level = level

        # 根据论文，第一层不使用跨模态注意力
        use_cross_modal = level > 1

        self.layers = nn.ModuleList([
            MMformerLayer(rgb_channels, dsm_channels, num_heads, use_cross_modal)
            for _ in range(num_layers)
        ])

    def forward(self, rgb_feat: torch.Tensor, dsm_feat: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            rgb_feat, dsm_feat = layer(rgb_feat, dsm_feat)

        # 返回RGB特征流作为输出
        return rgb_feat


class BRAFM(nn.Module):
    """边界区域注意力多级融合模块"""

    def __init__(self, channels: int):
        super(BRAFM, self).__init__()

        # 空间注意力用于边界区域
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # CBR块
        self.cbr_diff = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.cbr_out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, deep_feat: torch.Tensor, shallow_feat: torch.Tensor) -> torch.Tensor:
        # 上采样深层特征到浅层特征的尺寸
        if deep_feat.shape[2:] != shallow_feat.shape[2:]:
            deep_feat = F.interpolate(deep_feat, size=shallow_feat.shape[2:],
                                      mode='bilinear', align_corners=False)

        # 红色部分：边界区域注意力
        diff_feat = deep_feat - shallow_feat  # 特征差异
        attn_map = self.spatial_attn(diff_feat)  # 空间注意力
        weighted_feat = attn_map * deep_feat + shallow_feat  # 加权特征
        diff_out = self.cbr_diff(weighted_feat)  # CBR处理

        # 蓝色部分：特征相加
        add_feat = deep_feat + shallow_feat

        # 连接并输出
        concat_feat = torch.cat([add_feat, diff_out], dim=1)
        output = self.cbr_out(concat_feat)

        return output


class ResNetEncoder(nn.Module):
    """ResNet编码器"""

    def __init__(self, in_channels: int = 3, pretrained: bool = True, half_channels: bool = False):
        super(ResNetEncoder, self).__init__()

        if half_channels:
            # DSM编码器 - 手动构建减半通道的ResNet
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # 构建各层
            self.layer1 = self._make_layer(32, 32, 3, stride=1)  # 32 channels
            self.layer2 = self._make_layer(32, 64, 4, stride=2)  # 64 channels
            self.layer3 = self._make_layer(64, 128, 6, stride=2)  # 128 channels
            self.layer4 = self._make_layer(128, 256, 3, stride=2)  # 256 channels
        else:
            # RGB编码器 - 使用标准ResNet34
            if pretrained:
                resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet34(weights=None)

            # 修改第一层以适应输入通道数
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool

            self.layer1 = resnet.layer1  # 64 channels
            self.layer2 = resnet.layer2  # 128 channels
            self.layer3 = resnet.layer3  # 256 channels
            self.layer4 = resnet.layer4  # 512 channels

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """构建ResNet层"""
        layers = []

        # 第一个block可能需要下采样
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            downsample = None

        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))

        # 其余blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)  # 1/4
        features.append(x1)

        x2 = self.layer2(x1)  # 1/8
        features.append(x2)

        x3 = self.layer3(x2)  # 1/16
        features.append(x3)

        x4 = self.layer4(x3)  # 1/32
        features.append(x4)

        return features


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TMFNet(nn.Module):
    """Transformer-based Multi-modal Fusion Network"""

    def __init__(self, rgb_channels: int = 3, dsm_channels: int = 1, num_classes: int = 6):
        super(TMFNet, self).__init__()

        # 双分支编码器
        self.rgb_encoder = ResNetEncoder(in_channels=rgb_channels, pretrained=True, half_channels=False)
        self.dsm_encoder = ResNetEncoder(in_channels=dsm_channels, pretrained=False, half_channels=True)

        # 特征通道数
        rgb_feat_channels = [64, 128, 256, 512]
        dsm_feat_channels = [32, 64, 128, 256]  # DSM编码器通道数减半

        # 空间注意力模块 - 修复通道数匹配问题
        self.spatial_attns = nn.ModuleList([
            SpatialAttention(rgb_feat_channels[i], dsm_feat_channels[i]) for i in range(4)
        ])

        # MMformer模块
        mmformer_layers = [3, 3, 12, 2]  # 根据论文设置
        self.mmformers = nn.ModuleList([
            MMformer(rgb_feat_channels[i], dsm_feat_channels[i],
                     mmformer_layers[i], num_heads=2 ** (i + 1), level=i + 1)
            for i in range(4)
        ])

        # BRAFM模块
        self.brafms = nn.ModuleList([
            BRAFM(rgb_feat_channels[i]) for i in range(3)  # 前3层用于跳跃连接
        ])

        # 解码器
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(rgb_feat_channels[3], rgb_feat_channels[2],
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(rgb_feat_channels[2]),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(rgb_feat_channels[2], rgb_feat_channels[1],
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(rgb_feat_channels[1]),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(rgb_feat_channels[1], rgb_feat_channels[0],
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(rgb_feat_channels[0]),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(rgb_feat_channels[0], rgb_feat_channels[0],
                                   kernel_size=2, stride=2),
                nn.BatchNorm2d(rgb_feat_channels[0]),
                nn.ReLU(inplace=True)
            )
        ])

        # 分割头
        self.seg_head = nn.Conv2d(rgb_feat_channels[0], num_classes, kernel_size=1)

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        # 编码器特征提取
        rgb_features = self.rgb_encoder(rgb)
        dsm_features = self.dsm_encoder(dsm)

        # 多级特征融合
        fused_features = []
        for i in range(4):
            # 空间注意力
            rgb_weighted, dsm_weighted = self.spatial_attns[i](rgb_features[i], dsm_features[i])

            # MMformer融合
            fused_feat = self.mmformers[i](rgb_weighted, dsm_weighted)
            fused_features.append(fused_feat)

        # 解码器 - 逐步上采样并融合跳跃连接
        x = fused_features[3]  # 最深层特征

        for i in range(3):
            # 上采样
            x = self.decoder[i](x)

            # BRAFM融合跳跃连接
            skip_feat = fused_features[2 - i]  # 对应的跳跃连接特征
            x = self.brafms[2 - i](x, skip_feat)

        # 最后一次上采样到原始分辨率
        x = self.decoder[3](x)

        # 分割预测
        output = self.seg_head(x)

        return output


# 损失函数
class CombinedLoss(nn.Module):
    """组合损失函数：Cross-Entropy + Dice Loss"""

    def __init__(self, weight_ce: float = 0.5, weight_dice: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Dice损失计算"""
        pred_softmax = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)

        total_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss
        return total_loss


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = MMformer(rgb_channels=3, dsm_channels=1,num_layers=4)

    # 创建示例输入
    batch_size = 2
    rgb_input = torch.randn(batch_size, 3, 512, 512)
    dsm_input = torch.randn(batch_size, 1, 512, 512)
    target = torch.randint(0, 6, (batch_size, 512, 512))

    # # 前向传播
    # output = model(rgb_input, dsm_input)
    # print(f"Output shape: {output.shape}")
    #
    # # 损失计算
    # criterion = CombinedLoss()
    # loss = criterion(output, target)
    # print(f"Loss: {loss.item()}")
    #
    # # 计算参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")
