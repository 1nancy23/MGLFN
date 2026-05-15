import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from TeRN import TextureRefinementNetwork
from timm.models.layers import DropPath, trunc_normal_
import timm
import math
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            x = self.b4(self.pre_conv(res4))
            h4 = self.up4(x)

            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)

            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class GCSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class CharacterRefinement(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pooling_size=2):
        super(CharacterRefinement, self).__init__()

        # 第一个卷积层 - 输入到特征矩阵Xj
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # 自适应池化层 - 用于压缩特征矩阵
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(pooling_size)

        # 第二个卷积层 - 特征融合
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # 可学习的权重参数 - 对应图中的Δwj和Δbj
        self.delta_w = nn.Parameter(torch.randn(out_channels, 1, 1))
        self.delta_b = nn.Parameter(torch.randn(out_channels))

        # 批归一化
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Stage 1: 初始特征提取
        # 输入 -> ConvLayer -> 特征矩阵 Xj^(C*H*W)
        feature_map = F.relu(self.bn1(self.conv1(x)))

        # 自适应池化压缩特征
        compressed_features = self.adaptive_pooling(feature_map)

        # Character Refinement 过程
        # 应用可学习参数进行特征调整
        refined_features = compressed_features * self.delta_w + self.delta_b.view(-1, 1, 1)

        # 将压缩特征上采样回原始尺寸
        upsampled_features = F.interpolate(refined_features, size=feature_map.shape[2:],
                                           mode='bilinear', align_corners=False)

        # 特征融合 (对应图中的⊕操作)
        fused_features = feature_map + upsampled_features

        # 第二个卷积层处理融合后的特征
        output = F.relu(self.bn2(self.conv2(fused_features)))

        return output


class AdaptivePoolingWeighted(nn.Module):
    """实现基于高斯分布的加权池化"""

    def __init__(self, output_size):
        super(AdaptivePoolingWeighted, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # 计算池化窗口大小
        stride_h = height // self.output_size
        stride_w = width // self.output_size

        # 生成高斯权重
        gaussian_weights = self.generate_gaussian_weights(stride_h, stride_w)
        gaussian_weights = gaussian_weights.to(x.device)

        # 应用加权池化
        pooled = F.adaptive_avg_pool2d(x, self.output_size)

        return pooled

    def generate_gaussian_weights(self, h, w):
        """生成高斯分布权重"""
        center_h, center_w = h // 2, w // 2
        weights = torch.zeros(h, w)

        for i in range(h):
            for j in range(w):
                dist = ((i - center_h) ** 2 + (j - center_w) ** 2) ** 0.5
                weights[i, j] = math.exp(-0.5 * (dist / (min(h, w) / 4)) ** 2)

        return weights / weights.sum()


# 完整的Stage 1-res1,2 模块
class Stage1Res(nn.Module):
    def __init__(self, in_channels=3, feature_channels=64, num_blocks=2):
        super(Stage1Res, self).__init__()

        self.blocks = nn.ModuleList()

        # 第一个块
        self.blocks.append(CharacterRefinement(in_channels, feature_channels))

        # 后续块
        for _ in range(num_blocks - 1):
            self.blocks.append(CharacterRefinement(feature_channels, feature_channels))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
import sys
sys.path.append(r'D:\ZJF\种植作物分类\ZhongzhiCodes\UNetFormer_ResMMF_EdgeDe')
from UNetFormer_ResMMF_EdgeDe.MMFormer import MMformer
from UNetFormer_ResMMF_EdgeDe.Edge_detection import *
class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=False,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

        # self.TeRN1=TextureRefinementNetwork(in_channels=64,num_blocks=1)
        # self.TeRN2=TextureRefinementNetwork(in_channels=128,num_blocks=1)
        # self.TeRN3=TextureRefinementNetwork(in_channels=256,num_blocks=1)
        # self.TeRN4=TextureRefinementNetwork(in_channels=512,num_blocks=1)
        # self.label_edge = EdgeDetectionModule(method=method)
        self.feat_edgex = HybridEdgeDetection(methods=['sobel', 'laplacian'])
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.SSM1=VisionMambaBlock(64)
        # self.SSM2=VisionMambaBlock(128)
        # self.SSM3=VisionMambaBlock(256)
        # self.SSM4=VisionMambaBlock(512)
        # self.rf1= Stage1Res(64,64, 1)
        # self.rf2 = Stage1Res(128, 128, 1)
        # self.rf3 = Stage1Res(256, 256, 1)
        # self.rf4 = Stage1Res(512, 512, 1)
    #     self.GCSA1=GCSA(dim=64,num_heads=4,bias=True)
    #     self.GCSA2 = GCSA(dim=128, num_heads=4, bias=True)
    #     self.GCSA3 = GCSA(dim=256, num_heads=2, bias=True)
    #     self.GCSA4 = GCSA(dim=512, num_heads=2, bias=True)
        self.MMFormer1 = MMformer(rgb_channels=64,dsm_channels=1,num_layers=2,num_heads=4)
        self.MMFormer2 = MMformer(rgb_channels=128,dsm_channels=1,num_layers=2,num_heads=4)
        self.MMFormer3 = MMformer(rgb_channels=256,dsm_channels=1,num_layers=1,num_heads=2)
        self.MMFormer4 = MMformer(rgb_channels=512,dsm_channels=1,num_layers=1,num_heads=2)
    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        # Temp = res4
        # =self.feat_edgex(self.down(x))
        res1_dsm_return=self.down(self.feat_edgex(self.down(x)))
        # res1_dsm=torch.unsqueeze(torch.argmax(res1_dsm_return,dim=1),dim=1).float()
        res1_dsm = res1_dsm_return
        res1_array=res1_dsm
        # print(res1_dsm.shape)
        # res1_dsm=torch.unsqueeze(res1_dsm,dim=1)
        # print(res1_dsm.shape)

        # res1_array = res1_array.cpu().detach().numpy()
        # for i in range(len(res1_array)):
        #     # print(res1_array)
        #     res1_arrayx=np.transpose(res1_array[i],(1,2,0))
        #     plt.imshow(res1_arrayx)
        #     plt.colorbar()
        #     plt.show()

        res2_dsm=self.down(res1_dsm)
        res3_dsm=self.down(res2_dsm)
        res4_dsm=self.down(res3_dsm)
        # print(res1.shape)
        # print(res2.shape)
        # print(res3.shape)
        # print(res1.shape)
        # res1 = res1+self.TeRN1(res1)*0.4
        # res2 = res2+self.TeRN2(res2)*0.4
        # res3 = res3+self.TeRN3(res3)*0.4
        # res4 = res4+self.TeRN4(res4)*0.4
        res1 = res1+self.MMFormer1(res1,res1_dsm)
        res2 = res2+self.MMFormer2(res2,res2_dsm)
        res3 = res3+self.MMFormer3(res3,res3_dsm)
        res4 = res4+self.MMFormer4(res4,res4_dsm)
        Temp = res4
        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, h, w)
            return (x, ah),res1_dsm_return
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x
# modelx=UNetFormer(num_classes=9)
# A=torch.Tensor(2,3,512,512)
# B,_=modelx(A)
# print(B.shape)
# TestModel()