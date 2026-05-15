import torch
import torch.nn as nn
import math

class SinCosPositionEncoding2D(nn.Module):
    def __init__(self, height: int = 512, width: int = 512):
        super().__init__()
        self.height = height
        self.width = width

        # 生成[0, height-1]和[0, width-1]的坐标张量，float32类型
        y_pos = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # [H,1]
        x_pos = torch.arange(width, dtype=torch.float32).unsqueeze(0)   # [1,W]

        # 位置归一化到[0,1]
        self.register_buffer('y_pos', y_pos / (height - 1))
        self.register_buffer('x_pos', x_pos / (width - 1))

        # 计算position编码维度
        # 这里我们只要最终输出通道为1, 所以内部隐含编码维度我们设定为64作为映射维度
        self.dim = 64
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))
        self.register_buffer('div_term', div_term)  # [dim/2]

    def forward(self, batch_size: int):
        # 计算位置角度编码，y部分
        y_angles = self.y_pos * 2 * math.pi  # 归一化 * 2pi
        y_angles = y_angles.unsqueeze(-1) * self.div_term  # [H,1,dim/2]
        # print('-----------',y_angles.shape)
        # sin和cos interleaved
        y_embed = torch.cat([torch.sin(y_angles), torch.cos(y_angles)], dim=-1)  # [H,1,dim]

        # x部分
        x_angles = self.x_pos * 2 * math.pi
        x_angles = x_angles.unsqueeze(-1) * self.div_term  # [1,W,dim/2]
        x_embed = torch.cat([torch.sin(x_angles), torch.cos(x_angles)], dim=-1)  # [1,W,dim]

        # 扩展并复用x和y维度，使其广播和叠加
        y_embed = y_embed.expand(self.height, self.width, -1)  # [H,W,dim]
        x_embed = x_embed.expand(self.height, self.width, -1)  # [H,W,dim]

        # 按维度融合x和y编码（元素相加）
        pos_encoding = y_embed + x_embed  # [H,W,dim]

        # 将pos_encoding从(H,W,dim)变成(1, dim, H, W)再取平均法降维至1通道
        # 也可以通过线性层降维，这里简单取均值
        pos_encoding = pos_encoding.permute(2, 0, 1).unsqueeze(0)  # [1, dim, H, W]
        pos_encoding = pos_encoding.mean(dim=1, keepdim=True)  # [1,1,H,W]

        # 扩展批次维度
        pos_encoding = pos_encoding.repeat(batch_size, 1, 1, 1)  # [bs,1,H,W]

        # pos_encoding = pos_encoding.reshape(batch_size,feat_size,-1,feat_size).permute(0,2,1,3)

        return pos_encoding
def split_and_concat(tensor, tile_size=128):
    bs, c, h, w = tensor.shape
    assert c == 1, "通道数应为1"
    assert h % tile_size == 0 and w % tile_size == 0, "高宽必须能被tile_size整除"

    tiles = []
    num_h = h // tile_size
    num_w = w // tile_size

    # 逐个切块
    for i in range(num_h):
        for j in range(num_w):
            # 计算切割窗口
            h_start = i * tile_size
            h_end = (i + 1) * tile_size
            w_start = j * tile_size
            w_end = (j + 1) * tile_size

            # 切片
            tile = tensor[:, :, h_start:h_end, w_start:w_end]  # shape: (bs, 1, 128, 128)
            tiles.append(tile)

    # 按通道维度拼接
    out = torch.cat(tiles, dim=1)  # shape: (bs, 16, 128, 128)

    return out

# 测试代码:
if __name__ == "__main__":
    pe = SinCosPositionEncoding2D(512, 512)
    batch_size = 2
    pos_enc = pe(batch_size)
    print(pos_enc.shape,"----")
    # pos_enc = pos_enc.reshape(batch_size,128,-1,128,)
    pos_enc=split_and_concat(pos_enc,128)
    Temp=pos_enc[0,0,:,:].detach().cpu().numpy()
    # Temp=pos_enc[0][0].detach().cpu().numpy()
    from matplotlib import pyplot as plt
    plt.imshow(Temp)
    plt.colorbar()
    plt.show()
    print(f"Position encoding shape: {pos_enc.shape}")  # should be [2,1,512,512]