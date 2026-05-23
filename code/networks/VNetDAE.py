"""
VNetDAE: Denoising Auto-Encoder for Label Space Refinement
用于半监督学习中伪标签去噪和不确定性估计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='batchnorm'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='batchnorm'):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='batchnorm'):
        super(UpsamplingDeconvBlock, self).__init__()
        ops = []
        ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class VNetDAE(nn.Module):
    """
    Denoising Auto-Encoder based on VNet architecture
    用于对分割网络的伪标签进行去噪，提升半监督学习的鲁棒性
    
    核心思想：
    1. 输入带噪声的伪标签（来自教师模型的预测）
    2. 输出去噪后的标签，更接近真实分布
    3. 通过比较输入输出的差异来估计不确定性
    """
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='instancenorm', 
                 has_dropout=False, input_size=(112, 112, 80), is_LS_noise=True, emb_dim=512):
        super(VNetDAE, self).__init__()
        self.has_dropout = has_dropout
        self.is_LS_noise = is_LS_noise
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        
        # Encoder
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        # Bottleneck embedding layer
        # 计算bottleneck大小
        self.bottleneck_size = (input_size[0] // 16, input_size[1] // 16, input_size[2] // 16)
        bottleneck_features = n_filters * 16 * self.bottleneck_size[0] * self.bottleneck_size[1] * self.bottleneck_size[2]
        
        # 使用adaptive pooling来处理不同大小的输入
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc_encode = nn.Linear(n_filters * 16 * 64, emb_dim)
        self.fc_decode = nn.Linear(emb_dim, n_filters * 16 * 64)
        
        # Decoder
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)
        
        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        
        if has_dropout:
            self.dropout = nn.Dropout3d(p=0.5)
        
        # Label space noise injection (for training)
        self.noise_std = 0.1

    def add_label_noise(self, x):
        """添加标签空间噪声用于训练DAE"""
        if self.training and self.is_LS_noise:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            x = torch.clamp(x, 0, 1)
        return x

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        
        if self.has_dropout:
            x5 = self.dropout(x5)
        
        return [x1, x2, x3, x4, x5]

    def get_embedding(self, x5):
        """获取bottleneck embedding"""
        x_pooled = self.adaptive_pool(x5)
        x_flat = x_pooled.view(x_pooled.size(0), -1)
        emb = self.fc_encode(x_flat)
        return emb

    def decoder(self, features, emb=None, original_size=None):
        x1, x2, x3, x4, x5 = features
        
        x5_up = self.block_five_up(x5)
        # 处理尺寸不匹配
        if x5_up.shape != x4.shape:
            x5_up = F.interpolate(x5_up, size=x4.shape[2:], mode='trilinear', align_corners=True)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        if x6_up.shape != x3.shape:
            x6_up = F.interpolate(x6_up, size=x3.shape[2:], mode='trilinear', align_corners=True)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        if x7_up.shape != x2.shape:
            x7_up = F.interpolate(x7_up, size=x2.shape[2:], mode='trilinear', align_corners=True)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        if x8_up.shape != x1.shape:
            x8_up = F.interpolate(x8_up, size=x1.shape[2:], mode='trilinear', align_corners=True)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        out = self.out_conv(x9)
        
        # 确保输出尺寸与原始输入匹配
        if original_size is not None and out.shape[2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='trilinear', align_corners=True)
        
        return out

    def forward(self, x):
        """
        Args:
            x: 输入的伪标签 (one-hot或argmax后的标签)，shape: (B, 1, D, H, W) 或 (B, C, D, H, W)
        Returns:
            out: 去噪后的标签预测
            emb: bottleneck embedding
        """
        original_size = x.shape[2:]
        
        # 添加标签空间噪声（训练时）
        x = self.add_label_noise(x)
        
        features = self.encoder(x)
        emb = self.get_embedding(features[-1])
        out = self.decoder(features, emb, original_size)
        
        return out, emb


class LightweightDAE(nn.Module):
    """
    轻量级DAE，适用于资源受限场景
    使用更少的参数但保持去噪能力
    """
    def __init__(self, n_channels=1, n_classes=2, n_filters=8, normalization='instancenorm'):
        super(LightweightDAE, self).__init__()
        
        # 简化的encoder-decoder结构
        self.encoder = nn.Sequential(
            nn.Conv3d(n_channels, n_filters, 3, padding=1),
            nn.InstanceNorm3d(n_filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_filters, n_filters * 2, 3, stride=2, padding=1),
            nn.InstanceNorm3d(n_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_filters * 2, n_filters * 4, 3, stride=2, padding=1),
            nn.InstanceNorm3d(n_filters * 4),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(n_filters * 4, n_filters * 2, 2, stride=2),
            nn.InstanceNorm3d(n_filters * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(n_filters * 2, n_filters, 2, stride=2),
            nn.InstanceNorm3d(n_filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_filters, n_classes, 1),
        )
        
        self.noise_std = 0.1

    def add_noise(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return torch.clamp(x + noise, 0, 1)
        return x

    def forward(self, x):
        original_size = x.shape[2:]
        x = self.add_noise(x)
        features = self.encoder(x)
        out = self.decoder(features)
        if out.shape[2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='trilinear', align_corners=True)
        return out, features


def compute_dae_uncertainty(dae_output, original_pred, gamma=1.0):
    """
    计算DAE不确定性和确定性掩码
    
    Args:
        dae_output: DAE去噪后的输出 (B, C, D, H, W)
        original_pred: 原始模型的softmax预测 (B, C, D, H, W)
        gamma: 不确定性权重因子
    
    Returns:
        uncertainty: 不确定性图 (B, 1, D, H, W)
        certainty_mask: 确定性掩码 (B, C, D, H, W)
    """
    # 对DAE输出应用sigmoid获得概率
    dae_probs = torch.sigmoid(dae_output)
    
    # L2距离作为不确定性度量
    uncertainty = (dae_probs - original_pred) ** 2
    
    # 确定性掩码：低不确定性区域权重高
    certainty_mask = torch.exp(-gamma * uncertainty)
    
    return uncertainty, certainty_mask


def train_dae_epoch(dae_model, dataloader, optimizer, device, noise_ratio=0.3):
    """
    DAE预训练的一个epoch
    使用带标签数据对DAE进行训练
    
    Args:
        dae_model: DAE模型
        dataloader: 带标签数据的dataloader
        optimizer: 优化器
        device: 设备
        noise_ratio: 噪声比例
    """
    dae_model.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        labels = labels.to(device).float()
        
        # 将标签转换为one-hot并添加噪声
        if labels.dim() == 4:  # (B, D, H, W)
            labels = labels.unsqueeze(1)  # (B, 1, D, H, W)
        
        # 添加噪声
        noisy_labels = labels + torch.randn_like(labels) * noise_ratio
        noisy_labels = torch.clamp(noisy_labels, 0, 1)
        
        optimizer.zero_grad()
        
        # DAE前向传播
        denoised, _ = dae_model(noisy_labels)
        
        # 重建损失
        loss = F.mse_loss(torch.sigmoid(denoised), labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
