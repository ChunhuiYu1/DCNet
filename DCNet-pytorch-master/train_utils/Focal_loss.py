import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # 计算Focal Loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return torch.mean(focal_loss)

# 创建Focal Loss实例
# criterion = FocalLoss(alpha=0.25, gamma=2.0)
#
# # 计算模型的预测
# outputs = model(inputs)
#
# # 计算损失
# loss = criterion(outputs, targets)
#
# # 反向传播和优化
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
