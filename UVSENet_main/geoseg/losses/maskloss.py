import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossWithIgnore(nn.Module):
    def __init__(self, ignore_index=255, weight=None):
        super(CrossEntropyLossWithIgnore, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, pred, target):
        """
        pred: (B, C, H, W)  网络输出
        target: (B, H, W)   标签
        """
        # 检查尺寸
        if pred.size(2) != target.size(1) or pred.size(3) != target.size(2):
            raise ValueError(f"预测尺寸 {pred.shape} 和标签尺寸 {target.shape} 不匹配！")

        # 交叉熵 loss
        loss = F.cross_entropy(
            pred, 
            target.long(), 
            weight=self.weight, 
            ignore_index=self.ignore_index,  # 这里会自动忽略 255 的像素
            reduction="mean"
        )
        return loss
