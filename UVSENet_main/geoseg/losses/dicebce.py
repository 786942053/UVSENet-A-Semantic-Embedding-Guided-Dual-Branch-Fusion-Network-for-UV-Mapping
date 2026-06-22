import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, smooth=1e-6, bce_weight=0.5, dice_weight=0.5):
        """
        BCE + Dice Loss
        Args:
            pos_weight: (float) 给前景设置更高的权重，例如 pos_weight=背景像素数/前景像素数
            smooth: 防止除零
            bce_weight: BCE loss 权重
            dice_weight: Dice loss 权重
        """
        super(BCEDiceLoss, self).__init__()
        self.pos_weight = pos_weight
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits: (N,1,H,W) or (N,H,W) raw output from model
        targets: (N,H,W) binary ground truth {0,1}
        """
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        # BCE Loss
        bce_loss = self.bce(logits, targets.float())

        # Sigmoid + Dice Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1,2))
        dice_score = (2. * intersection + self.smooth) / (probs.sum(dim=(1,2)) + targets.sum(dim=(1,2)) + self.smooth)
        dice_loss = 1 - dice_score.mean()

        # Weighted sum
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss
