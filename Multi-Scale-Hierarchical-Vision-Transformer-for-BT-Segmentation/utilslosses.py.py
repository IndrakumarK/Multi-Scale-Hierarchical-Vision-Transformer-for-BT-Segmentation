import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, lambda_dice=0.5, lambda_ce=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.lambda_dice * dice + self.lambda_ce * ce