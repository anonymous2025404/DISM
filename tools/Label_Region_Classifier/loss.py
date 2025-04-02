import torch.nn as nn


class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, pred, target, mask):
        loss = self.bce(pred, target)
        masked_loss = loss * mask.unsqueeze(-1)
        return masked_loss.sum() / (mask.sum() + 1e-8)