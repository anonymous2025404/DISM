import torchvision
import torch.nn as nn

class DenseNetDualTask(nn.Module):
    def __init__(self, num_lesions=13, num_regions=29):
        super().__init__()
        
        self.backbone = torchvision.models.densenet121(pretrained=True).features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lesion_head = nn.Sequential(
            nn.Linear(1024, num_lesions),
            nn.Sigmoid()
        )
        self.region_head = nn.Sequential(
            nn.Linear(1024, num_lesions*num_regions),
            nn.Sigmoid()
        )
        
        self.num_lesions = num_lesions
        self.num_regions = num_regions

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features).view(x.size(0), -1)
        
        lesion_pred = self.lesion_head(pooled)
        region_pred = self.region_head(pooled).view(-1, self.num_lesions, self.num_regions)
        
        return lesion_pred, region_pred
    