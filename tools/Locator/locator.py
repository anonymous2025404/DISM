from collections import OrderedDict
from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
# from torchinfo import summary

from tools.Locator.custom_roi_heads import CustomRoIHeads
from tools.Locator.custom_rpn import CustomRegionProposalNetwork
from tools.Locator.image_list import ImageList


class ObjectDetector(nn.Module):
    def __init__(self, return_feature_vectors=False):
        super().__init__()
        self.return_feature_vectors = return_feature_vectors

        self.num_classes = 30

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.backbone.out_channels = 2048

        self.rpn = self._create_rpn()
        self.roi_heads = self._create_roi_heads()

    def _create_rpn(self):
        anchor_generator = AnchorGenerator(
            sizes=((20, 40, 60, 80, 100, 120, 140, 160, 180, 300),),
            aspect_ratios=((0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.1, 2.6, 3.0, 5.0, 8.0),),
        )

        rpn_head = RPNHead(self.backbone.out_channels, anchor_generator.num_anchors_per_location()[0])

        rpn = CustomRegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
            score_thresh=0.0,
        )

        return rpn

    def _create_roi_heads(self):
        feature_map_output_size = 8
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=feature_map_output_size, sampling_ratio=2)

        resolution = roi_pooler.output_size[0]
        representation_size = 1024

        box_head = TwoMLPHead(self.backbone.out_channels * resolution**2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, self.num_classes)

        roi_heads = CustomRoIHeads(
            return_feature_vectors=self.return_feature_vectors,
            feature_map_output_size=feature_map_output_size,
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.01,
            nms_thresh=0.0,
            detections_per_img=100,
        )

        return roi_heads

    def _check_targets(self, targets):
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")

        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            if not isinstance(boxes, torch.Tensor):
                torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

            torch._assert(
                len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
            )

            # x1 should always be < x2 and y1 should always be < y2
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width." f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    def _transform_inputs_for_rpn_and_roi(self, images, features):
        images = ImageList(images)
        features = OrderedDict([("0", features)])

        return images, features

    def forward(self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None):
        if targets is not None:
            self._check_targets(targets)

        features = self.backbone(images)

        images, features = self._transform_inputs_for_rpn_and_roi(images, features)

        proposals, proposal_losses = self.rpn(images, features, targets)
        roi_heads_output = self.roi_heads(features, proposals, images.image_sizes, targets)

        detector_losses = roi_heads_output["detector_losses"]

        if not self.training:
            detections = roi_heads_output["detections"]
            class_detected = roi_heads_output["class_detected"]

        if self.return_feature_vectors:
            top_region_features = roi_heads_output["top_region_features"]
            class_detected = roi_heads_output["class_detected"]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if not self.return_feature_vectors:
            if self.training:
                return losses
            else:
                return losses, detections, class_detected

        if self.return_feature_vectors:
            if self.training:
                return losses, top_region_features, class_detected
            else:
                return losses, detections, top_region_features, class_detected
