from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops


class CustomRoIHeads(RoIHeads):
    def __init__(
        self,
        return_feature_vectors,
        feature_map_output_size,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
        )
        # return_feature_vectors == True if we train/evaluate the object detector as part of the full model
        self.return_feature_vectors = return_feature_vectors

        # set kernel_size = feature_map_output_size, such that we average over the whole feature maps
        self.avg_pool = nn.AvgPool2d(kernel_size=feature_map_output_size)
        self.dim_reduction = nn.Linear(2048, 1024)

    def get_top_region_features_detections_class_detected(
        self,
        box_features,
        box_regression,
        class_logits,
        proposals,
        image_shapes
    ):
        pred_scores = F.softmax(class_logits, -1)
        pred_scores = pred_scores[:, 1:]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        num_images = len(boxes_per_image)
        pred_scores_per_img = torch.split(pred_scores, boxes_per_image, dim=0)

        if self.return_feature_vectors:
            region_features_per_img = torch.split(box_features, boxes_per_image, dim=0)
        else:
            region_features_per_img = [None] * num_images

        if not self.training:
            pred_region_boxes = self.box_coder.decode(box_regression, proposals)
            pred_region_boxes_per_img = torch.split(pred_region_boxes, boxes_per_image, dim=0)
        else:
            pred_region_boxes_per_img = [None] * num_images 

        output = {}
        output["class_detected"] = []
        output["top_region_features"] = []
        output["detections"] = {
            "top_region_boxes": [],
            "top_scores": []
        }

        for pred_scores_img, pred_region_boxes_img, region_features_img, img_shape in zip(pred_scores_per_img, pred_region_boxes_per_img, region_features_per_img, image_shapes):
            pred_classes = torch.argmax(pred_scores_img, dim=1)

            mask_pred_classes = torch.nn.functional.one_hot(pred_classes, num_classes=29).to(pred_scores_img.device)
            pred_top_scores_img = pred_scores_img * mask_pred_classes

            top_scores, indices_with_top_scores = torch.max(pred_top_scores_img, dim=0)

            num_predictions_per_class = torch.sum(mask_pred_classes, dim=0)

            class_detected = (num_predictions_per_class > 0)

            output["class_detected"].append(class_detected)

            if self.return_feature_vectors:
                top_region_features = region_features_img[indices_with_top_scores]
                output["top_region_features"].append(top_region_features)

            if not self.training:
                pred_region_boxes_img = box_ops.clip_boxes_to_image(pred_region_boxes_img, img_shape)
                pred_region_boxes_img = pred_region_boxes_img[:, 1:]
                top_region_boxes = pred_region_boxes_img[indices_with_top_scores, torch.arange(start=0, end=29, dtype=torch.int64, device=indices_with_top_scores.device)]

                output["detections"]["top_region_boxes"].append(top_region_boxes)
                output["detections"]["top_scores"].append(top_scores)

        output["class_detected"] = torch.stack(output["class_detected"], dim=0)

        if self.return_feature_vectors:
            output["top_region_features"] = torch.stack(output["top_region_features"], dim=0)
        if not self.training:
            output["detections"]["top_region_boxes"] = torch.stack(output["detections"]["top_region_boxes"], dim=0)
            output["detections"]["top_scores"] = torch.stack(output["detections"]["top_scores"], dim=0)

        return output

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError("target labels must of int64 type, instead got {t['labels'].dtype}")

        if targets is not None:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_roi_pool_feature_maps = self.box_roi_pool(features, proposals, image_shapes)

        box_feature_vectors = self.box_head(box_roi_pool_feature_maps)
        class_logits, box_regression = self.box_predictor(box_feature_vectors)

        detector_losses = {}

        if labels and regression_targets:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        roi_heads_output = {}
        roi_heads_output["detector_losses"] = detector_losses

        if self.return_feature_vectors or not self.training:
            box_features = self.avg_pool(box_roi_pool_feature_maps)
            box_features = torch.squeeze(box_features)

            output = self.get_top_region_features_detections_class_detected(box_features, box_regression, class_logits, proposals, image_shapes)

            roi_heads_output["class_detected"] = output["class_detected"]

            if self.return_feature_vectors:
                roi_heads_output["top_region_features"] = self.dim_reduction(output["top_region_features"])

            if not self.training:
                roi_heads_output["detections"] = output["detections"]

        return roi_heads_output
