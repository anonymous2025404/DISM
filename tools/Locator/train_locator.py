from ast import literal_eval
from copy import deepcopy
import logging
import os
import random
from typing import List, Dict

import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ANATOMICAL_REGIONS
from tools.Locator.custom_image_dataset_locator import CustomImageDataset
from tools.Locator.locator import ObjectDetector


device = torch.device("cuda:0")
model = ObjectDetector(return_feature_vectors=False)
# print(f"Number of available GPUs: {torch.cuda.device_count()}")
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = nn.DataParallel(model, device_ids=[1, 0])
model.to(device, non_blocking=True)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)


RUN = 1
RUN_COMMENT = """Enter comment here."""
SEED = 41
IMAGE_INPUT_SIZE = 512
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 0.2
BATCH_SIZE = 16
EFFECTIVE_BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 20
LR = 1e-3
EVALUATE_EVERY_K_STEPS = 1000
PATIENCE_LR_SCHEDULER = 5
THRESHOLD_LR_SCHEDULER = 1e-3
FACTOR_LR_SCHEDULER = 0.5
COOLDOWN_LR_SCHEDULER = 5

# set the seed value for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def get_title(region_set, region_indices, region_colors, class_detected_img):
    class_detected = [class_detected_img[region_index] for region_index in region_indices]
    region_set = [region + f" ({color})" if cls_detect else region + f" ({color}, nd)" for region, color, cls_detect in zip(region_set, region_colors, class_detected)]
    return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])


def plot_box(box, ax, clr, linestyle, class_detected=True):
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    ax.add_artist(
        plt.Rectangle(
            xy=(x0, y0),
            height=h,
            width=w,
            fill=False,
            color=clr,
            linewidth=1,
            linestyle=linestyle
        )
    )

    if not class_detected:
        ax.annotate("not detected", (x0, y0), color=clr, weight="bold", fontsize=10)


def plot_gt_and_pred_bboxes_to_tensorboard(writer, overall_steps_taken, images, detections, targets, class_detected, num_images_to_plot=2):
    from PIL import Image
    import io
    pred_boxes_batch = detections["top_region_boxes"]

    gt_boxes_batch = torch.stack([t["boxes"] for t in targets], dim=0)

    region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
    region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
    region_set_3 = ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"]
    region_set_4 = ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"]
    region_set_5 = ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]

    regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5]
    all_regions = []
    all_regions.extend(region_set_1)
    all_regions.extend(region_set_2)
    all_regions.extend(region_set_3)
    all_regions.extend(region_set_4)
    all_regions.extend(region_set_5)
    all_regions_list = []
    all_regions_list.append(all_regions)

    images = images.numpy().transpose(0, 2, 3, 1)

    for num_img, image in enumerate(images):

        gt_boxes_img = gt_boxes_batch[num_img]
        pred_boxes_img = pred_boxes_batch[num_img]
        class_detected_img = class_detected[num_img].tolist()

        # print(gt_boxes_img)
        # print(pred_boxes_img)

        for num_region_set, region_set in enumerate(all_regions_list):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()

            plt.imshow(image, cmap="gray")
            plt.axis("on")

            region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]
            region_colors = ["r", "g", "b", "c", "m", "w", "orange", "dodgerblue"]
            cnt = 0

            for region_index in region_indices:
                box_gt = gt_boxes_img[region_index].tolist()
                box_pred = pred_boxes_img[region_index].tolist()

                box_region_detected = class_detected_img[region_index]
                if box_region_detected:
                    color = region_colors[cnt]
                    cnt = (cnt+1) % len(region_colors)
                    print(f"region_index: {region_index} - box_gt:{box_gt} box_pred:{box_pred}, color:{color}")
                    # plot_box(box_gt, ax, clr=color, linestyle="solid")
                    plot_box(box_pred, ax, clr=color, linestyle="dashed")

            # title = get_plot_title(region_set, region_indices, region_colors, class_detected_img)
            # ax.set_title(title)

            buf = io.BytesIO()
            fig.savefig(buf, bbox_inches="tight")
            buf.seek(0)
            im = Image.open(buf)
            im = np.asarray(im)[..., :3]

            pic_save_path = ""
            cv2.imwrite(pic_save_path, im)
            plt.close(fig)


def compute_box_area(box):
    x0 = box[..., 0]
    y0 = box[..., 1]
    x1 = box[..., 2]
    y1 = box[..., 3]

    return (x1 - x0) * (y1 - y0)


def compute_intersection_and_union_area_per_class(detections, targets, class_detected):

    pred_boxes = detections["top_region_boxes"]

    gt_boxes = torch.stack([t["boxes"] for t in targets], dim=0)

    x0_max = torch.maximum(pred_boxes[..., 0], gt_boxes[..., 0])
    y0_max = torch.maximum(pred_boxes[..., 1], gt_boxes[..., 1])
    x1_min = torch.minimum(pred_boxes[..., 2], gt_boxes[..., 2])
    y1_min = torch.minimum(pred_boxes[..., 3], gt_boxes[..., 3])
    intersection_boxes = torch.stack([x0_max, y0_max, x1_min, y1_min], dim=-1)

    intersection_area = compute_box_area(intersection_boxes)
    pred_area = compute_box_area(pred_boxes)
    gt_area = compute_box_area(gt_boxes)

    valid_intersection = torch.logical_and(x0_max < x1_min, y0_max < y1_min)

    valid_intersection = torch.logical_and(valid_intersection, class_detected)

    intersection_area = torch.where(valid_intersection, intersection_area, torch.tensor(0, dtype=intersection_area.dtype, device=intersection_area.device))

    union_area = (pred_area + gt_area) - intersection_area

    intersection_area = torch.sum(intersection_area, dim=0)
    union_area = torch.sum(union_area, dim=0)

    return intersection_area, union_area


def get_val_loss_and_other_metrics(model, val_dl, writer, overall_steps_taken):
    model.eval()
    val_loss = 0.0
    num_images = 0

    sum_class_detected = torch.zeros(29, device=device)

    sum_intersection_area_per_class = torch.zeros(29, device=device)
    sum_union_area_per_class = torch.zeros(29, device=device)

    with torch.no_grad():
        for batch_num, batch in tqdm(enumerate(val_dl)):
            images, targets = batch.values()

            batch_size = images.size(0)
            num_images += batch_size

            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict, detections, class_detected = model(images, targets)

            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item() * batch_size

            sum_class_detected += torch.sum(class_detected, dim=0)

            intersection_area_per_class, union_area_per_class = compute_intersection_and_union_area_per_class(detections, targets, class_detected)
            sum_intersection_area_per_class += intersection_area_per_class
            sum_union_area_per_class += union_area_per_class

            # if batch_num == 0:
                # plot_gt_and_pred_bboxes_to_tensorboard(writer, overall_steps_taken, images, detections, targets, class_detected, num_images_to_plot=2)

    val_loss /= len(val_dl)
    avg_num_detected_classes_per_image = torch.sum(sum_class_detected / num_images).item()
    avg_detections_per_class = (sum_class_detected / num_images).tolist()
    avg_iou_per_class = (sum_intersection_area_per_class / sum_union_area_per_class).tolist()

    return val_loss, avg_num_detected_classes_per_image, avg_detections_per_class, avg_iou_per_class


def log_stats_to_console(
    train_loss,
    val_loss,
    epoch,
):
    log.info(f"Epoch: {epoch}:")
    log.info(f"\tTrain loss: {train_loss:.3f}")
    log.info(f"\tVal loss: {val_loss:.3f}")


def train_model(
    model,
    train_dl,
    val_dl,
    optimizer,
    scaler,
    lr_scheduler,
    epochs,
    weights_folder_path,
    writer
):

    lowest_val_loss = np.inf

    best_model_state = None

    overall_steps_taken = 0 

    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE

    for epoch in range(epochs):
        if epoch < 18:
            continue
        log.info(f"Training epoch {epoch}!")

        train_loss = 0.0
        steps_taken = 0
        for num_batch, batch in tqdm(enumerate(train_dl)):
            images, targets = batch.values()

            batch_size = images.size(0)

            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)

                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()

            if (num_batch + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * batch_size
            steps_taken += 1
            overall_steps_taken += 1

            if steps_taken >= EVALUATE_EVERY_K_STEPS or (num_batch + 1) == len(train_dl):
                log.info(f"Evaluating at step {overall_steps_taken}!")

                train_loss /= steps_taken

                val_loss, avg_num_detected_classes_per_image, avg_detections_per_class, avg_iou_per_class = get_val_loss_and_other_metrics(model, val_dl, writer, overall_steps_taken)

                writer.add_scalars("_loss", {"train_loss": train_loss, "val_loss": val_loss}, overall_steps_taken)
                writer.add_scalar("avg_num_predicted_classes_per_image", avg_num_detected_classes_per_image, overall_steps_taken)

                anatomical_regions = ["_".join(region.split()) for region in ANATOMICAL_REGIONS]

                for class_, avg_detections_class in zip(anatomical_regions, avg_detections_per_class):
                    writer.add_scalar(f"num_preds_{class_}", avg_detections_class, overall_steps_taken)

                for class_, avg_iou_class in zip(anatomical_regions, avg_iou_per_class):
                    writer.add_scalar(f"iou_{class_}", avg_iou_class, overall_steps_taken)

                current_lr = float(optimizer.param_groups[0]["lr"])
                writer.add_scalar("lr", current_lr, overall_steps_taken)

                log.info(f"Metrics evaluated at step {overall_steps_taken}!")

                model.train()

                lr_scheduler.step(val_loss)

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    best_epoch = epoch
                    best_model_save_path = os.path.join(
                        weights_folder_path, f"val_loss_{lowest_val_loss:.3f}_epoch_{epoch}.pth"
                    )
                    best_model_state = deepcopy(model.state_dict())

                if (num_batch + 1) == len(train_dl):
                    log_stats_to_console(train_loss, val_loss, epoch)

                train_loss = 0.0
                steps_taken = 0

        torch.save(best_model_state, best_model_save_path)

    log.info("Finished training!")
    log.info(f"Lowest overall val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
    return None


def collate_fn(batch: List[Dict[str, Tensor]]):
    batch = list(filter(lambda x: x is not None, batch))

    image_shape = batch[0]["image"].size()
    images_batch = torch.empty(size=(len(batch), *image_shape))

    for i, sample in enumerate(batch):
        images_batch[i] = sample.pop("image")

    targets = batch

    batch_new = {}
    batch_new["images"] = images_batch
    batch_new["targets"] = targets

    return batch_new


def get_data_loaders(train_dataset, val_dataset, test_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_transforms(dataset: str):
    mean = 0.471
    std = 0.302

    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )
    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms


def get_datasets_as_dfs(dataset_path):
    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels"]

    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval}

    datasets_as_dfs = {dataset: os.path.join(dataset_path, dataset) + "_13labels_01.csv"  for dataset in ["train", "valid", "test"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])
    total_num_samples_val = len(datasets_as_dfs["test"])


    log.info(f"Train: {total_num_samples_train} images")
    log.info(f"Val: {total_num_samples_val} images")
    log.info(f"Test: {total_num_samples_val} images")


    return datasets_as_dfs


def main_train():
    weights_folder_path = ""

    datasets_as_dfs = get_datasets_as_dfs(dataset_path="")

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")
    test_transforms = get_transforms("val")

    train_dataset = CustomImageDataset(datasets_as_dfs["train"], train_transforms)
    val_dataset = CustomImageDataset(datasets_as_dfs["valid"], val_transforms)
    test_dataset = CustomImageDataset(datasets_as_dfs["test"], test_transforms)

    train_loader, val_loader, test_loader = get_data_loaders(train_dataset, val_dataset)
    model.load_state_dict(torch.load(os.path.join(weights_folder_path, "")))
    
    model.train()

    scaler = torch.cuda.amp.GradScaler()

    opt = AdamW(model.parameters(), lr=LR)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=FACTOR_LR_SCHEDULER, patience=PATIENCE_LR_SCHEDULER, threshold=THRESHOLD_LR_SCHEDULER, cooldown=COOLDOWN_LR_SCHEDULER)
    # writer = SummaryWriter(log_dir=tensorboard_folder_path)

    log.info("\nStarting training!\n")
    train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        optimizer=opt,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        epochs=EPOCHS,
        weights_folder_path=weights_folder_path,
        # writer=writer
        writer=None
    )


if __name__ == "__main__":
    main_train()
