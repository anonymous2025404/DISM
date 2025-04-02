import sys, os
sys.path.append("")

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import cv2
from typing import List, Dict
from torch import Tensor
import numpy as np
import random
import logging
from ast import literal_eval
import pandas as pd
from tqdm import tqdm

from model.densenetDualHead import DenseNetDualTask
from tools.LabeltoRegion.loss import MaskedBCELoss
from tools.LabeltoRegion.lr_dataset import CustomImageDataset

log = logging.getLogger(__name__)

IMAGE_INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8
SEED = 3407
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ""
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 0.2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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
        ]
    )

    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms

def collate_fn(batch: List[Dict[str, Tensor]]):
    # each dict in batch (which is a list) is for a single image and has the keys "image", "boxes", "labels"
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

def get_datasets_as_dfs(dataset_path):
    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_anatomicalfinding"]
    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval, "bbox_anatomicalfinding": literal_eval}

    datasets_as_dfs = {dataset: os.path.join(dataset_path, dataset) + "_13labels_02.csv" for dataset in ["train", "valid"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])

    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Train: {new_num_samples_train} images")
    log.info(f"Val: {new_num_samples_val} images")

    # with open(config_file_path, "a") as f:
    #     f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
    #     f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    return datasets_as_dfs

def get_data_loaders(train_dataset, val_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader

def validate_stage1(model, val_loader):
    model.eval()

    val_loss = 0.0
    num_images = 0

    criterion = nn.BCELoss(size_average = True)
    with torch.no_grad():
        for batch_num, batch in tqdm(enumerate(val_loader)):
            images, targets = batch.values()

            batch_size = images.size(0)
            # num_images += batch_size

            images = images.to(DEVICE, non_blocking=True)
            # targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
            lesion_target = [d["findings"].to(DEVICE, non_blocking=True) for d in targets]
            lesion_target = torch.stack(lesion_target).float().squeeze()
            
            lesion_pred, _ = model(images)
            
            loss = criterion(lesion_pred, lesion_target)
            val_loss += loss.item() * batch_size
    val_loss /= len(val_loader)
    return val_loss

def validate_stage2(model, val_loader):
    model.eval()

    val_loss = 0.0

    criterion = nn.BCELoss(size_average = True)
    with torch.no_grad():
        for batch_num, batch in tqdm(enumerate(val_loader)):
            images, targets = batch.values()

            batch_size = images.size(0)
            # num_images += batch_size

            images = images.to(DEVICE, non_blocking=True)
            # targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
            lesion_target = [d["findings"].to(DEVICE, non_blocking=True) for d in targets]
            lesion_target = torch.stack(lesion_target).float().squeeze()
            region_target = [d["labels"].to(DEVICE, non_blocking=True) for d in targets]
            region_target = torch.stack(region_target).float().squeeze()
            
            _, region_pred = model(images)
            
            loss = criterion(region_pred, region_target)
            val_loss += loss.item() * batch_size

    val_loss /= len(val_loader)
    return val_loss


def train_stage1(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    for param in model.region_head.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters()},
        {'params': model.lesion_head.parameters()}
    ], lr=lr)
    
    criterion = nn.BCELoss(size_average = True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for num_batch, batch in tqdm(enumerate(train_loader)):
            img, targets = batch.values()
            batch_size = img.size(0)
            # print(type(targets))
            # print(targets)
            # print(targets[0])
            # print(targets[0]['findings'])
            # print(targets[0]['labels'])
            # print(targets)

            lesion_target = [d["findings"].to(DEVICE, non_blocking=True) for d in targets]
            lesion_target = torch.stack(lesion_target).float().squeeze()
            # lesion_target = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in lesion_target]
            
            img = img.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            lesion_pred, _ = model(img)
            # print(f"pred_type: {type(lesion_pred)}, target_type: {type(lesion_target)}\n")
            # print(f"Pred shape: \n{lesion_pred}")
            # print("-----------------------------------------------")
            # print(f"Target shape: \n{lesion_target}")
            # input()
            loss = criterion(lesion_pred, lesion_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()* batch_size

        vloss = validate_stage1(model, val_loader)
        train_loss /= len(train_loader)
        print(f"Stage: 1. Epoch {epoch+1}, TrainLoss: {train_loss}, ValLoss: {vloss}")
        torch.save(model.state_dict(), f"{CKPT_DIR}/model_s1_e{epoch+1}_vloss{vloss}.pth")

def train_stage2(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.region_head.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.Adam(model.region_head.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    train_loss = 0.0
    for epoch in range(num_epochs):
        if epoch < 10:
            continue
        for num_batch, batch in tqdm(enumerate(train_loader)):
            img, targets = batch.values()
            optimizer.zero_grad()
            batch_size = img.size(0)
            # targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
            lesion_target = [d["findings"].to(DEVICE, non_blocking=True) for d in targets]
            region_target = [d["labels"].to(DEVICE, non_blocking=True) for d in targets]
            img = img.to(DEVICE, non_blocking=True)
            lesion_target = torch.stack(lesion_target).float().squeeze()
            region_target = torch.stack(region_target).float().squeeze()
            _, region_pred = model(img)

            loss = criterion(region_pred, region_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()* batch_size
        vloss = validate_stage2(model, val_loader)
        train_loss /= len(train_loader)
        print(f"Stage: 2. Epoch {epoch+1}, TrainLoss: {train_loss}, ValLoss: {vloss}")
        torch.save(model.state_dict(), f"{CKPT_DIR}/model_s2_e{epoch+1}_vloss{vloss}.pth")


if __name__ == "__main__":
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    model = DenseNetDualTask(num_lesions=13, num_regions=29)
    model.load_state_dict(torch.load(""))
    model.to(DEVICE)

    # dataset_path = ""

    datasets_as_dfs = get_datasets_as_dfs("")
    train_dataset = CustomImageDataset(datasets_as_dfs["train"], transforms=get_transforms("train"))
    val_dataset = CustomImageDataset(datasets_as_dfs["valid"], transforms=get_transforms("valid"))

    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset)

    # train_stage1(model, train_loader, val_loader, num_epochs=10, lr=1e-4)
    
    train_stage2(model, train_loader, val_loader, num_epochs=20, lr=1e-4)