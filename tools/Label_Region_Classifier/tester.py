import sys, os

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
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score

from model.densenetDualHead import DenseNetDualTask
from tools.LabeltoRegion.loss import MaskedBCELoss
from tools.LabeltoRegion.lr_dataset import CustomImageDataset
from config import *

log = logging.getLogger(__name__)

IMAGE_INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8
SEED = 3407 # 赞美欧姆弥赛亚
DEVICE = torch.device("cuda")
CKPT_DIR = ""
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 1.0
PERCENTAGE_OF_TEST_SET_TO_USE = 1.0

CKPT_S2 = ""

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



def get_region_acc(true_list, pred_list, threshold=0.5):
    def convert_type(list): #batch * 13 * 29
        re_list = []
        for i in list:
            lr = []
            for j in i:
                r = []
                for k in j:
                    if k == 1:
                        r.append(1)
                    else:
                        r.append(0)
                lr.append(r)
            re_list.append(lr)
        return list
    true_labels = np.array(true_list)
    pred_labels = np.array(pred_list)

    pred_labels = np.where(pred_labels > threshold, 1, 0)
    # pred_labels = pred_labels.astype('int64')
    # true_labels = true_labels.astype('int64')
    # pred_labels = convert_type(pred_labels)
    # true_labels = convert_type(true_labels)

    # from sklearn.utils.multiclass import type_of_target
    # print(type_of_target(pred_labels))
    # print(type_of_target(true_labels))
    # print(true_labels.shape)
    # print(pred_labels.shape)
    # print(true_labels[0][0])

    # acc = np.sum(true_labels == pred_labels) / np.sum(true_labels != -1)
    # print("----------------------------\nRegion Acc:", acc)
    precision_all = precision_score(true_labels, pred_labels, average='micro')
    recall_all = recall_score(true_labels, pred_labels, average='micro')
    f1_all = f1_score(true_labels, pred_labels, average='micro')

    print("----------------------------\nRegion Precision:", precision_all)
    print("----------------------------\nRegion Recall:", recall_all)
    print("----------------------------\nRegion F1 score:", f1_all)
    return precision_all, recall_all, f1_all

def get_prec(true_list, pred_list, del_row, pathologies_list):
    true_labels = np.array(true_list)
    pred_labels = np.array(pred_list)
    if del_row != None:
        true_labels = np.delete(true_labels, del_row, axis=1)
        pred_labels = np.delete(pred_labels, del_row, axis=1)
    
    # print(true_labels.shape)
    # print(pred_labels.shape)
    # from sklearn.utils.multiclass import type_of_target
    # print(type_of_target(pred_labels))
    # print(type_of_target(true_labels))
    
    precision_all = precision_score(true_labels, pred_labels, average='micro')
    print("----------------------------\nPrecision:", precision_all)
    for i in range(len(pathologies_list)):
        gt_row = true_labels[:,[i]]
        pred_row = pred_labels[:,[i]]
        precision = precision_score(gt_row, pred_row)
        print(f"{pathologies_list[i]}: {precision}")
    return precision_all

def get_recall(true_list, pred_list, del_row, pathologies_list):
    true_labels = np.array(true_list)
    pred_labels = np.array(pred_list)
    if del_row != None:
        true_labels = np.delete(true_labels, del_row, axis=1)
        pred_labels = np.delete(pred_labels, del_row, axis=1)
    recall_all = recall_score(true_labels, pred_labels, average='micro')
    print("----------------------------\nRecall:", recall_all)
    for i in range(len(pathologies_list)):
        gt_row = true_labels[:,[i]]
        pred_row = pred_labels[:,[i]]
        recall = recall_score(gt_row, pred_row)
        print(f"{pathologies_list[i]}: {recall}")
    return recall_all

def get_f1(true_list, pred_list, del_row, pathologies_list):
    true_labels = np.array(true_list)
    pred_labels = np.array(pred_list)
    if del_row != None:
        true_labels = np.delete(true_labels, del_row, axis=1)
        pred_labels = np.delete(pred_labels, del_row, axis=1)
    f1_all = f1_score(true_labels, pred_labels, average='micro')
    print("----------------------------\nF1 score:", f1_all)
    for i in range(len(pathologies_list)):
        gt_row = true_labels[:,[i]]
        pred_row = pred_labels[:,[i]]
        f1 = f1_score(gt_row, pred_row)
        print(f"{pathologies_list[i]}: {f1}")
    return f1_all

def get_auc(true_list, pred_list, del_row, pathologies_list):
    true_labels = np.array(true_list)
    pred_labels = np.array(pred_list)
    if del_row != None:
        true_labels = np.delete(true_labels, del_row, axis=1)
        pred_labels = np.delete(pred_labels, del_row, axis=1)
    roc_auc_all = roc_auc_score(true_labels, pred_labels, average='micro')
    print("----------------------------\nROC AUC score:", roc_auc_all)
    for i in range(len(pathologies_list)):
        gt_row = true_labels[:,[i]]
        pred_row = pred_labels[:,[i]]
        roc_auc = roc_auc_score(gt_row, pred_row)
        print(f"{pathologies_list[i]}: {roc_auc}")    
    return roc_auc_all

def process_chexbert_file():
    test_df = pd.read_csv(TEST_PATH, header=0)
    test_df['study_id'] = test_df['study_id'].astype(str)
    test_df.fillna(0, inplace=True)
    test_df.replace(-1.0, 1.0, inplace=True)
    train_df = pd.read_csv(TRAIN_PATH, header=0)
    train_df['study_id'] = train_df['study_id'].astype(str)
    train_df.fillna(0, inplace=True)
    train_df.replace(-1.0, 1.0, inplace=True)
    val_df = pd.read_csv(VAL_PATH, header=0)
    val_df['study_id'] = val_df['study_id'].astype(str)
    val_df.fillna(0, inplace=True)
    val_df.replace(-1.0, 1.0, inplace=True)
    chexbert_all_df = pd.concat([train_df, val_df, test_df])
    
    return chexbert_all_df


def get_datasets_as_dfs(dataset_path):
    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_anatomicalfinding"]

    # since bbox_coordinates and bbox_labels are stored as strings in the csv_file, we have to apply
    # the literal_eval func to convert them to python lists
    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval, "bbox_anatomicalfinding": literal_eval}

    datasets_as_dfs = {dataset: os.path.join(dataset_path, dataset) + "_13labels_02.csv" for dataset in ["train", "valid", "test"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])
    total_num_samples_test = len(datasets_as_dfs["test"])

    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)
    new_num_samples_test = int(PERCENTAGE_OF_TEST_SET_TO_USE * total_num_samples_test)

    log.info(f"Train: {new_num_samples_train} images")
    log.info(f"Val: {new_num_samples_val} images")
    log.info(f"Test: {new_num_samples_test} images")

    # with open(config_file_path, "a") as f:
    #     f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
    #     f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]
    datasets_as_dfs["test"] = datasets_as_dfs["test"][:new_num_samples_test]
    return datasets_as_dfs

def get_test_datasets_as_dfs(dataset_path):
    usecols = ["image_id", "mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_anatomicalfinding"]

    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval, "bbox_anatomicalfinding": literal_eval, "image_id": str}

    datasets_as_dfs = {dataset: os.path.join(dataset_path, dataset) + "_13labels_02.csv" for dataset in ["test", "valid"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

    # datasets_as_dfs["test"] = datasets_as_dfs["test"][:10]
    # print(datasets_as_dfs["test"])
    # total_num_samples_test = len(datasets_as_dfs["test"])

    # new_num_samples_test = int(PERCENTAGE_OF_TEST_SET_TO_USE * total_num_samples_test)

    # log.info(f"Test: {new_num_samples_test} images")

    # datasets_as_dfs["test"] = datasets_as_dfs["test"][:new_num_samples_test]
    # datasets_as_dfs["test"] = datasets_as_dfs["valid"]
    return datasets_as_dfs

def get_data_loaders(train_dataset, val_dataset):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def get_test_data_loaders(val_dataset):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return val_loader

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

def find_chexbert_label(dicom_id: str):
    line = chexbert_df.loc[chexbert_df["dicom_id"] == dicom_id]
    if len(line) == 0:
        return []
    else:
        findings = []
        for p in CLASS:
            findings.append(line[p].values[0])
        findings.append(line['No Finding'].values[0])
        return findings


def test_2stage(model, test_loader):
    from tools.LabeltoRegion.prob_to_labels import load_prior_knowledge

    prior_dict = load_prior_knowledge()
    reverse_dict = {v: k for k, v in ANATOMICAL_REGIONS.items()}

    def prior_constranint(pathology:str, regions):
        for i, region in enumerate(regions):
            region_name = reverse_dict[i]
            if prior_dict[pathology][region_name] == 0:
                regions[i] = 0
        return regions
    
    model.eval()
    gt_list = []
    gt_13_list = []
    pred_list = []
    out_pred_list = []
    region_pred_pk_list = []
    region_gt_list = []
    region_pred_list = []

    out_all = {'dicom_id': [], 'findings': [], 'region': []}

    with torch.no_grad():
        for batch_num, batch in tqdm(enumerate(test_loader)):
            images, targets = batch.values()

            # num_images += batch_size

            images = images.to(DEVICE, non_blocking=True)
            # targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
            lesion_target = [d["findings"].to(DEVICE, non_blocking=True) for d in targets]
            lesion_target = torch.stack(lesion_target).float().squeeze()
            lesion_target = lesion_target.cpu().numpy().tolist()
            # print(lesion_target)

            region_target = [d["labels"].to(DEVICE, non_blocking=True) for d in targets] # batch * 29
            region_target = torch.stack(region_target).float().squeeze()
            region_target = region_target.cpu().numpy().tolist()
            # print(region_target)
            # input()

            lesion_pred, region_pred = model(images)
            lesion_pred = lesion_pred.cpu().numpy().tolist()
            region_pred = region_pred.cpu().numpy().tolist()
            # lesion_pred = np.where(lesion_pred > 0.5, 1, 0)
            
            # print(lesion_pred)
            # input()
            dicom_ids = [d["dicom_id"] for d in targets]
            # for target, pred, dicom_id in zip(lesion_target, lesion_pred, dicom_ids):
            for rpred, rgt, pred, dicom_id in zip(region_pred, region_target, lesion_pred, dicom_ids):
                out_pred_list.append([dicom_id] + pred)
                target = find_chexbert_label(dicom_id)
                if len(target) == 0:
                    print(f"No chexbert record in {dicom_id}")
                    continue

                gt_list.append(target) # 14
                pred_list.append(pred) # 13
                gt_13_list.append(target[:-1])


                out_all['dicom_id'].append(dicom_id)
                out_all['findings'].append(pred)
                out_all['region'].append(rpred)

                for ei, i in enumerate(pred):
                    # print(f"dicom_id: {dicom_id}, now label: {CLASS[ei]}")
                    if i > 0.36:
                        # print(np.where(np.array(rpred[ei]) > threshold, 1, 0))
                        pk_region = prior_constranint(CLASS[ei], rpred[ei])
                        # print(np.where(np.array(pk_region) > threshold, 1, 0))
                        # print(rgt[ei])
                        # input()
                        region_gt_list.append(rgt[ei])
                        region_pred_list.append(rpred[ei])
                        region_pred_pk_list.append(pk_region)
                # print(gt_list[-1])
                # print(gt_13_list[-1])
                # print(pred_list[-1])
                # input()
            
    
    
    metrics = [] # threshold, auc, precision, recall, f1
    metrics_pk = [] # threshold, auc, precision, recall, f1

    out_all_pd = pd.DataFrame(out_all, columns=["dicom_id", "findings", "region"])


    # precision_region, recall_region, f1_region = get_region_acc(region_gt_list, region_pred_pk_list, threshold=0.26)
    # print(f"AUC: {auc}")
    for threshold in np.arange(0.01, 0.9, 0.01):

        precision_region, recall_region, f1_region = get_region_acc(region_gt_list, region_pred_pk_list, threshold)
        metrics_pk.append([threshold, precision_region, recall_region, f1_region])
    metric_df_pk = pd.DataFrame(metrics_pk, columns=["threshold", "precision", "recall", "f1_score"])
    df_sorted_pk = metric_df_pk.sort_values(by='f1_score', ascending=False)
    for i in range(10):
        print(f"+ PK Best region threshold{i+1}: {df_sorted_pk.iloc[i]['threshold']}, metrics: precision-{df_sorted_pk.iloc[i]['precision']}, recall-{df_sorted_pk.iloc[i]['recall']}, f1-score-{df_sorted_pk.iloc[i]['f1_score']}")

    auc = get_auc(gt_13_list, pred_list, None, CLASS)

    threshold_f = 0.36
    pred_list_threshold = np.where(np.array(pred_list) > threshold_f, 1, 0)
    pred_list_threshold = pred_list_threshold.tolist()
    for i, line in enumerate(pred_list_threshold):
        if line == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] or line == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
            pred_list_threshold[i].append(1)
        else:
            pred_list_threshold[i].append(0)
            
    prec = get_prec(gt_list, pred_list_threshold, None, CLASS)
    rec = get_recall(gt_list, pred_list_threshold, None, CLASS)
    f1 = get_f1(gt_list, pred_list_threshold, None, CLASS)

chexbert_df = process_chexbert_file()

def test_2stage_from_file(test_loader):
    import ast
    from tools.LabeltoRegion.prob_to_labels import load_prior_knowledge

    prior_dict = load_prior_knowledge()
    reverse_dict = {v: k for k, v in ANATOMICAL_REGIONS.items()}

    def prior_constranint(pathology:str, regions):
        # for i, region in enumerate(regions):
        #     region_name = reverse_dict[i]
        #     if prior_dict[pathology][region_name] == 0:
        #         regions[i] = 0
        return regions

    pred_df = pd.read_csv("", header=0, index_col=['dicom_id'])
    pred_df['findings'] = pred_df['findings'].apply(ast.literal_eval)
    pred_df['region'] = pred_df['region'].apply(ast.literal_eval)

    gt_list = []
    gt_13_list = []
    pred_list = []
    out_pred_list = []
    region_pred_list = []
    region_gt_list = []

    out_all = {'dicom_id': [], 'findings': [], 'region': []}
    print(f"Start to test PK....")
    for batch_num, batch in tqdm(enumerate(test_loader)):
        images, targets = batch.values()

        lesion_target = [d["findings"].to(DEVICE, non_blocking=True) for d in targets]
        lesion_target = torch.stack(lesion_target).float().squeeze()
        lesion_target = lesion_target.cpu().numpy().tolist()

        region_target = [d["labels"].to(DEVICE, non_blocking=True) for d in targets] # batch * 29
        region_target = torch.stack(region_target).float().squeeze()
        region_target = region_target.cpu().numpy().tolist()

        dicom_ids = [d["dicom_id"] for d in targets]
        lesion_pred = []
        region_pred = []
        for dicom_id in dicom_ids:
            line = pred_df.loc[dicom_id]
            findings = line['findings']
            unpk_regions = line['region']
            pk_regions = []
            for i, region29 in enumerate(unpk_regions):
                pk_region = prior_constranint(CLASS[i], region29)
                pk_regions.append(pk_region)

            lesion_pred.append(findings)
            region_pred.append(pk_regions)

        # for target, pred, dicom_id in zip(lesion_target, lesion_pred, dicom_ids):
        for rpred, rgt, pred, dicom_id in zip(region_pred, region_target, lesion_pred, dicom_ids):
            out_pred_list.append([dicom_id] + pred)
            target = find_chexbert_label(dicom_id)
            if len(target) == 0:
                print(f"No chexbert record in {dicom_id}")
                continue

            gt_list.append(target) # 14
            pred_list.append(pred) # 13
            gt_13_list.append(target[:-1])


            out_all['dicom_id'].append(dicom_id)
            out_all['findings'].append(pred)
            out_all['region'].append(rpred)

            for i in pred:
                if i == 1:
                    for gt_reg, pred_reg in zip(rgt, rpred):

                        region_gt_list.append(gt_reg)
                        region_pred_list.append(pred_reg)
            # print(gt_list[-1])
            # print(gt_13_list[-1])
            # print(pred_list[-1])
            # input()

    out_all_pd = pd.DataFrame(out_all, columns=["dicom_id", "findings", "region"])
    out_all_pd.to_csv("")

    precision_region, recall_region, f1_region = get_region_acc(region_gt_list, region_pred_list, threshold=0.26)

    auc = get_auc(gt_13_list, pred_list, None, CLASS)

    threshold_f = 0.36
    pred_list_threshold = np.where(np.array(pred_list) > threshold_f, 1, 0)
    pred_list_threshold = pred_list_threshold.tolist()
    for i, line in enumerate(pred_list_threshold):
        if line == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] or line == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
            pred_list_threshold[i].append(1)
        else:
            pred_list_threshold[i].append(0)

    prec = get_prec(gt_list, pred_list_threshold, None, CLASS)
    rec = get_recall(gt_list, pred_list_threshold, None, CLASS)
    f1 = get_f1(gt_list, pred_list_threshold, None, CLASS)
    print("no_PK")

if __name__ == "__main__":

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    model = DenseNetDualTask(num_lesions=13, num_regions=29)
    model.load_state_dict(torch.load(CKPT_S2))
    model.to(DEVICE)

    datasets_as_dfs = get_test_datasets_as_dfs("")
    test_dataset = CustomImageDataset(datasets_as_dfs["test"], transforms=get_transforms("valid"))

    test_loader = get_test_data_loaders(test_dataset)


    test_2stage(model, test_loader)
    
    # test_2stage_from_file(test_loader)