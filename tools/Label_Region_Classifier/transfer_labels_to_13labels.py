import pandas as pd
import os, sys
import ast
from tqdm import tqdm
sys.path.append("")
from config import *

label_df = pd.read_csv("", index_col='label_all')

def convert_labels_to_13labels(label):
    c_label = label_df.loc[label]
    return c_label['label_13']

def transfer_labels(filedir, dataset):
    filepath = os.path.join(filedir, dataset + "_dup.csv")
    df = pd.read_csv(filepath, index_col='image_id')
    df['bbox_anatomicalfinding'] = df['bbox_anatomicalfinding'].apply(ast.literal_eval)
    for idx in tqdm(df.index):
        line = df.loc[idx]
        labels_list = line['bbox_anatomicalfinding'] # [[], [], []]
        label_dup_list = []
        for labels in labels_list:
            convert_labels_list = []
            for label in labels:
                try:
                    label = convert_labels_to_13labels(label)
                except:
                    pass
                if label != 'None':
                    convert_labels_list.append(label)
            # label_dup = list(set(labels))
            label_dup = list(set(convert_labels_list))
            label_dup_list.append(label_dup)
        df.at[idx, 'bbox_anatomicalfinding'] = label_dup_list
    df.to_csv(os.path.join(filedir, dataset + "_13labels.csv"))
    print(f"Finish convert [{dataset}] file.")

def transfer_labels_int(filedir, dataset):
    filepath = os.path.join(filedir, dataset + "_dup.csv")
    df = pd.read_csv(filepath, index_col='image_id')
    df['bbox_anatomicalfinding'] = df['bbox_anatomicalfinding'].apply(ast.literal_eval)
    for idx in tqdm(df.index):
        line = df.loc[idx]
        labels_list = line['bbox_anatomicalfinding'] # [[], [], []]
        label_dup_list = []
        for labels in labels_list:
            convert_labels_list = []
            label_01 = []
            class_labels = CLASS
            for label in labels:
                try:
                    label = convert_labels_to_13labels(label)
                except:
                    pass
                if label != 'None':
                    convert_labels_list.append(label)
            # label_dup = list(set(labels))
            label_dup = list(set(convert_labels_list))
            for class_label in class_labels:
                if class_label in label_dup:
                    label_01.append(1)
                else:
                    label_01.append(0)
            label_dup_list.append(label_01)
        df.at[idx, 'bbox_anatomicalfinding'] = label_dup_list
    df.to_csv(os.path.join(filedir, dataset + "_13labels_01.csv"))
    print(f"Finish convert [{dataset}] file.")

def transfer_labels_v_29regions(filedir, dataset):
    filepath = os.path.join(filedir, dataset + "_13labels_01.csv")
    df = pd.read_csv(filepath, index_col='image_id')
    df['bbox_anatomicalfinding'] = df['bbox_anatomicalfinding'].apply(ast.literal_eval)
    for idx in tqdm(df.index):
        line = df.loc[idx]
        labels_list = line['bbox_anatomicalfinding'] # [[], [], []]
        # region_list = line['bbox_labels'] # []

        l_region = [[0] * 29 for _ in range(13)]
        l_label = [0] * 13

        for i, labels in enumerate(labels_list): # 29
            for j, label in enumerate(labels): # 13
                try:
                    if label == 1:
                        l_region[j][i] = 1
                        l_label[j] = 1
                    else:
                        l_region[j][i] = 0
                except:
                    print(f"Error image ID: {idx}")
                    print(l_region)
                    print("-----------------")
                    print(l_label)
                    print("-----------------")
                    print(f"i: {i}, j: {j}")
                    input()
                    return
        df.at[idx, 'bbox_anatomicalfinding'] = l_label
        df.at[idx, 'bbox_labels'] = l_region
    df.to_csv(os.path.join(filedir, dataset + "_13labels_02.csv"))
    print(f"Finish convert [{dataset}] file.")


def transfer_to_str():
    for dataset in ["train", "valid", "test"]:
    # for dataset in ["test"]:
        transfer_labels("", dataset)

def transfer_to_int():
    for dataset in ["train", "valid", "test"]:
    # for dataset in ["test"]:
        transfer_labels_int("", dataset)

def transfer_label_to_region():
    for dataset in ["train", "valid", "test"]:
        transfer_labels_v_29regions("", dataset)  

if __name__ == '__main__':
    # transfer_to_str() # 转换为str格式的13个对应标签
    # transfer_to_int() # 转换为01矩阵格式的13个对应标签
    transfer_label_to_region() # 转换为标签13 * 区域29的矩阵，每个标签对应29个区域