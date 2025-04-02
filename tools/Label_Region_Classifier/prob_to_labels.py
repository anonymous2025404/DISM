import sys, os

import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from config import CLASS, ANATOMICAL_REGIONS, TEST_PATH, TRAIN_PATH, VAL_PATH

PRIOR_KNOWLEDGE = True
prior_dict = {} # dict[label:str][region:str] -> int(1)

def load_prior_knowledge():
    reverse_dict = {v: k for k, v in ANATOMICAL_REGIONS.items()}
    for label in CLASS:
        prior_dict[label] = {}
        for region in ANATOMICAL_REGIONS.values():
            prior_dict[label][reverse_dict[region]] = 0
    
    gt_file_path = ""
    df = pd.read_csv(gt_file_path, index_col='image_id')
    df['bbox_anatomicalfinding'] = df['bbox_anatomicalfinding'].apply(ast.literal_eval)
    df['bbox_labels'] = df['bbox_labels'].apply(ast.literal_eval)

    for idx in tqdm(range(len(df)), position=0):
        dicom_id = df.index[idx]
        findings = df['bbox_anatomicalfinding'][idx] # []*13
        regions = df['bbox_labels'][idx] # [[]*29]*13

        for l, label_i in enumerate(regions):
            for i, region in enumerate(label_i):
                if region != 0:
                    prior_dict[CLASS[l]][reverse_dict[i]] = 1
    # print(f"prior_dict: {prior_dict}")
    for number, i in enumerate(prior_dict):
        print(f"label: {CLASS[number]}\n{prior_dict[CLASS[number]]}")
    
    return prior_dict


def prob_to_labels(prob_file_path, save_path):
    reverse_dict = {v: k for k, v in ANATOMICAL_REGIONS.items()}

    
    def prior_constranint(pathology:str, regions):
        for i, region in enumerate(regions):
            region_name = reverse_dict[i]
            if prior_dict[pathology][region_name] == 0:
                regions[i] = 0
        return regions

    l_threshold = 0.36
    r_threshold = 0.26

    df = pd.read_csv(prob_file_path, index_col='dicom_id')
    df['findings'] = df['findings'].apply(ast.literal_eval)
    df['region'] = df['region'].apply(ast.literal_eval)
    
    findings_array = np.array(df['findings'].tolist())
    region_array = np.array(df['region'].tolist())

    findings_array = np.where(findings_array > l_threshold, 1, 0)
    region_array = np.where(region_array > r_threshold, 1, 0)

    findings_list = findings_array.tolist()
    region_list = region_array.tolist()
    dicom_id_list = df.index.tolist()

    # print(findings_array[0])
    # print(region_array[0])
    # input()
    out_list_1 = []
    out_list_2 = []
    for idx in range(len(findings_list)):
        dicom_id = dicom_id_list[idx]

        out_label_1 = []
        out_label_2 = []
        i = 0
        for finding, regions in zip(findings_list[idx], region_list[idx]):
            finding_label = ""
            region_label = ""
            
            if PRIOR_KNOWLEDGE:
                regions = prior_constranint(CLASS[i], regions)
            
            if finding == 1:
                finding_label = CLASS[i]

                for ri, region in enumerate(regions):
                    if region == 1:
                        if region_label == "":
                            region_label = reverse_dict[ri]
                        else:
                            region_label = region_label + "|" + reverse_dict[ri]
                out_label_1.append(finding_label)
                out_label_2.append(finding_label + "-" + region_label)
                i += 1
            else:
                i += 1
                continue
        
        out_list_1.append([dicom_id, out_label_1])
        out_list_2.append([dicom_id, out_label_2])
        # print(out_list_1[-1])
        # print(out_list_2[-1])
        # input()
    out_df_1 = pd.DataFrame(out_list_1, columns=['dicom_id', 'findings'])
    out_df_2 = pd.DataFrame(out_list_2, columns=['dicom_id', 'findings'])
    out_df_1.to_csv(f"{save_path}_l.csv", index=False)
    out_df_2.to_csv(f"{save_path}_lr.csv", index=False)



load_prior_knowledge()