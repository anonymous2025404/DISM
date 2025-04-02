import csv, os, sys
import pandas as pd
import tools.section_parser as sp
import json
from tools.CheXbert.src.label import label_api
from config import *

section_names_set = set()

def get_split_list(split_name):
    list = []
    filepath = MIMIC_SPLIT_PATH
    print(f"++Start to read {split_name} file.++")
    with open(filepath, newline='', encoding='utf-8') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            if row[3] == split_name:
                data = {}
                data['dicom_id'] = row[0]
                data['study_id'] = row[1]
                data['subject_id'] = row[2]
                list.append(data)
    print(f"++Finish to read {split_name} file.++")
    return list

def get_report(subject_id:str, study_id:str):
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    if f"s{study_id}" in custom_section_names or f"s{study_id}" in custom_indices:
        return -1  # skip all reports without "findings" sections

    path_to_report = os.path.join(REPORT_PATH, "files", f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")

    if not os.path.exists(path_to_report):
        # print(f"Missing report: {path_to_report}")
        return 0

    with open(path_to_report) as f:
        report = "".join(f.readlines())

    sections, section_names, _ = sp.section_text(report)
    section_names_set.update(section_names)

    if "findings" in section_names:
        findings_index = len(section_names) - section_names[-1::-1].index("findings") - 1
        report = sections[findings_index]
    else:
        # print(f"Missing findings section: {path_to_report}")
        return 0

    report = " ".join(report.split())
    return report

def print_DATAFILE(split_list, split_name):
    print(f"++Start to process {split_name} file.++")
    data = []
    missing_reports = []
    for i in range(len(split_list)):
        dict_data = {}
        dict_data['dicom_id'] = split_list[i]['dicom_id']
        dict_data['study_id'] = split_list[i]['study_id']
        dict_data['subject_id'] = split_list[i]['subject_id']
        dict_data['image_file'] = f"/files/p{dict_data['subject_id'][:2]}/p{dict_data['subject_id']}/s{dict_data['study_id']}/{dict_data['dicom_id']}.jpg"
        dict_data['Report Impression'] = get_report(dict_data['subject_id'], dict_data['study_id'])
        if dict_data['Report Impression'] == 0:
            missing_reports.append(dict_data['study_id'])
        else:
            data.append(dict_data)
        # print(dict_data['image_file'])
        # print(dict_data)
        # input()
    df = pd.DataFrame(data)
    csv_file_name = f'{DATASET_DIR}/{split_name}_chexbert.csv'
    df.to_csv(csv_file_name, index=False, header=True)
    # with open('./data_with_reference_report/missing_train_reports.txt', "w")as f:
    #     for i in missing_reports:
    #         f.write(i+'\n')
    print(f"++Start to label {split_name} file.++")
    labeled_df = label_api(csv_file_name, csv_file_name, CHEXBERT_CKPT)
    labeled_df.to_csv(csv_file_name, index=False, header=True)
    num_rows, num_cols = labeled_df.shape
    print(f"++{split_name} file done. {num_rows} data get.++")

def create_mimic_cxr_split_datafiles():
    test_list = get_split_list("test") # List
    train_list = get_split_list("train") # List
    val_list = get_split_list("validate") # List

    print_DATAFILE(test_list, "test")
    print_DATAFILE(train_list, "train")
    print_DATAFILE(val_list, "val")
    section_names = list(section_names_set)
    print("> Section names find: \n")
    print(section_names)

'''
'indications', 'indication', 'clinic indication', 'history',
'reason for the exam', 'reason of exam', 'reason  for examination',
'reason for indication'
'''