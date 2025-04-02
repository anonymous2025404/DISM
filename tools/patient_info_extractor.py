# 按照study id 提取病人的病历信息
import os, sys
import pandas as pd
from config import *
import numpy as np
import tools.section_parser as sp

info_sections = ['indications', 'indication', 'clinic indication', 'history', 'reason for the exam', 'reason of exam', 'reason  for examination', 'reason for indication']

def get_info(subject_id:str, study_id:str):
    # custom_section_names and custom_indices specify reports that don't have "findings" sections
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    if f"s{study_id}" in custom_section_names or f"s{study_id}" in custom_indices:
        return -1  # skip all reports without "findings" sections

    path_to_report = os.path.join(REPORT_PATH, "files", f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")

    if not os.path.exists(path_to_report):
        # print(f"Missing report: {path_to_report}")
        return 0

    with open(path_to_report) as f:
        report = "".join(f.readlines())

    # split report into sections
    # section_names is a list that specifies the found sections, e.g. ["indication", "comparison", "findings", "impression"]
    # sections is a list of same length that contains the corresponding text from the sections specified in section_names
    sections, section_names, _ = sp.section_text(report)

    source_info = ""
    for sec in info_sections:
        if sec in section_names:
            # get index of "findings" by matching from reverse (has to do with how section_names is constructed)
            findings_index = len(section_names) - section_names[-1::-1].index(sec) - 1
            if sections[findings_index] == "":
                continue
            info = sec.upper() + ": " + sections[findings_index]
            # remove unnecessary whitespaces
            info = " ".join(info.split())
            source_info = source_info + info

    return source_info

def extract_info():
    print("++Extracting patient info...")
    df = pd.read_csv(MIMIC_SPLIT_PATH, header=0)
    df['study_id'] = df['study_id'].astype(str)
    df['subject_id'] = df['subject_id'].astype(str)
    df.set_index(["dicom_id"])

    source_info = []
    for idx, row in df.iterrows():
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        patient_info = get_info(subject_id, study_id)
        try:
            source_info.append([subject_id, study_id, dicom_id, patient_info])
        except:
            print(f"Error extracting info for {subject_id}, {study_id}")
            continue

    df_source_info = pd.DataFrame(source_info, columns=["subject_id", "study_id", "dicom_id", "info"])
    df_source_info.to_csv(SOURCE_INFO_PATH, index=False)
    print("++Done extracting patient info.")

# extract_info()