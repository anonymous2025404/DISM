import csv
import pandas as pd
import os, sys, re
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score
from numpy.core.fromnumeric import argmax
from tqdm import *
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 测试用，记得删
from config import *
from tools.CheXbert.src.label import label_api

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from cider.cider import Cider

import spacy


EVAL_REPORT_FILE = ""


gt_df1 = pd.read_csv(TEST_PATH, header=0)
gt_df1['study_id'] = gt_df1['study_id'].astype(str)
# gt_df.set_index(["study_id"])
gt_df1.fillna(0, inplace=True)
gt_df1.replace(-1.0, 1.0, inplace=True)

gt_df2 = pd.read_csv(TRAIN_PATH, header=0)
gt_df2['study_id'] = gt_df2['study_id'].astype(str)
# gt_df.set_index(["study_id"])
gt_df2.fillna(0, inplace=True)
gt_df2.replace(-1.0, 1.0, inplace=True)

gt_df3 = pd.read_csv(VAL_PATH, header=0)
gt_df3['study_id'] = gt_df3['study_id'].astype(str)
# gt_df.set_index(["study_id"])
gt_df3.fillna(0, inplace=True)
gt_df3.replace(-1.0, 1.0, inplace=True)

gt_df = pd.concat([gt_df1, gt_df2, gt_df3])

pred_df = label_api(EVAL_REPORT_FILE, "", CHEXBERT_CKPT)
# pred_df['study_id'] = pred_df['study_id'].astype(str)
pred_df.fillna(0, inplace=True)
pred_df.replace(-1.0, 1.0, inplace=True)


origin_pred_df = pd.read_csv(EVAL_REPORT_FILE, header=0)
origin_pred_df.set_index(["dicom_id"])
# origin_pred_df['study_id'] = origin_pred_df['study_id'].astype(str)
origin_pred_df['Report Impression'] = origin_pred_df['Report Impression'].astype(str)

split_df = pd.read_csv(MIMIC_SPLIT_PATH, header=0)
split_df['study_id'] = split_df['study_id'].astype(str)

pathologies_all = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]

def get_prec(true_list, pred_list, del_row, pathologies_list):
    true_labels = np.array(true_list)
    pred_labels = np.array(pred_list)
    if del_row != None:
        true_labels = np.delete(true_labels, del_row, axis=1)
        pred_labels = np.delete(pred_labels, del_row, axis=1)
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




def compute_NLG_scores(nlg_metrics: list[str], gen_sents_or_reports: list[str], ref_sents_or_reports: list[str]) -> dict[str, float]:
    def convert_for_pycoco_scorer(sents_or_reports: list[str]):
        sents_or_reports_converted = {}
        for num, text in enumerate(sents_or_reports):
            sents_or_reports_converted[str(num)] = [re.sub(' +', ' ', text.replace(".", " ."))]

        return sents_or_reports_converted

    scorers = {}
    if "bleu" in nlg_metrics:
        scorers["bleu"] = Bleu(4)
    if "meteor" in nlg_metrics:
        scorers["meteor"] = Meteor()
    if "rouge" in nlg_metrics:
        scorers["rouge"] = Rouge()
    if "cider" in nlg_metrics:
        scorers["cider"] = Cider()

    gen_sents_or_reports = convert_for_pycoco_scorer(gen_sents_or_reports)
    ref_sents_or_reports = convert_for_pycoco_scorer(ref_sents_or_reports)

    nlg_scores = {}

    for metric_name, scorer in scorers.items():
        score, _ = scorer.compute_score(ref_sents_or_reports, gen_sents_or_reports)
        if metric_name == "bleu":
            nlg_scores["bleu_1"] = score[0]
            nlg_scores["bleu_2"] = score[1]
            nlg_scores["bleu_3"] = score[2]
            nlg_scores["bleu_4"] = score[3]
        else:
            nlg_scores[metric_name] = score

    return nlg_scores

def get_nlg_score(gen_reports, ref_reports):
    nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
    # nlg_metrics = ["meteor"]
    nlg_scores = compute_NLG_scores(nlg_metrics, gen_reports, ref_reports)
    print("------------------------------------NLG-----------------------------------")
    for nlg_metric_name, score in nlg_scores.items():
        print(f"{nlg_metric_name}: {score}")


def evaluate_reports():
    pred_list = []
    gt_list = []
    match_reports = []
    gt_reports = []
    for idx in pred_df.index:
        pa = pred_df.loc[idx]
        line = []
        # 第一列是报告文本
        for p in pathologies_all:
            line.append(pa[p])

        try:
            # gt_pa = gt_df[gt_df['dicom_id']==pa['dicom_id']] # 用于标签文件有dicom_id的情况
            split_line = split_df[pa['dicom_id'] == split_df['dicom_id']]  # 用于标签文件没有dicom_id的情况
            gt_study_id = split_line['study_id'].values[0]
            gt_pa = gt_df[gt_df['study_id'] == gt_study_id]
        
            # origin_pa = origin_pred_df[origin_pred_df['dicom_id']==pa['dicom_id']]
            gt_line = []
            # gt_line.append(gt_pa['study_id'].values[0])
            gt_line.append(gt_pa['Atelectasis'].values[0])
            gt_line.append(gt_pa['Cardiomegaly'].values[0])
            gt_line.append(gt_pa['Consolidation'].values[0])
            gt_line.append(gt_pa['Edema'].values[0])
            gt_line.append(gt_pa['Enlarged Cardiomediastinum'].values[0])
            gt_line.append(gt_pa['Fracture'].values[0])
            gt_line.append(gt_pa['Lung Lesion'].values[0])
            gt_line.append(gt_pa['Lung Opacity'].values[0])
            gt_line.append(gt_pa['No Finding'].values[0])
            gt_line.append(gt_pa['Pleural Effusion'].values[0])
            gt_line.append(gt_pa['Pleural Other'].values[0])
            gt_line.append(gt_pa['Pneumonia'].values[0])
            gt_line.append(gt_pa['Pneumothorax'].values[0])
            gt_line.append(gt_pa['Support Devices'].values[0])

            pred_list.append(line)
            gt_list.append(gt_line)
            
            
            match_reports.append(pa['Report Impression'])
            if type(match_reports[-1]) != str:
                # print(match_reports[-1])
                # print(pa['dicom_id'])
                match_reports[-1] = pa['Report Impression'].iloc[0]
            gt_reports.append(gt_pa['Report Impression'].iloc[0])
            if type(gt_reports[-1]) != str:
                print(gt_reports[-1])
                print(pa['dicom_id'])

        except:
            print(f"Error in dicom: {pa['dicom_id']}")
            continue
            # break

    # calculate the metrics
    get_prec(gt_list, pred_list, None, pathologies_all)
    get_recall(gt_list, pred_list, None, pathologies_all)
    get_f1(gt_list, pred_list, None, pathologies_all)
    # print(len(gt_reports))
    get_nlg_score(match_reports, gt_reports)

    print(f"Processed {len(pred_list)} reports.")

def evaluate_nlg():
    match_reports = []
    gt_reports = []
    for idx in origin_pred_df.index:
        pa = origin_pred_df.loc[idx]
        try:
            gt_pa = gt_df[gt_df['dicom_id']==pa['dicom_id']]
            match_reports.append(pa['Report Impression'])
            # print(match_reports[-1])
            if type(match_reports[-1]) != str:
                # print(match_reports[-1])
                # print(pa['dicom_id'])
                match_reports[-1] = pa['Report Impression'].iloc[0]
            gt_reports.append(gt_pa['Report Impression'].iloc[0])
            if type(gt_reports[-1]) != str:
                print(gt_reports[-1])
                print(pa['dicom_id'])
        except:
            # print(f"Error in study: {pa['study_id']}")
            continue
            # break
    # print(match_reports[0])
    # print(gt_reports[0])
    get_nlg_score(match_reports, gt_reports)

    print(f"Processed {len(match_reports)} reports.")

evaluate_reports()
# evaluate_nlg()