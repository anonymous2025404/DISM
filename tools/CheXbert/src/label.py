import os, sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import tools.CheXbert.src.utils as utils
from tools.CheXbert.src.models.bert_labeler import bert_labeler
from tools.CheXbert.src.bert_tokenizer import tokenize
from transformers import BertTokenizer
from collections import OrderedDict
from tools.CheXbert.src.datasets.unlabeled_dataset import UnlabeledDataset, UnlabeledReports
from tools.CheXbert.src.constants import *
from tqdm import tqdm

def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of 
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch

def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader

def load_unlabeled_report(report, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledReports(report)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader

def label(checkpoint_path, csv_path):
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint 
    @param csv_path (string): location of csv with reports

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report  
    """
    ld = load_unlabeled_data(csv_path)
    
    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0: #works even if only 1 GPU available
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)
             
    if was_training:
        model.train()

    y_pred = [t.tolist() for t in y_pred]
    return y_pred

def save_preds(y_pred, csv_path, out_path, save_to_file):
    """Save predictions as out_path/labeled_reports.csv 
    @param y_pred (List[List[int]]): list of predictions for each report
    @param csv_path (string): path to csv containing reports
    @param out_path (string): path to output directory
    """
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    
    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    reports = pd.read_csv(csv_path)['Report Impression']
    # study_id = pd.read_csv(csv_path)['study_id']
    dicom_id = pd.read_csv(csv_path)['dicom_id']
    # image_file = pd.read_csv(csv_path)['image_file']

    df['Report Impression'] = reports.tolist()
    # df['study_id'] = study_id.tolist()
    df['dicom_id'] = dicom_id.tolist()
    # df['image_file'] = image_file.tolist()
    # new_cols = ['Report Impression'] + CONDITIONS + ['study_id'] + ['dicom_id'] + ['image_file']
    # new_cols = ['Report Impression'] + CONDITIONS + ['study_id'] + ['dicom_id']
    new_cols = ['Report Impression'] + CONDITIONS + ['dicom_id']
    df = df[new_cols]

    df.replace(0, np.nan, inplace=True) #blank class is NaN
    df.replace(3, -1, inplace=True)     #uncertain class is -1
    df.replace(2, 0, inplace=True)      #negative class is 0 
    
    if save_to_file:
        df.to_csv(os.path.join(out_path, 'labeled_reports.csv'), index=False)
    else:
        return df

# 返回一个标注后的dataframe表
def label_api(csv_path, out_path, checkpoint_path): # 报告文本文件路径，输出文件路径（api返回无需填写），chexbert使用的模型权重路径
    y_pred = label(checkpoint_path, csv_path)
    return save_preds(y_pred, csv_path, out_path, False)

# 读取区域信息文件，并对每个分段句子进行标签标注
def label_for_region(csv_path, out_path, checkpoint_path):
    region_df = pd.read_csv(csv_path, header=0, index_col=['image_id'])
    # subject_id	study_id	image_id	mimic_image_file_path	bbox_coordinates	bbox_labels	bbox_phrases	bbox_phrase_exists	bbox_is_abnormal
    region_df['phrases_labels'] = None # 增加一列用于存储每一个区域的句子的标注结果
    region_df['phrases_labels'] = region_df['phrases_labels'].astype('object') # 转为object以存储列表

    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0: #works even if only 1 GPU available
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()
    

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        # phrases_labels = []
        for idx in tqdm(region_df.index):
            line = region_df.loc[idx]

            reports = line['bbox_phrases'].split('[')[1]
            reports = reports.split(']')[0]
            reports = reports.split('\', \'')
            
            phrases_label = []
            for report in reports:
                # if report == "":
                #     phrases_label.append([])
                #     continue
                ld = load_unlabeled_report(report)
                y_pred = [[] for _ in range(len(CONDITIONS))] #initialize empty lists for each condition
                # for i, data in enumerate(tqdm(ld)):
                for i, data in enumerate(ld):
                    batch = data['imp'] #(batch_size, max_len)
                    batch = batch.to(device)
                    src_len = data['len']
                    batch_size = batch.shape[0]
                    attn_mask = utils.generate_attention_masks(batch, src_len, device)

                    out = model(batch, attn_mask)

                    for j in range(len(out)):
                        curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                        y_pred[j].append(curr_y_pred)

                for j in range(len(y_pred)):
                    y_pred[j] = torch.cat(y_pred[j], dim=0)

                y_pred = [t.tolist() for t in y_pred]
                y_pred = np.array(y_pred)
                label = []
                for i in range(len(y_pred)-1):
                    if y_pred[i][1] == np.nan:
                        y_pred[i][1] = 0
                    elif y_pred[i][1] == 3:
                        y_pred[i][1] = 1
                    elif y_pred[i][1] == 2:
                        y_pred[i][1] = 0
                    # print(f"{CONDITIONS[i]}: {y_pred[i][1]}")
                    # if y_pred[i][1] == 1:
                        # label.append(CONDITIONS[i])
                    label.append(y_pred[i][1])
                phrases_label.append(label)
            # phrases_labels.append(phrases_label)
            # print(f"{region_df.info()} \nPhrases label: {phrases_label}")
            region_df.at[idx, 'phrases_labels'] = phrases_label
            # 如果文件不存在，则写入列名
            if not os.path.exists(out_path):
                region_df.loc[[idx]].to_csv(out_path, index=False)
            else:
                # 追加行数据，不写入列名
                region_df.loc[[idx]].to_csv(out_path, index=False, mode='a', header=False)
            # print(line['study_id'])
            # print(region_df.loc[idx, 'phrases_labels'])
            # print(phrases_labels[-1])
            # print(region_df.loc[idx])
            # input()
        # region_df['phrases_labels'] = phrases_labels
        # region_df.to_csv(out_path, index=False)
    print("Label for region file, done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                              under the \"Report Impression\" column')
    parser.add_argument('-o', '--output_dir', type=str, nargs='?', required=True,
                        help='path to intended output folder')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?', required=True,
                        help='path to the pytorch checkpoint')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_dir
    checkpoint_path = args.checkpoint

    y_pred = label(checkpoint_path, csv_path)
    save_preds(y_pred, csv_path, out_path, True)
