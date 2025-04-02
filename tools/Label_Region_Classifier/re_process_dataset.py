import pandas as pd
import os
import ast
from tqdm import tqdm

label_set = set()
def clean_labels(filedir, dataset):
    df = pd.read_csv(os.path.join(filedir, dataset + ".csv"), index_col='image_id')
    df['bbox_anatomicalfinding'] = df['bbox_anatomicalfinding'].apply(ast.literal_eval)
    # print(os.path.join(filedir, dataset + ".csv"))
    # print(df.info())
    # df.drop_duplicates(subset=['bbox_anatomicalfinding'], inplace=True)
    for idx in tqdm(df.index):
        line = df.loc[idx]
        labels_list = line['bbox_anatomicalfinding'] # [[], [], []]
        label_dup_list = []
        for labels in labels_list:
            label_dup = list(set(labels))
            label_dup_list.append(label_dup)
            for label in label_dup:
                if label != "":
                    label_set.add(label)
        df.at[idx, 'bbox_anatomicalfinding'] = label_dup_list
        # print(df.loc[idx]['bbox_anatomicalfinding'])
        # input()
    df.to_csv(os.path.join(filedir, dataset + "_dup.csv"))
    print(f"Finish duplicate [{dataset}] file.")
    
def output_label(filedir):
    label_set_list = list(label_set)
    label_set_df = pd.DataFrame(label_set_list, columns=['label_all'], index=None)
    label_set_df.to_csv(os.path.join(filedir, "label_count.csv"), index=False)
    print("Finish output [label_count] file.")

def clean_and_extract_labels():
    for dataset in ["train", "valid", "test"]:
    # for dataset in ["test"]:
        clean_labels("", dataset)
    output_label("")

if __name__ == '__main__':
    clean_and_extract_labels()