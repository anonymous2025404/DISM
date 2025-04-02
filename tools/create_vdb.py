from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
import pandas as pd
import sys, os
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

import urllib3
# import weaviate
# from weaviate.embedded import EmbeddedOptions
import ssl
ssl._create_default_https_context = ssl._create_unverified_context()

http = urllib3.PoolManager(
    cert_reqs = 'CERT_NONE'
)
import requests
s = requests.Session()
s.verify = False

# def create_db():
#     loader = TextLoader("")
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = text_splitter.split_documents(documents)

#     model_name = ""
#     model_kwargs = {'device': 'cpu'}
#     embedding = HuggingFaceBgeEmbeddings(
#         model_name=model_name,
#         model_kwargs=model_kwargs
#     )

#     persist_directory = 'sentence'
#     db = FAISS.from_documents(
#         chunks,
#         embedding
#     )
#     db.save_local(f"db/{persist_directory}")
#     print(f"> Finish creating database! Saved in db/{persist_directory}")

def create_db_csv(filepath, db_path):
    loader = CSVLoader(file_path=filepath)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    chunks = text_splitter.split_documents(data)

    model_name = ""
    model_kwargs = {'device': 'cpu'}
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    db = FAISS.from_documents(
        chunks, 
        embedding
    )
    db.save_local(f"{db_path}/chexbert_csv")
    print(f"> Finish creating database! Saved in {db_path}/chexbert_csv")

def create_db(file_path, db_path):
    list_of_documents = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        for line in tqdm(lines, position=0):
            idx += 1
            doc = Document(page_content=line, metadata=dict(page=idx))
            list_of_documents.append(doc)
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=RAG_CKPT,
        model_kwargs={'device': 'cuda:0'}
    )
    db = FAISS.from_documents(list_of_documents, embeddings)
    db.save_local(f"{db_path}")
    print(f"> Finish creating database! Saved in {db_path}")


def init_datafile():
    df = pd.read_csv(TRAIN_PATH, header=0)
    df['study_id'] = df['study_id'].astype(str)
    df.drop_duplicates(subset='study_id', keep='first', inplace=True)
    df.set_index(["dicom_id"])
    df.fillna(0, inplace=True)
    df.replace(-1.0, 1.0, inplace=True)


    info_df = pd.read_csv(SOURCE_INFO_PATH, header=0)
    info_df['study_id'] = info_df['study_id'].astype(str)
    info_df.set_index(["dicom_id"])
    info_df.fillna(0, inplace=True)
    info_df.replace(-1.0, 1.0, inplace=True)

    vdb_p = []
    vdb_p_info = []
    for idx in tqdm(df.index, position=0):
        line = df.loc[idx]
        report = line['Report Impression']
        if str(report) == '-1':
            continue
        info = info_df.loc[idx]['info']

        p_id = line['study_id']
        
        label = ""
        for pa in CLASS[:-1]:
            if line[pa] == 1:
                label += pa + '.'
        if label == "":
            label = "No findings present."
        if line[CLASS[-1]] == 1:
            label += CLASS[-1] + '.'

        out_line = f"[PATIENT {p_id}. Radiological manifestations: {label} REPORT: {report}]"
        out_line_withinfo = f"[PATIENT {p_id}. {info}. Radiological manifestations: {label} REPORT: {report}]"
        vdb_p.append(out_line)
        vdb_p_info.append(out_line_withinfo)
    
    with open(VDB_P_PATH, 'w') as f:
        for line in vdb_p:
            f.write(line + '\n')
    with open(VDB_P_INFO_PATH, 'w') as f:
        for line in vdb_p_info:
            f.write(line + '\n')
    print("> create_vdb.py: Finish init datafile!")

if __name__ == '__main__':
    if os.path.exists(VDB_P_PATH) == False or os.path.exists(VDB_P_INFO_PATH) == False:
        init_datafile()
    print("> Start to create database...")
    create_db(VDB_P_PATH, VDB_PATH)
    create_db(VDB_P_INFO_PATH, VDB_INFO_PATH)
    print("> Done.")