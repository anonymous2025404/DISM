# 仅使用llm进行报告的整理
import os, sys, time
from tqdm import tqdm
from openai import OpenAI
import openai
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import urllib3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context()
http = urllib3.PoolManager(
    cert_reqs = 'CERT_NONE'
)
import requests
s = requests.Session()
s.verify = False

from config import *

if MATCH_TYPE == 'single': # 单句匹配的情况
    if INFO_USE:
        LLM_REPORT_PATH = LLM_0SHOT_REPORT_S_PATH_INFO
    else:
        LLM_REPORT_PATH = LLM_0SHOT_REPORT_S_PATH
else:
    if INFO_USE:
        LLM_REPORT_PATH = LLM_0SHOT_REPORT_G_PATH_INFO
    else:
        LLM_REPORT_PATH = LLM_0SHOT_REPORT_G_PATH

if RAG_USE == True:
    if INFO_USE == True:
        VDB = VDB_INFO_PATH # 有病人病历数据的向量数据库
        LLM_REPORT_PATH = LLM_RAG_INFO

    else:
        VDB = VDB_INFO_PATH # 无病人病历数据的向量数据库
        LLM_REPORT_PATH = LLM_RAG

    persist_directory = VDB
    db = FAISS.load_local(persist_directory, HuggingFaceBgeEmbeddings(model_name=RAG_CKPT, model_kwargs={'device': 'cpu'}), allow_dangerous_deserialization=True)

if LLM_USE == "qwen7b":
    model_name = QWEN_7B
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)


# 读取预测的测试集的标签
label_df = pd.read_csv(PRED_LABEL_PATH, header=0, index_col=['dicom_id'])
label_df['study_id'] = label_df['study_id'].astype(str)

if INFO_USE:
    info_df = pd.read_csv(SOURCE_INFO_PATH, header=0, index_col=['dicom_id'])

# 不依靠句子匹配，直接生成报告，包含病历信息
prompt_info = '''
Write the findings section of the medical report based on the patient's indication, medical history, and initial interpretation of the chest x-ray. Identify relevant details from the patient's medical history, especially existing diagnostics and medical device information, to use in writing the findings section of the imaging report.
Patient's indication, medical history, and other information: [{info}]
Initial interpretation of the x-ray: [{report}]

Rules:
Start directly with the report findings.
Do not add any labels, prefixes, or interpretations.
Do not write impressions or other sections, just the findings section of the medical report.
Write only what is necessary to reflect the patient's lesion findings and medical device information.
Make sure the report is concise, objective, and focused on the current findings, avoiding stating the patient's historical records and personal information.
'''
# 不依靠句子匹配，直接生成报告，不包含病历信息
prompt_no_info = '''
Provide only the report content without any labels or prefixes. As a medical report editor, modify this radiology finding labels to a complete reports:  
{report}

Rules:
Only write what's necessary to reflect the symptoms and medical manifestations which included in the simple report, do not add other symptoms and medical manifestations that not mentioned and do not ignore any symptoms and medical manifestations that mentioned.
Start directly with the report content.
Do not add any labels, prefixes or explanations.
Do not write impression or other parts, just findings part of chest x-ray medical reports.
'''

template_no_info = '''
Provide only the report content without any labels or prefixes. As a medical report editor, modify this simple radiology report to a complete reports: 
{report}

Rules:
Only modify what's necessary to reflect the symptoms and medical manifestations which included in the simple report, do not add other symptoms and medical manifestations that not mentioned and do not ignore any symptoms and medical manifestations that mentioned.
Keep the original format and style.
Start directly with the report content.
Do not add any labels, prefixes or explanations.
Do not write impression or other parts, just findings part of medical reports.
'''

template_with_info = '''
Write the findings section of a medical report based on the patient's indication, history, and the preliminary interpretation of the chest X-ray image. Identify and include relevant details from the patient's medical history, specifically any existing diagnoses and medical equipment-related information, to enhance the findings section of the imaging report.
Patient's indication, history, and other informations: [{info}]
Preliminary interpretation of X-ray: [{report}]

Rules:
Start directly with the report findings content.
Use the original format and style of Preliminary interpretation of X-ray.
Do not add any labels, prefixes or explanations.
Do not write impression or other parts, just findings part of medical reports.
Only modify what's necessary to reflect the symptoms and medical manifestations and devices information which included in the patient's indication, history, and the preliminary interpretation.
Do not directly write any background information or context from the patient's medical history.
Ensure that the findings are concise, objective, focus on current findings and avoid stating the patient's history and personal information.
'''

template_no_info_rag = '''
Provide only the report content without any labels or prefixes. As a medical report editor, modify this simple radiology report to a complete reports: 
Radiological evaluation of X-ray:{report}

Rules:
Only modify what's necessary to reflect the symptoms and medical manifestations which included in the simple report, do not add other symptoms and medical manifestations that not mentioned and do not ignore any symptoms and medical manifestations that mentioned.
Start directly with the report content.
Do not add any labels, prefixes or explanations.
Do not write impression or other parts, just findings part of medical reports.
Follow the format and writing style of the Examples.

{rag_info}
'''

template_with_info_rag = '''
Write the findings section of the medical report based on the patient's indication, medical history, and Preliminary evaluation of the chest x-ray. Identify relevant details from the patient's medical history, especially existing diagnostics and medical device information, to use in writing the findings section of the imaging report.
Patient's indication, history, and other informations: [{info}]
Preliminary evaluation of X-ray: [{report}]

Rules:
Start directly with the report findings.
Do not add any labels, prefixes, or interpretations.
Do not write impressions or other sections, just findings part of chest x-ray medical reports.
Write only what is necessary to reflect the patient's lesion findings and medical device information.
Make sure the report is concise, objective, and focused on the current findings, avoiding stating the patient's history and personal information.
Follow the format and writing style of the Examples.

{rag_info}
'''

template_guide_rag_info = '''
Write the findings section of a medical report based on the patient's indication, history, and radiological manifestations. Identify and include relevant details from the patient's medical history, specifically any existing diagnoses and medical equipment-related information, to enhance the findings section of the imaging report.
Rules:
Start directly with the report findings content.
Do not add any labels, prefixes or explanations.
Do not write impression or other parts, just findings part of medical reports.
Only modify what's necessary to reflect the symptoms,  medical manifestations and devices information which included in the patient's indication, history, and the radiological manifestations.
Do not directly write any background information or context from the patient's medical history.
Ensure that the findings are concise, objective, focus on current findings and avoid stating the patient's history and personal information.

{rag_info}Question 4: PATIENT X. {info} Radiological manifestations: {report}\nAnswer 4: REPORT: 
'''


# ------------------------------测试用--------------------------------

def test_qwen7b():
    model_name = QWEN_7B

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    while True:
        prompt = input("Please input prompt:")
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("LLM response: ")
        print(response)

def test_retrieve():
    query = f"Which patient has the similarist situation as following: INDICATION: ___-year-old male with history of metastatic melanoma, presenting with confusion and somnolence. Evaluate for acute cardiopulmonary process. Radiological manifestations: Atelectasis.Enlarged Cardiomediastinum.Lung Opacity.Pleural Effusion."

    k = 3
    retriever = db.similarity_search(query, k=k)
    examples = ""

    # print(f"Query: {query}")
    # print(retriever[0].page_content)
    # input()

    for i in range(k):
        print(type(retriever[i].page_content))
        
        content = str(retriever[i].page_content)
        print(type(content))
        examples += content.split('REPORT:')[1]
        examples += retriever[i].page_content.split('REPORT:')[1]
    print(examples)

# -----------------------------------无rag---------------------------------------

def get_llm_report_qwen72b(simple_report:str, info:str):
    prompt = template_no_info.format(report=simple_report)
    print(f"Prompt:\n{prompt}\n")
    client = OpenAI(
        api_key=API_KEY, 
        base_url="",
    )
    completion = client.chat.completions.create(
        model="",
        messages=[
            {'role': 'user', 'content': prompt}],
        )
    res = completion.model_dump_json()
    res = json.loads(res)
    content = res['choices'][0]['message']['content']
    report = content.replace("\n", " ")
    report = report.lower()
    print("Response:\n")
    print(report)
    return report

def get_llm_report_qwen7b(simple_report:str, info:str):
    role_prompt = "You are a medical report editor."
    if INFO_USE:
        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": template_with_info.format(info=info, report=simple_report)}
            # {"role": "user", "content": prompt_info.format(info=info, report=simple_report)}
        ]
    else:
        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": template_no_info.format(report=simple_report)}
            # {"role": "user", "content": prompt_no_info.format(report=simple_report)}
        ]
    print(f"Prompt:\n{messages[1]['content']}")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 仅使用大模型对报告进行处理
def llm_0shot(llm_type):
    reports = []
    for idx in tqdm(pred_df.index, position = 0):
        pa = pred_df.loc[idx]

        label_line = label_df.loc[idx]
        label = ""
        for cls in CLASS[:-1]:
            if label_line[cls] == 1:
                label += cls + '.'
        if label == "":
            label = "No findings present."
        if label_line[CLASS[-1]] == 1:
            label += CLASS[-1] + '.'

        try:      
            simple_report = pa['Report Impression']
            if INFO_USE:
                pa_info = info_df.loc[idx]['info']
                if llm_type == "qwen72b":
                    llm_report = get_llm_report_qwen72b(pa['Report Impression'], pa_info)
                elif llm_type == "qwen7b":
                    # llm_report = get_llm_report_qwen7b(pa['Report Impression'], pa_info)
                    llm_report = get_llm_report_qwen7b(label, pa_info)
            else:
                if llm_type == "qwen72b":
                    llm_report = get_llm_report_qwen72b(pa['Report Impression'], "")
                elif llm_type == "qwen7b":
                    # llm_report = get_llm_report_qwen7b(pa['Report Impression'], "")
                    llm_report = get_llm_report_qwen7b(label, "")
            # print(pa['study_id'])
            # print(pa_info)
            # print(f"Simple Report:\n{simple_report}\n")
            print(f"LLM Report:\n{llm_report}\n")
            input()
            reports.append([pa['study_id'], idx, simple_report, llm_report])
        except:
            break
            # print(f"Error: {idx}")
            continue
    
    reports_df = pd.DataFrame(data = reports, index = None, columns = ['study_id', 'dicom_id', 'match_report', 'Report Impression'])
    reports_df.to_csv(LLM_REPORT_PATH)
    print(f"All reports are saved in the path: {LLM_REPORT_PATH}")
# ----------------------------------有rag----------------------------------

# 明确使用rag
def get_llm_report_qwen7b_rag(simple_report:str, info:str, rag_info:str, labels:str):
    role_prompt = "You are a medical report editor."
    if INFO_USE:
        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": template_with_info_rag.format(info=info, report=labels, rag_info=rag_info)}
            # {"role": "user", "content": template_guide_rag_info.format(info=info, report=labels, rag_info=rag_info)}
        ]
    else:
        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": template_no_info_rag.format(report=labels, rag_info=rag_info)}
        ]
    # print(f"Prompt:{messages[1]['content']}")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # print(f"Prompt: {messages[1]['content']}")
    
    return response

# 使用大模型+RAG对报告进行处理
def llm_rag(llm_type):
    reports = []
    for idx in tqdm(pred_df.index, position = 0):
        pa = pred_df.loc[idx]
        
        try:
            simple_report = pa['Report Impression']
            if INFO_USE:
                pa_info = info_df.loc[idx]['info']
                label_line = label_df.loc[idx]
                label = ""
                for cls in CLASS[:-1]:
                    if label_line[cls] == 1:
                        label += cls + '.'
                if label == "":
                    label = "No findings present."
                if label_line[CLASS[-1]] == 1:
                    label += CLASS[-1] + '.'

                query = f"Which patient has the similarist situation as following: {pa_info}. Radiological manifestations: {label}"

                k = 3
                retriever = db.similarity_search(query, k=k) # 因为数据库中存在一些病人有多个记录的问题，所以k=1或3比较好
                examples = ""

                # print(f"Query: {query}")
                # print(retriever[0].page_content)
                # input()

                for i in range(k):
                    content = retriever[i].page_content.split('REPORT:')[1]
                    content = content.split(']')[0]
                    examples += f"Example {i+1}: {content}\n"

                    # content = retriever[i].page_content.split('[')[1]
                    # content = content.split(']')[0]
                    # examples += f"Question {i+1}: {content.split('REPORT:')[0]}\nAnswer {i+1}: REPORT: {content.split('REPORT:')[1]}\n"

                if llm_type == "qwen72b":
                    llm_report = get_llm_report_qwen72b(pa['Report Impression'], pa_info)
                elif llm_type == "qwen7b":
                    llm_report = get_llm_report_qwen7b_rag(pa['Report Impression'], pa_info, examples, label)
            else:
                label_line = label_df.loc[idx]
                label = ""
                for cls in CLASS[:-1]:
                    if label_line[cls] == 1:
                        label += cls + '.'
                if label == "":
                    label = "No lesions present."
                if label_line[CLASS[-1]] == 1:
                    label += CLASS[-1] + '.'

                query = f"Which patient has the similarist situation as following: Radiological manifestations: {label}"
                k = 3
                retriever = db.similarity_search(query, k=k)
                examples = ""
                for i in range(k):
                    content = retriever[i].page_content.split('REPORT:')[1]
                    content = content.split(']')[0]
                    examples += f"Example {i+1}: {content}\n"

                if llm_type == "qwen72b":
                    llm_report = get_llm_report_qwen72b(pa['Report Impression'], "")
                elif llm_type == "qwen7b":
                    llm_report = get_llm_report_qwen7b_rag(pa['Report Impression'], "", examples, label)

            # print(pa['study_id'])
            # print(pa_info)
            # print(f"Simple Report:\n{simple_report}\n")
            # print(f"LLM Report:\n{llm_report}\n")
            # input()

            reports.append([pa['study_id'], idx, simple_report, llm_report])
        except:
            # break
            continue
    
    reports_df = pd.DataFrame(data = reports, index = None, columns = ['study_id', 'dicom_id', 'match_report', 'Report Impression'])
    reports_df.to_csv(LLM_REPORT_PATH)
    print(f"All reports are saved in the path: {LLM_REPORT_PATH}")


# --------------------------------------------------------------------


if __name__ == "__main__":
    print(f"> INFO: match_type:{MATCH_TYPE}, LLM_USE:{LLM_USE}, RAG_USE:{RAG_USE}, INFO_USE:{INFO_USE}")
    # test_qwen7b()
    # test_retrieve()
    if RAG_USE == True:
        llm_rag(LLM_USE)
    else:
        llm_0shot(LLM_USE) # qwen72b, qwen7b