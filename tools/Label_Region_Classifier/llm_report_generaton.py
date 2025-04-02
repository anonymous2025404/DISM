import os, sys, time
from tqdm import tqdm
from openai import OpenAI
import openai
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import ast


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

if LLM_USE == "qwen7b":
    model_name = QWEN_7B
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

if LLM_USE == "qwen72b":
    client = OpenAI(
        api_key=API_KEY, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


pred_df = pd.read_csv("", header=0, index_col=['dicom_id'])
pred_df['findings'] = pred_df['findings'].apply(ast.literal_eval)

# info_df = pd.read_csv(SOURCE_INFO_PATH, header=0, index_col=['dicom_id'])

gt_df1 = pd.read_csv(TRAIN_PATH, header=0, index_col=['dicom_id'])
gt_df2 = pd.read_csv(VAL_PATH, header=0, index_col=['dicom_id'])
gt_df3 = pd.read_csv(TEST_PATH, header=0, index_col=['dicom_id'])
gt_df = pd.concat([gt_df1, gt_df2, gt_df3], axis=0)

template_no_info = '''
As a medical report editor, modify this simple radiology report to a complete reports: 
{report}

Rules:
Do not add other symptoms and medical manifestations that are not mentioned, and do not omit any symptoms and medical manifestations that are mentioned.
Start directly with the report content.
Do not add any labels, prefixes or explanations.
Do not write impression or other parts, just findings part of medical reports.
Write as the findings section of the medical report.
'''

template_info_rag = '''
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
Do not add descriptions that were not included in the initial X-ray assessment.

{rag_info}
'''

template_info_rag_2 = '''
Write the findings section of the medical report based on the patient's indication, history, and initial chest x-ray evaluation. Identify relevant details from the patient's history, especially existing diagnostic and medical device information, to use in writing the findings section of the imaging report.
Rule:
Start directly with the report findings.
Do not add any labels, prefixes, or explanations.
Do not write impressions or other sections, just the findings section of the chest x-ray medical report.
Write only what is necessary to reflect the patient's pathological findings and medical device information from the initial x-ray evaluation.
Make sure the report is concise, objective, and focused on the current findings, avoiding stating the patient's medical history and personal information.
Follow the format and writing style of the example.
Do not write medical devices without supporting devices mentioned in the initial x-ray evaluation.

{rag_info}

Current patient's indication, history, and other information: [{info}]
Current patient's initial x-ray evaluation: [{report}]
'''

template_info_0shot = '''
Write the findings section of the medical report based on the patient's indication, history, and initial chest x-ray evaluation. Identify relevant details from the patient's history, especially existing diagnostic and medical device information, to use in writing the findings section of the imaging report.
Rule:
Start directly with the report findings.
Do not add any labels, prefixes, or explanations.
Do not write impressions or other sections, just the findings section of the chest x-ray medical report.
Write only what is necessary to reflect the patient's pathological findings and medical device information from the initial x-ray evaluation.
Make sure the report is concise, objective, and focused on the current findings, avoiding stating the patient's medical history and personal information.
Follow the format and writing style of the findings part of chest X-ray radiology report.
Do not write medical devices without supporting devices mentioned in the initial x-ray evaluation.

Current patient's indication, history, and other information: [{info}]
Current patient's initial x-ray evaluation: [{report}]
'''

template_noinfo_rag = '''
As a medical report editor, modify this simple radiology report to a complete reports: 
{report}

Rules:
Do not add other symptoms and medical manifestations that are not mentioned, and do not omit any symptoms and medical manifestations that are mentioned.
Start directly with the report content.
Do not add any labels, prefixes or explanations.
Do not write impression or other parts, just findings part of medical reports.
Write as the findings section of the medical report.
Follow the format and writing style of the example.

{rag_info}
'''


def process_findings_list(findings_list):
    report = ""
    for findings in findings_list:
        pathology = findings.split("-")[0]
        
        region = findings.split("-")[1]
        regions = region.split("|")
        if len(regions) == 0:
            report += f"{pathology} is present."
        else:
            rs = ""
            for r in regions:
                if rs == "":
                    rs += f"{r}"
                else:
                    rs += f", {r}"
            if rs != "":
                report += f"{pathology} is present in {rs}."
            else:
                report += f"{pathology} is present."
    if report == "":
        report = "No findings."
    return report

def llm_lr_0shot_no_info():
    out_list = []

    for idx in tqdm(pred_df.index, position = 0):
        dicom_id = idx
        findings = pred_df.loc[dicom_id, 'findings']
        role_prompt = "You are a medical report editor."
        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": template_no_info.format(report=process_findings_list(findings))}
            # {"role": "user", "content": prompt_no_info.format(report=simple_report)}
        ]
        # print(f"Prompt:\n{messages[1]['content']}")
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
        # print(f"Response:\n{response}")
        # input()
        reference_report = gt_df.loc[dicom_id, 'Report Impression']
        out_list.append([dicom_id, reference_report, response])
    
    out_df = pd.DataFrame(out_list, columns=['dicom_id', 'reference_report', 'Report Impression'])
    out_df.to_csv('llm_lr_0shot_no_info.csv', index=False)

def llm_lr_info_rag():
    info_df = pd.read_csv(SOURCE_INFO_PATH, header=0, index_col=['dicom_id'])
    persist_directory = VDB_INFO_PATH
    db = FAISS.load_local(persist_directory, HuggingFaceBgeEmbeddings(model_name=RAG_CKPT, model_kwargs={'device': 'cpu'}), allow_dangerous_deserialization=True)

    out_list = []

    for idx in tqdm(pred_df.index, position = 0):
        try:
            dicom_id = idx
            findings = pred_df.loc[dicom_id, 'findings']
            pa_info = info_df.loc[idx]['info']
            simple_report = process_findings_list(findings)

            query = f"Which patient has the similarist report as following: INIDCATION {pa_info}. REPORT: {simple_report}"
            k = 3
            retriever = db.similarity_search(query, k=k)
            examples = ""
            for i in range(k):
                content = retriever[i].page_content.split('REPORT:')[1]
                content = content.split(']')[0]
                examples += f"Example {i+1}: {content}\n"

            role_prompt = "You are a medical report editor."
            messages = [
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": template_info_rag_2.format(info=pa_info, report=simple_report)}
            ]
            # print(f"Prompt:\n{messages[1]['content']}")
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
            # print(f"Response:\n{response}")
            # input()
            reference_report = gt_df.loc[dicom_id, 'Report Impression']
            out_list.append([dicom_id, reference_report, response])
        except Exception as e:
            print(e)
            continue

    out_df = pd.DataFrame(out_list, columns=['dicom_id', 'reference_report', 'Report Impression'])
    out_df.to_csv('llm_lr_info_rag.csv', index=False)

def llm_lr_info_0shot():
    info_df = pd.read_csv(SOURCE_INFO_PATH, header=0, index_col=['dicom_id'])
    
    out_list = []

    for idx in tqdm(pred_df.index, position = 0):
        try:
            dicom_id = idx
            findings = pred_df.loc[dicom_id, 'findings']
            pa_info = info_df.loc[idx]['info']
            simple_report = process_findings_list(findings)

            role_prompt = "You are a medical report editor."
            messages = [
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": template_info_0shot.format(info=pa_info, report=simple_report)}
            ]
            # print(f"Prompt:\n{messages[1]['content']}")
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
            # print(f"Response:\n{response}")
            # input()
            reference_report = gt_df.loc[dicom_id, 'Report Impression']
            out_list.append([dicom_id, reference_report, response])
        except Exception as e:
            print(e)
            continue

    out_df = pd.DataFrame(out_list, columns=['dicom_id', 'reference_report', 'Report Impression'])
    out_df.to_csv('llm_lr_info_0shot.csv', index=False)

def llm_prompt_chain():

    def llm_generate(prompt):
        role_prompt = "You are a medical report editor."
        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": prompt}
        ]
        # print(f"Prompt:\n{messages[1]['content']}")
        if LLM_USE == 'qwen7b':
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
            # print(f"Response:\n{response}")
            return response
        if LLM_USE == 'qwen72b':
            completion = client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": role_prompt},
                    {'role': 'user', 'content': prompt}],
                )
            res = completion.model_dump_json()
            res = json.loads(res)
            content = res['choices'][0]['message']['content']
            response = content.replace("\n", " ")
            response = response.lower()
            # print("Response:\n")
            # print(response)
            return response


    info_df = pd.read_csv(SOURCE_INFO_PATH, header=0, index_col=['dicom_id'])

    persist_directory = VDB_INFO_PATH
    db = FAISS.load_local(persist_directory, HuggingFaceBgeEmbeddings(model_name=RAG_CKPT, model_kwargs={'device': 'cpu'}), allow_dangerous_deserialization=True)

    out_list = []

    for idx in tqdm(pred_df.index, position = 0):
        try:
            dicom_id = idx
            findings = pred_df.loc[dicom_id, 'findings']
            pa_info = info_df.loc[idx]['info']
            simple_report = process_findings_list(findings)

            query = f"Which patient has the similarist report as following: {pa_info}. REPORT: {simple_report}"
            k = 3
            retriever = db.similarity_search(query, k=k)
            examples = ""
            for i in range(k):
                content = retriever[i].page_content.split('REPORT:')[1]
                content = content.split(']')[0]
                examples += f"Example {i+1}: {content}\n"
            
            prompt1 = f'''
Extract important information from the following patient medical records, including the patient's current symptoms, current illnesses, and the name and location of medical support equipment in use if clearly mentioned. If it is not mentioned in the record, do not write it.
Current Patient info: {pa_info}.
Note: Condense all important information into one sentence without unnecessary symbols, headings, and explanations. If the medical device is not mentioned in the patient information, do not mention it.
            '''
            response1 = llm_generate(prompt1)

            prompt2 = f'''
This is the report of the chest X-ray of the current patient. I need you to complete this report based on the patient's clinical indications and other information.
The current patient's clinical indications: {response1}
The current patient's examination report: {simple_report}
Note: 
1、Tell me the completed report directly without any other symbols or explanations. The report should not include recommendations, but only state the imaging findings.
2、In examination results, Pleural Other means "pleural thickening", "fibrosis" or "pleural scar". Support Devices means "lines", "picc", "tube" or "marker", choose a reasonable device name of it instead of "Support Devices". Lung lesion means "mass", "nodule" or "tumor", choose a reasonable lesion class name of it instead of "Lung Lesion".
3、If report is "No findings", you can write "The lungs are clear without focal consolidation. No pleural effusion or pneumothorax is seen. The cardiac and mediastinal silhouettes are unremarkable. ".
4、Remember, you are writing a findings part of rediology report for a chest x-ray and must follow its writing rules, don't write patient's history and indication.
5、Don't write sentence like "no support devices or xxx are present.". Focus on findings which mentioned in report.
            '''
            response2 = llm_generate(prompt2)

            prompt3 = f'''
This is the report of an X-ray examination of a patient. I need you to imitate the format and writing method of the three samples I gave you to standardize the writing of this examination result. It is important to note that you should not write about lesions and medical support devices that are not included in the examination results. When writing, you should pay attention to include all lesions, medical support devices, and areas mentioned in the examination results. 
{examples}
Current patient's report: {response2}
Note: 
1、Start directly with the final report without any explanations or extra symbols.
2、If no abnormalities are found in important parts, you can just mention it in one or two sentences.
3、Do not confuse the findings and medical devices in the example with those in your current patient.
4、Do not write "no other abnormalities are identified.". If there are no other abnormalities, you can write "xxx (lesions visible on chest X-ray) not present".
            '''
            response3 = llm_generate(prompt3)

            reference_report = gt_df.loc[dicom_id, 'Report Impression']
            out_list.append([dicom_id, reference_report, response3])
            # input()
        except Exception as e:
            print(e)
            continue

    out_df = pd.DataFrame(out_list, columns=['dicom_id', 'reference_report', 'Report Impression'])
    out_df.to_csv('llm_lr_info_rag_chain.csv', index=False)

def llm_lr_noinfo_rag():

    persist_directory = VDB_PATH
    db = FAISS.load_local(persist_directory, HuggingFaceBgeEmbeddings(model_name=RAG_CKPT, model_kwargs={'device': 'cpu'}), allow_dangerous_deserialization=True)

    out_list = []

    for idx in tqdm(pred_df.index, position = 0):
        try:
            dicom_id = idx
            findings = pred_df.loc[dicom_id, 'findings']
            simple_report = process_findings_list(findings)

            query = f"Which patient has the similarist report as following: REPORT: {simple_report}"
            k = 3
            retriever = db.similarity_search(query, k=k)
            examples = ""
            for i in range(k):
                content = retriever[i].page_content.split('REPORT:')[1]
                content = content.split(']')[0]
                examples += f"Example {i+1}: {content}\n"
            
            role_prompt = "You are a medical report editor."
            messages = [
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": template_noinfo_rag.format(report=simple_report, rag_info=examples)}
            ]
            # print(f"Prompt:\n{messages[1]['content']}")
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
            # print(f"Response:\n{response}")
            # input()
            reference_report = gt_df.loc[dicom_id, 'Report Impression']
            out_list.append([dicom_id, reference_report, response])
        except Exception as e:
            print(e)
            continue

    out_df = pd.DataFrame(out_list, columns=['dicom_id', 'reference_report', 'Report Impression'])
    out_df.to_csv('llm_lr_noinfo_rag.csv', index=False)


if __name__ == '__main__':
    # llm_lr_0shot_no_info()
    # llm_lr_info_rag()
    # llm_lr_info_0shot()
    # llm_prompt_chain()
    llm_lr_noinfo_rag()