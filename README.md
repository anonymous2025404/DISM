# DISM
Code for our paper ***Beyond End-to-end: A Decoupled and Interactive LLM-Based Framework for Structured Medical Report Generation***.

## Environment
Follow the file *requirements.txt*.

## Config
Check the config.py and change the filepath to your own path.
Some of them may not be used.

## Data prepare
There are three type of data files should be prepared.
1. Patients' clinical context.
2. MIMIC reports with 14 labels.
3. Previous case database(vdb).
4. Finding-region data including bbox_coordinates, bbox_labels and bbox_anatomicalfinding for patients extracted from Chest imaGenome dataset.
You can follow /tools/create_database.py, /tools/patient_info_extractor.py, /tools/Label_Region_Classifier/transfer_labels_to_13labels.py and re_process_dataset.py to process them.

## Train
* For finding-region classifier, run /tools/Label_Region_Classifier/trainer.py
* For Locator, run /tools/Locator/train_locator.py

## Test the generated reports
1. Run /tools/llm_qwen.py to generate reports for test datasets.
2. Run /tools/evaluate_reports.py to get the NLG scores and CE scores.
