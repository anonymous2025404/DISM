import sys
sys.path.append("")

import csv
import json
import logging
import os
import re

import imagesize
import spacy
import torch
from tqdm import tqdm
from config import *
import tools.section_parser as sp
# from src.path_datasets_and_weights import path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg, path_full_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# to log certain statistics during dataset creation
txt_file_for_logging = "log_file_dataset_creation.txt"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# constant specifies how many rows to create in the customized csv files
# can be useful to create small sample datasets (e.g. of len 200) for testing things
# if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is None, then all possible rows are created
NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES = None

# settings
path_full_dataset = ""
path_mimic_cxr = REPORT_PATH
path_mimic_cxr_jpg = ""
path_chest_imagenome = ""


def write_stats_to_log_file(
    dataset: str,
    num_images_ignored_or_avoided: int,
    missing_images: list[str],
    missing_reports: list[str],
    num_faulty_bboxes: int,
    num_images_without_29_regions: int
):
    with open(txt_file_for_logging, "a") as f:
        f.write(f"{dataset}:\n")
        f.write(f"\tnum_images_ignored_or_avoided: {num_images_ignored_or_avoided}\n")

        f.write(f"\tnum_missing_images: {len(missing_images)}\n")
        for missing_img in missing_images:
            f.write(f"\t\tmissing_img: {missing_img}\n")

        f.write(f"\tnum_missing_reports: {len(missing_reports)}\n")
        for missing_rep in missing_reports:
            f.write(f"\t\tmissing_rep: {missing_rep}\n")

        f.write(f"\tnum_faulty_bboxes: {num_faulty_bboxes}\n")
        f.write(f"\tnum_images_without_29_regions: {num_images_without_29_regions}\n\n")


def write_rows_in_new_csv_file(dataset: str, csv_rows: list[list]) -> None:
    log.info(f"Writing rows into new {dataset}.csv file...")

    if dataset == "test":
        csv_rows, csv_rows_less_than_29_regions = csv_rows

    new_csv_file_path = os.path.join(path_full_dataset, dataset)
    new_csv_file_path += ".csv" if not NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES else f"-{NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES}.csv"

    header = ["subject_id", "study_id", "image_id", "mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_phrases", "bbox_phrase_exists", "bbox_is_abnormal", "bbox_anatomicalfinding"]
    if dataset in ["valid", "test"]:
        header.append("reference_report")

    with open(new_csv_file_path, "w") as fp:
        csv_writer = csv.writer(fp)

        csv_writer.writerow(header)
        csv_writer.writerows(csv_rows)

    # for the test set, we put all images that do not have bbox coordinates (and corresponding bbox labels) for all 29 regions
    # into a 2nd csv file called test-2.csv
    if dataset == "test":
        new_csv_file_path = new_csv_file_path.replace(".csv", "-2.csv")

        with open(new_csv_file_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            csv_writer.writerows(csv_rows_less_than_29_regions)


def check_coordinate(coordinate: int, dim: int) -> int:

    if coordinate < 0:
        coordinate = 0
    elif coordinate > dim:
        coordinate = dim
    return coordinate


def coordinates_faulty(height, width, x1, y1, x2, y2) -> bool:
    area_of_bbox_is_zero = x1 == x2 or y1 == y2
    smaller_than_zero = x2 <= 0 or y2 <= 0
    exceeds_limits = x1 >= width or y1 >= height

    return area_of_bbox_is_zero or smaller_than_zero or exceeds_limits


def determine_if_abnormal(attributes_list: list[list]) -> bool:
    for attributes in attributes_list:
        for attribute in attributes:
            if attribute == "nlp|yes|abnormal":
                return True

    return False


def convert_phrases_to_single_string(phrases: list[str], sentence_tokenizer) -> str:
    def remove_substrings(phrases):
        def remove_wet_read(phrases):
            # since there can be multiple WET READS's, collect the indices where they start and end in index_slices_to_remove
            index_slices_to_remove = []
            for index in range(len(phrases)):
                if phrases[index:index + 8] == "WET READ":

                    # curr_index searches for "AM" or "PM" that signals the end of the WET READ substring
                    for curr_index in range(index + 8, len(phrases)):
                        # since it's possible that a WET READ substring does not have an"AM" or "PM" that signals its end, we also have to break out of the iteration
                        # if the next WET READ substring is encountered
                        if phrases[curr_index:curr_index + 2] in ["AM", "PM"] or phrases[curr_index:curr_index + 8] == "WET READ":
                            break

                    # only add (index, curr_index + 2) (i.e. the indices of the found WET READ substring) to index_slices_to_remove if an "AM" or "PM" were found
                    if phrases[curr_index:curr_index + 2] in ["AM", "PM"]:
                        index_slices_to_remove.append((index, curr_index + 2))

            # remove the slices in reversed order, such that the correct index order is preserved
            for indices_tuple in reversed(index_slices_to_remove):
                start_index, end_index = indices_tuple
                phrases = phrases[:start_index] + phrases[end_index:]

            return phrases

        phrases = remove_wet_read(phrases)
        phrases = re.sub(SUBSTRINGS_TO_REMOVE, "", phrases, flags=re.DOTALL)

        return phrases

    def remove_whitespace(phrases):
        phrases = " ".join(phrases.split())
        return phrases

    def capitalize_first_word_in_sentence(phrases, sentence_tokenizer):
        sentences = sentence_tokenizer(phrases).sents

        # capitalize the first letter of each sentence
        phrases = " ".join(sent.text[0].upper() + sent.text[1:] for sent in sentences)

        return phrases

    def remove_duplicate_sentences(phrases):
        # remove the last period
        if phrases[-1] == ".":
            phrases = phrases[:-1]

        # dicts are insertion ordered as of Python 3.6
        phrases_dict = {phrase: None for phrase in phrases.split(". ")}

        phrases = ". ".join(phrase for phrase in phrases_dict)

        # add last period
        return phrases + "."

    # convert list of phrases into a single phrase
    phrases = " ".join(phrases)

    # remove "PORTABLE UPRIGHT AP VIEW OF THE CHEST:" and similar substrings from phrases, since they don't add any relevant information
    phrases = remove_substrings(phrases)

    # remove all whitespace characters (multiple whitespaces, newlines, tabs etc.)
    phrases = remove_whitespace(phrases)

    # for consistency, capitalize the 1st word in each sentence
    phrases = capitalize_first_word_in_sentence(phrases, sentence_tokenizer)

    phrases = remove_duplicate_sentences(phrases)

    return phrases

def convert_attributes_to_label_list(attributes_list: list[list]) -> list[str]:
    label_list = []
    for attributes in attributes_list:
        for attribute in attributes:
            attribute = attribute.split("|")
            if attribute[0] == "anatomicalfinding" and attribute[1] == "yes":
                label_list.append(attribute[2])
            if attribute[0] == "disease" and attribute[1] == "yes":
                label_list.append(attribute[2])
            if attribute[0] == "tubesandlines" and attribute[1] == "yes":
                label_list.append(attribute[2])
            if attribute[0] == "devices" and attribute[1] == "yes":
                label_list.append(attribute[2])
    return label_list


def get_attributes_dict(image_scene_graph: dict, sentence_tokenizer) -> dict[tuple]:
    attributes_dict = {}
    for attribute in image_scene_graph["attributes"]:
        region_name = attribute["bbox_name"]

        # ignore region_names such as "left chest wall" or "right breast" that are not part of the 29 anatomical regions
        if region_name not in ANATOMICAL_REGIONS:
            continue

        phrases = convert_phrases_to_single_string(attribute["phrases"], sentence_tokenizer)
        is_abnormal = determine_if_abnormal(attribute["attributes"])
        anatomicalfindings = convert_attributes_to_label_list(attribute["attributes"])

        attributes_dict[region_name] = (phrases, is_abnormal, anatomicalfindings)

    return attributes_dict


def get_reference_report(subject_id: str, study_id: str, missing_reports: list[str]):
    # custom_section_names and custom_indices specify reports that don't have "findings" sections
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    if f"s{study_id}" in custom_section_names or f"s{study_id}" in custom_indices:
        return -1  # skip all reports without "findings" sections

    path_to_report = os.path.join(path_mimic_cxr, "files", f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")

    if not os.path.exists(path_to_report):
        shortened_path_to_report = os.path.join(f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")
        missing_reports.append(shortened_path_to_report)
        return -1

    with open(path_to_report) as f:
        report = "".join(f.readlines())

    # split report into sections
    sections, section_names, _ = sp.section_text(report)

    if "findings" in section_names:
        findings_index = len(section_names) - section_names[-1::-1].index("findings") - 1
        report = sections[findings_index]
    else:
        return -1  # skip all reports without "findings" sections

    report = " ".join(report.split())

    return report


def get_total_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        next(csv_reader)

        return sum(1 for row in csv_reader)


def get_rows(dataset: str, path_csv_file: str, image_ids_to_avoid: set) -> list[list]:

    csv_rows = []
    num_rows_created = 0

    if dataset == "test":
        csv_rows_less_than_29_regions = []

    total_num_rows = get_total_num_rows(path_csv_file)

    sentence_tokenizer = spacy.load("en_core_web_trf")

    num_images_ignored_or_avoided = 0
    num_faulty_bboxes = 0
    num_images_without_29_regions = 0
    missing_images = []
    missing_reports = []

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        next(csv_reader)

        for row in tqdm(csv_reader, total=total_num_rows):
            subject_id = row[1]
            study_id = row[2]
            image_id = row[3]

            if image_id in IMAGE_IDS_TO_IGNORE or image_id in image_ids_to_avoid:
                num_images_ignored_or_avoided += 1
                continue

            image_file_path = row[4].replace(".dcm", ".jpg")
            mimic_image_file_path = os.path.join(path_mimic_cxr_jpg, image_file_path)

            if not os.path.exists(mimic_image_file_path):
                missing_images.append(mimic_image_file_path)
                continue

            if dataset in ["valid", "test"]:
                reference_report = get_reference_report(subject_id, study_id, missing_reports)

                # skip images that don't have a reference report with "findings" section
                if reference_report == -1:
                    continue


            chest_imagenome_scene_graph_file_path = os.path.join(path_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"

            with open(chest_imagenome_scene_graph_file_path) as fp:
                image_scene_graph = json.load(fp)

            anatomical_region_attributes = get_attributes_dict(image_scene_graph, sentence_tokenizer)

            new_image_row = [subject_id, study_id, image_id, mimic_image_file_path]
            bbox_coordinates = []
            bbox_labels = []
            bbox_phrases = []
            bbox_phrase_exist_vars = []
            bbox_is_abnormal_vars = []
            bbox_anatomicalfindings = [] 

            width, height = imagesize.get(mimic_image_file_path)

            num_regions = 0

            region_to_bbox_coordinates_dict = {}
            for obj_dict in image_scene_graph["objects"]:
                region_name = obj_dict["bbox_name"]
                x1 = obj_dict["original_x1"]
                y1 = obj_dict["original_y1"]
                x2 = obj_dict["original_x2"]
                y2 = obj_dict["original_y2"]

                region_to_bbox_coordinates_dict[region_name] = [x1, y1, x2, y2]

            for anatomical_region in ANATOMICAL_REGIONS:
                bbox_coords = region_to_bbox_coordinates_dict.get(anatomical_region, None)

                if bbox_coords is None or coordinates_faulty(height, width, *bbox_coords):
                    num_faulty_bboxes += 1
                else:
                    x1, y1, x2, y2 = bbox_coords
                    x1 = check_coordinate(x1, width)
                    y1 = check_coordinate(y1, height)
                    x2 = check_coordinate(x2, width)
                    y2 = check_coordinate(y2, height)

                    bbox_coords = [x1, y1, x2, y2]

                    class_label = ANATOMICAL_REGIONS[anatomical_region] + 1

                    bbox_coordinates.append(bbox_coords)
                    bbox_labels.append(class_label)

                    num_regions += 1

                bbox_phrase, bbox_is_abnormal, bbox_anatomicalfinding = anatomical_region_attributes.get(anatomical_region, ("", False, [""]))
                bbox_phrase_exist = True if bbox_phrase != "" else False

                bbox_phrases.append(bbox_phrase)
                bbox_phrase_exist_vars.append(bbox_phrase_exist)
                bbox_is_abnormal_vars.append(bbox_is_abnormal)
                bbox_anatomicalfindings.append(bbox_anatomicalfinding)

            new_image_row.extend([bbox_coordinates, bbox_labels, bbox_phrases, bbox_phrase_exist_vars, bbox_is_abnormal_vars, bbox_anatomicalfindings])
            print(new_image_row)
            input()

            if dataset == "train" or (dataset in ["valid", "test"] and num_regions == 29):
                if dataset in ["valid", "test"]:
                    new_image_row.append(reference_report)

                csv_rows.append(new_image_row)

                num_rows_created += 1
            elif dataset == "test" and num_regions != 29:
                new_image_row.append(reference_report)
                csv_rows_less_than_29_regions.append(new_image_row)

            if num_regions != 29:
                num_images_without_29_regions += 1

            if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES and num_rows_created >= NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES:
                break

    write_stats_to_log_file(dataset, num_images_ignored_or_avoided, missing_images, missing_reports, num_faulty_bboxes, num_images_without_29_regions)

    if dataset == "test":
        return csv_rows, csv_rows_less_than_29_regions
    else:
        return csv_rows


def create_new_csv_file(dataset: str, path_csv_file: str, image_ids_to_avoid: set) -> None:
    log.info(f"Creating new {dataset}.csv file...")

    csv_rows = get_rows(dataset, path_csv_file, image_ids_to_avoid)

    write_rows_in_new_csv_file(dataset, csv_rows)

    log.info(f"Creating new {dataset}.csv file... DONE!")


def create_new_csv_files(csv_files_dict, image_ids_to_avoid):
    if os.path.exists(path_full_dataset):
        log.error(f"Full dataset folder already exists at {path_full_dataset}.")
        log.error("Delete dataset folder or rename variable path_full_dataset before running script to create new folder!")
        return None

    os.mkdir(path_full_dataset)
    for dataset, path_csv_file in csv_files_dict.items():
        create_new_csv_file(dataset, path_csv_file, image_ids_to_avoid)


def get_images_to_avoid():
    path_to_images_to_avoid = os.path.join(path_chest_imagenome, "silver_dataset", "splits", "images_to_avoid.csv")

    image_ids_to_avoid = set()

    with open(path_to_images_to_avoid) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        next(csv_reader)

        for row in csv_reader:
            image_id = row[2]
            image_ids_to_avoid.add(image_id)

    return image_ids_to_avoid


def get_train_val_test_csv_files():
    """Return a dict with datasets as keys and paths to the corresponding csv files in the chest-imagenome dataset as values"""
    path_to_splits_folder = os.path.join(path_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}

def main():
    csv_files_dict = get_train_val_test_csv_files()
    image_ids_to_avoid = get_images_to_avoid()
    create_new_csv_files(csv_files_dict, image_ids_to_avoid)

if __name__ == "__main__":
    main()
