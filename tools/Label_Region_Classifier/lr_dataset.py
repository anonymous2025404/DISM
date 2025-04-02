import cv2
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, dataset_df, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        try:
            image_path = self.dataset_df.iloc[index, 1]

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            class_labels = self.dataset_df.iloc[index, 3]
            findings = self.dataset_df.iloc[index, 4]
            dicom_id = self.dataset_df.iloc[index, 0]

            transformed = self.transforms(image=image)
            transformed_image = transformed["image"]

            sample = {
                "image": transformed_image,
                "labels": torch.tensor(class_labels, dtype=torch.int64),
                "findings": torch.tensor(findings, dtype=torch.int64),
                "dicom_id": dicom_id,
            }
        except Exception:
            return None

        return sample