from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


def get_dataset_class(use_metadata: bool):
    return ImageAndMetadataDataset if use_metadata else ImageOnlyDataset


class ImageOnlyDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.at[idx, 'image_path']
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        target = self.dataframe.at[idx, 'target']

        return image, target


class ImageAndMetadataDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.metadata_feature_columns = self.dataframe.columns.get_indexer(
            [col for col in self.dataframe.columns if col not in {'image', 'image_path', 'target', 'class_name'}]
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.at[idx, 'image_path']
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        target = self.dataframe.at[idx, 'target']
        features = self.dataframe.iloc[idx, self.metadata_feature_columns].tolist()

        return image, features, target

