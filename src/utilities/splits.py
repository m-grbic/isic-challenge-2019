import pandas as pd
import os
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

IMAGES_PATH = "./data/ISIC_2019_Training_Input"
METADATA_PATH = "./data/ISIC_2019_Training_Metadata.csv"
OUTPUTS_PATH = "./data/ISIC_2019_Training_GroundTruth.csv"
DATASETS_PATH = "./data/datasets"


def get_images_df() -> pd.DataFrame:
    """Lists all images and creates DataFrame with image and image path columns."""
    images = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg') or f.endswith('.png')]
    image_paths = [os.path.join(IMAGES_PATH, image) for image in images]
    images = [image.replace('.jpg', '').replace('.png', '') for image in images]
    return pd.DataFrame({'image': images, 'image_path': image_paths})


def get_metadata_df() -> pd.DataFrame:
    """Returns DataFrame with metadata."""
    outputs_df = pd.read_csv(METADATA_PATH)
    outputs_df['age_approx'] = outputs_df['age_approx'].fillna(outputs_df['age_approx'].median())
    outputs_df['sex'] = outputs_df['sex'].ffill()
    outputs_df['lesion_id'] = outputs_df['lesion_id'].fillna('unknown')
    outputs_df['anatom_site_general'] = outputs_df['anatom_site_general'].fillna('unknown')
    outputs_df['lesion_id'] = outputs_df['lesion_id'].str.split('_').str[0]
    return outputs_df


def get_outputs_df() -> pd.DataFrame:
    """Loads DataFrame with expected outputs and creates DataFrame with image and target columns."""
    outputs_df = pd.read_csv(OUTPUTS_PATH)
    classes = [col for col in outputs_df.columns if col != 'image']
    outputs_df['target'] = outputs_df.loc[:, classes].apply(
        lambda x: classes[[i for i, val in enumerate(x.tolist()) if val == 1][0]], axis=1
    )
    return outputs_df[['image', 'target']]


def create_dataset_df() -> pd.DataFrame:
    """Loads images, outputs and metadata DataFrames, merges them and returns a single DataFrame."""
    images_df = get_images_df()
    outputs_df = get_outputs_df()
    metadata_df = get_metadata_df()

    if len(images_df) != len(outputs_df):
        raise RuntimeError(
            f'Different size of input and output DataFrames: {len(images_df)} != {len(outputs_df)}'
            )

    if len(images_df) != len(metadata_df):
        raise RuntimeError(
            f'Different size of input and metadata DataFrames: {len(images_df)} != {len(metadata_df)}'
            )
    
    dataset = pd.merge(images_df, metadata_df, on='image')
    if len(dataset) != len(images_df):
        raise RuntimeError(
            f'Different size of merged and input DataFrames: {len(images_df)} != {len(metadata_df)}'
            )
    
    dataset = pd.merge(dataset, outputs_df, on='image')
    if len(dataset) != len(outputs_df):
        raise RuntimeError(
            f'Different size of input and output DataFrames: {len(dataset)} != {len(outputs_df)}'
            )

    return dataset


def calculate_dataset_stats(df: pd.DataFrame):
    """Calculates DataFrame statistics."""
    print(df.groupby('class_name').agg({'image': 'nunique'}).reset_index())


def calculate_class_weights(df: pd.DataFrame) -> list:
    """
    Calculate class weights based on the distribution of class labels in a DataFrame.

    Args:
        df: DataFrame containing the target column.
        target_column: Name of the target column.

    Returns:
        Class weights
    """
    # Count occurrences of each class label
    class_counts = df['target'].value_counts().sort_index()
    
    # Compute class weights
    class_weights = 1.0 / class_counts
    
    # Normalize weights to sum up to 1
    class_weights /= class_weights.sum()
    
    return class_weights.tolist()


def create_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Does one hot encoding and standard scaling od the features."""
    df = df.copy()
    # Apply one hot encoding to 'anatom_site_general', 'lesion_id', 'sex'
    one_hot_dfs = [pd.get_dummies(df[col], prefix=col, dtype=float) for col in ['anatom_site_general', 'lesion_id', 'sex']]
    # Apply standard scaller to age_approx
    df['age'] = StandardScaler().fit_transform(df[['age_approx']])
    # Concatenate metadata features
    return pd.concat((df[['image', 'age']], *one_hot_dfs), axis=1)


def create_split(
        train_perc: int = 0.8,
        valid_perc: int = 0.1,
        test_perc: int = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates train-valid-test split based on metadata features.

    Pipeline:
        - Get input DataFrame and create metadata features
        - Create column to split on based on unique combination of target, is age less than median, anatom site general, lesion id and sex
        - For each unique combination sample train-valid-test data
        - Concatenate DataFrames and make sure there is no overlap of images
        - Calculate statistics for each DataFrame
    """
    np.random.seed(random_state)

    if abs((train_perc + valid_perc + test_perc) - 1) > 1e-4:
        raise RuntimeError(
            f'Train-valid-test percentages do not sum up to one: {train_perc=}, {valid_perc=} and {test_perc}'
            )
    
    # 
    dataset_df = create_dataset_df()
    metadata_features = create_metadata_features(dataset_df)

    dataset_df['age_less_median'] = (dataset_df['age_approx'] < dataset_df['age_approx'].median()).astype(str)
    dataset_df['split_column'] = dataset_df[['target', 'age_less_median', 'anatom_site_general', 'lesion_id', 'sex']].apply(lambda x: '_'.join(x), axis=1)
    
    classes = dataset_df['split_column'].unique().tolist()
    train_dfs, valid_dfs, test_dfs = [], [], []
    for c in classes:
        # Extract DatAFrame for the unique value
        class_df  = dataset_df[dataset_df['split_column'] == c]
        # Sample train data
        train_class_df = class_df.sample(frac=train_perc)
        # Extract the remaining data
        remaining_df = class_df[~class_df.image.isin(train_class_df.image)]
        # Sample validation data
        valid_class_df = remaining_df.sample(frac=valid_perc/(valid_perc + test_perc))
        # Use the remaining data for testing
        test_class_df = remaining_df[~remaining_df.image.isin(valid_class_df.image)]

        assert len(class_df) == len(train_class_df) + len(valid_class_df) + len(test_class_df)
        
        train_dfs.append(train_class_df)
        valid_dfs.append(valid_class_df)
        test_dfs.append(test_class_df)

    train_df = pd.concat(train_dfs, ignore_index=True)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    train_df = pd.merge(train_df[['image', 'image_path', 'target']], metadata_features, on='image')
    valid_df = pd.merge(valid_df[['image', 'image_path', 'target']], metadata_features, on='image')
    test_df = pd.merge(test_df[['image', 'image_path', 'target']], metadata_features, on='image')

    assert len(set(train_df.image.unique()).intersection(set(valid_df.image.unique()))) == 0
    assert len(set(valid_df.image.unique()).intersection(set(test_df.image.unique()))) == 0
    assert len(set(train_df.image.unique()).intersection(set(test_df.image.unique()))) == 0

    class_encoder = {t: i for i, t in enumerate(train_df.target.unique())}
    for df in [train_df, valid_df, test_df]:
        df['class_name'] = df['target']
        df['target'] = df['class_name'].map(class_encoder)

    print("Training DataFrame statistics")
    calculate_dataset_stats(train_df)
    print("Validation DataFrame statistics")
    calculate_dataset_stats(valid_df)
    print("Testing DataFrame statistics")
    calculate_dataset_stats(test_df)

    return train_df, valid_df, test_df


def save_dataset(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, dataset_name: str) -> None:
    """Saves dataset."""
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    Path(dataset_path).mkdir(exist_ok=True, parents=True)

    train_df.to_pickle(os.path.join(dataset_path, "train.pkl"))
    valid_df.to_pickle(os.path.join(dataset_path, "valid.pkl"))
    test_df.to_pickle(os.path.join(dataset_path, "test.pkl"))


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads dataset."""
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)

    train_df = pd.read_pickle(os.path.join(dataset_path, "train.pkl"))
    valid_df = pd.read_pickle(os.path.join(dataset_path, "valid.pkl"))
    test_df = pd.read_pickle(os.path.join(dataset_path, "test.pkl"))

    return train_df, valid_df, test_df
