import torch
from torch.utils.data import WeightedRandomSampler
import pandas as pd


def get_weighted_sampler(df: pd.DataFrame):
    # Calculate class weights based on the distribution of target labels
    class_counts = df['target'].value_counts().sort_index()
    class_weights = 1.0 / class_counts

    # Create weights array based on target labels
    weights = torch.tensor([class_weights[label] for label in df['target']], dtype=torch.double)

    # Create a WeightedRandomSampler instance
    return WeightedRandomSampler(weights, len(weights), replacement=True)
