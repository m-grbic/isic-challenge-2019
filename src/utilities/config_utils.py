import yaml
import os
from pydantic.dataclasses import dataclass

CONFIG_PATH = "configs/"


def load_config(name: str) -> dict:
    """Loads yaml config."""
    with open(os.path.join(CONFIG_PATH, name), "r") as f:
        return yaml.safe_load(f)
    

def save_config(d: dict, path: str) -> None:
    """Saves config as yaml file."""
    with open(path, "w") as f:
        yaml.safe_dump(d, f)


@dataclass
class DatasetConfig:
    dataset_name: str
    train_perc: float
    valid_perc: float
    test_perc: float


@dataclass
class TrainConfig:
    dataset_name: str
    batch_size: int
    num_epochs: int
    num_workers: int
    early_stopping_patience: int
    backbone: str
    use_metadata: bool
    learning_rate: float


@dataclass
class EvalConfig:
    dataset_name: str
    batch_size: int
    num_workers: int
    model_path: str
    backbone: str
    use_metadata: bool

