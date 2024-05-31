import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import datetime
import os

from utilities import (load_dataset, calculate_class_weights, get_dataset_class, 
                       get_train_transforms, get_test_transforms, get_model, 
                       get_averaged_metrics, train_model, load_config, save_config, 
                       TrainConfig)

config = TrainConfig(**load_config("train.yaml"))
print(config)



# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Preparing data")
    train_df, valid_df, _ = load_dataset(config.dataset_name)

    num_classes = train_df.target.nunique()
    metrics = get_averaged_metrics(num_classes)

    dataset_class = get_dataset_class(use_metadata=config.use_metadata)
    train_ds = dataset_class(train_df, get_train_transforms())
    valid_ds = dataset_class(valid_df, get_test_transforms())

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, pin_memory=True, num_workers=config.num_workers, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=config.batch_size, pin_memory=True, num_workers=config.num_workers, shuffle=False)

    print("Load model")
    model = get_model(num_classes=num_classes, use_metadata=config.use_metadata, model_name=config.backbone)
    model.to(device)

    # Define loss function and optimizer
    weights = torch.Tensor(calculate_class_weights(train_df)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print("Training..")
    st = time.perf_counter()
    train_model(
        model=model,
        train_loader=train_dl,
        val_loader=valid_dl,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.num_epochs,
        metrics=metrics,
        early_stopping_patience=config.early_stopping_patience,
        use_metadata=config.use_metadata
    )
    et = time.perf_counter()
    print(f'Finished training after {str(datetime.timedelta(seconds=et-st))}')

    save_config(config.__dict__, f"./runs/{sorted(os.listdir('./runs'))[-1]}/train.yaml")

if __name__ == "__main__":
    main()