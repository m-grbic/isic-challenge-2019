import os

import traceback
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model: nn.Module, test_loader: DataLoader, averaged_metrics: dict, class_metrics: dict, use_metadata: bool):
    model.to(device).eval()  # Set the model to evaluation mode

    st = time.perf_counter()
    with torch.no_grad():  # Disable gradient calculation during testing

        outputs_lst, labels_lst = [], []

        for data in tqdm(test_loader):
            if use_metadata:
                inputs, metadata, labels = data[0].to(device), torch.stack(data[1], dim=1).float().to(device), data[2].to(device)
                outputs = model(inputs, metadata)
            else:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)

            outputs_lst.append(outputs.cpu())
            labels_lst.append(labels.cpu())
    et = time.perf_counter()

    outputs=torch.cat(outputs_lst)
    labels=torch.cat(labels_lst)
    print(f"Inference time: {labels.shape[0] / (et - st)} FPS")


    avg_metrics_dict = {metric_name: metric(outputs, labels).item() for metric_name, metric in averaged_metrics.items()}
    cls_metrics_dict = {metric_name: metric(outputs, labels).numpy() for metric_name, metric in class_metrics.items()}

    return avg_metrics_dict, cls_metrics_dict


def validate_model(model: nn.Module, val_loader: DataLoader, criterion, metrics: dict, use_metadata: bool):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation during validation

        outputs_lst, labels_lst = [], []

        for data in tqdm(val_loader):
            if use_metadata:
                inputs, metadata, labels = data[0].to(device), torch.stack(data[1], dim=1).float().to(device), data[2].to(device)
                outputs = model(inputs, metadata)
            else:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            outputs_lst.append(outputs.cpu())
            labels_lst.append(labels.cpu())

    outputs=torch.cat(outputs_lst)
    labels=torch.cat(labels_lst)

    val_metrics = {metric_name: metric(outputs, labels).item() for metric_name, metric in metrics.items()}
    avg_loss = val_loss / len(val_loader)

    return avg_loss, val_metrics


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion,
        optimizer: optim,
        num_epochs: int,
        metrics: dict,
        early_stopping_patience: int,
        use_metadata: bool
    ):

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, threshold=1e-3, verbose=True)
    try:
        writer = SummaryWriter()

        train_losses = []
        val_losses = []
        best_valid_loss = None
        patience_cnt = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            outputs_lst, labels_lst = [], []
            for data in tqdm(train_loader):
                optimizer.zero_grad()

                if use_metadata:
                    inputs, metadata, labels = data[0].to(device), torch.stack(data[1], dim=1).float().to(device), data[2].to(device)
                    outputs = model(inputs, metadata)
                else:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                outputs_lst.append(outputs.cpu())
                labels_lst.append(labels.cpu())

            outputs=torch.cat(outputs_lst)
            labels=torch.cat(labels_lst)
            train_metrics = {metric_name: metric(outputs, labels).item() for metric_name, metric in metrics.items()}

            # Compute average training loss for the epoch
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validate the model
            val_loss, val_metrics = validate_model(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                metrics=metrics,
                use_metadata=use_metadata
            )
            val_losses.append(val_loss)

            # LR scheduler step
            scheduler.step(val_loss)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Train/Loss', train_loss, epoch)
            for metric, value in train_metrics.items():
                writer.add_scalar(f'Train/{metric}', value, epoch)
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            for metric, value in val_metrics.items():
                writer.add_scalar(f'Validation/{metric}', value, epoch)

            # Print epoch statistics
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

            if best_valid_loss is None or val_losses[-1] <= best_valid_loss:
                print("New best model found! Saving..")
                save_model(model)
                best_valid_loss = val_losses[-1]
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt == early_stopping_patience:
                return

        writer.close()
    except KeyboardInterrupt:
        print("Training stopped. Proceeding with saving model.")
    except Exception as e:
        print(traceback.format_exc())


def save_model(model):
    model_name = sorted(os.listdir("./runs"))[-1]
    torch.save(model.state_dict(), f"./models/{model_name}.pth")
