import torch
from torchmetrics.classification import Accuracy, Recall, Specificity, AUROC, AveragePrecision, F1Score, Precision


def get_averaged_metrics(num_classes: int):
    return {
        'Accuracy': Accuracy(task="multiclass", average='macro', num_classes=num_classes),
        'Accuracy (weighted)': Accuracy(task="multiclass", average='weighted', num_classes=num_classes),
        'Recall': Recall(task="multiclass", average='macro', num_classes=num_classes),
        'Specificity': Specificity(task="multiclass", average='macro', num_classes=num_classes),
        'AUROC': AUROC(task="multiclass", average='macro', num_classes=num_classes),
        'AveragePrecision': AveragePrecision(task="multiclass", average='macro', num_classes=num_classes),
        'F1Score': F1Score(task="multiclass", average='macro', num_classes=num_classes),
    }

def get_class_metrics(num_classes: int):
    return {
        'Accuracy': Accuracy(task="multiclass", average=None, num_classes=num_classes),
        'OneVsOtherAccuracy': OneVsOtherAccuracy(num_classes=num_classes),
        'Recall': Recall(task="multiclass", average=None, num_classes=num_classes),
        'Specificity': Specificity(task="multiclass", average=None, num_classes=num_classes),
        'AUROC': AUROC(task="multiclass", average=None, num_classes=num_classes),
        'AveragePrecision': AveragePrecision(task="multiclass", average=None, num_classes=num_classes),
        'F1Score': F1Score(task="multiclass", average=None, num_classes=num_classes),
        'PPV': Precision(task="multiclass", average=None, num_classes=num_classes),
        'NPV': NegativePredictiveValue(num_classes=num_classes)
    }


class OneVsOtherAccuracy:

    def __init__(self, num_classes: int):
        self._num_classes = num_classes

    def __call__(self, outputs, labels):

        _, predicted = torch.max(outputs, 1)

        accuracies = []
        for cls in range(self._num_classes):
            cls_predicted = (predicted == cls)
            cls_labels = (labels == cls)
            cls_true_positives = (cls_predicted == cls_labels).sum(axis=0)
            accuracies.append(cls_true_positives / len(cls_predicted))
        return torch.Tensor(accuracies)



class NegativePredictiveValue:

    def __init__(self, num_classes: int):
        self._num_classes = num_classes

    def __call__(self, outputs, labels):

        _, predicted = torch.max(outputs, 1)

        npv_list = [
            negative_predictive_value(predicted, labels, class_label) for class_label in range(self._num_classes)
        ]
        return torch.Tensor(npv_list)


def negative_predictive_value(predicted, labels, class_label):
    """
    Compute Negative Predictive Value (NPV) for a specific class in multiclass classification.

    Args:
        outputs (torch.Tensor): Predicted probabilities or logits, shape (batch_size, num_classes).
        labels (torch.Tensor): True class labels, shape (batch_size,).
        class_label (int): Label of the class for which NPV is computed.
        threshold (float): Threshold for prediction (default: 0.5).

    Returns:
        float: Negative Predictive Value (NPV) for the specified class.
    """
    class_mask = (labels == class_label)
    
    # True negatives (TN): true negative cases (labels != class_label) that are correctly identified
    tn = torch.sum((predicted != class_label) & ~class_mask).item()
    
    # Predicted negatives (PN): cases predicted as negative (predicted != class_label)
    pn = torch.sum(predicted != class_label).item()
    
    if pn == 0:
        return 0.0  # Avoid division by zero
    
    # Negative Predictive Value (NPV) for the specified class
    npv = tn / pn
    return npv
