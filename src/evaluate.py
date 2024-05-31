import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

from utilities import (test_model, load_dataset, get_dataset_class, get_test_transforms, get_model, 
                       get_averaged_metrics, get_class_metrics, load_config, EvalConfig)

config = EvalConfig(**load_config("eval.yaml"))
print(config)


def main():
    print("Creating splits")
    _, _, test_df = load_dataset(config.dataset_name)

    # Drop duplicates from columns A and B
    unique_target_class_name_df = test_df.drop_duplicates(subset=['target', 'class_name'])

    # Create the dictionary mapping unique values of column A to column B values
    target2class_name = dict(zip(unique_target_class_name_df['target'], unique_target_class_name_df['class_name']))
    num_classes = test_df.target.nunique()
    assert len(target2class_name) == num_classes

    averaged_metrics = get_averaged_metrics(num_classes=num_classes)
    class_metrics = get_class_metrics(num_classes=num_classes)

    print("Create datasets")
    dataset_class = get_dataset_class(config.use_metadata)
    test_ds = dataset_class(test_df, get_test_transforms())

    print("Create DataLoaders")
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, pin_memory=True, num_workers=config.num_workers, shuffle=False)

    print("Load model")
    model = get_model(num_classes=num_classes, use_metadata=config.use_metadata, model_name=config.backbone)
    model.load_state_dict(torch.load(config.model_path))

    print("Evaluating..")
    _, cls_metrics_dct = test_model(
        model=model,
        test_loader=test_dl,
        averaged_metrics=averaged_metrics,
        class_metrics=class_metrics,
        use_metadata=config.use_metadata
    )
    print('Finished evaluation')

    class2metrics = {
        class_name: [v[target] for v in cls_metrics_dct.values()] for target, class_name in target2class_name.items()
    }
    metrics_df = pd.DataFrame(class2metrics)
    
    metrics_df['Average'] = metrics_df.mean(axis=1)
    metrics_df['Metric'] = list(cls_metrics_dct.keys())
    
    columns_order = ['Metric', 'Average'] + [col for col in metrics_df.columns if col not in ['Metric', 'Average']]
    metrics_df = metrics_df[columns_order]

    metrics_df.to_excel(f'results/{os.path.basename(config.model_path)}'.replace('.pth', '.xlsx'))
    print(metrics_df)

if __name__ == "__main__":
    main()