from utilities import load_config, DatasetConfig, create_split, save_dataset

config = DatasetConfig(**load_config("dataset.yaml"))


def main():

    # Create split
    train_df, valid_df, test_df = create_split(
        config.train_perc,
        config.valid_perc,
        config.test_perc
    )

    # TODO: add some function that extracts statistics

    # Save dataset
    save_dataset(train_df, valid_df, test_df, config.dataset_name)


if __name__ == "__main__":
    main()