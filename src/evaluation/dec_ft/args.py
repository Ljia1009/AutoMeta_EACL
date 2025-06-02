import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="",
        help="Full path of the file used for training. ",
    )
    parser.add_argument(
        "--train_data_option",
        type=str,
        default="train",
        help="Option for the training data. ",
    )
    parser.add_argument(
        "--valid_data_option",
        type=str,
        default="dev",
        help="Option for the validation data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Full path to save the output; if not provided, the output will be saved in the data/preprocessed/ directory with data options in the name.",
    )

    parser.add_argument(
        "--data_option",
        type=str,
        default="dev",
        help="Option for the file used for testing, ignored when full path is provided. Valid options are dev, test, or train.",
    )
    args = parser.parse_args()
    return args
