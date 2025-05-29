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
        "--test_data_path",
        type=str,
        default="dev",
        help="Full path of the file used for testing.",
    )
    parser.add_argument(
        "--key_option",
        type=str,
        default="review",
        help="Option for the keys extracted from each review. Valid options are review, all.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of samples to run summarization for, default to dataset length.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Full path to save the output; if not provided, the output will be saved in the output/ directory with model options in the name.",
    )

    parser.add_argument(
        "--data_option",
        type=str,
        default="dev",
        help="Option for the file used for testing, ignored when full path is provided. Valid options are dev, test, or train.",
    )
    args = parser.parse_args()
    return args
