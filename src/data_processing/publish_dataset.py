""" 
Script to publish the dataset to huggingface hub.
"""

import argparse
from configparser import ConfigParser
from posixpath import abspath, dirname

from datasets import Dataset, load_from_disk


def push_to_hub(dataset_dict, repo_id, private=True, commit_message=None):
    for key in dataset_dict.keys():
        df = dataset_dict[key].to_pandas()
        dataset = Dataset.from_pandas(df)

        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            split=key,
            create_pr=True,
            commit_message=commit_message,
        )


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_location", type=str, help="Database location")
    parser.add_argument(
        "--dataset_name", type=str, help="Name of the dataset on hugging face"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        help="Message you want to add for updating the dataset",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--private",
        type=bool,
        help="If the dataset is private or public",
        required=False,
        default=True,
    )
    return parser.parse_args()


def load_config():
    """Loads configuration from config.ini file."""
    directory = dirname(abspath(__file__))
    config = ConfigParser()
    config.read("config.ini")
    config.read(f"{directory}/config.ini")
    return config["DATASET_PUSH"]


def main():
    args = load_args()
    config = load_config()

    push_to_hub(
        dataset_dict=load_from_disk(args.db_location),
        repo_id=f"{config['repo_group']}/{args.dataset_name}",
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
