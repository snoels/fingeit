"""Download script that will download all the datasets specified in the config file and will save them in the provided location.

"""
import argparse
import os

from config import load_config
from datasets import load_dataset


def load_args():
    """Loads command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_location", type=str, default="./data", help="Target database location"
    )
    return parser.parse_args()


def download_datasets(datasets: list[str], db_location: str) -> None:
    """Download the provided datasets and places them in the db_location."""
    for dataset_link in datasets:
        print(f"Downloading {dataset_link}...")
        _download_dataset(dataset_link, db_location)


def _download_dataset(dataset_link: str, db_location: str) -> None:
    dataset_link = dataset_link.strip()
    try:
        dataset = load_dataset(dataset_link)
        save_path = os.path.join(db_location, dataset_link)
        dataset.save_to_disk(save_path)
        print(f"Saved {dataset_link} to {save_path}")
    except Exception as e:
        print(f"Failed to download {dataset_link} due to {str(e)}")


def main():
    args = load_args()
    config = load_config()
    datasets = config["dataset_links"].split(",")
    download_datasets(datasets, args.db_location)


if __name__ == "__main__":
    main()
