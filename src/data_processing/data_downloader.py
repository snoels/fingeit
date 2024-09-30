""" Script to download datasets
All datasets mentioned in the config.ini file will be downloaded and saved under the directory passed in the arguments (--db_location).
"""

import argparse
import os

from datasets import load_dataset

from src.data_processing.config import load_config


def get_args():
    """Loads command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_location", type=str, default="./data", help="Target database location"
    )
    return parser.parse_args()


def get_datasets() -> list[str]:
    config = load_config()["DATASETS"]
    return [x.strip() for x in config.get("dataset_links").split(",")]


def download_datasets(datasets_links: list[str], db_location: str) -> None:
    """Downloads datasets mentioned in config.ini and save under ./data directory"""
    for dataset_link in datasets_links:
        print(f"Downloading {dataset_link}...")
        try:
            dataset = load_dataset(dataset_link)
            save_path = os.path.join(db_location, dataset_link)
            dataset.save_to_disk(save_path)
            print(f"Saved {dataset_link} to {save_path}")
        except Exception as e:
            print(f"Failed to download {dataset_link} due to {str(e)}")


if __name__ == "__main__":
    args = get_args()
    datasets = get_datasets()
    download_datasets(datasets, args.db_location)
