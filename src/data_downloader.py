import argparse
import os
from configparser import ConfigParser

from datasets import load_dataset


def load_args():
    """Loads command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_location", type=str, default="./data", help="Target database location"
    )
    return parser.parse_args()


def load_config():
    """Loads configuration from config.ini file."""
    config = ConfigParser()
    config.read("./src/config.ini")
    return config


def download_datasets(config, db_location):
    """Downloads datasets mentioned in config.ini and save under ./data directory"""
    print(dict(config["DEFAULT"]))
    dataset_links = config.get("DATASETS", "dataset_links").split(",")

    for dataset_link in dataset_links:
        dataset_link = dataset_link.strip()
        print(f"Downloading {dataset_link}...")
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
    download_datasets(config, args.db_location)


if __name__ == "__main__":
    main()
