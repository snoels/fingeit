import argparse
import copy
from functools import partial
from typing import Any, Dict

import fasttext
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from translation_service import call_chatgpt_sync, load_config, save_dataset
from utils import filter_dataset, identify_language


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_location", type=str, help="Database location")
    parser.add_argument("--db_new_location", type=str, help="New database location")
    parser.add_argument(
        "--max_retries", type=int, default=3, help="Maximum number of allowed retries"
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default="Dutch",
        help="Specify target language. It adjusts translation model and prompts. Default is 'Dutch'.",
    )
    return parser.parse_args()


def update_translation(
    example: Dict[str, Any], idx: int, idx_to_change: int, new_value: Dict[str, Any]
) -> Any:
    return new_value if idx == idx_to_change else example


def replace_options(example):
    example["translation"] = example["text"].replace("Options:", "Opties:")
    return example


def post_process_wrong_transaltions(dataset):
    for dataset_keys in dataset.keys():
        dataset[dataset_keys] = dataset[dataset_keys].map(replace_options)

    return dataset


def filter_translations(args: argparse.Namespace, dataset):
    config = load_config(args)

    dataset = post_process_wrong_transaltions(dataset)

    for dataset_keys, dataset_values in dataset.items():
        remove_indexes = []
        for idx, row in enumerate(dataset_values):
            row = copy.deepcopy(row)

            if filter_dataset(row, "translation"):
                continue

            for retry in range(args.max_retries):
                try:
                    translation = call_chatgpt_sync(config, row["prompt"])
                    row["translation"] = translation
                    if translation and filter_dataset(row, "translation"):
                        dataset[dataset_keys] = dataset_values.map(
                            partial(
                                update_translation, idx_to_change=idx, new_value=row
                            ),
                            with_indices=True,
                        )
                        print(
                            f"Row {idx} translation succeeded after {retry + 1} attempts."
                        )
                        break

                except Exception as e:
                    print(
                        f"Row {idx} translation failed on attempt {retry + 1}: {str(e)}"
                    )
            else:
                print(
                    f"Row {idx} translation failed after {args.max_retries} attempts."
                )
                remove_indexes.append(idx)

        print("to_remove", remove_indexes)
        dataset[dataset_keys] = dataset_values.filter(
            lambda _, idx: idx not in remove_indexes, with_indices=True
        )

    return dataset


def check_language(args: argparse.Namespace, target_language="nld"):
    dataset = load_from_disk(args.db_location)
    config = load_config(args)

    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification", filename="model.bin"
    )
    model = fasttext.load_model(model_path)

    for dataset_keys, dataset_values in dataset.items():
        remove_indexes = []
        for idx, row in enumerate(dataset_values):
            row = copy.deepcopy(row)

            if identify_language(row, "translation", model) == target_language:
                continue

            for retry in range(args.max_retries):
                try:
                    translation = call_chatgpt_sync(config, row["prompt"])
                    row["translation"] = translation
                    if (
                        translation
                        and identify_language(row, "translation", model)
                        == target_language
                    ):
                        dataset[dataset_keys] = dataset_values.map(
                            partial(
                                update_translation, idx_to_change=idx, new_value=row
                            ),
                            with_indices=True,
                        )
                        print(
                            f"Row {idx} translation succeeded after {retry + 1} attempts. The translation has been changed to Dutch."
                        )
                        break

                except Exception as e:
                    print(
                        f"Row {idx} translation failed on attempt {retry + 1}: {str(e)}"
                    )
            else:
                print(
                    f"Row {idx} translation failed after {args.max_retries} attempts. Couldn't convert to Dutch."
                )
                remove_indexes.append(idx)

        print("to_remove", remove_indexes)
        dataset[dataset_keys] = dataset_values.filter(
            lambda _, idx: idx not in remove_indexes, with_indices=True
        )

    return dataset


if __name__ == "__main__":
    args = parse_args()

    # check languages
    dataset = check_language(args)

    # filter translations
    dataset = filter_translations(args, dataset)

    # Save dataset to new location
    save_dataset(dataset, args)
