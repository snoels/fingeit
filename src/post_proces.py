import argparse
import asyncio
import time

import fasttext
import pandas as pd
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from translation_service import call_chatgpt_bulk, load_config, save_dataset
from utils import filter_dataset, is_language


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


def replace_options(example):
    example["translation"] = example["translation"].replace("Options:", "Opties:")
    return example


def post_process_wrong_transaltions(dataset):
    for dataset_keys in dataset.keys():
        dataset[dataset_keys] = dataset[dataset_keys].map(replace_options)

    return dataset


def filter_bad_translations(dataset):
    for key in dataset.keys():
        df = pd.DataFrame(dataset[key])

        dataset_with_index = dataset[key].add_column("index", df.index.tolist())

        indices_of_bad_translations = df[
            ~df["translation"].apply(lambda x: filter_dataset(x))
        ].index.tolist()

        print(indices_of_bad_translations)

        dataset[key] = dataset_with_index.filter(
            lambda example: example["index"] not in indices_of_bad_translations
        ).remove_columns("index")

    return dataset


def check_language(args: argparse.Namespace, target_language="nld"):
    dataset = load_from_disk(args.db_location)

    dataset = post_process_wrong_transaltions(dataset)

    config = load_config(args)

    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification", filename="model.bin"
    )
    model = fasttext.load_model(model_path)

    for retry in range(args.max_retries):
        for dataset_keys in dataset.keys():
            pd_df = pd.DataFrame(dataset[dataset_keys])
            not_in_target_df = pd_df[
                ~pd_df["translation"].apply(
                    lambda x: is_language(x, model, target_language)
                )
            ]

            prompts_not_in_target_language = not_in_target_df["prompt"].values.tolist()
            indices_not_in_target_language = not_in_target_df.index.tolist()

            if prompts_not_in_target_language:
                # Retrieve translations
                translations = asyncio.run(
                    call_chatgpt_bulk(prompts_not_in_target_language, config)
                )

                # Assign translations back to the dataset at appropriate indices
                for idx, translation in zip(
                    indices_not_in_target_language, translations
                ):
                    pd_df.at[idx, "translation"] = translation

                new_dataset = dataset[dataset_keys].remove_columns("translation")
                dataset_with_new_translation = new_dataset.add_column(
                    "translation", list(pd_df["translation"])
                )
                dataset[dataset_keys] = dataset_with_new_translation

    for key in dataset.keys():
        df = pd.DataFrame(dataset[key])

        dataset_with_index = dataset[key].add_column("index", df.index.tolist())

        indices_not_in_target_language = df[
            ~df["translation"].apply(lambda x: is_language(x, model, target_language))
        ].index.tolist()

        dataset[key] = dataset_with_index.filter(
            lambda example: example["index"] not in indices_not_in_target_language
        ).remove_columns("index")

    return dataset


def calculate_elapsed_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time % 60)
    print(f"Total time: {elapsed_minutes} minutes and {elapsed_seconds} seconds")


if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()

    # check languages
    dataset = check_language(args)

    # filter translations
    dataset = filter_bad_translations(dataset)

    # Save dataset to new location
    save_dataset(dataset, args)

    # Calculate elapsed time
    calculate_elapsed_time(start_time)
