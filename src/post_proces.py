import argparse
import asyncio
import time

import fasttext
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
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


def replace_options(text):
    return text.replace("Options:", "Opties:")


def post_process_wrong_transaltions(df):
    df["translation"] = df["translation"].apply(replace_options)
    return df


def filter_bad_translations(df):
    indices_of_bad_translations = df[~df["translation"].apply(filter_dataset)].index
    print(indices_of_bad_translations)
    return df.drop(indices_of_bad_translations)


def check_language(config, df, target_language="nld"):
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification", filename="model.bin"
    )
    model = fasttext.load_model(model_path)

    for retry in range(args.max_retries):
        not_in_target_df = df[
            ~df["translation"].apply(lambda x: is_language(x, model, target_language))
        ]
        prompts_not_in_target_language = not_in_target_df["prompt"].tolist()

        if prompts_not_in_target_language:
            translations = asyncio.run(
                call_chatgpt_bulk(prompts_not_in_target_language, config)
            )
            for prompt, translation in zip(
                prompts_not_in_target_language, translations
            ):
                df.loc[df["prompt"] == prompt, "translation"] = translation

    indices_not_in_target_language = df[
        ~df["translation"].apply(lambda x: is_language(x, model, target_language))
    ].index
    df = df.drop(indices_not_in_target_language)

    return df


def calculate_elapsed_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time % 60)
    print(f"Total time: {elapsed_minutes} minutes and {elapsed_seconds} seconds")


if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()
    config = load_config(args)

    dataset = load_from_disk(args.db_location)

    new_ds = DatasetDict()

    for key in dataset.keys():
        df = dataset[key].to_pandas()

        df = check_language(config, df, target_language="nld")

        df = filter_bad_translations(df)

        new_ds[key] = Dataset.from_pandas(df, preserve_index=False)

    print(new_ds)

    save_dataset(new_ds, args)

    calculate_elapsed_time(start_time)
