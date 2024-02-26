"""This script performs post-processing on a given dataset, involving language checks and filtering based on regex. 

Initially, the script checks the language of each prompt using the fasttext language identification method. If the language does not match the target language, the prompt is retranslated up to a maximum number of retries. Once the language check is completed, the script proceeds to filter out incorrect translations based on regex checks. The script ends by saving the new dataset in the specified storage path.

Instructions:

1. Open terminal and navigate to the directory containing the script. Activate the virtual environment if necessary.
2. Execute the script using the following command:

    python -m src.data_processing.post_process --db_location <path_to_data> --db_new_location <path_to_new_data> --max_retries <number_of_attempts> --target_language <language>

    Please replace:

    - <path_to_data> with the path of the dataset to be processed
    - <path_to_new_data> with the desired location for the new dataset
    - <number_of_attempts> with the number of retries to be allowed for language retranslation
    - <language> with the target language (default is 'Dutch').
"""

import argparse
import asyncio
import time

import fasttext
from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import hf_hub_download

from src.data_processing.config import get_config
from src.data_processing.translators import ChatGptTranslator
from src.data_processing.utils import filter_dataset, is_language, save_dataset


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


def filter_bad_translations(df):
    indices_of_bad_translations = df[~df["translation"].apply(filter_dataset)].index
    print(indices_of_bad_translations)
    return df.drop(indices_of_bad_translations)


def check_language(df, translator: ChatGptTranslator, target_language="nld"):
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
                translator.call_chatgpt_bulk(prompts_not_in_target_language)
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
    config = get_config(args)

    dataset = load_from_disk(args.db_location)
    translator = ChatGptTranslator(config)

    new_ds = DatasetDict()

    for key in dataset.keys():
        df = dataset[key].to_pandas()

        df = check_language(df, translator, target_language="nld")

        df = filter_bad_translations(df)

        new_ds[key] = Dataset.from_pandas(df, preserve_index=False)

    save_dataset(new_ds, args)

    calculate_elapsed_time(start_time)
