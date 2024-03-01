"""
This is a script for managing the translation of a dataset using the OpenAI Completion API. It's responsible for adding prompts to data, translating the data, and storing the translations in a new dataset. The script operates with the help of the 'translation_service.py' module and receives its configuration settings from 'config.ini'.

This script offers flexibility in terms of the translation prompts used and the location of the original and new datasets. The script is asynchronous, using Python's asyncio and aiohttp modules. This allows it to handle multiple translation requests at the same time, improving performance.

How to run the script:
1. Navigate to the location of your code in the terminal. If you're using a virtual environment, make sure it's active.
2. Use the following command structure to run the script:

    poetry run python translation_service.py --db_location "<location_of_database_to_be_translated>" --db_new_location "<location_to_save_translated_database>" --prompt_type "<type_of_prompt>" --target_language "<target_language>"

Replace "<location_of_database_to_be_translated>", "<location_to_save_translated_database>", "<type_of_prompt>", and "<target_language>" with your specific values.

Command Example:

    poetry run python translation_service.py --db_location "../data/FinGPT/fingpt-sentiment-train" --db_new_location "../data/FinGPT-translated" --prompt_type "alpaca_prompt" --target_language "Dutch"

In the command example above, the script takes the dataset stored in the "../data/FinGPT/fingpt-sentiment-train" directory, translates it into Dutch using the "alpaca_prompt" prompt type, and stores the translated dataset in the "../data/FinGPT-translated" directory.
"""

import argparse
import asyncio
import os

import pandas as pd
from datasets import load_from_disk

from src.data_processing.config import load_config
from src.data_processing.prompters import AlpacaEmptyInputPrompter, AlpacaPrompter
from src.data_processing.translators import ChatGptTranslator


def load_args():
    """Loads command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_location", type=str, help="Database location")
    parser.add_argument("--db_new_location", type=str, help="New database location")
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="alpaca_prompt",
        choices=["alpaca_prompt", "alpaca_empty_input_prompt"],
        help="alpaca_prompt or alpaca_empty_input_prompt",
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default="Dutch",
        help="Specify the target language for translations. The translation model and prompts will be adjusted based on this language. Default is 'Dutch'.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Enable retry functionality. Default is '0'. 1 enables retry.",
    )

    return parser.parse_args()


def get_config(args):
    """Loads configuration from config.ini file."""
    sub_section = "TRANSLATE"
    config = load_config()
    target_language = args.target_language

    system_prompt = config.get(sub_section, "system_prompt")
    config.set(
        sub_section,
        "system_prompt",
        system_prompt.replace("<target_language>", target_language),
    )

    prompt = config.get(sub_section, "prompt")
    config.set(sub_section, "prompt", prompt.replace("<target_language>", target_language))
    return config[sub_section]


def retry_translate_and_add_response(dataset, translator: ChatGptTranslator):
    """Translates dataset from prompts and adds response to dataset."""
    for dataset_keys in dataset.keys():
        pd_df = pd.DataFrame(dataset[dataset_keys])
        # Extract prompts that are yet to be translated (translation is None)
        prompts_to_translate = pd_df[pd_df["translation"].isna()][
            "prompt"
        ].values.tolist()
        indices_to_translate = pd_df[pd_df["translation"].isna()].index.tolist()

        if prompts_to_translate:
            # Retrieve translations
            translations = asyncio.run(
                translator.call_chatgpt_bulk(prompts_to_translate)
            )

            # Assign translations back to the dataset at appropriate indices
            for idx, translation in zip(indices_to_translate, translations):
                pd_df.at[idx, "translation"] = translation

            new_dataset = dataset[dataset_keys].remove_columns("translation")
            dataset_with_new_translation = new_dataset.add_column(
                "translation", list(pd_df["translation"])
            )
            dataset[dataset_keys] = dataset_with_new_translation
    return dataset


def translate_and_add_response(dataset, translator: ChatGptTranslator):
    """Translates dataset from prompts and adds response to dataset."""
    for dataset_keys in dataset.keys():
        pd_df = pd.DataFrame(dataset[dataset_keys])
        prompts = list(pd_df["prompt"])

        responses = asyncio.run(translator.call_chatgpt_bulk(prompts))

        dataset_with_translation = dataset[dataset_keys].add_column(
            "translation", responses
        )
        dataset[dataset_keys] = dataset_with_translation

    return dataset


def save_dataset(dataset, args):
    """Saves dataset to new location."""
    original_folder_name = args.db_location.split("/")[-1]
    new_db_location = os.path.join(args.db_new_location, original_folder_name)
    dataset.save_to_disk(new_db_location)


if __name__ == "__main__":
    args = load_args()
    config = get_config(args)

    # Load and filter dataset
    dataset = load_from_disk(args.db_location)
    translator = ChatGptTranslator(config)

    if args.retry:
        dataset = retry_translate_and_add_response(dataset, translator)
    else:
        prompter = (
            AlpacaPrompter()
            if args.prompt_type == "alpaca_prompt"
            else AlpacaEmptyInputPrompter()
        )
        dataset = prompter.add_prompts(dataset)
        dataset = translate_and_add_response(dataset, translator)

    # Save dataset to new location
    save_dataset(dataset, args)
