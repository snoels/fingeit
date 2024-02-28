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
import ssl
import time
from configparser import ConfigParser

import aiohttp
import certifi
import pandas as pd
import requests
from datasets import load_from_disk


def load_args():
    """Loads command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_location", type=str, help="Database location")
    parser.add_argument("--db_new_location", type=str, help="New database location")
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="alpaca_prompt",
        help="alpaca_prompt or alpaca_empty_input_prompt",
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default="Dutch",
        help="Specify the target language for translations. The translation model and prompts will be adjusted based on this language. Default is 'Dutch'.",
    )

    return parser.parse_args()


def load_config(args):
    """Loads configuration from config.ini file."""
    config = ConfigParser()
    config.read("config.ini")

    target_language = args.target_language

    system_prompt = config.get("TRANSLATE", "system_prompt")
    config.set(
        "TRANSLATE",
        "system_prompt",
        system_prompt.replace("<target_language>", target_language),
    )

    prompt = config.get("TRANSLATE", "prompt")
    config.set(
        "TRANSLATE", "prompt", prompt.replace("<target_language>", target_language)
    )

    return config


async def call_chatgpt_async(session, config, target: str):
    payload = {
        "model": config.get("TRANSLATE", "model"),
        "messages": [
            {"role": "system", "content": config.get("TRANSLATE", "system_prompt")},
            {
                "role": "user",
                "content": config.get("TRANSLATE", "prompt") + "\n\n" + target,
            },
        ],
        "max_tokens": config.getint("TRANSLATE", "max_tokens"),
        "temperature": config.getfloat("TRANSLATE", "temperature"),
    }
    try:
        async with session.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.get('TRANSLATE', 'openai_secret_key')}",
            },
            json=payload,
            ssl=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
            response = await response.json()
        if "error" in response:
            print(f"OpenAI request failed with error {response['error']}")
        return response["choices"][0]["message"]["content"]
    except:
        print("Request failed.")


async def call_chatgpt_bulk(prompts, config):
    async with aiohttp.ClientSession() as session:
        responses = await asyncio.gather(
            *[call_chatgpt_async(session, config, prompt) for prompt in prompts]
        )
    return responses


def call_chatgpt_sync(config, target: str):
    payload = {
        "model": config.get("TRANSLATE", "model"),
        "messages": [
            {"role": "system", "content": config.get("TRANSLATE", "system_prompt")},
            {
                "role": "user",
                "content": config.get("TRANSLATE", "prompt") + "\n\n" + target,
            },
        ],
        "max_tokens": config.getint("TRANSLATE", "max_tokens"),
        "temperature": config.getfloat("TRANSLATE", "temperature"),
    }
    try:
        response = requests.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.get('TRANSLATE', 'openai_secret_key')}",
            },
            json=payload,
            verify=certifi.where(),
        )
        response = response.json()
        if "error" in response:
            print(f"OpenAI request failed with error {response['error']}")
        return response["choices"][0]["message"]["content"]
    except:
        print("Request failed.")


def alpaca_prompt(instruction, input, response):
    return f"""### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"""


def alpaca_empty_input_prompt(instruction, response):
    return f"""### Instruction:\n{instruction}\n\n### Response:\n{response}"""


def add_prompts(dataset, prompt_type):
    """Adds prompts to each item in the dataset."""

    for dataset_keys in dataset.keys():
        if prompt_type == "alpaca_prompt":
            dataset[dataset_keys] = dataset[dataset_keys].map(
                lambda x: {
                    "prompt": alpaca_prompt(x["instruction"], x["input"], x["output"])
                }
            )
        elif prompt_type == "alpaca_empty_input_prompt":
            dataset[dataset_keys] = dataset[dataset_keys].map(
                lambda x: {
                    "prompt": alpaca_empty_input_prompt(x["instruction"], x["output"])
                }
            )
    return dataset


def translate_and_add_response(dataset, config):
    """Translates dataset from prompts and adds response to dataset."""
    for dataset_keys in dataset.keys():
        pd_df = pd.DataFrame(dataset[dataset_keys])
        prompts = list(pd_df["prompt"])

        start_time = time.time()
        responses = asyncio.run(call_chatgpt_bulk(prompts, config))
        elapsed_time = time.time() - start_time

        print(f"The translation took {elapsed_time} seconds to run.")
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


def main():
    args = load_args()
    config = load_config(args)

    # Load and filter dataset
    dataset = load_from_disk(args.db_location)
    for dataset_keys in dataset.keys():
        dataset[dataset_keys] = dataset[dataset_keys].select(range(100))

    # Add prompts to the dataset
    dataset = add_prompts(dataset, args.prompt_type)

    # Translate and add response to the dataset
    dataset = translate_and_add_response(dataset, config)

    # Save dataset to new location
    save_dataset(dataset, args)


if __name__ == "__main__":
    main()
