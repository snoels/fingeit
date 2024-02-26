""" Script to add system and user prompts to the dataset.

Given a dataset (--db_location), this script adds system and user prompts to the dataset and saves the new dataset to a new location (--db_new_location).

"""

import argparse
import os

from datasets import DatasetDict, load_from_disk
from src.data_processing.utils import save_dataset


ALPACA_INTROMESSAGE_INPUT = """Hieronder staat een instructie die een taak beschrijft, samen met een input die context voorziet
Schrijf een reactie die op een passende manier voldoet aan de vraag.\n\n
### Instructie:\n{instruction}\n\n### Input:\n{input}\n\n### Reactie:
"""


ALPACA_INTROMESSAGE_NO_INPUT = """Hieronder staat een instructie die een taak beschrijft. Schrijf een reactie die op een passende manier voldoet aan het verzoek.\n\n### Instructie:\n{instruction}\n\n### Reactie:"""


def get_args():
    """Loads command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_location",
        type=str,
        default="./data/input",
        help="Will look for all subfolders and tread them as dataset(dicts)",
    )
    parser.add_argument(
        "--db_new_location",
        type=str,
        default="./data/messages",
        help="Where to store the datasets",
    )
    return parser.parse_args()


def add_train_messages(record):
    instruction = record["instruction"]
    input_ = record["input"]
    record["messages"] = [
        {
            "content": "Je bent een behulpzame financiële assistent. help met zorg, respect en waarheid. Reageer met de grootste nuttigheid maar wel veilig. Vermijd schadelijke, onethische, bevooroordeelde of negatieve inhoud. Zorg ervoor dat antwoorden eerlijkheid en positiviteit promoten.",
            "role": "system",
        },
        {
            "content": ALPACA_INTROMESSAGE_INPUT.format(
                instruction=instruction, input=input_
            )
            if input_
            else ALPACA_INTROMESSAGE_NO_INPUT.format(instruction=instruction),
            "role": "user",
        },
        {"content": record["output"], "role": "assistant"},
    ]
    return record


def add_prediction_messages(record):
    instruction = record["instruction"]
    input_ = record["input"]
    record["messages"] = [
        {
            "content": "Je bent een behulpzame financiële assistent. help met zorg, respect en waarheid. Reageer met de grootste nuttigheid maar wel veilig. Vermijd schadelijke, onethische, bevooroordeelde of negatieve inhoud. Zorg ervoor dat antwoorden eerlijkheid en positiviteit promoten.",
            "role": "system",
        },
        {
            "content": ALPACA_INTROMESSAGE_INPUT.format(
                instruction=instruction, input=input_
            )
            if input_
            else ALPACA_INTROMESSAGE_NO_INPUT.format(instruction=instruction),
            "role": "user",
        },
    ]
    return record


def process_datasets(args):
    dataset_path = os.path.abspath(f"{args.db_location}")

    ds = load_from_disk(dataset_path)
    new = DatasetDict()

    for key in ds.keys():
        ds_w_message = ds[key].map(add_train_messages)
        new[key] = ds_w_message.shuffle(seed=42)

    save_dataset(new, args)


if __name__ == "__main__":
    process_datasets(get_args())
