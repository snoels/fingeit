""" Script to reformat the translated datasets.

"""

import argparse

from datasets import Dataset, DatasetDict, load_from_disk
from src.data_processing.utils import save_dataset



def split_translation(text):
    if "input:" in text:
        text, _, _reponse = text.partition("response:")
        text, _, _input = text.partition("input:")
        text, _, _instruction = text.partition("instruction:")
    else:
        _input = ""
        text, _, _reponse = text.partition("response:")
        text, _, _instruction = text.partition("instruction:")

    return _instruction.strip(), _input.strip(), _reponse.strip()


def update_instruction_input_response(x):
    """Update the instruction, input and output columns based on the translation column."""
    _instruction, _input, _response = split_translation(x["translation"])
    x["instruction"] = _instruction
    x["input"] = _input
    x["output"] = _response

    return x


def format_datasetdict(original: DatasetDict):
    """Update all datasets with the translation of input, output and instruction."""
    new = DatasetDict()
    for key in original.keys():
        print(f"\t> Formatting {key} dataset")
        dataset = original[key]
        reformatted_dataset = dataset.map(update_instruction_input_response).map(
            remove_columns=["translation", "prompt"]
        )
        new[key] = Dataset.from_pandas(reformatted_dataset.to_pandas())
    return new


def parse_args():
    """Loads command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_location", type=str, help="Database location")
    parser.add_argument("--db_new_location", type=str, help="New database location")
    return parser.parse_args()


def main():
    args = parse_args()
    print("> Loading dataset")
    original = load_from_disk(args.db_location)
    print("> Formatting dataset")
    new = format_datasetdict(original)
    print("> Saving dataset")
    save_dataset(new, args)


if __name__ == "__main__":
    main()
