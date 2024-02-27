""" Script to reformat the translated datasets

"""

import argparse

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from utils import word_indexes


def split_into_parts(text: str) -> tuple[str, str, str]:
    """Extracts the instruction, input and output from a Lama prompt.
    Params:
        text: text as formatted for a Lama prompt
    Returns:
        In order: instruction, input, output
    Raises:
        AttributeError when the word is not found
    """
    _, instruction_end = word_indexes(text, "instruction")
    input_start, input_end = word_indexes(text, "input")
    response_start, response_end = word_indexes(text, "response")

    instruction = text[instruction_end + 1 : input_start].strip()
    input_ = text[input_end + 1 : response_start].strip()
    response = text[response_end + 1 :].strip()

    return instruction, input_, response


def update_instruction_input_response(x: pd.Series) -> pd.Series:
    """Update the instruction, input and output collumns based on the translation collumn."""
    try:
        instruction, _input, response = split_into_parts(x["translation"])
        x["instruction"] = instruction
        x["input"] = _input
        x["output"] = response
        x["success"] = True
    except Exception as e:
        print("Could not split in parts")
        x["success"] = False
    return x


def parse_args():
    """Loads command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_location", type=str, help="Database location")
    parser.add_argument("--db_new_location", type=str, help="New database location")
    return parser.parse_args()


def format_datasetdict(original: DatasetDict) -> DatasetDict:
    """Update all datasets with the translation of input, output and instruction.

    Given that the translation is property in the dataset formatted as a Lama prompt,
    it will extract the translated input, output and instruction from the lama prompt translation and
    update it in the datasets

    Params:
        original: datasetdict containing datasets that need to be updated
    returns:
        the updated datsetdict
    """
    new = DatasetDict()
    for key in original.keys():
        print(f"\t> Formatting {key} dataset")
        df = original[key].to_pandas()
        df = df.apply(update_instruction_input_response, axis=1)
        df = df[~df["success"]]
        # if df.shape[0] > 0:
        #     raise Exception("Dataset not fully correct translated")
        new[key] = Dataset.from_pandas(
            df[["input", "output", "instruction"]], preserve_index=False
        )
    return new


def format_datasetdict2(original: DatasetDict) -> DatasetDict:
    """Update all datasets with the translation of input, output and instruction.

    Given that the translation is property in the dataset formatted as a Lama prompt,
    it will extract the translated input, output and instruction from the lama prompt translation and
    update it in the datasets

    Params:
        original: datasetdict containing datasets that need to be updated
    returns:
        the updated datsetdict
    """
    new = DatasetDict()
    for key in original.keys():
        print(f"\t> Formatting {key} dataset")
        df = original[key].to_pandas()
        df = df.apply(update_instruction_input_response, axis=1)
        df = df[~df["success"]]
        # if df.shape[0] > 0:
        #     raise Exception("Dataset not fully correct translated")
        new[key] = Dataset.from_pandas(
            df[["input", "output", "instruction"]], preserve_index=False
        )
    return new


def main():
    args = parse_args()
    print("> Loading dataset")
    original = load_from_disk(args.db_location)
    print("> Formatting dataset")
    new = format_datasetdict(original)
    print("> Saving dataset")
    new.save_to_disk(args.db_new_location)


if __name__ == "__main__":
    main()
