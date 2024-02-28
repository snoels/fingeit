import argparse
import copy
import os
import shutil
from functools import partial
from typing import Any, Dict
from uuid import uuid4

from datasets import load_from_disk
from translation_service import call_chatgpt_sync, load_config
from utils import filter_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_location", type=str, help="Database location")
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


def main(args: argparse.Namespace):
    dataset = load_from_disk(args.db_location)
    config = load_config(args)

    temporary_location = os.path.join(os.path.dirname(args.db_location), str(uuid4()))
    os.makedirs(temporary_location, exist_ok=True)

    for dataset_keys, dataset_values in dataset.items():
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

    # save the modified dataset to a temporary location
    dataset.save_to_disk(temporary_location)

    # replace the original dataset with the updated one from temporary location
    shutil.rmtree(args.db_location)
    shutil.move(temporary_location, args.db_location)


if __name__ == "__main__":
    main(parse_args())
