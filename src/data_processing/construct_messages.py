import argparse
import os
from datasets import load_from_disk


ALPACA_INTROMESSAGE_INPUT ="""Hieronder staat een instructie die een taak beschrijft, samen met een input die context voorziet
Schrijf een reactie die op een passende manier voldoet aan de vraag.\n\n
### instructie:\n{instruction}\n\n### Input:\n{input}\n\n### Reactie:
"""


ALPACA_INTROMESSAGE_NO_INPUT = """Hieronder staat een instructie die een taak beschrijft. Schrijf een reactie die op een passende manier voldoet aan het verzoek.\n\n### Instructie:\n{instruction}\n\n### Reactie:"""

def get_args():
    """Loads command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./data/input", help="Will look for all subfolders and tread them as dataset(dicts)")
    parser.add_argument("--output_folder", type=str, default="./data/messages", help="Where to store the datasets")
    return parser.parse_args()

def add_messages(record):
    instruction = record['instruction']
    input_ = record['input']
    record['messages'] = [
        {
            'content': "Je bent een behulpzame financiële assistent. help met zorg, respect en waarheid. Reageer met de grootste nuttigheid maar wel veilig. Vermijd schadelijke, onethische, bevooroordeelde of negatieve inhoud. Zorg ervoor dat antwoorden eerlijkheid en positiviteit promoten.",
            'role': 'system'
        },
        {
            'content': ALPACA_INTROMESSAGE_INPUT.format(instruction =instruction, input=input_) if input_ else ALPACA_INTROMESSAGE_NO_INPUT.format(instruction=instruction),
            'role': "user"
        },
        {
            'content': record['output'],
            'role': "assistant"
        }
    ]
    return record


def process_datasets(args):
    for dir_ in os.listdir(args.input_folder):
        dataset_path = os.path.abspath(f"{args.input_folder}/{dir_}")
        output_path = os.path.abspath(f"{args.output_folder}/{dir_}")
        load_from_disk(dataset_path).map(add_messages).save_to_disk(output_path)


if __name__ == "__main__":
    process_datasets(get_args())



