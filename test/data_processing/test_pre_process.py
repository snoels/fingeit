from data_processing.translation.pre_process import (
    AlpacaEmptyInputPromptCreator,
    AlpacaPromptCreator,
)


def test_alpaca_prompt_creator():
    alpaca_creator = AlpacaPromptCreator()
    row = {
        "instruction": "Do something",
        "input": "Input text",
        "output": "Output text",
    }
    new_row = alpaca_creator._insert_prompt_to_row(row)

    assert (
        new_row["prompt"]
        == "### Instruction:\nDo something\n\n### Input:\nInput text\n\n### Response:\nOutput text"
    )


def test_alpaca_empty_input_prompt_creator():
    alpaca_empty_creator = AlpacaEmptyInputPromptCreator()
    row = {"instruction": "Do something", "output": "Output text"}
    new_row = alpaca_empty_creator._insert_prompt_to_row(row)

    assert (
        new_row["prompt"]
        == "### Instruction:\nDo something\n\n### Response:\nOutput text"
    )
