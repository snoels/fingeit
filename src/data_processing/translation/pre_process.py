from datasets import DatasetDict


class PromptCreator:
    def add_prompts(self, dataset: DatasetDict) -> DatasetDict:
        """Adds prompts to each item in the dataset."""

        for split in dataset.keys():
            dataset[split] = dataset[split].map(self._insert_prompt_to_row)
        return dataset


class AlpacaPromptCreator(PromptCreator):
    PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"

    def _insert_prompt_to_row(self, row):
        row["prompt"] = self.PROMPT_TEMPLATE.format(
            instruction=row["instruction"], input=row["input"], response=row["output"]
        )
        return row


class AlpacaEmptyInputPromptCreator(PromptCreator):
    PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

    def _insert_prompt_to_row(self, row):
        row["prompt"] = self.PROMPT_TEMPLATE.format(
            instruction=row["instruction"], response=row["output"]
        )
        return row
