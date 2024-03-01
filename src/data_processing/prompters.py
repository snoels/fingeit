class Prompter:
    def add_prompts(self, dataset):
        for dataset_keys in dataset.keys():
            dataset[dataset_keys] = dataset[dataset_keys].map(
                lambda x: self._add_prompt_to_row(x)
            )
        return dataset


class AlpacaPrompter(Prompter):
    TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"

    def _add_prompt_to_row(self, row):
        row["prompt"] = self.TEMPLATE.format(
            instruction=row["instruction"], input=row["input"], response=row["output"]
        )
        return row


class AlpacaEmptyInputPrompter(Prompter):
    TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

    def _add_prompt_to_row(self, row):
        row["prompt"] = self.TEMPLATE.format(
            instruction=row["instruction"], response=row["output"]
        )
        return row