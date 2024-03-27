from dataclasses import dataclass

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.file_access import FileAccess


@dataclass
class Metric:
    name: str
    value: float


@dataclass
class Evaluation:
    df: pd.DataFrame
    metrics: list[Metric]

    def __str__(self) -> str:
        metrics_str = "\n".join([f"{x.name}: {x.value}" for x in self.metrics])
        return f"Scoring metrics: \n {metrics_str}"


class BaseEvaluator:
    PREDICTION_COLUMN_NAME = "prediction"
    GROUND_TRUTH_COLUMN_NAME = "output"

    def __init__(
        self,
        model_path: str,
        remove_temp_file: bool | None = None,
        temp_storage_file: str | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model_path = model_path
        self._log_interval = 20
        self.file_access = FileAccess(temp_storage_file)
        self.remove_temp_file = remove_temp_file if remove_temp_file else False

    def _load_dataset(self, dataset_name: str, split: str) -> Dataset:
        return load_dataset(dataset_name, revision="main")[split]

    def _maybe_insert_system_message(self, messages):
        if messages[0]["role"] == "system":
            return

        # chat template can be one of two attributes, we check in orders
        chat_template = self.tokenizer.chat_template
        if chat_template is None:
            chat_template = self.tokenizer.default_chat_template

        # confirm the jinja template refers to a system message before inserting
        if "system" in chat_template or "<|im_start|>" in chat_template:
            messages.i

    def _apply_chat_template(
        self,
        example,
    ):
        messages = example["messages"][:-1]  # Remove the GT answer
        self._maybe_insert_system_message(messages)
        example["text"] = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return example

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self._apply_chat_template, num_proc=1, desc="Applying chat template"
        )

    def _get_response(self, response) -> str:
        lines = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for g in response
        ]
        assistant = "".join(lines).split("<|assistant|>")[-1]
        return assistant.replace("\n", "")

    def _get_start_index(self):
        return len(self.file_access.read())

    def _to_dataloader(self, dataset: Dataset) -> DataLoader:
        def collate_fn(batch):
            inputs = self.tokenizer(
                [f["text"] for f in batch],
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            )
            return inputs

        if (start_index := self._get_start_index()) != 0:
            self.file_access.mark_recovery()
            dataset = dataset.select(range(start_index, dataset.num_rows))

        return DataLoader(dataset, batch_size=10, collate_fn=collate_fn, shuffle=False)

    def _get_predictions(self, dataset: Dataset) -> Dataset:
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map="cuda"
        )  # do not move this to the constructor, out of memory
        for idx, inputs in enumerate(tqdm(self._to_dataloader(dataset))):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            res = model.generate(
                **inputs, max_length=2048, eos_token_id=self.tokenizer.eos_token_id
            )
            res_sentences = [self._get_response(i) for i in res]
            if (idx + 1) % self._log_interval == 0:
                tqdm.write(f"{idx}: {res_sentences[0]}")
            self.file_access.store(res_sentences)

            del res
            del inputs
            torch.cuda.empty_cache()

        return self.file_access.read()

    def _add_predictions(self, dataset: Dataset) -> Dataset:
        try:
            predictions = self._get_predictions(dataset)
            dataset = dataset.add_column(self.PREDICTION_COLUMN_NAME, predictions)
            if self.remove_temp_file:
                self.file_access.remove()
            return dataset
        except Exception as e:
            print(
                f"woepsie, somehting happend during prediction time. You can view already processed data in the temp file: {self.file_access.file}",
                e,
            )

    def _evaluate(self, dataset: Dataset) -> Evaluation:
        ...

    def evaluate(self, dataset_name: str, split: str) -> Evaluation:
        """
        Params:
            dataset_name: name of the dataset, can be a huggingface repo or folder
        """
        dataset = self._load_dataset(dataset_name, split)
        dataset = self._preprocess_dataset(dataset)
        dataset = self._add_predictions(dataset)
        return self._evaluate(dataset)
