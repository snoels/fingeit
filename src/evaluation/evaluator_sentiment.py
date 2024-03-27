from typing import Literal

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.evaluator_base import BaseEvaluator, Evaluation, Metric


class SentimentEvaluator(BaseEvaluator):
    def _change_target(self, x: str) -> Literal["positief", "negatief", "neutraal"]:
        if "positief" in x or "Positief" in x:
            return "positief"
        elif "negatief" in x or "Negatief" in x:
            return "negatief"
        return "neutraal"

    def _evaluate(self, dataset: Dataset) -> Evaluation:
        df = dataset.to_pandas()
        df["new_target"] = df[self.GROUND_TRUTH_COLUMN_NAME].apply(self._change_target)
        df["new_out"] = df[self.PREDICTION_COLUMN_NAME].apply(self._change_target)

        return Evaluation(
            df=df,
            metrics=[
                Metric(
                    name="acc", value=accuracy_score(df["new_target"], df["new_out"])
                ),
                Metric(
                    name="f1_macro",
                    value=f1_score(df["new_target"], df["new_out"], average="macro"),
                ),
                Metric(
                    name="f1_micro",
                    value=f1_score(df["new_target"], df["new_out"], average="micro"),
                ),
                Metric(
                    name="f1_weighted",
                    value=f1_score(df["new_target"], df["new_out"], average="weighted"),
                ),
            ],
        )


if __name__ == "__main__":
    evaluation = SentimentEvaluator("models/fingeitje").evaluate(
        "ice-hands/finred-messages", "test"
    )
    print(evaluation)
