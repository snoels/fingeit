""" Module contains the evaluator for the Headline dataset.

Using the `src.evaluation.evaluator_base.BaseEvaluator` as a base class, the `HeadlineEvaluator` class implements the `_evaluate` method to evaluate the Headline dataset.

"""


import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.evaluator_base import BaseEvaluator, Evaluation, Metric

class HeadlineEvaluator(BaseEvaluator):
    
    def __init__(self, language = 'NL'):
        self.language = language

    def _map_output(self, row):
        label = pred = None

        if self.language == "EN":
            true_val = "yes"
            false_val = "no"
        else:
            true_val = "ja"
            false_val = "nee"

        if true_val in row[self.GROUND_TRUTH_COLUMN_NAME].lower():
            label = 1
        elif false_val in row[self.GROUND_TRUTH_COLUMN_NAME].lower():
            label = 0

        if true_val in row[self.PREDICTION_COLUMN_NAME].lower():
            pred = 1
        elif false_val in row[self.PREDICTION_COLUMN_NAME].lower():
            pred = 0

        if "unknown" in row[self.PREDICTION_COLUMN_NAME].lower():
            pred = 1 - label

        if pred is None:
            pred = 1 - label

        return pd.Series({"label": label, "pred": pred})

    def _evaluate(self, dataset: Dataset) -> Evaluation:
        df = dataset.to_pandas()
        eval_df = df.apply(self._map_output, axis=1)

        return Evaluation(
            df=eval_df,
            metrics=[
                Metric(
                    name="Acc", value=accuracy_score(eval_df["label"], eval_df["pred"])
                ),
                Metric(
                    name="F1 binary",
                    value=f1_score(eval_df["label"], eval_df["pred"], average="binary"),
                ),
            ],
        )

if __name__ == "__main__":
    evaluation = HeadlineEvaluator("models/fingeitje", "EN").evaluate(
        "ice-hands/finred-messages", "test"
    )
    print(evaluation)