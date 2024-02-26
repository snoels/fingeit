""" Module contains the evaluator for the Sentiment dataset.

Using the `src.evaluation.evaluator_base.BaseEvaluator` as a base class, the `SentimentEvaluator` class implements the `_evaluate` method to evaluate the Sentiment dataset.
"""

from typing import Literal

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.evaluator_base import BaseEvaluator, Evaluation, Metric


class SentimentEvaluator(BaseEvaluator):
    def __init__(self, language='NL'):
        self.language = language

    def _change_target(self, x: str) -> Literal["positief", "negatief", "neutraal", "positive", "negative", "neutral"]:
        if self.language == 'EN':
          if "positive" in x or "Positive" in x:
              return "positive"
          elif "negative" in x or "Negative" in x:
              return "negative"
          return "neutral"
        else:
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
    evaluator = SentimentEvaluator("models/FinGEITje-7B-sft")
    evaluation = evaluator.evaluate("snoels/sentiment", "test")
    print(evaluation)

    evaluator = SentimentEvaluator('EN')
    evaluation = evaluator.evaluate("snoels/sentiment", "test")
    print(evaluation)