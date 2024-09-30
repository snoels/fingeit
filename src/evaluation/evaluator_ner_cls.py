""" Module contains the evaluator for the Ner cls dataset.

Using the `src.evaluation.evaluator_base.BaseEvaluator` as a base class, the `NERCLSEvaluator` class implements the `_evaluate` method to evaluate the Ner cls dataset.
"""

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.evaluator_base import BaseEvaluator, Evaluation, Metric

class NERCLSEvaluator(BaseEvaluator):
    relations = {
        'EN': [
        "person",
        "organisatie",
        "organisatie",
        ],
        'NL': [
        "persoon",
        "organization",
        "location",
        ],
    }

    def __init__(self, language='NL'):
        self.language = language

    def _change_target(self, x: str):
        if x in self.relations[self.language]:
            return x
        else:
            return "unknown"

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
    evaluation = NERCLSEvaluator('NL').evaluate(
        "ice-hands/finred-messages", "test"
    )
    print(evaluation)