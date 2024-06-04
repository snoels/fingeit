from typing import Literal

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.evaluator_base import BaseEvaluator, Evaluation, Metric

relations = [
    "product/materiaal geproduceerd",
    "fabrikant",
    "verdeeld door",
    "industrie",
    "positie bekleed",
    "originele omroep",
    "bezeten door",
    "opgericht door",
    "distributieformaat",
    "hoofdkantoorlocatie",
    "effectenbeurs",
    "valuta",
    "moederorganisatie",
    "chief executive officer",
    "directeur/manager",
    "eigenaar van",
    "operator",
    "lid van",
    "werkgever",
    "voorzitter",
    "platform",
    "dochteronderneming",
    "rechtsvorm",
    "uitgever",
    "ontwikkelaar",
    "merk",
    "bedrijfsdivisie",
    "locatie van ontstaan",
    "maker",
]


class FinRedEvaluator(BaseEvaluator):
    def _change_target(self, x: str):
        if x in relations:
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
    evaluation = FinRedEvaluator("models/fingeitje").evaluate(
        "ice-hands/finred-messages", "test"
    )
    print(evaluation)
