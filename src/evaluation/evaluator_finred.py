""" Module contains the evaluator for the Fin Red dataset.

Using the `src.evaluation.evaluator_base.BaseEvaluator` as a base class, the `FinRedEvaluator` class implements the `_evaluate` method to evaluate the Fin Red dataset.

"""

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.evaluator_base import BaseEvaluator, Evaluation, Metric

class FinRedEvaluator(BaseEvaluator):
    relations = {
        'EN': [
        "product_or_material_produced",
        "manufacturer",
        "distributed_by",
        "industry",
        "position_held",
        "original_broadcaster",
        "owned_by",
        "founded_by",
        "distribution_format",
        "headquarters_location",
        "stock_exchange",
        "currency",
        "parent_organization",
        "chief_executive_officer",
        "director_/_manager",
        "owner_of",
        "operator",
        "member_of",
        "employer",
        "chairperson",
        "platform",
        "subsidiary",
        "legal_form",
        "publisher",
        "developer",
        "brand",
        "business_division",
        "location_of_formation",
        "creator",
        ],
        'NL': [
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
    evaluation = FinRedEvaluator('NL').evaluate(
        "snoels/finred", "test"
    )
    print(evaluation)