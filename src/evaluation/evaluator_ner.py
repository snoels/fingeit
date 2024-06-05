import re

import pandas as pd
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score

from src.evaluation.evaluator_base import BaseEvaluator, Evaluation, Metric


class NEREvaluator(BaseEvaluator):
    def __init__(self, language='NL'):
        if language == 'EN':
            self.ent_dict = {
                "PER": "person",
                "ORG": "organisation",
                "LOC": "location",
            }
            self.lang_match_re = r"^(.*) is a? (.*)$"
        else:
            self.ent_dict = {
                "PER": "persoon",
                "ORG": "organisatie",
                "LOC": "locatie",
            }
            self.lang_match_re = r"^(.*) is een? (.*)$"
        
        self.ent_dict_rev = {v: k for k, v in self.ent_dict.items()}

    def _cvt_text_to_pred(self, tokens, text):
        preds = ["O" for _ in range(len(tokens))]
        for pred_txt in text.lower().strip(".").split(","):
            pred_match = re.match(self.lang_match_re, pred_txt)
            if pred_match is not None:
                entity, entity_type = (
                    pred_match.group(1).strip(),
                    pred_match.group(2).strip(),
                )
                entity_pred = self.ent_dict_rev.get(entity_type, "O")
                entity_tokens = entity.split()

                n = len(entity_tokens)
                for i in range(len(tokens) - n + 1):
                    if (
                        tokens[i : i + n] == entity_tokens
                        and preds[i : i + n] == ["O"] * n and entity_pred in self.ent_dict.keys()
                    ):
                        preds[i : i + n] = ["B-" + entity_pred] + [
                            "I-" + entity_pred
                        ] * (n - 1)
                        break
            else:
                continue

        return preds

    def _map_output(self, row):
        tokens = row["input"].lower().split()
        label = self._cvt_text_to_pred(tokens, row[self.GROUND_TRUTH_COLUMN_NAME])
        pred = self._cvt_text_to_pred(tokens, row[self.PREDICTION_COLUMN_NAME])
        return pd.Series({"label": label, "pred": pred})

    def _evaluate(self, dataset: Dataset) -> Evaluation:
        df = dataset.to_pandas()
        eval_df = df.apply(self._map_output, axis=1)

        return Evaluation(
            df=eval_df,
            metrics=[
                Metric(
                    name="F1",
                    value=f1_score(eval_df["label"], eval_df["pred"]),
                ),
                Metric(
                    name="Classification Report",
                    value=classification_report(eval_df["label"], eval_df["pred"]),
                ),
            ],
        )
        
if __name__ == "__main__":
    evaluation = NEREvaluator(language="EN").evaluate(
        "ice-hands/finred-messages", "test"
    )
    print(evaluation)

    evaluation = NEREvaluator(language="NON-EN").evaluate(
        "ice-hands/finred-messages", "test"
    )
    print(evaluation)