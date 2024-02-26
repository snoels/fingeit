import re

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score

from src.evaluation.evaluator_base import BaseEvaluator, Evaluation, Metric


class ConvFinQaEvaluator(BaseEvaluator):
    def _cvt_text_to_pred(self, text: str):
        # cvt_text_to_pred always takes the first match - this is not always correct
        # the idea is now that the answer is always the last number in the generation

        if not text:
            return "nan"
        pred_match = re.findall(r"\d+[\.,]?\d*", text)
        if pred_match:
            pred = pred_match[-1]
        else:
            pred = "0.0"
        return pred

    def _map_output(self, feature):
        label = self._cvt_text_to_pred(feature[self.GROUND_TRUTH_COLUMN_NAME])
        pred = self._cvt_text_to_pred(feature[self.PREDICTION_COLUMN_NAME])
        return pd.Series({"label": label, "pred": pred})

    def convert_to_float(self, s):
        s = str(s)
        if len(re.findall("\.", s)) == 1 and len(re.findall(",", s)) > 1:
            return str(float(s.replace(",", "")))
        elif len(re.findall(",", s)) == 1 and len(re.findall("\.", s)) > 1:
            return str(float(s.replace(".", "")))
        elif len(re.findall("\,", s)) == 1:
            return str(float(s.replace(",", ".")))
        elif len(re.findall("\.", s)) == 1:
            return str(float(s))
        else:
            return str(float(s))

    def _evaluate(self, dataset: Dataset) -> Evaluation:
        df = dataset.to_pandas()
        eval_df = df.apply(self._map_output, axis=1)
        eval_df = eval_df[eval_df["pred"] != "nan"]

        eval_df["label"] = [self.convert_to_float(d) for d in eval_df["label"]]
        eval_df["pred"] = [self.convert_to_float(d) for d in eval_df["pred"]]

        return Evaluation(
            df=eval_df,
            metrics=[
                Metric(
                    name="Accuracy",
                    value=accuracy_score(eval_df["label"], eval_df["pred"]),
                )
            ],
        )


if __name__ == "__main__":
    evaluation = ConvFinQaEvaluator("models/FinGEITje-7B-sft").evaluate(
        "snoels/convfinqa", "test"
    )
    print(evaluation)
