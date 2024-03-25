from datasets import Dataset
from scripts.evaluation.evaluator_base import BaseEvaluator, Evaluation, Metric
from sklearn.metrics import accuracy_score
import pandas as pd
import re

class ConvFinQaEvaluator(BaseEvaluator):

    def _cvt_text_to_pred(self, text:str):
        # cvt_text_to_pred always takes the first match - this is not always correct
        # the idea is now that the answer is always the last number in the generation

        if not text:
            return 'nan'
        pred_match = re.findall(r'\d+[\.,]?\d*', text) # new regex added that allows , and . 
        if pred_match:
            pred = pred_match[-1]
        else:
            pred = '0.0'
        return pred


    def _map_output(self, feature):
        label = self._cvt_text_to_pred(feature[self.GROUND_TRUTH_COLUMN_NAME])
        pred = self._cvt_text_to_pred(feature[self.PREDICTION_COLUMN_NAME])
        return pd.Series({'label': label, 'pred': pred})

    def convert_to_float(s):
        s = str(s)
        if len(re.findall("\.", s)) == 1 and len(re.findall(",", s)) > 1:
            # Replace commas and convert to float
            return str(s.replace(',', ''))
        elif len(re.findall(",", s)) == 1 and len(re.findall("\.", s)) > 1:
            # Replace dots and convert to float
            return str(s.replace('.', ''))
        elif len(re.findall("\,", s)) == 1:
            # Converts European decimal format (e.g., "1234,56" to 1234.56)
            return str(s.replace(',', '.'))
        elif len(re.findall("\.", s)) == 1:
            # Converts US decimal format (e.g., "1234.56" to 1234.56)
            return str(s)
        else:
            # Default: attempts to convert to a float directly
            return str(s)


    def _evaluate(self, dataset: Dataset) -> Evaluation:
        df = dataset.to_pandas()
        eval_df = df.apply(self._map_output, axis=1)
        eval_df = eval_df[eval_df['pred'] != 'nan']

        label = [self.convert_to_float(d) for d in eval_df['label']]
        pred = [self.convert_to_float(d) for d in eval_df['pred']]

        return Evaluation(
            df=eval_df,
            metrics=[
                Metric(
                    name="Accuracy",
                    value = accuracy_score(label, pred)
                )
            ]
        )


if __name__ == "__main__":
    evaluation = FinQaEvaluator('models/fingeitje').evaluate('ice-hands/finred-messages', 'test')
    print(evaluation)