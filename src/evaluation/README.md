# Evaluation package

To evaluate the model a set of metrics are defined per task.
the tasks can be grouped per dataset so every dataset is evaluated separatly.

## How to use the metrics
To use the metrics instantiate the class and call the evaluate method. 
eg.

```python
evaluator = HeadlineEvaluator(model_path='models/FinGEITje-7B-sft')
evaluation = evaluator.evaluate(dataset_name='snoels/finred', split='test')
print(evaluation)
```

To gain more info about the evaluation you can extract the processed pandas dataframe from the evaluation class

```python
df_to_investigate = evaluation.df
```

### Crashes
If for some reason the process fails not everything is lost. To mitigate system failures all predictions are stored in a text file. The path will be printed whenever the process has crashed. Recovery method:


``` python
evaluator = HeadlineEvaluator(model_path='models/FinGEITje-7B-sft', temp_storage_file=<PATH_TO_THE_EXISTING_FILE>)
evaluation = evaluator.evaluate(dataset_name='snoels/finred', split='test')
print(evaluation)
```

The base class will determine what the start index will be and continue from there on. It will also leave a mark where it is continued for investigation purpose later on.
Notice that the marks will be deleted before adding to the set


## How to create your own metrics
To create your own metrics you can relie on the [evaluator_base](./evaluator_base.py) module.
The module holds a baseclass called `BaseEvaluator`. The base class relies on the template method design pattern that provides a skelleton to create predictions based on a provided dataset and lets the superclass overwrite the `_evaluate` function for specific dataset evaluations.

What you need to do:
1. Create a new class
2. Extend the BaseEvaluator class
3. Add a function called `_evaluate` with defined as: `def _evaluate(self, dataset:Dataset) -> Evaluation:`
4. Implement the _evaluate function with your custom metric.