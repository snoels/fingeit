# Data Processing

This folder contains scripts to translate the financial data as described in [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) from English to Dutch. The approach is crafted so any other language to translate to should be a trivial job.

## Data downloading

Starting point of the data processing and can be found in [data_downloader.py](./data_downloader.py). This will download all datasets as described in the [config](./config.ini) file from [huggingface](https://huggingface.co/)

Run the script: \
`python -m src.data_processing.data_downloader --db_location <path to store the data>`

## Data translation

When the data is downloaded it needs to be translated. To do so:
Firstly the instruction, input and output are set in Alpaca prompt as follows

``` markdown
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

After that the prompts are translated using OpenAI GPT in an async fashion. Once translated the dataset is saved.

Note:
When we tried out of the box translations, it became clear that not everything was propperly translated. To mitigate the problem we did some preprocessing on the downloaded datasets. The preprocessing can be found in the [preprocessing notebook](../../notebooks/preprocess.ipynb)

Run the script: \
`python -m src.data_processing.translation_service --db_location <path to get the data> --db_new_location <path to store the data> --prompt_type <prompt type> --target_language <target language>`

## Data post processing

The [post_process.py](./post_process.py) script performs post-processing on the dataset. This involves language checks and filtering based on regex. It starts by checking the language of each prompt using the FastText language identification method. If the language doesn't match the target language, the prompt is retranslated (maximum retries can be specified). After the language check, it filters out incorrect translations based on regex checks. The script finally saves the new dataset to the specified location.

To run the script from your terminal:

`python -m src.data_processing.post_process --db_location <path to data> --db_new_location <path to new data> --max_retries <number of attempts> --target_language <language>`

## Data Reformatting

The purpose of the [translation_formatter.py](./translation_formatter.py) script is to reformat the translated datasets. This script ensures that the translated content is correctly distributed across the instruction, input, and output columns.The completed script can be found [here](./translation_formatter.py). 

Run the script: \
`python -m src.data_processing.translation_formatter --db_location <path to data> --db_new_location <path to new data>`

## Final step

The translated data is now ready for further data processing and modeling tasks in downstream analysis and model training.