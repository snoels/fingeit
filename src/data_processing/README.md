# Data Processing

This folder contains scripts to translate the financial data as described in [FinGPT]() from English to dutch. The approach is crafted so any other language to translate to should be a trivial job.

## Data downloading
Starting point of the data processing and can be found in [data_downloader.py](./data_downloader.py). This will download all datasets as described in the [config](./config.ini) file from [huggingface](https://huggingface.co/)

Run the script: \
`python -m src.data_processing.data_downloade --db_location <path to store the data>`