# 🐐 FinGEITje 7B: een groot open Financieel Nederlands taalmodel

[📄 README](./README.md)

GEITje is een financieel Nederlandstalig groot open taalmodel met 7 miljard parameters, gebaseerd op Mistral 7B. Het is (verder) getraind op 10 miljard tokens aan Nederlandstalige financiële teksten. Daardoor heeft het beter Nederlands geleerd, en meer kennis over Nederlandse financiële onderwerpen.

![DALL·E 3: "Create a logo for a Dutch large language model's Github readme. Incorporate a hyper realistic cute baby goat painting on a Dutch landscape with a few finance skyscrapers. The cute baby goat wears a business suit and has a financial background."](./resources/fingeitje-logo.jpeg)

## Getting started
1. RUN `poetry install`. Fingeitje uses [poetry](https://python-poetry.org/) as dependency manager. By running the command a venv will be created and the necessary python packages will be installed
2. Run [data_downloader](./src/data_processing/data_downloader.py) to download the original dataset
3. Run [translation service](./src/data_processing/translation_service.py) to translate the original dataset
4. Run [post processing service](./src/data_processing/post_process.py) to post process the translated dataset
5. Run [translation formatter](./src/data_processing/translation_formatter.py) to format the translation into the original dataset format

TODO @snoels can you add trainging and evaluating or delete the above? should we mention the alignment handbook and geitje 7b? also proabably your paper and results? link the hugginface dataset, leaderbord etc.


