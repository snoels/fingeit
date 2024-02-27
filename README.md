# 🐐 FinGEITje 7B: een groot open Financieel Nederlands taalmodel

[📄 README](./README.md)

GEITje is een financieel Nederlandstalig groot open taalmodel met 7 miljard parameters, gebaseerd op Mistral 7B. Het is (verder) getraind op 10 miljard tokens aan Nederlandstalige financiële teksten. Daardoor heeft het beter Nederlands geleerd, en meer kennis over Nederlandse financiële onderwerpen.

![DALL·E 3: "Create a logo for a Dutch large language model's Github readme. Incorporate a hyper realistic cute baby goat painting on a Dutch landscape with a few finance skyscrapers. The cute baby goat wears a business suit and has a financial background."](./resources/fingeitje-logo.jpeg)

## Getting started
1. Run [data_downloader](./src/data_downloader.py) to download the original dataset
2. Run [translation service](./src/translation_service.py) to translate the original dataset
3. Run [translation formatter](./src/translation_formatter.py) to format the translation into the original dataset format
