# 🐐 FinGEITje 7B: a large open Dutch Financial language model

[📄 README](./README.md)

GEITje is a large open Dutch financial language model with 7 billion parameters, based on Mistral 7B. It has been (further) trained on Dutch financial texts. As a result, it has learned better Dutch and has more knowledge about Dutch financial topics.

![DALL·E 3: "Create a logo for a Dutch large language model's Github readme. Incorporate a hyper realistic cute baby goat painting on a Dutch landscape with a few finance skyscrapers. The cute baby goat wears a business suit and has a financial background."](./resources/fingeitje-banner.png)

## Getting Started

1. **Install Dependencies**: Run `poetry install`. FinGEITje uses [Poetry](https://python-poetry.org/) as a dependency manager. Running this command will create a virtual environment and install the necessary Python packages.

2. **Download the Original Dataset**: Run [data_downloader](./src/data_processing/data_downloader.py) to download the original dataset.

3. **Translate the Dataset**: Run [translation_service](./src/data_processing/translation_service.py) to translate the original dataset.

4. **Post-Process the Translated Dataset**: Run [post_process](./src/data_processing/post_process.py) to post-process the translated dataset.

5. **Format the Translation**: Run [translation_formatter](./src/data_processing/translation_formatter.py) to format the translation into the original dataset format.

6. **Training Configuration**: The training configuration can be found in [config_qlora](./src/training/sft/config_qlora.yaml). This is a recipe as described in the [Alignment Handbook](https://github.com/huggingface/alignment-handbook), and we used the Alignment Handbook to spawn the whole training pipeline.

7. **Evaluation Package**: The evaluation package can be found in [evaluation](./src/evaluation/). To evaluate the model, a set of metrics are defined per task. The tasks can be grouped per dataset so every dataset is evaluated separately.

## Notebooks

- **[data_processing.ipynb](./notebooks/data_processing.ipynb)**: Contains the code that shows the [exact translations](./src/data_processing/translations_fingpt.py) that can be done before passing them to our [translation service](./src/data_processing/translation_service.py).

- **[combine_datasets.ipynb](./notebooks/combine_datasets.ipynb)**: Contains the code that shows how the translated instruction tuning data is filtered by a Dutch language check and some predefined checks, and combined into one instruction tuning dataset.

- **[evaluation_nl.ipynb](./notebooks/evaluation_nl.ipynb)**: Contains the evaluation of [snoels/FinGEITje-7B-sft](https://huggingface.co/datasets/snoels/FinGEITje-7B-sft) on the Dutch financial benchmark [snoels/FinDutchBench](https://huggingface.co/datasets/snoels/FinDutchBench) with and without automated answer extraction.

- **[evaluation_en.ipynb](./notebooks/evaluation_en.ipynb)**: Contains the evaluation of [snoels/FinGEITje-7B-sft](https://huggingface.co/datasets/snoels/FinGEITje-7B-sft) on the English financial benchmark with and without automated answer extraction.

## Paper

This repository is based on the following paper:

**A Dutch Financial Large Language Model**  
[Link to the paper](https://arxiv.org/abs/2410.12835) <!-- Replace with actual link when available -->

### Citation

If you use FinGEITje in your work, please cite:

```bibtex
@article{FinGEITje2024,
  title={A Dutch Financial Large Language Model},
  author={Noels, Sander and De Blaere, Jorne and De Bie, Tijl},
  journal={arXiv preprint arXiv:2410.12835},
  year={2024},
  url={https://arxiv.org/abs/2410.12835}
}
```

## Contributing

Contributions are welcome! If you have ideas, suggestions, or find issues, please open an issue or submit a pull request. We appreciate your help in improving FinGEITje.

## Acknowledgements

We would like to thank:

- **Rijgersberg** ([GitHub](https://github.com/Rijgersberg)) for creating one of the first Dutch foundation models called [GEITje](https://github.com/Rijgersberg/GEITje): a Dutch large open language model with 7 billion parameters, based on Mistral 7B. It has been further trained on 10 billion tokens of Dutch text. The model can be found here: [Rijgersberg/GEITje-7B](https://huggingface.co/Rijgersberg/GEITje-7B).

- **Bram Vanroy** ([GitHub](https://github.com/BramVanroy)) for creating one of the first Dutch open source chat models, [GEITje-7B-ultra](https://huggingface.co/BramVanroy/GEITje-7B-ultra), and for open-sourcing its training, translation ([dutch-instruction-datasets](https://github.com/BramVanroy/dutch-instruction-datasets)), and evaluation details.

We also extend our gratitude to the contributors of the [Alignment Handbook](https://github.com/huggingface/alignment-handbook) for providing valuable resources that aided in the development of FinGEITje.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Contact

For any inquiries or questions, please contact [Sander Noels](mailto:sander.noels@ugent.be).
