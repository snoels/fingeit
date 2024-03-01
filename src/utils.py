import re

INSTRUCTION_INPUT_RESPONSE_REGEX = re.compile(
    r"(?=.*instruction:\s*)(?=.*input:\s*)(?=.*response:\s*)",
    flags=re.IGNORECASE,
)

AI_NAMES_REGEX = re.compile(
    r"|".join(
        [
            rf"als (?:een )?{name}|ben (?:een )?{name}|{name} ben"
            for name in [
                "AI-assistent",
                "AI-gebaseerde assistent",
                "virtuele assistent",
                "digitale assistent",
                "tekst-assistent",
                "AI tekstgebaseerde asssistent",
                "tekstgebaseerde asssistent",
                "assistent",
                "taalmodel",
                "AI-taalmodel",
                "AI taalmodel",
            ]
        ]
    ),
    flags=re.IGNORECASE,
)

BRANDS_REGEX = re.compile(
    r"|".join(
        [
            "ChatGPT",
            "Chat GPT",
            "GPT3",
            "GPT 3",
            "gpt-3",
            "gpt-3.5-turbo",
            "GPT4",
            "GPT 4",
            "gpt-4",
            "gpt-4-turbo",
            "OpenAI",
            "ShareGPT",
        ]
    ),
    flags=re.IGNORECASE,
)

KNOWLEDGE_REGEX = re.compile(
    r"|".join(
        [
            "kennisafsluiting in 2023",
            "kennisstop in 2023",
            "kennisafsnijdatum van 2023",
            "cutoff in 2023",
            "Tot mijn kennis die bijgewerkt is tot begin 2023",
            "Voor zover mijn kennis reikt tot 2023",
            "Vanaf mijn kennis tot begin 2023",
            "As of my last update in 2023",
        ]
    ),
    flags=re.IGNORECASE,
)

INCORRECT_LANGUAGE_REGEX = re.compile(
    r"|".join(["It seems like there was a typo", "assistant"]), flags=re.IGNORECASE
)
APOLOGIES_REGEX = re.compile(
    r"|".join(["spijt me", "spijt mij", "sorry", "mijn excuses"]), flags=re.IGNORECASE
)


def filter_dataset(text: str) -> bool:
    if text is None:
        return False

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    # check if any of 'ai_names' exist in `text`
    if AI_NAMES_REGEX.search(text):
        return False
    # check if any of 'brands' regex matches exist in `text`
    if BRANDS_REGEX.search(text):
        return False

    # check if any 'knowledge_cut_offs' regex matches exist in `text`
    if KNOWLEDGE_REGEX.search(text):
        return False

    # check if any 'incorrect_language' regex matches exist in `text`
    if INCORRECT_LANGUAGE_REGEX.search(text):
        return False

    # check if any 'apologies' regex matches exist in `text`
    if APOLOGIES_REGEX.search(text):
        return False

    if len(text.split()) <= 3:
        return False

    if not INSTRUCTION_INPUT_RESPONSE_REGEX.search(text):
        return False

    return True


def is_language(text, model, target_language="nld"):
    if text is None:
        return False

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    predictions = model.predict(text, k=1)
    return (
        predictions[0][0].replace("__label__", "").replace("_Latn", "")
        == target_language
    )
