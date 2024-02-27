import re
import unicodedata as ud

latin_letters = {}


def word_indexes(text: str, lookup: str) -> tuple[int, int]:
    """Will return begin and index of a word to lookup.

    Params:
        text: text to look into
        lookup: word to look for
    returns:
        Start and end index of the word to look for
    Raises:
        AttributeError when the word is not found
    """
    lookupstr = r"\b(" + lookup + r")\b"
    result = re.search(lookupstr, text)
    begin = result.start()
    end = result.end()
    return begin, end


def is_translation_valid(text: str) -> bool:
    if not text:
        return False
    try:
        word_indexes(text, "instruction: ")
        word_indexes(text, "input: ")
        word_indexes(text, "response: ")
        return True
    except Exception:
        return False


def is_latin(unicode_chr: str):
    try:
        return latin_letters[unicode_chr]
    except KeyError:
        return latin_letters.setdefault(unicode_chr, "LATIN" in ud.name(unicode_chr))


def only_roman_chars(unicode_text: str):
    return all(is_latin(uchr) for uchr in unicode_text if uchr.isalpha())


def filter_dataset(row, column_name: str):
    ai_names = [
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
    regex_names = re.compile(
        r"|".join(
            [
                rf"als (?:een )?{name}|ben (?:een )?{name}|{name} ben"
                for name in ai_names
            ]
        ),
        flags=re.IGNORECASE,
    )

    brands = [
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
    regex_brands = re.compile(r"|".join(brands), flags=re.IGNORECASE)

    knowledge_cut_offs = [
        "kennisafsluiting in 2023",
        "kennisstop in 2023",
        "kennisafsnijdatum van 2023",
        "cutoff in 2023",
        "Tot mijn kennis die bijgewerkt is tot begin 2023",
        "Voor zover mijn kennis reikt tot 2023",
        "Vanaf mijn kennis tot begin 2023",
        "As of my last update in 2023",
    ]
    regex_knowledge = re.compile(r"|".join(knowledge_cut_offs), flags=re.IGNORECASE)

    incorrect_language = ["It seems like there was a typo", "assistant"]
    regex_language = re.compile(r"|".join(incorrect_language), flags=re.IGNORECASE)

    apologies = ["spijt me", "spijt mij", "sorry", "mijn excuses"]
    regex_apologies = re.compile(r"|".join(apologies), flags=re.IGNORECASE)

    text = row[column_name]
    if text is None:
        return False

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    if regex_names.search(text):
        return False

    if regex_brands.search(text):
        return False

    if regex_knowledge.search(text):
        return False

    if regex_language.search(text):
        return False

    if regex_apologies.search(text):
        return False

    if not only_roman_chars(text):
        return False

    if len(text.split()) <= 3:
        return False

    if not is_translation_valid(text):
        return False

    return True
