from configparser import ConfigParser
from posixpath import abspath, dirname


def load_config() -> ConfigParser:
    """Loads configuration from config.ini file."""
    directory = dirname(abspath(__file__))
    config = ConfigParser()
    config.read(f"{directory}/config.ini")
    return config



def get_config(args):
    """Loads configuration from config.ini file."""
    sub_section = "TRANSLATE"
    config = load_config()
    target_language = args.target_language

    system_prompt = config.get(sub_section, "system_prompt")
    config.set(
        sub_section,
        "system_prompt",
        system_prompt.replace("<target_language>", target_language),
    )

    prompt = config.get(sub_section, "prompt")
    config.set(
        sub_section, "prompt", prompt.replace("<target_language>", target_language)
    )
    return config[sub_section]