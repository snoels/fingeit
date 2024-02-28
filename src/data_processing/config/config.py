from configparser import ConfigParser
from posixpath import abspath, dirname


def load_config(subsection=None):
    """Loads configuration from config.ini file."""
    directory = dirname(abspath(__file__))
    config = ConfigParser()
    config.read(f"{directory}/config.ini")
    return config[subsection] if subsection else config
