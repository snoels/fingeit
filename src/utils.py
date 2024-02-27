import re

def word_indexes(text:str, lookup:str) -> tuple[int,int]:
    """ Will return begin and index of a word to lookup.
    
    Params:
        text: text to look into
        lookup: word to look for
    returns:
        Start and end index of the word to look for
    Raises:
        AttributeError when the word is not found
    """
    lookupstr = r'\b(' + lookup + r')\b'
    result = re.search(lookupstr, text)
    begin = result.start()
    end = result.end()
    return begin, end


def is_translation_valid(text:str) -> bool:
    try:
        word_indexes(text,'instruction')
        word_indexes(text,'input')
        word_indexes(text,'response')
        return True
    except AttributeError:
        return False
