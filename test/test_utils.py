import pytest
import utils

def test_word_index_should_return_indexes():
    #given
    text = "This is a test"
    #when
    start, end = utils.word_indexes(text, "test")

    #then
    assert start == 10
    assert end == 14


@pytest.mark.parametrize("text, expected",
  [
    ("instruction: a\ninput: b\nresponse: c", True),
    ("\ninput: b\nresponse: ", False),
    ("instruction: b\nresponse: c", False),
    ("instruction: a\ninput: c", False),
   ]                       
)

def test_is_translation_valid(text, expected):
    #when
    result = utils.is_translation_valid(text)
    #then
    assert result == expected