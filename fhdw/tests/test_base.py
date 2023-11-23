"""Test case definitions for modelling package."""
# pylint: disable=missing-function-docstring

from fhdw.modelling.base import clean_string


# Test case for a basic string with special characters
def test_clean_string_with_special_characters():
    input_string = "Hello! World @123"
    expected_output = "hello-world-123"
    assert clean_string(input_string) == expected_output


# Test case for a string with leading and trailing spaces
def test_clean_string_with_spaces():
    input_string = "   Spaces   "
    expected_output = "spaces"
    assert clean_string(input_string) == expected_output


# Test case for an empty string
def test_clean_string_empty_string():
    input_string = ""
    expected_output = ""
    assert clean_string(input_string) == expected_output


# Test case for a string with no special characters
def test_clean_string_no_special_characters():
    input_string = "NoSpecialCharacters123"
    expected_output = "nospecialcharacters123"
    assert clean_string(input_string) == expected_output


# Test case for a string with consecutive special characters
def test_clean_string_consecutive_special_characters():
    input_string = "Testing...123"
    expected_output = "testing-123"
    assert clean_string(input_string) == expected_output


# Test case for a string with a mix of upper and lower case characters
def test_clean_string_mixed_case():
    input_string = "MiXeDcAsE"
    expected_output = "mixedcase"
    assert clean_string(input_string) == expected_output


# Test case for a string with special characters at the beginning and end
def test_clean_string_special_characters_at_boundary():
    input_string = "!@#BoundaryTest!@#"
    expected_output = "boundarytest"
    assert clean_string(input_string) == expected_output
