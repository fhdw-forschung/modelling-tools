"""Test case definitions for modelling package."""
# pylint: disable=missing-function-docstring

from pathlib import Path

from fhdw.modelling.base import clean_string
from fhdw.modelling.base import make_experiment_name
from fhdw.modelling.base import validate_path


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


# Test cases for make_experiment_name function


def test_make_experiment_name_basic():
    """Test make_experiment_name with a basic target."""
    result = make_experiment_name("My-Target nr 1")
    assert result == "my-target-nr-1"


def test_make_experiment_name_with_prefix():
    """Test make_experiment_name with a custom prefix."""
    result = make_experiment_name("My_Target", prefix="custom_")
    assert result == "custom_my_target"


def test_make_experiment_name_special_characters():
    """Test make_experiment_name with a target containing special characters."""
    result = make_experiment_name("@My_Target!")
    assert result == "my_target"


def test_make_experiment_name_empty_prefix():
    """Test make_experiment_name with an empty prefix."""
    result = make_experiment_name("My_Target", prefix="")
    assert result == "my_target"  # underscore defuined to be a word character in regex


def test_valid_path_no_files(tmp_path):
    """Test for a folder with no files.

    Folder newly created; here a temp folder (pytest fixture).
    """
    assert validate_path(tmp_path) is True


def test_valid_path_with_files(tmp_path):
    """Test for a valid folder path with at least one model file."""
    file_path = f"{tmp_path}/model1.pkl"
    Path(file_path).touch()
    assert validate_path(file_path) is False
    assert validate_path(str(Path(file_path).parent)) is True


def test_invalid_path():
    """Test for an invalid folder path."""
    folder_path = "/path/to/nonexistent/folder"
    assert validate_path(folder_path) is False


def test_empty_path():
    """Test for an empty folder path."""
    folder_path = ""
    # resolves to current folder, therfore True
    assert validate_path(folder_path) is True
