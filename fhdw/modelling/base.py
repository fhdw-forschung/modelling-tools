"""General Modelling Resources."""

import re


def clean_string(string: str):
    """Clean a string by removing special characters and converting it to lowercase.

    Special characters in the beginning and end of the string are dropped.
    Those inbetween are replaced with a hyphen '-' character.

    Parameters:
    - string (str): The input string to be cleaned.

    Returns:
    - str: The cleaned string with special characters removed and converted to
    lowercase.
    """
    cleaned = re.sub(r"^\W+|\W+$", "", string)
    cleaned = re.sub(r"\W+", "-", cleaned)
    return cleaned.lower()


def make_experiment_name(target: str, prefix: str = ""):
    """Generate a standardized experiment name based on the target variable.

    Args:
    - target (str): The target variable for the experiment.

    Returns:
    - str: A formatted experiment name in the pattern 'myNEO_pycaret_{cleaned_target}'.
    """
    clean_target_name = clean_string(target)
    return f"{prefix}{clean_target_name}"


# myNEO_pycaret_
