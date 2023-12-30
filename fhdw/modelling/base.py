"""General Modelling Resources."""

import re


def clean_string(string: str):
    """Clean a string by removing special characters and converting it to lowercase.

    Special characters in the beginning and end of the string are dropped.
    Those inbetween are replaced with a hyphen '-' character.

    Args:
        string (str): The input string to be cleaned.

    Returns:
        str: The cleaned string with special characters removed and converted to
        lowercase.
    """
    cleaned = re.sub(r"^\W+|\W+$", "", string)
    cleaned = re.sub(r"\W+", "-", cleaned)
    return cleaned.lower()


def make_experiment_name(target: str, prefix: str = "", sep: str = "-"):
    """Generate a standardized experiment name based on the target variable.

    Clean the target variable by removing special characters and converting it to
    lowercase using the `clean_string` function.

    Args:
        target (str): The target variable for the experiment.

        prefix (str, optional): An optional prefix to be added to the experiment name.

        sep (str, optional): An optional seperator, placed between prefix and name.

    Returns:
    - str: A formatted experiment name in the pattern '[{prefix}{sep}]{cleaned_target}'.
    """
    clean_target_name = clean_string(target)
    clean_prefix_name = clean_string(prefix)
    return sep.join(part for part in [clean_prefix_name, clean_target_name] if part)
