import numpy as np


def convert_to_float(data):
    """Converts a string to a float, replacing commas with dots.

    Args:
        data (str): The string to be converted to float.

    Returns:
        float: The float value of the string, or None if it can't be converted.
    """
    # if data is instance of float, return data
    if isinstance(data, float):
        return data

    try:
        return float(data.replace(".", "").replace(",", "."))

    except Exception as e:
        if data == "-":
            return None
        else:
            print(f"Error converting {data} to float: {e}")


# include sine and cosine transformations for the hour and day of the week
def encode_feature(data, column_name: str, max_value: int):
    """Encodes a feature in a way that captures the periodicity of the data.

    Args:
        data (dataframe): the dataframe containing the data to be transformed.
        column_name (str): the name of the column to be transformed.
        max_value (int): the maximum value of the feature to be transformed.

    Returns:
        _type_: _description_
    """
    data[column_name + "_sin"] = np.sin(2 * np.pi * data[column_name] / max_value)
    data[column_name + "_cos"] = np.cos(2 * np.pi * data[column_name] / max_value)
    return data
