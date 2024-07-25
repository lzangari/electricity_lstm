def convert_to_float(data):
    """Converts a string to a float, replacing commas with dots.

    Args:
        data (str): The string to be converted to float.

    Returns:
        float: The float value of the string, or None if it can't be converted.
    """
    try:
        return float(data.replace(".", "").replace(",", "."))

    except Exception as e:
        if data == "-":
            return None
        else:
            print(f"Error converting {data} to float: {e}")
