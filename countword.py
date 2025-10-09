import re

s ='Wow that is a nice price, very nice Price! I said price 3 times.'

def count_price(text_string):
    """
    Counts the number of times the word "price" occurs in a string,
    accounting for capitalization and punctuation adjacency.

    Args:
        text_string (str): The input string to search within.

    Returns:
        int: The number of occurrences of "price".
    """
    # Use re.findall with a case-insensitive regex to find "price"
    # The regex \b ensures whole word matching, and [.,!?;:]* handles trailing punctuation
    # re.IGNORECASE makes the search case-insensitive
    occurrences = re.findall(r'\bprice[.,!?;:]*\b', text_string, re.IGNORECASE)
    return len(occurrences)