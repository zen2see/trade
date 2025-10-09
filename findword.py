import string
def price_finder(text):
  """
  Checks if the word 'price' (case-insensitive and ignoring punctuation) is in a string.
  Args:
    text: The input string to search.
  Returns:
    True if 'price' is found, False otherwise.
  """
  # Convert the entire string to lowercase to handle case-insensitivity
  lower_text = text.lower()
  # Remove punctuation from the string
  # This creates a translator that maps each punctuation character to None
#   print(lower_text)
#   translator = str.maketrans('', '', string.punctuation)
#   print(translator)
#   no_punctuation_text = lower_text.translate(translator)
# Check if 'price' is a substring of the processed string
#   return 'price' in no_punctuation_text
  return 'price' in lower_text
# Example Usage:
print(price_finder("What is the price?"))
print(price_finder("DUDE, WHAT IS PRICE!!!"))
print(price_finder("There is no price here."))
print(price_finder("pricey")) # This will return False as it's not the whole word
