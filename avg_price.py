def avg_price(stock_prices: list[float]) -> float:
    """
    Calculates the average of a list of stock price numbers.

    Args:
        stock_prices: A list of numbers representing stock prices.

    Returns:
        The average price as a float.
    """
    if not stock_prices:
        return 0.0
    
    total = sum(stock_prices)
    average = total / len(stock_prices)
    
    return average

def main():
    """
    Main function to run the standalone application.
    """
    # Example with a list of floats
    float_prices = [10.50, 11.25, 12.00]
    average_of_floats = avg_price(float_prices)
    print(f"The average of the float list is: {average_of_floats}")

    # Example with a list of integers
    integer_prices = [3, 4, 5]
    average_of_integers = avg_price(integer_prices)
    print(f"The average of the integer list is: {average_of_integers}")

if __name__ == "__main__":
    main()

