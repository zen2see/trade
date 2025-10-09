# INSTALL
python -m venv ./venv
source venv/bin/activate
pip install tensorflow
pip install QuantConnect

# GIT
Instead of syncing the virtual environment itself, you should:
Create a .gitignore file: Add the virtual environment's directory (e.g., venv/ or .venv/) to your .gitignore file to prevent it from being committed to the repository.
Generate a requirements.txt file: Use pip freeze > requirements.txt to generate a list of your project's dependencies.
Commit requirements.txt: Add and commit the requirements.txt file to your Git repository.
Recreate the virtual environment locally: When someone clones your repository, they can then create their own virtual environment and install the dependencies using pip install -r requirements.txt.

# SITES
quantpedia.com

# SYMBOL OBJECTS
VALUE - SECURITY TYPE - MARKET - HASUNDERLYING
DATE - OPTIONTYPE - OPTIONRIGH - STRIKEPRICE


# trd.py
```Javascript
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most TensorFlow logs
from datetime import datetime
from tensorflow import Sequential
from tensorflow import Dense, Flatten
from tensorflow import utils
import json

from QuantConnect.Research import QuantBook

start = datetime(2024, 1, 1)
end = datetime(2025, 1, 1)

qb = QuantBook()
symbol = qb.AddCrypto("BTCUSD", Resolution.Daily).Symbol
history = qb.History(symbol, start, end).loc(symbol) 
```
# PYTHON - FINANCE AND ALGO TRADING WITH QUANTCONNECT
 NumPy = Numerical processor
 Pandas = Data Analysis - excel for python
 Mathpotlib = Visualize data
# FINANCE
 Financial data
 Pandas and Finance
# FINANCIAL CONCEPTS
 Theory - Moden Portfolio Theory - optimizing holdings, CAPM - asset pricing module, Sharpe Ratio - relative performance/volatility
# ALGORITHMNIC TRADING
 LEAN Engine - Research - Technical Indicators - Derivative Contracts
# SETTING UP ENVIRONMENT
# VIRTUAL ENV
python -m venv ./venv
source venv/bin/activate
python -m pip freeze > requirements.txt
# JUPYTER NOTEBOOK
pip install notebook
jupyter notebook
CREATE NEW p3nb
`shift + enter` = new cell
`cell - cell type - MD` = convert to markdown
EX. print("Hello my name is {} and my number is {}".format(name,number)) OR
Ex. print("Hello my name is {x} and my number is {y}".format(x=name,y=number))
# LIST = [1,2,3] LIST.append(4) LIST[-1], LIST[0]

# nested = [1,2,['a','b']] nested[2][0] = 'a'

# dictionairy = {'key':10, 'key2':'seconditem'} d['key2'] = 'seconditem'

# TUPLE = (1,2,3) TUPLE[0] = 1 immutable

# SET = {1,2,3} = unique instances
1 = 1 True, 'string' == 'string' True 1 == '1' False and, or, (1==1) and not (1==2)

```python
if True:
    print 'True'
elif 2=2:
    print('Two')
else:
  print('Not True')

```
# seq = [1,2,3,4]
```python
for num in seq:
    print(num**2)
    i = 1
while i < 5:
      print('i is currently {}'.format(i))
      i = i+1
i is currently 1...
```
# range()
```python
range(5)
range(0,5)
for item in range(0,20,2):
    print(item)
```
# list(range(1,11))
# LIST COMPREHENSION
```python
x = [1,2,3,4]
out = []
for num in x:
    out.append(num**2) OR
[num**2 for num in x]
```
# FUNCTIONS
```python
def my_func(param='default'):
    """
    Docstring explaining what the func does goes here!
    """
    print(param)
my_func('hello')
def my_func2(argument):
   return argument*5
x = my_func2(5)
print(x) = 25
def times_two(var):
    return var*2
result = times_two(4)
result = 8
lambda var: var*2
lambda var: var*2
<funciton __main__.<lambda>>
```
```python
seq = [1,2,3,4,5]
list(map(lambda num:num*2,seq)) = [2,4,6,8,10]
def is_even(num):
    return num%2 == 0 OR
list (filter(lambda num:num%2 == 0,seq))
st = "hello my name is "
st.upper() = 'HELLO MY...'
tweet = "Go Sports! #cool"
tweet.split('#')[1] = 'cool'
d = {'key': 10, 'key2':'seconditem'}
d.keys() = dict_keys(['key', 'key2'])
d.items() = dict_items([('key', 10), ('key2', 'seconditem')])
```
1 in my_list = True or False

# EXERCISES TASK 1 
1 Given price = 300 , use python to figure out the square root of the price.
```python
import math
ans = math.sqrt(300)
```
2 stock_index = "SP500" - Grab '500' from the string using indexing.
```python
ans = stock_index[2:]
```
3 ** Given the variables:**
    stock_index = "SP500"
    price = 300
** Use .format() to print the following string: **
    The SP500 is at 300 today.
```python
stock_index  = "SP500"
price = 300
msg = "The {} is at {} today.".format(stock_index, price)
print(msg)
```
4 Given the variable of a nested dictionary with nested lists: **
stock_info = {'sp500':{'today':300,'yesterday': 250}, 'info':['Time',[24,7,365]]}
** Use indexing and key calls to grab the following items:**
Yesterday's SP500 price (250)
The number 365 nested inside a list nested inside the 'info' key.
```python
yesterday_sp500_price = stock_info['sp500']['yesterday']
number_365 = stock_info['info'][1][2]
print(f"Yesterday's SP500 price: {yesterday_sp500_price}")
print(f"The number 365: {number_365}")
```
5 ** Given strings with this form where the last source value is always separated by two dashes -- **
"PRICE:345.324:SOURCE--QUANDL"
Create a function called source_finder() that returns the source. For example, the above string passed into the function would return "QUANDL"
```python
def source_finder(data_string):
  """
  Extracts the source from a string formatted as "PRICE:VALUE:SOURCE--SOURCENAME".
  Args:
    data_string: The input string containing the source information.
  Returns:
    The source name as a string.
  """
  delimiter = "--"
  index = data_string.find(delimiter)
  if index != -1:
    return data_string[index + len(delimiter):]
  else:
    return None # Or raise an error, depending on desired error handling
```
6 ** Create a function called price_finder that returns True if the word 'price' is in a string. Your function should work even if 'Price' is capitalized or next to punctuation ('price!') **
```python
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
  translator = str.maketrans('', '', string.punctuation)
  no_punctuation_text = lower_text.translate(translator)
  # Check if 'price' is a substring of the processed string
  return 'price' in no_punctuation_text
# Example Usage:
print(price_finder("What is the price?"))
print(price_finder("DUDE, WHAT IS PRICE!!!"))
print(price_finder("There is no price here."))
print(price_finder("pricey")) # This will return False as it's not the whole word
```
7 **Create a function called avg_price that takes in a list of stock price numbers and calculates the average (Sum of the numbers divided by the number of elements in the list). It should return a float. **
```python
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
```

# NUMPY
Library for creating N-dimensional arrays, ability to quickly broadcast functions, buil-in algebra, stats, distributions, 
trigonometric, and random number capabilities
`import numpy as np`
# NumPy Arrays
Vectors(1 dimensional) and Matrices (2 dimensional)
see numpexample.py
# NumPy VS List
Memory efficiency
Expands to N-dimensional objects
Speed of calculations
Broadcasting ops and functions with numpy
All data science and ML use it
`my_list = [1,2,3]`
`np.array(my_list)` = `array([1, 2, 3])`
`my_matrix = [[1,2,3],[4,5,6],[7,8,9]]`
`np.array(my_matrix)` = 
`array([1,2,3],`
      `[4,5,6],`
      `[7.8.9])`
# BUILT IN METHODS TO CREATE ARRAYS
`np.arange(0,10)` = evenly spaced values withn a given interval
`arry([0,1,2,3,4,5,6,7,8,9])
# CREATE AN ARRAY OF ZEROS OR ONES
`np.zeros(3)` = `array([0., 0., 0.])`
# LINSPACE 
`np.linspace(0,10,3)` = `array([0., 5., 10.])`= return evenly spaced numbers over specified interval
# EYE 
`np.eye(4)` = creates an identity matrix
`array([[1., 0., 0., 0.],`
`       [0., 1., 0., 0.],`
`       [0., 0., 1., 0.],`
`       [0., 0., 0., 1.]])`
# RANDOM = Creates an array of the given shape and populates it with random samples
# from a uniform distribution over [0,1]
`np.random.rand(2)` = `array([0.37065108, 0.89813878])`
`np.random.randn(2)` = `array([-0.36633217, -1.40298731])` Returns from "standard normal" dist,
values closer to zero are more likely to appear
# RANDINT
`np.random.ranint(1,100)`
# SEED 
`np.random.seed(42)` set the random state, so that the same "random" results can be reproduced
`np.random.rand(4)` = `array([0.37454012, 0.95071431, 0.73199394, 0.59865848])`
# ARRAY ATTRIBUTES AND METHODS
`arr = np.arrange(25)` = `array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,`
`17, 18, 19, 20, 21, 22, 23, 24])`
`ranarr = np.random.randint(0,50,10)` = `array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])`
# RESHAPE
`arr.reshape(5,5)` Returns an array containing the same data with a new shape.
 `array([[ 0,  1,  2,  3,  4],`
       `[ 5,  6,  7,  8,  9],`
       `[10, 11, 12, 13, 14],`
       `[15, 16, 17, 18, 19],`
       `[20, 21, 22, 23, 24]])`
# MAX, MIN, ARGMAX, ARGMIN 
useful methods for finding max or min values. Or to find their index locations using argmin or argmax
`ranarr = array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])`
`ranarr.max` = 39 `ranarr.argmax()` = 7
# SHAPE
Shape is an attribute that arrays have (not a method)
`Vector arr.shape = (25,)` `arr.reshape(1,25)` = array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]) 
`arr.reshape(1,25).shape` = (1,25)
`arr.reshape(25,1)` = `array([[ 0],`
                             `[ 1],`
                             `...,`
                             `[24]])`
# DTYPE
`arr.dtype` = `dtype('int32')`
`arr2 = np.array([1.2, 3.4, 5.6])` 
`arr2.dtype` = `dtype('float64')`

# EXERCISES TASK 2 
1 import numpy as np
`import numpy as np`
2 Create an Array of 10 zeros
`array_of_zeros = np.zeros((10))`
3 Create an Array of 10 ones
`array_of_ones = np.ones((10))`
4 Create an Array of 10 fives
`array_of_fives = np.ones(10) * 5`
5 Create an Array of integers from 10 to 50
`array_of_integers = np.arange(10, 51) `
6 Create an array of all the even integers from 10 to 50
`even_integers_array = np.arange(10, 51, 2)`
7 Create a 3x3 matrix with values ranging from 0 to 8
`values = np.arange(0, 9).reshape(3, 3)`
8 Create a 3x3 identity matrix
`identity_matrix = np.identity(3)`
9 Use NumPy to generate a random number between 0 and 1
`random_number = np.random.rand(1)`
10 Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution
`random_numbers = np.random.randn(25)`
11 Create the following matrix:
Create a 1D array from 0.01 to 1.00 with a step of 0.01
The stop value in arange is exclusive, so we go up to 1.01 to include 1.00
`data = np.arange(0.01, 1.01, 0.01)` 
12 Create an array of 20 linearly spaced points between 0 and 1
`array_of_points = np.linspace(0, 1, 20)`
13 Replicate the resulting matrix outputs
`mat = np.arange(1,26).reshape(5,5)`
`mat[2:]`
14 Write code that reproduces the output shown below.
`mat[3,4]`
15 Write code that reproduces the output shown below.
`mat[:3,1:2]` GET ALL THREE ROWS AND THE 1 COLUMN 
16 Write code that reproduces the output shown below.
`mat[4,:]` OR `mat[4]`
17 Write code that reproduces the output shown below.
`mat[3:5,:]` OR `mat[3:5]`
18 Get the sum of all the values in mat
`mat.sum()`
19 Get the standard deviation of the values in mat
`mat.std()`
20 Get the sum of all the columns in mat
`mat.sum(axis=0)`
BONUS How to ensure that we always get the same random numbers
`np.random.seed(101)`
`np.random.rand(1)` = should get the same number

# CORE PANDAS
by Wes McKinney as a library for handling financial time series data sets. Now handles many types.
Series, DataFrames, conditional Filtering, Combining DataFrames, Inputs and Outputs
# PANDAS SERIES PT1
A Series is a data structure in Pandas that holds an array of info along with a named index (which differentiates it from NumPy array)
Formal Definition: One-dimensional ndarray with axis labels
Pandas Series add on a labled index (as opposed to numerical, it is still numerical)
`import numpy as np`
`import pandas as pd`
`help(pd.Series)`
`myindex = ['USA', 'Canada', 'Mexico']`
`mydata = [1776,1867,1821]`
`myseries = pd.Series(data=mydata)`
`myseries` = 0    1776
             1    1867
             2    1821
             dtyp: int64
`myserieswlabel = pd.Series(data=mydata,index=myindex)`

`pd.Series`
# PANDAS FROM A DICTIONARY
`ages = {'Sammy':5,'Frank':10,'Spike':7}`
{'Frank': 10, 'Sammy':5, 'Spike':7}
`pd.Series(ages)`
Sammy     5
Frank    10
Spike     7
dtype: int64
# PANDAS SERIES
# Imaginary Sales Data for 1st and 2nd Quarters for Global Company
`1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}`
`2 = {'Brazil': 100,'China': 500, 'India': 210,'USA': 260}`
`sales_q1 = pd.Series(q1)`
`sales_q2 = pd.Series(q2)`
`print(sales_q1,'\n')`
`print(sales_q2,'\n')`
`print('Sales q1 Japan:', sales_q1['Japan'])`
`print('Sales.q1.keys()', sales_q1.keys())`
`print('Sales.q1 * 2, can do +,/\n', sales_q1 * 2)`
`print('sales_q1.add(sales_q2,fill_value=0), aligns by index, add fill value if values don\'t match up\n', sales_q1.add``(sales_q2,fill_value=0),)`
`print('sales_q1.dtype', sales_q1.dtype)`
# DATAFRAME a table of columns and rows in pandas that we can restructure and filter, a group of Pandas Series # objects
# that share the same index