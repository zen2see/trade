from tkinter import ALL
import numpy as np
import pandas as pd
import timeit 
import os
from sqlalchemy import create_engine # https://www.sqlalchemy.org/
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec, ticker
from datetime import datetime
import re
# import pandas_datareader.data as web
from pandas_datareader import data as pdr
import yfinance as yf

""" 
#### PYTHON BASICS #########################################################################################
"""
# COMMENTS 
"""
# MULTIPLE ROW COMMENTS
"""

"""
# MATH
## 1+1=2, 1-3=-2, 1*3=6, 1/2=0.5, 2**4=16, 4%2=0, 5%2=1, 2+3*5+5=22, (2+3)*(5+5)=50

# VARIABLES
##x=1, y=2, x+y=3, z=x+y, # z now = 3

# STRINGS
## 'single', "double", "I don't care"

# PRINT
x = "hello"
print(x)
print("DOUBLE")
name = "Jose Portilla"
print("Hello my name is {}".format(name))
number = 12
print(f"Hi my name is {name} and my number is {number}".format(name, number)) # OR
print("Hi my name is {} and my number is {}".format(name, number))

# LIST mutable
['hi',0,1]
my_list = [1, 2 ,3] 
my_list.append(4) # [1,2,3,4] 
my_list = ['a','b','c','d']
my_list[0] = 'a'
my_list[-1] = 'd'
my_list[0:2] = ['a', 'b'] # OR my_list[0:2] or nt_list[:2]
nested = [1,2,['a','b']]
nested[2][0] = 'a'

# DICTIONARY
dict = {'key':10,'key2':'seconditem'}
dict['key2'] # 'seconditem'
dict.keys() # RETRUNS dict_keys(['key', 'key2])
# TUPLES like a list but non-mutable
t = (1,2,3)
print(t[0])

# SETS = unorderderd colleciton of UNIQUE items
## {1,2,3} = {1,1,1,2,2,2,3,3,3} 
import math
math.sqrt(4) # SHIFT+TAB for DOCUMENTATION

# COMPARISON OPERATORS
# 1 < 2, 1 >=3, 1 != 3, 'string' == 'string', 1 == '1'# FALSE (1==2 ) AND NOT (1==1), OR, NOT
if True:
    print('hello')
elif 2==2:
    print('two')
else:
    print('false')

# FOR LOOPS
seq = [1,2,3,4,5]
for item in seq:
    print(item)

# WHILE LOOPS
i = 1
while i < 5:
    print(f"i is currently", {i})
    i = i + 1

# RANGE
range(5) # = range(0,5)
for item in range(0,20,2):
    print(item) # prints 0,2,4...

# LIST    
list(range(1,11)) # [1,2,3...10]
my_list = [1,2,3,4]
my_list.pop() # RETURNS 4
my_list.pop(0) # RETURN 1
my_list # RETURNS [2,3]
2 in my_list # RETURNS TRUE

# LIST COMPREHENSION
x =[1,2,3,4]
out = []
for num in x:
    out.append(num**2)
out = [1, 4, 9, 16]
[num**2 for num in x] # = [1, 4, 9, 16]
 
# FUNCTIONS
def my_func():
    print('hello')
def my_func2(param='default'):
    # triple quotes
    #Docstring goes here between triple quotes! 
    # triple quotes
my_func(param)
# python3 my_func() # would print doc info and then the word default    
def my_func3(argument):
    return argument*5 
x = my_func3(5) # returns 25
print(x) # prints 25
def times_two(var):
    return var*2
result = times_two(4)
result # prints8

# EXAMPLE LAMDA AND MAP FUNCTION
lambda var: var*2 # SAME AS THE times_two(var) funciton above
seq = [1,2,3,4,5]
list(map(lambda num:num*2,seq)) # [2,4,6,8,10]
def is_even(num):
    return num%2 == 0 
(filter(is_even,seq)) # RETURNS <filter at 0x1ddb513a5f8)
list(filter(is_even,seq)) # RETRURNS, seq v=[1,2,3,4,5]
list(filter(lambda num:num%2 == 0,seq)) # RETURNS [2,4]

# STRING FUNCTIONS
st = 'hello my name is Sam'
st.lower # RETURNS 'hello my name is sam'
st.upper() # RETURNS 'HELLO MY NAME IS SAME'
tweet = "Go Sports! #cool"
tweet.split() # RETURNS ['Go', 'Sports!', '#cool']
tweet.split('#') # RETURNS 'cool'
# TYPE tweet. TO SEE THE LIST POSSIBLE  FUNCTIONS

# EXERCISES
#1 Given price = 300, figure out the squre root of the price in python.
price = 300
price ** 0.5 # OR
import math 
math.sqrt(price)  
#2 Grab '500' from the string using indexing
stock_index = "SP500"
stock_index[2:]
#3 Use .format() to print The SP500 is at 300 today.
stock_index = "SP500"
price = 300
"The {} is at {} today".format(stock_index,price)
#3 Given the variable of a nested dictionary w\ nested lists grab certain items using indexing + key calls
stock_info = {'sp500':{'today':300, 'yesterday':250},'info':['Time',[24,7,365]]}
stock_info.keys() # RETURNS dict_keys{['sp500', 'info']}
stock_info['sp500']['yesterday'] # RETURNS 250
stock_info['info'][1][2] # RETRUNS 365
stock_info['info'][1] # RETURNS [24, 7, 365]
#4 Create a () called source_finder() that returns the source.
def source_finder(s):
    return s.split('--')[-1]
source_finder("PRICE:345.324:SOURCE--QUANDL") # returns 'QUANDL'
#5 Create a () called price_finder that returns True if the word 'price' is in a string.
def price_finder(s):
    return 'price' == s.lower()
#6 Create a () called count_price() taht counts the # of times the word 'price' occurs in a str.
def count_price(s):
    count = 0
    for word in s.lower().split():
        if 'price' in word:
            count = count + 1
    return count
# OR
def count_price(s):
    return s.lower().count('price')
#7 Create a () called avg_price that takes in a list of stock price #'s and calculates the average in float.
def avg_price(stocks):
    return sum(stocks)/len(stocks)
avg_price([3,4,5]) # Returns 4.0

#### NUMPY ################################################################################################# 
## NumPy is a data science library

# ARRAYS - A Python library for creating N-dimensional arrays, quickly broadcast functions.
## Has Built-in linear algebra, statistical distributions, trigonometric, and random number capabilities.
## Look similar to standard Python lists, they are much more efficient!

# BROADCASTING capabilities are extremely useful for quickly applying () to our data sets.
import numpy as np
mylist = [1,2,3]
type(mylist) # = list
np.array(mylist) # = array([1,2,3])
myarray = np.array(mylist) # myarray = array([1,2,3])
type(myarray)  # numpy.ndarray
my_matrix  = [[1,2,3],[4,5,6],[7,8,9]] # my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
np.array(my_matrix) 
# array([[1,2,3],
#        [4,5,6],
#        [7,8,9]])
np.arange(0,101,20) # array([0,20,40,60,80,100])
np.zeros(5) # array([0.,0.,0.,0.,0.])
np.zeros((2,5)) 
# array([0.,0.,0.,0.,0.],
#        [0.,0.,0.,0.,0.]])
np.ones(5) # array([1., 1., 1., 1., 1.])
np.linspace(0,10,11)# array([0.    , 1.11111111, 2.22222222, ... 10.       ]) 21 numbers
np.eye(5) # array([1.,0.,0.,0.,0.,],
#                  [0.,1.,2.,3.,4.,],)
np.random.rand(1) # Gives a random number between 0 and 1 array([0.11242691])
np.random.rand(5,2) # Five rows of two random numbers
np.random.randn # Mean is 0 and variance 1 array([-0.90387279, -0.33609234, 1.54333558, etc.])
np.random.randint # Returns random integers
np.random.randint(0,101,(2,3)) # array([[ 0.13570505, 0.736018705, 2.01277515], ...])
np.random.randint(0,101,10) # array([8,30,33,49,66,32,69,14,96,62])
np.random.seed(42) # seed 42 from hitchhikers guide to the galaxy
np.random.rand(4) # array([0.3745012, ...])
arr = np.arange(0,25) # arr = array([0,1,2,3...24])
arr.reshape(5,5) # array([0,1,2,3,4],
#                        [5,6,7,8,9],...
ranarr = np.random.randint(0,101,10)
ranarr.max() # 93
ranarr.min() # 8
ranarr.argmax() # 1 position of max argument
ranarr.dtype # dtype('int32')
arr.shape # 25
arr = arr.reshape(5,5) # the numbers in an array of 5 by 5

# INDEXING AND SELECTION
anarr = np.arange(0,11) # arr = array([0,1,2,3..1])
anarr[8] # 8
anarr[1:5] # array([0,1,2,3,4]) SAME AS anarr([:5])
anarr[5:] # array([5,6,7,8,9,10])

# BROADCASTING 
anarr[0:5] = 100 # anarr = array([100, 100, 100, 100, 100, 5, 6, 7, 8, 9, 10])
newarr = np.arange(0,11) # newarr = array([0, 1, 2, ...10])
slice_of_newarr = newarr[0:5] # slice_of_newarr = array([0, 1, 2, 3, 4])
slice_of_newarr[:] = 99 # = array([99, 99, 99, 99, 99]) 
newarr # = array([99, 99, 99, 99, 99, 5, 6, 7, 8, 9, 10])
arr_copy = newarr.copy() 
arr_copy[:] = 100 # arr_copy = array([99, 99, 99, 99, 99, 5, 6, 7, 8, 9, 10])

# INDEXING ON 2D ARRAYS
arr_2d = np.array([[5,10,15],[20,25,30],[35,30,45]])
arr_2d # = array([5,  10, 15],
#                [20, 25, 30],
#                 [35, 40, 45])

arr_2d.shape # (3,3)
arr_2d[2] # = array([35, 40, 45])
arr_2d[1][1] # 25
arr_2d[1,1] # 25
arr_2d[:2] # array([ 5, 10, 15],
#                  [20, 25, 30]]) 
arr_2d[:2,1:] # array([10, 15],
#                      [25, 30]])

                      # OPERATORS
arrop = np.arange(0,10) # array([0,1,2,3,4,5,6,7,8,9])
arrop + 5 # array([5,6,7,8,9,10,11,12,13,14])
arrop - 2 # array([-2,-1,0,1,2,3,4,5,6,7])
arrop + arrop # array([0,2,4,6,8,10,12,14,16,18]) CAN DO *,-
np.sqrt(arrop) # array([0.        , 1.        , 1.41421356, ...])
np.sin(arrop) # array([0.        , 0.84147098,  0.90929743, ...])
np.log(arrop) # array([        -ing, 0.        , 0.69314718, ...])
arrop.sum() # sum all elements in array
arrop.mean() # arrop.max() arr.var() arr.std()
arr2d = np.arange(0,25).reshape(5,5) # arr2d.shape (5,5)
arr2d # array[0,1,2,3,4],
      #      [5,6,7,8...
arr2d.sum() # 300
arr2d.sum(axis=0) # array([50,55,60,65,70]) columns
arr2d.sum(axis=1) # array([10,35,60,85,110]) rows

# NUMPY EXERCISES
# 1 import NumPy an np
# 2 Create an array of 10 zeros zeros = np.zeros(10)
# 3 Create an array of 10 ones ones = np.ones(10)
# 4 Create an array of 10 fives five = np.fives(10)
# 5 Create an array of the integers from 10 to 50 = np.arange(10,51)
# 6 Create an array of all even integers from 10 to 50 = np.arange(10,51,2)
# 7 Create a 3x3 matrix with values ranging form 0 to 8 = np.arange(0,9).reshape(3,3)
# 8 Create a 3x3 identity matrix = np.identity(3)
# 9 Use Numpy to generge a random number between 0 and 1 = np.random.rand()
# 10 Generate an array of 25 random numbers sampled from a standard normal dist
# = np.random.randn(25)
# 11 Create an array from 0.01 to 1.0 w\step of 0.01 np.arange(0.01, 1.01, 0.01)
# 12 Create linearly spaced matrix between 0-1: = np.linspace(0,1,20)
# 13 Write code to grab row 3 on: mat = np.arange(1,26).reshape(5,5) = mat[2:,1:]
# 14 Write code to grab 20 = mat[3,4]
# 15 Write code to grab the 2, 7 and 12 = mat[:3,1:2]
# 16 Write code to grab 21-25 = mat[4,:] OR mat[4]
# 17 Write code to grab 16-25 = mat[3:5] OR mat[3:5,:]
# 18 Get the sum of all the values in mat = mat.sum()
# 19 Get the standard deviation of the values in mat = mat.std()
# 20 Get the sum of all the columns in mat = mat.sum(axis=0)
# BONUS = Always get same random #'s = np.random.seed(101), np.random.rand(1)

#### CORE PANDAS ###########################################################################################
# A Series - data structure that holds an array of info w/a named index
# Named index differentiates this from a NumPy array. Formal Def: A long-dimensional ndarray w\ axis labels
# import numpy as np
# import pandas as pd
# help(pd.Series)

# LIST REVIEW - PANDAS SERIES
myindex = ['USA', 'Canada', 'Mexico']
mydata = [1776,1867,1821]
myseries = pd.Series(data=myindex)
print(myseries,'\n') # OR myseries
# 0    1776
# 1    1867
# 2    1821
myseries2 = pd.Series(data=mydata, index=myindex)
print(myseries2, '\n') # OR myseries2
# USA       1776
# Mexico    1867
# Canada    1821
# dtype: int64
print(myseries[0], '\n') # 1776
#print(myseries['USA']) # 1776
ages = {'Sam': 5, 'Frank': 10, 'Spike': 7}
pd.Series(ages)
print(pd.Series(ages), '\n') 
# Sam    5
# Frank  10
# Spike  7
# dtype:  in64 
# Imaginary Sales Data for 1st and 2nd Quarters for Global Company
q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}
q2 = {'Brazil': 100,'China': 500, 'India': 210,'USA': 260}
sales_q1 = pd.Series(q1)
sales_q2 = pd.Series(q2)
print(sales_q1,'\n')
# JAPAN     80      Brazil      100
# China     450     China       500 
# India     200     India       210
# USA       250     USA         260
# dtyp: int64
print(sales_q2,'\n')
print('Sales q1 Japan:', sales_q1['Japan'])
print('Sales.q1.keys()', sales_q1.keys()) # Index(['Japan', 'China', ' India', 'USA'], dtype='object')
print('Sales.q1 * 2, can do +,/\n', sales_q1 * 2)
print('sales_q1.add(sales_q2,fill_value=0), aligns by index, add fill value if values don\'t match up\n'/
, sales_q1.add(sales_q2,fill_value=0),)
print('sales_q1.dtype', sales_q1.dtype)

#### DATAFRAME 
# DATAFRAME a table of columns and rows in pandas that we can restructure and filter, group of Pandas Series 
# objects that share the same index
# import numpy as np
# import pandas as pd
np.random.seed(101) # Enusre a set of random integers
dfranddata = np.random.randint(0,101,(4,3)) # randint(low, high=None, size=None, dtype='1')
print("\ndf data: \n", dfranddata)
# array([95, 11, 81],
#        70, 63, 87],
#        75,  9, 77],
#        40,  4, 63]])
dfindex = ['CA','NY','AZ','TX']
dfcolumns = ['Jan', 'Feb', 'Mar']
df = pd.DataFrame(data=dfranddata, index=dfindex, columns=dfcolumns)
#     Jan  Feb  Mar
# CA   95   11   81
# NY   70   63   87
# AZ   75    9   77
# TX   40    4   63
print('\nDataFrame of df data:\n', df,'\n','\nDF Info:')
print(df.info()) 

# HOW TO READ A CSV FILE FROM a PANDAS DATAFRAME (CAN READ OTHER FILES LIKE HTML)
dfcvs = pd.read_csv('tips.csv') # pd.read_csv)'C:\\Users\\CSV_FILE\\LOCATION\.file.csv
print('\nTIPS:\n',dfcvs) # PRINT WHOLE FILE
print('COLUMNS:\n',dfcvs.columns,'\n') # Index(['total_bill', 'tip', ...])
print('INDEX:\n',dfcvs.index,'\n') # RangeIndex(start=0, stop=244, step=1)
print('HEAD:\n',dfcvs.head(),'\n') # dfcvs.head(10)  First 10 rows
print('TAIL:\n',dfcvs.tail(5),'\n') # dfcvs.tail(10) last 10 rows
print('INFO:\n')
print(dfcvs.info())
print('DESCRIBE:\n',dfcvs.describe(),'\n') # DESCRIBE STATISTICAL OPS MIN, MAX, MEAN, COUNT, STD, 25%, 50%
print('DESCRIBE.TRANSPOSE:\n',dfcvs.describe().transpose(),'\n') # Transposes columns                                                                                                                                                                                                                                                           

# USING OS
# import os
# os.getcwd() GET WORKING DIR

# WORKING WITH COLUMNS
print('WORKING WITH COLUMNS:\n',dfcvs.head(),'\n')
print('TOTAL_BILL:\n',dfcvs['total_bill'], '\n') # OR dfcvs.total_bill
print('TYPE:\n',type(dfcvs['total_bill']))
someColumns = ['total_bill','tip']
print('COLUMNS:\n', dfcvs[someColumns],'\n')

# CALCULATE PERCENTAGE OF TIP
print('PERCENT OF TIP:\n',100 * dfcvs['tip'] / dfcvs['total_bill'], '\n')
dfcvs['tip_pct'] = 100 * dfcvs['tip'] / dfcvs['total_bill']
print(dfcvs.head())

# ROUND OFF TO 2 DECIMAL PLACES PRICE PER PERSON
dfcvs['price_per_person'] = np.round(dfcvs['total_bill'] / dfcvs['size'],2)
print(dfcvs.head())

# REMOVING COLUMS
dfcvs.drop('tip_pct', axis=1, inplace=True) # axis=0 =rows, axis=1 =columns inplace=True =column kept in bg

# WORKING WITH ROWS
print('INDEX\n',dfcvs.index)
print('TEMPORARILY SET INDEX\n,dfcvs.set_index(\'Payment ID\')','\n',)
print('PERMANENTLY SET INDEX by dfcvs = dfcvs.set_index(...')
print('RESET INDEX\ndfcvs.reset_index(),\n')
print('LOC - LABEL BASED INDEXING\n,dfcvs.loc[\'Sun2959\'], OR numerical with dfcvs.iloc[0]\n')
print('SLICE DATA\ndfcvs.iloc[0:4]\n') 
# COULD DO dfcvs.loc[\'Sun2959\']: OR dfcvs.loc[['Sun2959','Sun5260']] FOR MULTIPLE ROWS
print('REMOVING ROWS,dfcvs.drop(\'Sun2959\', axis=0)') # axis=0 is rows, axis=1 is columns
print('PERMANENTLY DROP First 3 WOULD BE, dfcvs = dfcvs.iloc[3:3]') # DROP FIRST 3 ROWS AND KEEP REST)
print('ADDING A ROW = SAY WE ADD A ROW one_row = dfcvs.iloc[0], dfcvs.append(one_row), /
PERMANENT = dfcvs = dfcvs.append(one_row)')
a_row = dfcvs.iloc[0]
print('added',a_row)
"""

"""
# TYPICALLY IN DATA ANALYSIS OUR DATASETS ARE LARGE ENOUGH WE DON'T FILTER BASED ON POSITION, USE CONDITION
# CONDITIONAL FILTERING ALLOWS US TO SELECT ROWS BASED A CONDITION ON A COLUMN

myindex2 = ['USA', 'Canada', 'Mexico']
mydatayear = [1776,1867,1821]
mydatapop = [ 328, 38, 126]
mydatagdp = [20.5, 1.7,1.22]
df2 = pd.DataFrame(data={'Year': mydatayear, 'Population': mydatapop, 'GDP': mydatagdp}, index=myindex2)
print('\nDATAFRAME df2:\n', df2)
print('\ndf2["Population"] > 50:\n',df2['Population'] > 50,) # RETURNS A BOOLEAN 
print('\ndf2[df2["Population"] > 50]:\n',df2[df2['Population'] > 50],) # RETURNS ROWS WHERE CONDITION IS TRUE

# FILTERING WITH (MULTIPLE) CONDITIONS
dftips = pd.read_csv('tips.csv')
print('\nREAD HEAD OF TIPS DataFrame\n', dftips.head())
print('\nFILTER TIPS SERIES WHERE TOTAL BILL > 50\n', dftips[dftips['total_bill'] > 50])
print('\nFILTER TIPS SERIES WHERE TOTAL BILL > 22 AND DAY IS FRI\n', /
 dftips[(dftips['total_bill'] > 22) & (dftips['day'] == 'Fri')])
# print(dftips.columns)
options = ['Sat', 'Sun']
dftips['day'].isin(options) # Will print 0   True...

### PANDAS USEFUL METHODS ###
print('\nUSEFUL METHODS:\napply()\n')
print('EX. 1 dftips.head()\n',dftips.head())
print('GRAB LAST 4 DIGIST OF CC#\n',dftips['CC Number'].apply(lambda x: str(x)[-4:]),'\n')
print('CREATE A CUSTOM FUNCTION GRAB LAST FOUR and apply it to a /
column\ndef grab_last_four(num):\n return str(num)[-4:]\n')
print('def grab_last_four(num):')
print('    return str(num)[-4:]\n') 
def grab_last_four(num):
    return str(num)[-4:]    
print('dftips[\'CC Number\'].apply(grab_last_four),\n',dftips['CC Number'].apply(grab_last_four),'\n')
print('CREATE A CUSTOM FUNCTION THAT TAKES A PRICE AND RETURNS $ DEPENDING ON PRICE\n')
def yelp(price: int):
    if price < 10:
        return "$"
    elif price < 30:
        return "$$"
    else:
        return "$$$"
print('\ndftips[\'yelp\'] = dftips[\'total_bill\'].apply(yelp)')
dftips['yelp'] = dftips['total_bill'].apply(yelp)
print(dftips.head())

# MULTIPLE FUNCTION ON MULTIPLE COLUMNS
print('APPLY MULTIPLE FUNCTIONS ON MULTIPLE COLUMNS')
print('EXAMPLE LAMBDA FUNCTION, SUPPOSE WE HAD\n def simple(num):\n return num * 2\n')
print('LAMBDA FUNCTION WOULD BE\n lambda num: num *2')
print('dftips[\'TOTAL_BILL,tip\'].apply(lambda num: num * 2)\n', /
dftips[['total_bill','tip']].apply(lambda num: num * 2),'\n')
def Quality(total_bill,tip):
    if tip / total_bill >= 0.25:
        return 'Generous'
    elif tip / total_bill >= 0.15:
        return 'Average'
    else:
        return 'Cheap'
print('dftips[[\'total_bill\', \'tip\']]/
.apply(lambda dftips: Quality(dftips[\'total_bill\'], dftips[\'tip\']), axis=1)\n')
dftips['Quality']=dftips[['total_bill', 'tip']]/
.apply(lambda dftips: Quality(dftips['total_bill'], dftips['tip']), axis=1)
print(dftips['Quality'])
print('\nHAVE IT RUN FASTER WITH VECTORIZE - df[\'Quality\'] = /
np.vectorize(Quality)(dftips[\'total_bill\'], dftips[\'tip\'])')

# NP VECTORIZE TRANSFORMS FUNTION THAT ARE NOT NUMPY AWARE 
dftips['Quality'] = np.vectorize(Quality)(dftips['total_bill'], dftips['tip'])
print(dftips.head())
# dftips['Quality'] = dftips['total_bill', 'tip'].apply(Quality)

# TIMELY CODE STARTS HERE AFTER ADDING TRIPLE QUOTES
# TIMING CODE WITH TIMEIT - NOTE TRIPLE QUOTES BELOW ARE NEEDED FOR TIMELY CODE TO WORK/
# SO PLACE 3 QUOTES BEFORE THIS LINE
# TIMELY CODE TO BE EXECUTED ONCE FOR TIMELY
"""
pdtimely = '''
import numpy as np
import pandas as pd
dftips = pd.read_csv('tips.csv')
def qualitytimely(total_bill,tip):
    if tip / total_bill >= 0.25:
        return 'Timely: Generous'
    elif tip / total_bill >= 0.15:
        return 'Timely: Average'
    else:
        return 'Timely: Cheap'
'''

# CODE SNIPPET WHOSE EXECUTION TIME IS TO BE MEASURED BY TIMELY
stmt_one ='''
dftips['Tip Quality'] = dftips[['total_bill', 'tip']].apply(lambda dftips: qualitytimely(dftips['total_bill'], dftips['tip']), axis=1)
'''
stmt_two = '''
dftips['Tip Quality'] = np.vectorize(qualitytimely)(dftips['total_bill'], dftips['tip'])
''' 
## timelyone = timeit.timeit(setup=pdtimely,stmt=stmt_one,number=1000)
## timelytwo = timeit.timeit(setup=pdtimely,stmt=stmt_two,number=1000)
## print('\nUsing timely: ',timelyone)
## print('Using timely with vectorize: ',timelytwo)
# TIMELY CODE ENDS HERE

"""
# USEFUL METHODS - DESCRIBING AND SORTING
print('\nUSEFUL METHODS - DESCRIBING AND SORTING')
dfsort = pd.read_csv('tips.csv')
print('dfsort.describe()\n',dfsort.describe())
print('\nSORT VALUES BY TIP HI-LOW\ndfsort.sort_values(\'tip\')',dfsort.sort_values('tip',ascending=False))
print('\nSORT VALUES BY MULTIPLE VALUES\ndfsort.sort_values(\'tip\',\'size\')',dfsort.sort_values(['tip','size']))
print('\nSORT VALUES BY TOTAL BILL MAX\ndfsort.sort_values[\'total_bill\'].max():',dfsort['total_bill'].max())
print('\nGET TOTAL BILL MAX INDEX LOCATION\ndfsort.sort_values[\'total_bill\'].idxmax():',dfsort['total_bill'].idxmax())

# USEFUL METHODS - CORRELATION VALUES
print('\nCORRELATION VALUES\ndfsort.corr()',dfsort.corr(numeric_only=1))

# USEFUL METHODS - COUNT OF VALUES
print('\nVALUE CONTS\ndfsort[\'sex\'].value_counts()\n',dfsort['sex'].value_counts(), sep='')

# USEFUL METHODS - REPLACE METHOD
print('\nREPLACE VALUES\ndfsort[\'sex\'].replace([\'Female\'],[\'Male\'],[\'F\',\'M\'])\n',dfsort['sex'].replace(['Female','Male'],['F','M']), sep='')

# USEFUL METHODS - REPLACE MAP METHOD
print('\nmymap = {\'Female\':\'F\',\'Male\':\'M\'}', sep='')
mymap = {'Female':'F','Male':'M'}
print('\nREPLACE VALUES MAP METHOD\ndfsort[\'sex\'].map(mymap)\n',dfsort['sex'].map(mymap), sep='')

# DUPLICATED ROWS
simpledf = pd.DataFrame([1,2,2,2],['a','b','c','d',])
print('\nDUPLICATED = simpledf.duplicated() (SUBSEQUENT DUPLICATES)\n',simpledf.duplicated(), sep='')

# DROP DUPLICATES
print('\nDROP DUPLICATES = simpledf.drop_duplicates()\n',simpledf.drop_duplicates(), sep='')

# INCLUSIVE (LIKE BILL BETWEEN $10-20)
print('\nBETWEEN 10-20 simpledf[\'total_bill\'].between(10,20,inclusive=both)\n',dfsort['total_bill'].between(10,20,inclusive='both'), sep='') 
print('\n',dfsort[dfsort['total_bill'].between(10,20,inclusive='both')])

# GRAB LARGEST
print('\nLARGEST TIPS = dfsort.nlargest(2,\'tip\')\n',dfsort.nlargest(2,'tip'))

# GRAB BY ORDER
print('\nSORT VALUES TIP = dfsort.sort_values(\'tip\',ascending=True).iloc[0:2])\n',dfsort.sort_values('tip',ascending=True).iloc[0:2])
# RANDOM SAMPLES
print('\nSAMPLE FIVE RANDOM ROWS =  dfsort.sample(5)\n', dfsort.sample(5))

# OFTEN THE DATA NEEDED EXIST IN TWO SEPARATE SOURCES, FORTUNATELY, PANDAS MAKES IT EASY TO COMBINE. 
# SAME FORMAT (*SIMPLEST) A CONCATENATION (PAST TWO DF TOGETHER) THROUH pd.concat() IS ALL THAT IS NEEDED
# PANDAS WILL AUTO FILL FOR NAN VALUES

# import numpy as np
# import pandas as pd
print('\ndata_one = {\'A\': [\'A0\', \'A1\', \'A2\', \'A3\'], \'B\': [\'B0\', \'B1\', \'B2\', \'B3\']}',)
print('data_two = {\'C\': [\'C0\', \'C1\', \'C2\', \'C3\'], \'D:\' [\'D0\', \'D1\', \'D2\', \'D3\']}')
data_one = {'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0','B1','B2','B3']}
data_two = {'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0','D1','D2','D3']}
print('\none = pd.DataFrame(data_one)')
print('two = pd.DataFrame(data_two)\n')
one = pd.DataFrame(data_one)
two = pd.DataFrame(data_two)
print('one:\n',one,'\n', sep='')
print('two:\n',two,'\n', sep='')
print('\nCONCAT THEM BOTH = pd.concat([one,two],axis=1)\n',pd.concat([one,two],axis=1))
print('\nSEPARATE BY AB and CD = pd.concat([one,two],axis=0)\n',pd.concat([one,two],axis=0))

# MATCHING COLUMNS
print('\ntwo.columns = one.columns')
two.columns = one.columns
print('\ntwo:\n',two)
print('\nMERGE THEM = pd.concat([one,two],axis=0):')
concatdf = pd.concat([one,two],axis=0)
print(concatdf)

# RESET INDEX
print('\nRESET INDEX -  concatdf.index = range(len(concatdf))')
concatdf.index = range(len(concatdf))
print(concatdf)

# COMBINING\MERGING DATAFRAMES
print('\nOFTEN DATAFRAMES ARE NOT IN EXACT SAME ORDER OR FORMAT, IN THIS CASE, WE NEED TO MERGE THEM')
print('THE .merge() TAKES IN A KEY ARG LABELED HOW: 3 WAYS INNER, OUTER, LEFT/RIGHT')
print("MAIN ARGUMENT IS TO DECIDE HOW TO DEAL WITH INFO ONLY PRESENT IN ONE OF THE JOINED TABLES")
print('DECIDE ON WHAT COLUMN TO MERGE TOGETHER, ON COLUMN SHOULD BE PRIMARY, PRESENT AND REPRESENT THE SANE THING')
print('DECIDE HOW TO MERGE TABLES ON THE NAME COLUMN')
print('REGISTRATIONS - reg_id name -- LOGINS - log_id name')
print('MERGERS ARE OFTEN SHOWN AS A VENN DIAGRAM')
print('pd.merge(registrations,logins,how=\'inner\',on=\'name\')')
print('RESULTS = reg_id name log_id')
print('\nHELP = help(pd.merge)')
print('\nregistrations = pd.DataFrame({\'reg_id\':[1,2,3,4],\'name\':[\'Andrew\',\'Bobo\',\'Claire\',\'David\']}')
print('logins = pd.DataFrame({\'log_id\':[1,2,3,4],\'name\':[\'Xavier\',\'Andrew\',\'Yolanda\',\'Bobo\']}')
registrations = pd.DataFrame({'reg_id':[1,2,3,4],'name':['Andrew','Bobo','Claire','David']})
logins = pd.DataFrame({'log_id':[1,2,3,4],'name':['Xavier','Andrew','Yolanda','Bobo']})
print('pd.merge(registrations,logins,how=\'inner\',on=\'name\')\n',pd.merge(registrations,logins,how='inner',on='name'))
print('pd.merge(logins,registrations,how=\'inner\',on=\'name\')\n',pd.merge(logins,registrations,how='inner',on='name'))
print('\nLEFT MERGE\npd.merge(registrations,logins,how=\'left\',on=\'name\')')
mergedL = pd.merge(left=registrations,right=logins,how='left',on='name')
print(mergedL)
print('\nRIGHT MERGE\npd.merge(registrations,logins,how=\'right\',on=\'name\')')
mergedR = pd.merge(left=registrations,right=logins,how='right',on='name')
print(mergedR)

# PANDAS IO - CVS
# import pandas as pd
# import os
# os.getcwd()
# dfsomecvs = pd.read_csv('example.csv', header=None) # Sets A B C D as a row entry 0 a b c d
# dfsomecvs = pd.read_csv('example.csv', index_col=0) # Sets a as index
# dfsinecvs.to_csv('somefilename_or_filepath_+filename, index=false)

# PANDAS IO - HTML (pip install lxml) (can also copy source file)
# import pandas as pd
#urlfile = "worldpop.html" # https://en.wikipedia.org/wiki/World_population"
urlfile2 = "https://www.worldometers.info/world-population/population-by-country/"
#try:
#    with open(urlfile, 'r', encoding='utf-8') as f:
#        html_content = f.read()
#except FileNotFoundError:
#    print(f"Error: The file '{urlfile}' was not found.")
#except Exception as e:
#    print(f"An error occurred: {e}")
#tableshtml = pd.read_html(urlfile)
#print(len(urlfile))
# print(len(tableshtml))
# world_topten = tableshtml[4]
# print(world_topten.iloc[1:1, 0:1])
#world_topten = world_topten['World population (millions, UN estimates)[15]'] 
#world_topten = world_topten.drop(11,axis=0) # Clean up a bit
#world_topten = world_topten.drop('#',axis=1)
#print(world_topten.columns = ['Region','2022 (percent)','2030 (percent)','2050 (percent)'])
#world_topten
tableshtml2 = pd.read_html(urlfile2)
print(len(tableshtml2)) # only 1
world_top_ten = tableshtml2[0] # if more than one tables could do tableshtml2[1]
world_top_ten = world_top_ten.drop('#',axis=1) # DROP that # Column
# world_top_ten.columns = ['Country', 'Pop. 25'...] # RENAME Column
# world_top_ten.to_html('WORLD TOP TEN BY POP.html', index=False)
print(world_top_ten)

# PANDAS IO - EXCEL
# import pandas as pd 
# import openpyxl
# import xlrdc # python-excel.org
# dfexcel = pd.read_excel('Excel_Sample.xlsx', sheet_name='Sheet1') # sheet_name='Sheet2'
# workbook = pd.ExcelFile('Excel_Sample.xlsx')
# workbook.sheet_names # ['Sheet1', 'Sheet2']
# excelsheetdic = pd.read_excel('myexcelfile.xlsx', sheet_name=None)
# type(excelsheetdic) # dict
# excelsheetdic.keys() # dict_keys(['Sheet1', 'Sheet2'])
# df1 = excelsheetdic['Sheet1']
# dfl.to_excel('example.xlsx', sheet_name='Sheet1')

# PANDAS IO - SQL
# import pandas as pd 
# from sqlalchemy import create_engine # https://www.sqlalchemy.org/ pip3 install sqlalchemy
# Use sqlalchemy to connect to your SQL database with the driver:
# docs.sqlaclchemy.rog/en/13/dialects/index.html
# Use the sqlalchemy driver connection with pandas read_sql method
# Pandas can read in entire tables as a DataFrame or pars a SQL query through SELECT * FROM table;
temp_sqlldb = create_engine('sqlite:///:memory:') # TEMP DB IN MEMORY
sqldf = pd.DataFrame(data=np.random.randint(low=0,high=100,size=(4,4)),columns=['A','B','C','D'],index=['0','1','2','3'])
sqldf.to_sql(name='sqltable', con=temp_sqlldb, index=False) # CREATE TABLE mytable
print(sqldf,'\n') 
sqldf2 = pd.read_sql(sql='sqltable', con=temp_sqlldb) # READ TABLE
print(sqldf2,'\n')
sqldf3 = pd.read_sql_query(sql='SELECT a,c FROM sqltable', con=temp_sqlldb) # SQL QUERY
print(sqldf3,'\n')

# EXERCISES PANDAS IO - SQL
## 1. import neccessary libraries
# import pandas as pd
## 2. List all columns in the two datasets. Additionally retrieve the datatype for each column.
#for col, dtype in .df1.dtypes.items(): print(f'{col}: {dtype}') # OR
consituents = pd.read_csv('constituents.csv')
financials = pd.read_csv('constituents-financials.csv')
#print(consituents.columns, financials.columns) # OR (BETTER)
consituents.info(),financials.info(), consituents.head()
## 3. Print first 5 rows of each dataframe
print('\n',consituents.head(), financials.head()) # OR consituents.head(5), financials.head(5)
## 4. Drop the SEC Filings column from the financials dataset 
financials = financials.drop('SEC Filings', axis=1) # DROP SEC FILINGS COLUMN
print('\n',financials) 
## 5. Set the Symbol column to the index in the financials dataset
print('\nSet the Symbol column to the index')
financials = financials.set_index('Symbol')
print(financials)
## 6. What are the 10 largest companies according to the market cap?
print('\n10 largest companies according to the market cap')
print(financials['Market Cap'].nlargest(10)) # OR financials.sort_values('Market Cap', ascending=False).head(10)
## 7. DROP GOOG - Difference between GOOG and GOOGL is that GOOG shares have no voting rights
financials.drop('GOOG', axis=0) # DROP GOOG OR financials.drop('GOOG', axis=0, inplace=True) for permanent
## 8. How many companies have a dividend yield > 4%
print('\nCompanies that have a dividend yield > 4%')
print(len(financials[financials['Dividend Yield'] > 4])) # OR
## 9.What is the mean Earnings per Share for all companies with a market cap > 1e+11
print('\nMean Earnings per Share for all companies with a market cap > 1e+11')
print(financials[financials['Market Cap']>1e+11]['Earnings/Share'].mean())
## 10. How many companies have a positive earnings per shares ratio?
print('\nCompanies that have a positive earnings per shares ratio?')
print(financials['Price/Earnings'] > 0) # OR print(financials['Price/Earnings'] > 0).sum() # OR
print('\n',len(financials[financials['Price/Earnings'] > 0])) # OR print((financials['Price/Earnings'] > 0).sum())
## 11 Which company pays the highest dividend yield? What was its 52 week high?
print('\nCompany that pays the highest dividend yield and its 52 week high')
print(financials['Dividend Yield'].idxmax()) #financials['Dividend Yield'].max()) 
print(consituents[consituents['Symbol'] == 'CTL'])
print('\nFinancials.describe()\n',financials.describe())
print ('\nCompanies with the largest spread betwen 52 Week High and Low')
print(abs(financials['52 Week High'] - financials['52 Week Low']).idxmax())
print('\nReturn all companies with a price between $50 and $100')
print(financials['Price'].between(50,100))
print('\nCreate a column called "Market Cap in Billion')
financials['Market Cap in Billion'] = financials['Market Cap'] / 1e+9
print(financials['Market Cap in Billion'])
print('\nIs there a correlation between Market Cap and Dividend Yield')
print(financials[['Market Cap','Dividend Yield']].corr())
print('\nMerge the financials dataframe with the constituents dataframe')
merged = pd.merge(financials, consituents, on='Symbol', suffixes=('_fin', '_const'))
print(merged)
print('\nHow often does each sector appear in the dataset')
print(merged['Sector'].value_counts()) # OR merged['Sector'].unique()
print('\nReplace Information Technology with IT')
itmap = {'Information Technology':'IT'}
print('\nmerged[\'Sector\'].map(itmap)\n',merged['Sector'].replace(itmap))#,inplace=True)
print('\nAdd a $ before the stock price',merged['Price'].apply(lambda x: f'${x}'))


# MATPLOTLIB - ################################################################################################# 
# plotting with Python https://matplotlib.org https://matplotlib.org/gallery.html
# Ability to heavily customize a plot, 2 separate approaches to creating plots, functional and OOP
# Two main goals - plot out a functional relationship - y=2x, also relationship bet two points x=[1,2,3], y=[2,4,6]

# MATPLOTLIB - BASICS
# Difference between displaying plots w/in a notebook vs running a script (need to add plt.show())
# import matplotlib.pyplot as plt
# import numpy as np
xnp = np.arange(0,10)
ynp = 2*xnp
print('\nxnp np.arange(0,10) = ',xnp)
print('\nynp = 2*xnp',ynp)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.xlim(0,6) # LIMIT X ONLY SHOWS 0-6 ON X CAN ALSO DO YLIM
plt.title('Simple Plot')
print('\nplt.plot(xnp,ynp)', plt.plot(xnp,ynp))
plt.savefig('simpleplot.png') # SAVE FIGURE AS PNG FILE
plt.show() # NEED THIS LINE IF RUNNING AS SCRIPT TO DISPLAY FIGURE

# MATPLOTLIB UNDERSTANDING THE FIGURE OBJECT - add axes and then plot on those axes.
# import numpy as np
# import matplotlib.pyplot as plt
# DATA FOR PLOTTING
a = np.linspace(0,10,11) # 11 linearly spaced points between 0-10
b = a**4
print(' a=np.linspace(0,10,11):',a,'\n','b=a**4:',b)
# plt.figure() # <Figure size 640x480 with 0 Axes>
# plt.figure(figsize=(10,10)) # FIGURE SIZE IN INCHES WIDTH, HEIGHT
# plt.show() # NEED THIS LINE IF RUNNING AS SCRIPT TO DISPLAY FIGURE EMPTY AS OF NOW
fig = plt.figure()
print('\nfig = plt.figure()',fig)
# LARGE AXES
axes1 = fig.add_axes([0,0,1,1]) # 0,0 = lower left corner, 1,1 = width, height
axes1.plot(a,b) # Add axis, can add more add; to get rid
axes1.set_xlim(0,8)
axes1.set_ylim(0,8000)
axes1.set_xlabel('A')
axes1.set_ylabel('B')
axes1.set_title('LARGE PLOT B = A^4')
# SMALL AXES
axes2 = fig.add_axes([0.2,0.5,0.25,0.25])
axes2.plot(a,b)
axes2.set_xlim(1,2)
axes2.set_ylim(0,50)
axes2.set_xlabel('A')
axes2.set_ylabel('B')
axes2.set_title('ZOOMED IN 1 TO 2')
plt.show()
xnparr = np.arange(0,10) # spaced out by 1
print("\nxnparr = np.arange(0,10):", xnparr) 
pri

# MATPLOTLIB - FIGURE PARAMETERS
a = np.linspace(0,10,11) # 11 linearly spaced points between 0-10
b = a**4
fig2 = plt.figure(figsize=(2,2),dpi=100) # dpi = dots per inch
axesf2 = fig2.add_axes([0,0,1,1])
axesf2.plot(a,b)
fig2.savefig('Figure Parameters.png', bbox_inches='tight') # SAVE FIGURE AS PNG FILE
plt.show()

# MATPLOTLIB - SUBPLOTS FUNCTIONALITY
# Matplotlib comes with a pre-confitgured fuc call plt.subplots()
# plt.subplots() returns a tuple (fig,axes) containing the figure canvas and then
# a numpy array holding the axes objects. fig = entrie canvas, axes = numpy array of axes by pos
# import numpy as np
# import matplotlib.pyplot as plt
asp = np.linspace(0,10,11)
bsp = asp ** 4
xsp = np.arange(0,10)
ysp = 2 * xsp
fig3, axes3 = plt.subplots(nrows=1,ncols=2) # 1 row and 2 columns
# plt.show()
print(type(axes3))
print('axes3.shape: ',axes3.shape)
print(axes3)
#axes3[0].plot(xsp,ysp)
# plt.show()
#axes3[1].plot(asp,bsp)
# plt.show()
fig3, axes3 = plt.subplots(nrows=2,ncols=2) # Now 2 rows and 2 columns
print('axes3.shape: ',axes3.shape)
#axes3[0][0].plot(xsp,ysp)
#axes3[0][1].plot(asp,bsp)
axes3[1][1].plot(asp,bsp)
axes3[1][1].set_title('TITLE 1,1')
fig3.suptitle('Figure Level', fontsize=12)
plt.tight_layout() # ADJUST SPACING BETWEEN PLOTS OR fig.subplots_adjust(hspace=0.5)
plt.show()
fig3.savefig('Subplots Functionality.png', bbox_inches='tight') # SAVE FIGURE AS PNG FILE
# fig3.set_figwidth(10) # SOME ADDITIONAL OPTIONS OR figsize=(4,10),dpi=200)

# MATPLOTLIB - LEGENDS 
# Visual Styling, colors, editing lines, markers
xfll = np.linspace(0,11,10)
figl = plt.figure()
ax = figl.add_axes([0,0,1,1])
ax.plot(xfll, xfll, label='xfll vs xfll')
ax.plot(xfll,xfll**2, label='xfill vs xfll^2')
ax.legend(loc='lower right') # loc=0 is best location, can also do 'upper left' etc
figl.savefig('Matplotlib Legends.png', bbox_inches='tight') # SAVE FIGURE AS
plt.show()

# MATPLOTLIB - COLOR & STYLES Google hex color picker
# import numpy as np
# import matplotlib.pyplot as plt
xstyl = np.linspace(0,11,10)
figstl = plt.figure()
# LINES
axesstl = figstl.add_axes([0,0,1,1])
axesstl.plot(xstyl, xstyl, label='x v x', lw=10, linestyle=':') # can use linewidth=lw
axesstl.plot(xstyl, xstyl+2, color='green', label='x v x+2') # can use hex '#008000'
axesstl.legend()
lines = axesstl.plot(xstyl,xstyl+.5,color='purple', label='x v x+1', linewidth=2, marker='o',\
                     markerfacecolor='red', markeredgewidth=6,markeredgecolor='orange')
#lines[0].set_dashes([5,2,10,2]) # 5 on, 2 off, 10 on, 2 off 
lines[0].set_dashes([1,2,1,2,10,2]) # 
axesstl.plot(xstyl,xstyl+1,color='#FF5733', label='x v x+1', lw=2, marker='^', markersize=10)
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS
# LOGARITHMIC SCALE
# Define the x variable for plotting
x = np.linspace(0.1, 10, 100) # Example x values, adjust as needed
# Create the figure and subplots
fig, axeslog = plt.subplots(1, 2, figsize=(10,4))
# Plot on the first subplot with normal scale
axeslog[0].plot(x,x**2,x,np.exp(x),label='x^2')
axeslog[0].set_title('Normal Scale')
axeslog[0].legend() # Add legend to distinguish lines
axeslog[1].plot(x,x**2,x,np.exp(x),label='exp(x)')
axeslog[1].set_yscale("log")
axeslog[1].set_title('Log Scale (y)')
axeslog[1].legend() # Add legend to distinguish lines
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS CUSTOMIZE PLACEMENT OF TICKS AND CUSTIM TICK LABELS
x = np.linspace(0.1, 10, 100) # Example x values, adjust as needed
fig, axestick = plt.subplots(figsize=(10,4))
axestick.plot(x,x**2,x,x**3,lw=2)
axestick.set_xticks([1,2,3,4,5])
axestick.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'])
yticks = [0,50,100,150]
axestick.set_yticks(yticks)
axestick.set_yticklabels(['$%.1f$' % y for y in yticks], fontsize=18)
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS SCIENTIFIC NOTATION
# from matplotlib import ticker
x = np.linspace(.001, 5, 10) # Example x values, adjust as needed
fig, axsn = plt.subplots(1,1)
##axsn.plot(x, x**2, x, np.exp(x),) # It plots two separate lines on the same axes:
lines = axsn.plot(x, x**2, x, np.exp(x),) # Plot both lines with one call and capture the artists
axsn.set_title('Scientific Notation')
axsn.set_yticks([0,50,100,150])
##axsn.legend()
axsn.legend(lines, ['$x^2$', 'e$^x$']) # Pass the captured lines and labels to the legend in a single line
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
axsn.yaxis.set_major_formatter(formatter)
plt.show()  

# MATHPLOTLIB - ADVANCED COMMANDS AXIS NUMBER AND AXIS LABEL SPACING ADJUSTMENTS
x = np.linspace(0.1, 5, 100) # Example x values, adjust as needed
plt.rcParams['xtick.major.pad'] = 5
plt.rcParams['ytick.major.pad'] = 5
fig, ax = plt.subplots(1,1)
ax.plot(x, x**2, x, np.exp(x))
ax.set_yticks([0,50,100,150])
ax.set_title('Label and Axis spacing')
## Padding between axis label and axis numbers
ax.xaxis.labelpad = 5
ax.yaxis.labelpad = 5
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
# Adjustments
fig.subplots_adjust(left=0.15, right=.9, bottom=0.1, top=0.0) 
# Restore defaults
##matplotlib.rcParams['xtick.major.pad'] = 3
##matplotlib.rcParams['ytick.major.pad'] = 3

# MATHPLOTLIB - ADVANCED COMMANDS AXIS GRID turn on/off and customize grid appearance 
# http://www.matplotlib.org/gallery.html
x = np.linspace(0.1, 5, 100)
fig, axgrid = plt.subplots(1,2, figsize=(10,3))
# Default Grid Appearance
axgrid[0].plot(x, x**2, x, x**3, lw=2)
axgrid[0].grid(True)
axgrid[0].set_title('Default Grid')
# Custom Grid Appearance
axgrid[1].plot(x, x**2, x, x**3, lw=2)
axgrid[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
axgrid[1].set_title('Custom Grid')
plt.show()
## #Default Grid Appearance
##axes[0].plot(x, x**2, x, x**3. lw=2)
##axes[0].grid(True)

## MATHPLOTLIB - ADVANCED COMMANDS AXIS SPINES
x = np.linspace(0.1, 5, 100)
fig, axspines = plt.subplots(figsize=(6,2))
axspines.spines['bottom'].set_color('blue')
axspines.spines['top'].set_color('blue')
axspines.spines['left'].set_color('red')
axspines.spines['left'].set_linewidth(2)
# Turn off right spine
axspines.spines['right'].set_color('none')
axspines.yaxis.tick_left() # Only ticks on the left side
plt.show()

## MATHPLOTLIB - ADVANCED COMMANDS TWIN AXES
x = np.linspace(0.1, 5, 100)
fig, axtwin = plt.subplots()
axtwin.plot(x, x**2, lw=2,color='blue')
axtwin.set_ylabel(r'area $(m^2)$', color='blue', fontsize=18)
for label in axtwin.get_yticklabels():
    label.set_color('blue')
axtwin2 = axtwin.twinx()
axtwin2.plot(x, x**3, lw=2,color='red')
axtwin2.set_ylabel(r'volume $(m^3)$', color='red', fontsize=18)
for label in axtwin2.get_yticklabels():
    label.set_color('red')
plt.show()

## MATHPLOTLIB - ADVANCED COMMANDS AXES WHERE X AND Y IS ZERO
xx = np.linspace(-0.75, 1., 100)
fig, axxyzero = plt.subplots()
axxyzero.spines['right'].set_color('none')
axxyzero.spines['top'].set_color('none')
axxyzero.xaxis.set_ticks_position('bottom')
axxyzero.spines['bottom'].set_position(('data',0)) # Set position of x spine to x=0
axxyzero.yaxis.set_ticks_position('left')
axxyzero.spines['left'].set_position(('data',0)) # Set position of y spine to y=0
xx = np.linspace(-0.75, 1., 100)
axxyzero.plot(xx, xx**3)
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS OTHER 2D PLOT STYLES
xx = np.linspace(-0.75, 1., 100)
n = np.array([0,1,2,3,4,5])
fig, axes = plt.subplots(1,4, figsize=(12, 3))
axes[0].scatter(xx, xx + 0.25*np.random.rand(len(xx)))
axes[0].set_title('Scatter')
axes[1].step(n, n**2, lw = 2)
axes[1].set_title('Step')
axes[2].bar(n, n**2, align='center', width=0.5, alpha=0.5)
axes[2].set_title('Bar')
axes[3].fill_between(xx, xx**2, xx**3, color='green', alpha=0.5)
axes[3].set_title('fill_between')
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS TEXT ANNOTATION
xx = np.linspace(-0.75, 1., 100)
fig, xt = plt.subplots()
xt.plot(xx, xx**2, xx, xx**3)
# set x and y limits
xt.set_xlim(-0.8, 1.0)
xt.set_ylim(-0.5, 1.0)
xt.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")
xt.text(0.05, 0.1, r"$y=x^3$", fontsize=20, color="green")
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS MULTIPLE SUBPLOTS AND INSETS
# Axes can be added to a matplotlib Figure canvas manually using fig.add_axes
# or using a sub-figure layout manager - subplots, subplot2grid or gridspec.
xx = np.linspace(.1, 1.0, 5)
fig, ax = plt.subplots(2, 3)
# show x-axis from 0.0 to 1.0 with ticks every 0.2 on all subplots
xticks = np.arange(0.0, 1.01, 0.2)
for a in ax.flat:
    a.set_xlim(0.0, 1.0)
    a.set_xticks(xticks)
fig.tight_layout()
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS SUBPLOT2GRID
xsp = np.linspace(0.0, 1.01, 5)
fig = plt.figure()
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
ax4 = plt.subplot2grid((3,3), (2,0))
ax5 = plt.subplot2grid((3,3), (2,1))
fig.tight_layout()
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS GRIDSPEC
# from matplotlib import gridspec, ticker
xsp = np.linspace(0.0, 1.01, 5)
fig = plt.figure()
gs = gridspec.GridSpec(2, 3, height_ratios=[2,1], width_ratios=[1, 2, 1])
for g in gs:
    ax = fig.add_subplot(g)
fig.tight_layout()
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS ADD AXES
xax = np.linspace(0.0, 1.01, 5)
fig, ax = plt.subplots()
ax.plot(xax, xax**2, xax, xax**3)
fig.tight_layout()
# INSET
inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35]) # X,Y,Width,Height
inset_ax.plot(xax, xax**2, xax, xax**3)
inset_ax.set_title('zoom near origin')
# SET AXIS RANGE
inset_ax.set_xlim(-.2, .2)
inset_ax.set_ylim(-.005, .01)
# SET AXIS TICK LOCATIONS
inset_ax.set_yticks([0, 0.005, 0.01])
inset_ax.set_xticks([0-.1,0,.1])
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS  COLORMAPS AND CONTOUR FIGURES https://www.scipy.org/Cookbook/Matplotlib/Show_colormaps
alpha = 0.7
phi_ext = 2 * np.pi * 0.5
def flux_qubit_potential(phi_m, phi_p):
    return 2 + alpha - 2 *np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext)
phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T
fig, ax = plt.subplots()
# use pcolormesh with shading='auto'; set symmetric vmin/vmax for 
# #RdBu and add a colorbar
vmax = np.abs(Z).max()
p = ax.pcolor(X/(2*np.pi), Y/(2*np.pi), Z, cmap=plt.cm.RdBu, vmin=-vmax, vmax=vmax)#vmin=abs(Z))
# p = ax.pcolormesh(X/(2*np.pi), Y/(2*np.pi), Z, cmap=plt.cm.RdBu, shading='auto',
#                   vmin=-vmax, vmax=vmax)
fig.colorbar(p, ax=ax)
ax.set_xlabel(r'$\phi_p / 2\pi$')
ax.set_ylabel(r'$\phi_m / 2\pi$')
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS IMSHOW
fig, ax = plt.subplots()
vmax = np.abs(Z).max()
im = ax.imshow(Z, cmap=plt.cm.RdBu, vmin=-vmax, vmax=vmax, \
origin='lower', extent=(X.min()/(2*np.pi), X.max()/(2*np.pi), \
                        Y.min()/(2*np.pi), Y.max()/(2*np.pi)),
    interpolation='bilinear'
)
cb = fig.colorbar(im, ax=ax)
ax.set_xlabel(r'$\phi_p / 2\pi$')
ax.set_ylabel(r'$\phi_m / 2\pi$')
plt.show()

# MATHPLOTLIB - ADVANCED COMMANDS CREATE A FIGURE
fig = plt.figure()
ax = fig.add_subplot()
grams = np.arange(0, 11)
print('\nxnp np.arange(0,10) = ',xnp)
print('\nynp = 2*xnp',ynp)
plt.xlabel('X axis')
plt.ylabel('Y axis')
###vmin=abs(Z).min(). vmax=abs(Z.max(),

## MATHPLOTLIB - EXERCISES 
## TASK 1
# 1. CREATING DATA FROM AN EQUATION
# E = mc(squared) 
# Create two arrays: E and m, m is simply 11 evenly spaced values representing
# 0-10 grams. E should be the equivalent energy for the mass. Figure out what to
# provide for c for the units m/s c= 3.00x10^8 186,000 miles per second or 299,792,458 meters p/s.
# import numpy as np
m = np.linspace(0,10,11)
c = 3 * 10**8
E = m*c**2
print(E)

# 2. PLOTTING E=MC^2
# 1. Create a figure and axis object using plt.subplots() with 1 row and
# 2 columns. Plot the following data on each axis:
# import matplotlib.pyplot as plt
plt.title('E = mc^2')
plt.xlabel('M in grams')
plt.ylabel('E in Joules')
plt.plot(m,E,color='red',lw=5)
plt.xlim(0,10)
plt.show()
# OR
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(m,E,color='red',lw=5)
#plt.yscale('log') # WAY TO DO BONUS BELOW
#plot.grid(which='both',axis='y') # WAY TO DO BONUS BELOW
plt.show()

#3 BONUS Figure out how to plot this on a logarthimic scale on the y axis. Grid on ytikcksplt.title('E = mc^2')
plt.title('E = mc^2')
plt.xlabel('M in grams')
plt.ylabel('E in Joules')
plt.plot(m,E,color='red',lw=5)
plt.xlim(0,10)
plt.yscale('log')
plt.grid
plt.grid(which='both',axis='y')
plt.show()

## TASK 2
# CREATING TWO PLOTS FROM DATA POINTS
# In finance, the yield curve is a curve showing several yields to maturity or interest rates across
# different contract lengths (2 months, 2 year, 20 year, etc...) for a similar debt contract.  The 
# curve shows the relation between the (level of the) interest rate (or cost of borrowing) and time to
# maturity, known as the "term", of the debt for a given borrower in a given currency.
# The US dollar interest rates paid on US Treasury securities for various maturities are closely watched
# by many traders, and are commonly plotted on a graph called "the yield cruve"
# The data shows the interest paid for a US Treasury bond for a certain contract lengtrh. 
# The lable lists shows the corresponding contract length per index position.
# Run the cell below to create the lists for plotting.
# Plot both curves on the same Figure. Add a legend to show which curve corresponds to a certain year.
labels = ['1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
july16_2007 = [4.75,4.98,5.08,5.01,4.89,4.89,4.95,4.99,5.05,5.21,5.14]
july16_2020 = [0.12,0.11,0.13,0.14,0.16,0.17,0.28,0.46,0.62,1.09,1.31]
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(labels,july16_2007,label='july16_2007')
axes.plot(labels,july16_2020,label='july16_2020')
axes.legend() # or plt.legend() MOVE IT OUTSIDE PLOT axes.legend(loc=(1.04,0.5)) 
plt.show()
# USE SUBPLOTS TO SHWO EACH YEAR'S YIELD CURVE
fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(12,8))
axes[0].plot(labels,july16_2007,label='july16_2007')
axes[0].set_title('july16_2007')
axes[1].plot(labels,july16_2020,label='july16_2020')
axes[1].set_title('july16_2020')
plt.show()

## TASK 3
# Recreate the plot below that uses twin axes.
labels = ['1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
july16_2007 = [4.75,4.98,5.08,5.01,4.89,4.89,4.95,4.99,5.05,5.21,5.14]
july16_2020 = [0.12,0.11,0.13,0.14,0.16,0.17,0.28,0.46,0.62,1.09,1.31]
fig, ax1 = plt.subplots() # sublots(figsize=(6,2))
ax1.spines['left'].set_color('blue')
ax1.spines['left'].set_linewidth(4)
ax1.spines['right'].set_color('red')
ax1.spines['right'].set_linewidth(2)
ax1.plot(labels,july16_2007, lw=2, color='blue')
ax1.set_ylabel('July 16 2007', fontsize=18, color='blue')
for label in ax1.get_yticklabels():
    label.set_color('blue')
ax2 = ax1.twinx()
ax2.plot(labels,july16_2020, lw=2, color='red')
ax2.set_ylabel('July 16 2020', fontsize=18, color='red') 
for label in ax2.get_yticklabels():
    label.set_color('red')
plt.show()


# PANDAS AND FINANCE ##############################################################
# Core Pandas time Methods, Visualizations, Time Series, Rolling Statistics
# Time shifts and Row calculations, Data Sources

# CORE PANDAS TIME METHODS
# Allow us to extract info from the timestamp, such as: DOW WE vs WD AM vs PM
# from datetime import datetime
# import pandas as pd
# import numpy as np
myyear = 2025
mymonth = 1
myday = 1
myhour = 2
mymin = 30
mysec = 15
mydate = datetime(myyear, mymonth, myday)
print(mydate)
mydatetime = datetime(myyear, mymonth, myday, myhour, mymin, mysec)
print(mydatetime)
print(mydatetime.year)

#myser_dt = pd.Series(['Nov 3, 1990', '2000-01-01', None], format="ISO8601", utc=True)
myser = pd.Series(['Nov 3, 1990','2000-01-01',None])
print("pd.Series([\'Nov 3, 1990\', \'2000-01-01\',None], print(myser):")
print(myser)
# print('\ntimeseries = pd.to_datetime(myser,format=\'mixed)')
# Robust parsing: let pandas infer mixed formats, be tolerant to failures, and set UTC
print("\ntimeseries = pd.to_datetime(myser, errors='coerce', utc=True, infer_datetime_format=True)")
# timeseries = pd.to_datetime(myser, format='mixed')
timeseries = pd.to_datetime(myser, errors='coerce', utc=True, infer_datetime_format=True)
print(timeseries)
# Fallback: if any values are NaT, try a second pass with an explicit common format
mask = timeseries.isna() & myser.notna()
if mask.any():
    print("\nFallback parsing for entries that failed inference:")
    fallback = pd.to_datetime(myser[mask], format='%b %d, %Y', errors='coerce', utc=True)
    timeseries.loc[mask] = fallback
    print(timeseries)
print('\ntimeseries[0].year =', timeseries[0].year)
print('\nobvi_euro_date = \'31-12-2000\'')
obvi_euro_date='31-12-2000'
# Example of European-style date parsing — specify dayfirst to avoid warnings
print('pd.to_datetime(obvi_euro_date, dayfirst=True):')
print(pd.to_datetime(obvi_euro_date, dayfirst=True))
# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
print('\nstyle_date = \'12--Dec--2000\''), 
style_date = '12--Dec--2000'
print("pd.to_datetime(style_date,format=\'%d--%b--%Y:")
print(pd.to_datetime(style_date,format='%d--%b--%Y'))
custom_date = '12th of Dec 2000'
print('\ncustom_date = \'12th of Dec 2000\', pd.to_datetime(custom_date):')
print(pd.to_datetime(custom_date),'\n')

# READ FROM CSV
sales_path = 'PWQuantConnect/03-Core-Pandas/RetailSales_BeerWineLiquor.csv'
try:
    sales = pd.read_csv(sales_path)
except FileNotFoundError:
    print(f"Error: The file '{sales_path}' was not found.")
except Exception as e:
     print(f"An error occurred: {e}")
print(sales.head(),'\n')
print('sales[\'DATE\'] = pd.to_datetime(sales["DATE"])')
sales['DATE'] = pd.to_datetime(sales["DATE"])
print('sales[\'DATE\'][0].year:',sales['DATE'][0].year)
print('\nsales = pd.read_csv(sales_path, parse_dates=[0])')
sales = pd.read_csv(sales_path, parse_dates=[0])
print('sales[\'DATE\']:')
print(sales['DATE'])
print("\nsales = sales.set_index(\'DATE\'), sales=")
sales = sales.set_index('DATE')
print(sales)
print('\nsales.resample(rule=\'YE\').mean() =')
ruleYE = sales.resample(rule='YE').mean()
print(ruleYE)
print('\nsales = pd.read_csv(sales_path, parse_dates=[0])')
sales = pd.read_csv(sales_path, parse_dates=[0])
print('sales.head() and info():\n', sales.head())
sales.info()
print('\nsales[\'DATE\'].dt.year:')
print(sales['DATE'].dt.year) # coulddo dt.year, etc

# PANDAS VISUALIZATIONS
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# READ FROM CSV
fb_path = 'PWQuantConnect/DATA/FB.csv'
try:
    fb = pd.read_csv(fb_path)
except FileNotFoundError:
    print(f"Error: The file '{fb_path} was not found.")
except Exception as e:
    print(f"An error occured: {e}")
# print(fb,'\n')
fb.plot()
plt.show()
fb[['Adj Close','High']].plot()
plt.show()
fb[['Adj Close','High']].plot(kind='box')
plt.show()
fb['Volume'].plot(kind='hist',bins=100)
plt.show()
fb['Volume'].plot(kind='kde',c='red')
plt.show()
plt.figure(figsize=(10,3),dpi=150)
fb['Volume'].plot(kind='line',c='red')
plt.show()
new_fb_path = 'PWQuantConnect/DATA/TGT.csv'
try:
    new_fb = pd.read_csv(new_fb_path)
except FileNotFoundError:
    print(f"Error: The file '{new_fb_path} was not found.")
except Exception as e:
    print(f"An error occured: {e}")
new_fb['Adj Close'].plot(label='TGT')
fb['Adj Close'].plot(label='FB')
plt.legend(loc=(0.4,1))
plt.show()
# MAKE ON THE SAME GRAPH
ax = new_fb['Adj Close'].plot(label='TGT')
fb[['Adj Close','High']].plot(ax=ax)
plt.legend()
plt.show()
fig,ax = plt.subplots(dpi=75)
new_fb['Adj Close'].plot(ax=ax,ls='--',color='red',lw=3)
fb[['Adj Close','High']].plot(ax=ax)
plt.xlabel('PLT LABEL')
plt.savefig('TGT plots.png')
plt.show()


# VISUALIZING TIME SERIES DATA WITH PANDAS - PART 1
# import pandas as pd
# import matplotlib.pyplot as plt
cost_path = 'PWQuantConnect/DATA/COST.csv'
try:
    cost_fb = pd.read_csv(cost_path,index_col='Date')
except FileNotFoundError:
    print(f"Error: The file '{cost_path} was not found.")
except Exception as e:
    print(f"An error occured: {e}")
print('cost_fb = pd.read_csv(cost_path,index_col=\'Date\')\n',cost_fb,'\n')
print('cost_fb.index:\n',cost_fb.index)
cost_fb = pd.read_csv(cost_path,index_col='Date',parse_dates=True)
print('\ncost_fb.index_col=\'Date\',parse_dates=True:\n',cost_fb)
print('\nAdd index keyword = cost_fb.index:\n',cost_fb.index)
print('\ncost_fb.plot\nplt.show():\n')
cost_fb.plot()
plt.show()
print('Only plot the Adj Close = cost_fb[\'Adj Close\'].plot()):')
cost_fb['Adj Close'].plot()
plt.show()
#print('cost_fb[\'Adj Close\'][\'2018-01-01\']:',cost_fb['Adj Close']['2018-01-01'])
print('cost_fb[\'Adj Close\'][\'2018-01-01\':\'2020-01-01\'].plot():')
print(cost_fb['Adj Close']['2018-01-01':'2020-01-01'].plot())
plt.show()
print('OR')
print('cost_fb[\'Adj Close\'].plot(xlim=[\'2018-01-01\', \'2020-01-01\'])\n')
print(cost_fb['Adj Close'].plot(xlim=['2018-01-01', '2020-01-01'],ylim=[150,290]))
plt.show()

# SHOW TICKS - locator (Set location of ticks) and formatter (Dates have ++choices)
# from matplotlib import dates
cost_path = 'PWQuantConnect/DATA/COST.csv'
cost_fb = pd.read_csv(cost_path,index_col='Date',parse_dates=True)
plt.figure(dpi=100)
# LOCATOR
print('aticks = cost_fb[\'Close\'][\'2018-01-01\':\'2018-03-01\'].plot():')
aticks = cost_fb['Close']['2018-01-01':'2018-03-01'].plot()
print('\naticks.xaxis.set_major_locator(dates.WeekdayLocator())') # Can specify Weekday
aticks.xaxis.set_major_locator(dates.WeekdayLocator()) # Can be MonthlyLocator, etc.
aticks.xaxis.set_major_locator(dates.YearLocator())
aticks.xaxis.set_major_locator(dates.MonthLocator())
plt.show()
# FORMATTER
aticks.xaxis.set_major_formatter(dates.DateFormatter('%Y-%B'))
plt.show()


# VISUALIZING TIME SERIES DATA WITH PANDAS - PART 2
cost_path = 'PWQuantConnect/DATA/COST.csv'
try:
    cost_vd = pd.read_csv(cost_path,index_col='Date',parse_dates=True)
except FileNotFoundError:
    print(f"Error: The file '{cost_path} was not found.")
except Exception as e:
    print(f"An error occured: {e}")
print('vtsd = cost_vd[\'Close\'][\'2018-01-01\':\'2020-01-01\'].plot():')
vtsd = cost_vd['Close']['2018-01-01':'2020-01-01']
print('\nplt.figure(figsize=(8,4),dpi=100)',plt.figure(figsize=(9,6),dpi=60))
print('\nvts = vtsd[\'Close\'][\'2018-01-01\':\'2020-01-01\'].plot()\n')
atx = cost_vd['Close']['2018-01-01':'2020-01-01'].plot()
atx.xaxis.set_major_locator(dates.YearLocator())
atx.xaxis.set_major_formatter(dates.DateFormatter('%Y,   %B'))
atx.xaxis.set_minor_locator(dates.MonthLocator())
atx.xaxis.set_minor_formatter(dates.DateFormatter('%B'))
atx.tick_params(axis='x',which='minor',rotation=90)
atx.tick_params(axis='x',which='major',rotation=90)
atx.yaxis.grid(True)
plt.xticks(ha='center')
plt.show()


# PANDAS - ROLLING STATISTICS
# Allow us toget info over a 'window' of time, the window moves along
# with thedataset, allowing us to see a moving or 'rolling' statistic
# import  pandas as pd
cost_wmtpath = 'PWQuantConnect/DATA/WMT.csv'
try:
    cost_vd = pd.read_csv(cost_wmtpath,index_col='Date',parse_dates=True)
except FileNotFoundError:
    print(f"Error: The file '{cost_wmtpath} was not found.")
except Exception as e:
    print(f"An error occured: {e}")
plt.figure(figsize=(10,3),dpi=100)
cost_vd['Adj Close'].plot()
plt.show()
# Rolling average with a 7-day window = 3 (Days)
plt.figure(figsize=(10,3),dpi=100)
print('cost_vd[\'Adj Close\'].rolling(window=3).mean().plot()\n')
cost_vd['Adj Close'].rolling(window=3).mean().plot()
plt.show()
# Now a larger set
plt.figure(figsize=(10,3),dpi=100)
print('cost_vd[\'Adj Close\'].rolling(window=200).mean().plot()\n')
cost_vd['Adj Close'].plot(label='Adj Close')
cost_vd['Adj Close'].rolling(window=200).mean().plot(label='200 MA')
plt.legend()
plt.show()
plt.figure(figsize=(10,3),dpi=100)
print('cost_vd[\'Adj Close\'].plot(labels=\'Adj Close\')')
cost_vd['Adj Close'].plot(label='Adj Close')
print('cost_vd[\'Adj Close\'].rolling(window=14).std().plot(label=\'14 std\')')
cost_vd['Adj Close'].rolling(window=14).std().plot(label='14 std')
plt.legend()
plt.show()
print('cost_vd[\'Adj Close\'].rolling(window=3).std().plot(label=\'14 std\')')
cost_vd['Adj Close'].rolling(window=3).std().plot(label='14 std')
plt.legend()
plt.show()



# PANDAS TIME SHIFTING AND ROW CALCULATIONS
cost_wmt2path = 'PWQuantConnect/DATA/WMT.csv'
try:
    cost_rowcalc = pd.read_csv(cost_wmt2path,index_col='Date',parse_dates=True)
except FileNotFoundError:
    print(f"Error: The file '{cost_wmt2path} was not found.")
except Exception as e:
    print(f"An error occured: {e}")
plt.figure(figsize=(10,3),dpi=100)
print('\ncost_rowcalc.head()',cost_rowcalc.head())
# SHIFTING BY FORWARD OR BACK ONE DAY
print('\ncost_rowcalc.shift(1).head()',cost_rowcalc.shift(1).head())
print('\ncost_rowcalc.shift().tail()',cost_rowcalc.shift().tail())
print('\ncost_rowcalc.shift(-1).tail()',cost_rowcalc.shift(-1).tail())
print('\ncost_rowcalc.shift(1).head()',cost_rowcalc.shift(-1).head())
print('\ncost_rowcalc.shift(periods=1,freq=\'ME\')',cost_rowcalc.shift(periods=1,freq='ME'))
# WHAT IS THE DIFFERENCE, FROM THE PREVIOUS DAY
print('\ncost_rowcalc[\'Adj Close\'].diff(1)\n',cost_rowcalc['Adj Close'].diff(1))
# WHAT IS THE DIFFERENCE, (in percent terms) FROM THE PREVIOUS DAY
print('\ncost_rowcalc[\'Adj Close\'].pct_change(1)\n',cost_rowcalc['Adj Close'].pct_change(1))
# ABILITY TO PERFORM CUMALIVE SUMS FOR PRODUCTS
print('\ncumalitive_sum_series = pd.Series([1,2,3,4])')
cumalitive_sum_series = pd.Series([1,2,3,4])
print(cumalitive_sum_series)
print(f'\ncumalitive_sum_series.cumsum().\n{cumalitive_sum_series.cumsum()}')
print(f'\ncumalitive_sum_series.cumprod().\n{cumalitive_sum_series.cumprod()}')


# PANDAS PYTHON API BASED DATA SOURCES
# pip install pandas-datareader (Ext on pandas to connect with APIs across websites)
# pip install yfinance (additional functionality than pandas-datareader) CHECK FOR UPDATES
# import pandas_datareader.data as web https://pandas-datareader.readthedocs.io
# import yfinance as yf
# from pandas_datareader import data as pdr

# print('aapl_df = web.DataReader(\'AAPL\',\'yahoo\',start=\'2023-01-01\', end=\'2024-01-01\'')
# AAPL THIS WAY DIDN'T WORK
#aapl_df = web.DataReader('AMD','yahoo',start='2023-01-01', end='2024-01-01')


# MAKE PANDAS-DATAREADER USE YFINANCE UNDER THE HOOD
print('\nge_df = web.DataReader() USING yf.download(\'GE\', start=\'2019-09-10\', end=\'2019-10-09\', auto_adjust=True):')
ge_df = yf.download('GE', start='2019-09-10', end='2019-10-09', auto_adjust=True)
print(ge_df.head())
print('\naapl_df = web.DataReader() USING yf.download(\'AAPL\',\'yahoo\',start=\'2023-01-01\', end=\'2024-01-01\':')
aapl_df = yf.download('AMD',start='2023-01-01', end='2024-01-01', auto_adjust=True)
print(aapl_df.head())
# USING FEDERAL RESERVE DATASET
print('\ninflation_df = pdr.DataReader(\'T10YIE\', \'fred\', start=\'2004-01-01\',end=\'2020-01-01\, auto_adjust=True\')')
inflation_df = pdr.DataReader('T10YIE', 'fred', start='2004-01-01',end='2020-01-01')
print('inflation_df\n',inflation_df)
print('\ninflation_df.plot())',inflation_df.plot())
plt.show()
# GETTING OTHER DETAILS
print('\n\apple_ticker = yf.Ticker(\'AAPL\')')
apple_ticker = yf.Ticker('AAPL')
print('apple_ticker.get_balance_sheet()',apple_ticker.get_balance_sheet())
"""

# PANDAS AND FINANCE EXERCISE
# TASK 1 Import neccessary files 
# GIVEN DATASET SP500 https://finance.yahoo.com/quote/SPY
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# from matplotlib import dates
# TASK 2 Grab the SPY historical data from Jan-1-2000 to Jan-1-2021
print('\nPANDAS AND FINANCES EXERCISE')
print('1. IMPORT NECCESSARY FILES')
finex = 'PWQuantConnect/05-Pandas-and-Finance/SPY2000_2021.csv'
try:
    finex = pd.read_csv(finex,index_col='Date',parse_dates=True)
except FileNotFoundError:
    print(f"Error: The file '{finex} was not found.")
except Exception as e:
    print(f"An error occured: {e}")
print('2. Grab the SPY historical data from Jan-1-2000 to Jan-1-2021 FROM YAHOO FINANCE')
print('finex = yf.download(\'SPY\', start=\'2000-01-01\', end=\'2021-01-01\', auto_adjust=False)')
finex = yf.download('SPY', start='2000-01-01', end='2021-01-01', auto_adjust=False)
print(finex)
# TASK 3 Check the head of the ten first entires in the dataset
print('\n3. Check the head of the ten first entires in the dataset')
print('finex.head(10)')
print(finex.head(10))
# TAKS 4 Check the datatype of all entires
print('\n4. Check the datatype of all entires')
print('finex.dtypes')
print(finex.dtypes) # also finex.info()
# TASK 5 Plot the Adj Closing price of the SP500, with the price on the y axis
# and the year on the x axis. Use Locator() and Formatter() so you can see a tick 
# for every year in the dataset(only showing the year numb er, not the full YYYY-MM-DD)
# Choose a resonable fig size, Set the dpi to 300 and save as sp500.png
# from matplotlib import dates
print('\n5. Plot the Adj Closing price of the SP500, with the price on the y axis\n\
and the year on the x axis. Use Locator() and Formatter() so you can see a tick\n\
for every year in the dataset (only showing the year number, not the full YYYY-MM-DD)\n\
Choose a reasonable fig size, Set the dpi to 300 and save as sp500.png\n')
# from matplotlib import dates 
# Create the figure and axes with specified size and dpi print('plt.figure(figsize=(10,4),dpi=100')
#plt.figure(figsize=(20,4),dpi=200)
fig, ax = plt.subplots(figsize=(10,6), dpi=100)
#print('ax = finex[\'Adj Close\'].plot()')
#ax = finex['Adj Close'].plot()
print('ax.plot(finex.index, finex[\'Adj Close\'], label=\'SPY Adj Close\')')
ax.plot(finex.index, finex['Adj Close'], label='SPY Adj Close')
print('ax.xaxis.set_major_locator(dates.YearLocator()):')
ax.xaxis.set_major_locator(mdates.YearLocator())
print('ax.xaxis.set_major_formatter(dates.DateFormatter(\'%Y\')):')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
print('\nOptionally, format minor ticks or rotate labels for better readability,\nplt.xticks(rotation=45, ha=\'right\'')
plt.xticks(rotation=45, ha='right')
print('\nAdd labels and title if needed,plt.xlabel(\'Year\'),plt.ylabel(\'Adjusted Close Price\',plt.title(...\'')
plt.xlabel('Year')
plt.ylabel('Adjusted Close Price')
plt.title('S&P 500 Adjusted Close Over Time')
print('\nSave Plot = plt.savefig(\'AdjClosePriceSP500.png\')') # SAVE BEFORE plt.show()
plt.savefig('AdjClosePriceSP500.png')
plt.show()


