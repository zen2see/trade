from tkinter import ALL
import numpy as np
import pandas as pd
import timeit 

# COMMENTS 
"""
# MULTIPLE ROW COMMENTS
"""

"""
# MATH
1+1=2, 1-3=-2, 1*3=6, 1/2=0.5, 2**4=16, 4%2=0, 5%2=1, 2+3*5+5=22, (2+3)*(5+5)=50

# VARIABLES
x=1, y=2, x+y=3, z=x+y, # z now = 3

# STRINGS
'single', "double", "I don't care"

# PRINT
x=hello print(x)='hello', print("DOUBLE")=DOUBLE, name ="Jose Portilla"
print("Hello my name is {}".format(name))
number = 12
print("Hi my name is {} and my number is {}".format(name, number)) OR
print("Hi my name is {x} and my number is {y}".format(x=name, y=number))

LIST mutable
['hi',0,1]
[1,2,3] = [1, 2 ,3] 
mylist.append(4) = [1,2,3,4]
my_list = ['a','b','c','d']
my_list[0] = 'a' my_list[-1] = 'd'
my_list[0:2] = ['a', 'b'] OR my_list[0:2] or nt_list[:2]
nested = [1,2['a','b']] nested[2][0] = 'a'

DICTIONARY
dict = {'key':10,'key2':'seconditem;}
d['key2'] = 'seconditem'
d.keys() # RETRUNS dict_keys(['key', 'key2])
TUPLES like a list but non-mutable
t = (1,2,3) t[0] = 1

SETS = unorderderd colleciton of UNIQUE items
{1,2,3} = {1,1,1,2,2,2,3,3,3} 
import math
math.sqrt() SHIFT+TAB for DOCUMENTATION

COMPARISON OPERATORS
1 < 2, 1 >=3, 1 != 3, 'string' == 'string' 1 == '1' FALSE (1==2 ) AND NOT (1==1), OR, NOT
if True:
    print('hello')
elif 2==2:
    print('two')
else:
    print('false')

FOR LOOPS
seq = [1,2,3,4,5]
for item in seq:
    print(item)

WHILE LOOPS
i = 1
while i < 5:
    print(i is currently {}.format(i))
    i = i + 1

RANGE
range(5) # = range(0,5)
for item in range(0,20,2):
    print(item) # prints 0,2,4...

LIST    
list(range(1,11)) # [1,2,3...10]
my_list = [1,2,3,4]
my_list.pop() # RETURNS 4
my_list,pop(0) # RETURN 1
my_list # RETURNS [2,3]
2 in my_list # RETURNS TRUE

LIST COMPREHENSION
x =[1,2,3,4]
out = []
fro num in x:
    out.append(num**2)
out = [1, 4, 9, 16]
[num**2 for num in x] # = [1, 4, 9, 16]
 
FUNCTIONS
def my_func():
    print('hello')
def my_func2(param='default'):
    \"""
    Docstring goes here!
    \"""
    print(param)
# python3 my_func() # would print doc info and then the word default    
def my_func3(argument):
    return argument*5 # returns 25
x = my_func(5)
print(x) # prints 25
def times_two(var):
    return var*2
result = times_two(4)
result # prints8

# EXAMPLE LAMDA AND MAP FUNCTION
lambda var: var*2 # SAME AS THE times_two(var) funciton above
seq = [1,2,3,4,5]
list(map(lambda num:num*2,seq)) = [2,4,6,8,10]
def is_even(num):
    return num%2 == 0 
(filter(is_even,seq)) RETURNS <filter at 0x1ddb513a5f8)
list(filter(is_even,seq)) RETRURNS, seq v=[1,2,3,4,5]
list(filter(lambda num:num%2 == 0,seq)) RETURNS [2,4]

STRING FUNCTIONS
st = 'hello my name is Sam'
st.lower # RETURNS 'hello my name is sam'
st.upper() # RETURNS 'HELLO MY NAME IS SAME'
tweet = "Go Sports! #cool"
tweet.split() # RETURNS ['Go', 'Sports!', '#cool']
tweet.split('#') # RETURNS 'cool'
# TYPE tweet. TO SEE THE LIST POSSIBLE  FUNCTIONS

EXERCISES
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
"The {} is at {} today".format(stock_index,price))
#3 Given the variable of a nested dictionary with nested lists grab certain items using indexing and key calls
stock_info = {'sp500':{'today':300, 'yesterday':250},'info':['Time',[24,7,365]]}
stock_info.keys() # RETURNS dict_keys{['sp500', 'info']}
stock_info['sp500']['yesterday'] # RETURNS 250
stock_info['info'][1][2] # RETRUNS 365
stock_info['info'][1] RETURNS [24, 7, 365]
#4 Create a () called source_finder() that returns the source.
def source_finder(s):
    return s.split('--')[-1]
source_finder("PRICE:345.324:SOURCE--QUANDL") # returns 'QUANDL'
#5 Create a () called price_finder that returns True if the word 'price' is in a string.
def price_finder(s):
    return 'price' is s.lower()
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

# NUMPY 
NumPy is a data science library

ARRAYS - A Python library for creating N-dimensional arrays, quickly broadcast functions.
Has Built-in linear algebra, statistical distributions, trigonometric, and random number capabilities.
Look similar to standard Python lists, they are much more efficient!

BROADCASTING capabilities are extremely useful for quickly applying () to our data sets.
import numpy as np
mylist = [1,2,3]
type(mylist) # = list
np.array(mylist) # = array([1,2,3])
myarray = np.array(mylist) # myarray = array([1,2,3])
type(array) = # numpy.ndarray
my_matrix  = [[1,2,3],[4,5,6],[7,8,9]] # my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
np.array(my_matrix) 
# array([[1,2,3],
        [4,5,6],
        [7,8,9]])
np.arange(0,101,20) # array([0,20,40,60,80,100])
np.zeros(5) # array([0.,0.,0.,0.,0.])
np.zeros(2,5)) 
# array([0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.]])
np.ones(5) # array([1., 1., 1., 1., 1.])
np.linspace(0,10,11)# array([0.    , 1.11111111, 2.22222222, ... 10.       ]) 21 numbers
np.exe(5) # array([1.,0.,0.,0.,0.,],
                  [0.,1.,2.,3.,4.,],)
np.random.rand(1) # Gives a random number between 0 and 1 array([0.11242691])
np.random.rand(5,2) # Five rows of two random numbers
np.random.randn # Mean is 0 and variance 1 array([-0.90387279, -0.33609234, 1.54333558, etc.])
np.random.randint # Returns random integers
np.random.randint(0,101(2,3)) # array([[ 0.13570505, 0.736018705, 2.01277515], ...])
np.random.randint(0,101,10) # array([8,30,33,49,66,32,69,14,96,62])
np.random.seed(42) # seed 42 from hitchhikers guide to the galaxy
np.random.rand(4) # array([0.3745012, ...])
arr = np.arrange(0,25) # arr = array([0,1,2,3...24])
arr.reshape(5,5) # array([0,1,2,3,4],
                        [5,6,7,8,9],...
ranarr = np.random.randint(0,101,10)
ranarr.max() # 93
ranarr.min() # 8
ranarr.argmax() # 1 position of max argument
ranarr.dtype # dtype('int32')
arr.shape # 25
arr = arr.reshape(5,5) # the numbers in an array of 5 by 5

# INDEXING AND SELECTION
anarr = np.arrange(0,11) # arr = array([0,1,2,3..1])
anarr[8] # 8
anarr[1:5] # array([0,1,2,3,4]) SAME AS anarr([:5])
anarr2[5:] # array([5,6,7,8,9,10])

# BROADCASTING 
anarr[0:5] = 100 # anarr = array([100, 100, 100, 100, 100, 5, 6, 7, 8, 9, 10])
newarr = np.arange(0,11) # newarr = array([0, 1, 2, ...10])
slice_of_newarr = newarr[0:5] # slice_of_newarr = array([0, 1, 2, 3, 4])
slice_of_newarr[:] = 99 # = array([99, 99, 99, 99, 99]) 
newarr # = array([99, 99, 99, 99, 99, 5, 6, 7, 8, 9, 10])
arr_copy = newarr.copy() 
arr_copy[:] = 100 # arr_copy = array([99, 99, 99, 99, 99, 5, 6, 7, 8, 9, 10])

# INDEXING ON 2D ARRAYS
arr_2d = np.array([5,10,15],[20,25,30],[35,30,45])
arr_2d # = array([5,  10, 15],
                 [20, 25, 30],
                 [35, 40, 45])

arr_2d.shape # (3,3)
arr_2d[2] # = array([35, 40, 45])
arr_2d[1][1] # 25
arr_2d[1,1] # 25
arr_2d[:2] # array([ 5, 10, 15],
#                  [20, 25, 30]]) 
arr_2d[:2,1:] # array([10, 15],
                      [25, 30]])

                      # OPERATORS
arrop =- np.arrange(0,10) # array([0,1,2,3,4,5,6,7,8,9])
arrop + 5 # array([5,6,7,8,9,10,11,12,13,14])
arrop - 2 # array([-2,-1,0,1,2,3,4,5,6,7])
arrop + arrop # array([0,2,4,6,8,10,12,14,16,18]) CAN DO *,-
np.sqrt(arrop) # array([0.        , 1.        , 1.41421356, ...])
np.sin(arrop) # array([0.        , 0.84147098,  0.90929743, ...])
np.log(arrop) # array([        -ing, 0.        , 0.69314718, ...])
arrop.sum() # sum all elements in array
arrop.mean() # arrop.max() arr.var() arr.std()
arr2d = np.arange(0,25).reshap(5,5) # arr2d.shape (5,5)
arr2d # array[0,1,2,3,4],
      #      [5,6,7,8...
arr2d.sum() # 300
arr2d.sum(axis=0# array([50,55,60,65,70]) columns
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

# PANDAS
# A Series - data structure that holds an array of info w/a named+# index
# Named index differentiates this from a NumPy array. Formal Definition: A long-dimensional ndarray with axis labels
# import numpy as np
# import pandas as pd
# help(pd.Series)

# LIST REVIEW - PANAS SERIES
myindex = ['USA', 'Canada', 'Mexico']
mydata = [1776,1867,1821]
myseries = pd.Series(data=mydata)
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
myseries['USA'] # 1776
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
print('sales_q1.add(sales_q2,fill_value=0), aligns by index, add fill value if values don\'t match up\n', sales_q1.add(sales_q2,fill_value=0),)
print('sales_q1.dtype', sales_q1.dtype)

# DATAFRAME
# DATAFRAME a table of columns and rows in pandas that we can restructure and filter, a group of Pandas Series 
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

# HOW TO READ A CSV FILE FROM a PANDAS DATAFRAME (CAN READ OTHER FILES)
dfcvs = pd.read_csv('tips.csv') # pd.read_csv)'C:\\Users\\CSV_FILE\\LOCATION\.file.csv
print('\nTIPS:\n',dfcvs) # PRINT WHOLE FILE
print('COLUMNS:\n',dfcvs.columns,'\n') # Index(['total_bill', 'tip', ...])
print('INDEX:\n',dfcvs.index,'\n') # RangeIndex(start=0, stop=244, step=1)
print('HEAD:\n',dfcvs.head(),'\n') # dfcvs.head(10)  First 10 rows
print('TAIL:\n',dfcvs.tail(5),'\n') # dfcvs.tail(10) last 10 rows
print('INFO:\n')
print(dfcvs.info())
print('DESCRIBE:\n',dfcvs.describe(),'\n') # DESCRIBE DOES STATISTICAL OPS - MIN, MAX, MEAN, COUNT, STD, 25%, 50%, 75%
print('DESCRIBE.TRANSPOSE:\n',dfcvs.describe().transpose(),'\n') # Transposes columns                                                                                                                                                                                                                                                           

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
dfcvs.drop('tip_pct', axis=1, inplace=True) # axis=0 is rows, axis=1 is columns inplace=True means column is kept in bg

# WORKING WITH ROWS
print('INDEX\n',dfcvs.index)
print('TEMPORARILY SET INDEX\n,dfcvs.set_index(\'Payment ID\')','\n',)
print('PERMANENTLY SET INDEX by dfcvs = dfcvs.set_index(...')
print('RESET INDEX\ndfcvs.reset_index(),\n')
print('LOC - LABEL BASED INDEXING\n,dfcvs.loc[\'Sun2959\'], OR numerical with dfcvs.iloc[0]\n')
print('SLICE DATA\ndfcvs.iloc[0:4]\n') # COULD DO dfcvs.loc[\'Sun2959\']: FOR MULTIPLE ROWS
print('REMOVING ROWS,dfcvs.drop(\'Sun2959\', axis=0)') # axis=0 is rows, axis=1 is columns
print('PERMANENTLY DROP First 3 WOULD BE, dfcvs = dfcvs.iloc[3:3]') # DROP FIRST 3 ROWS AND KEEP REST)
print('ADDING A ROW = SAY WE ADD A ROW one_row = dfcvs.iloc[0], dfcvs.append(one_row), PERMANENT = dfcvs = dfcvs.append(one_row)')
a_row = dfcvs.iloc[0]
print('added',a_row)

# TYPICALLY IN DATA ANALYSIS OUR DATASETS ARE LARGE ENOUGH WE DON'T FILTER BASED ON POSITION, INSTEAD BASED ON A CONDITION
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
print('\nFILTER TIPS SERIES WHERE TOTAL BILL > 22\n', dftips[dftips['total_bill'] > 50])
print('\nFILTER TIPS SERIES WHERE TOTAL BILL > 22 AND DAY IS FRI\n', dftips[(dftips['total_bill'] > 22) & (dftips['day'] == 'Fri')])
# print(dftips.columns)

# PANDAS USEFUL METHODS
print('\nUSEFUL METHODS:\napply()\n')
print('EX. 1 dftips.head()\n',dftips.head())
print('GRAB LAST 4 DIGIST OF CC#\n',dftips['CC Number'].apply(lambda x: str(x)[-4:]),'\n')
print('CREATE A CUSTOM FUNCTION GRAB LAST FOUR and apply it to a column\ndef grab_last_four(num):\n return str(num)[-4:]\n')
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
print('dftips[\'TOTAL_BILL,tip\'].apply(lambda num: num * 2)\n', dftips[['total_bill','tip']].apply(lambda num: num * 2),'\n')
def Quality(total_bill,tip):
    if tip / total_bill >= 0.25:
        return 'Generous'
    elif tip / total_bill >= 0.15:
        return 'Average'
    else:
        return 'Cheap'
print('dftips[[\'total_bill\', \'tip\']].apply(lambda dftips: Quality(dftips[\'total_bill\'], dftips[\'tip\']), axis=1)\n')
dftips['Quality']=dftips[['total_bill', 'tip']].apply(lambda dftips: Quality(dftips['total_bill'], dftips['tip']), axis=1)
print(dftips['Quality'])
print('\nHAVE IT RUN FASTER WITH VECTORIZE - df[\'Quality\'] = np.vectorize(Quality)(dftips[\'total_bill\'], dftips[\'tip\'])')
# NP VECTORIZE TRANSFORMS FUNTION THAT ARE NOT NUMPY AWARE 
dftips['Quality'] = np.vectorize(Quality)(dftips['total_bill'], dftips['tip'])
print(dftips.head())
# dftips['Quality'] = dftips['total_bill', 'tip'].apply(Quality)
# TIMELY CODE STARTS HERE AFTER ADDING TRIPLE QUOTES
# TIMING CODE WITH TIMEIT - NOTE TRIPLE QUOTES BELOW ARE NEEDED FOR TIMELY CODE TO WORK SO PLACE 3 QUOTES BEFORE THIS LINE
# TIMELY CODE TO BE EXECUTED ONCE FOR TIMELY
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
"""
# CODE SNIPPET WHOSE EXECUTION TIME IS TO BE MEASURED BY TIMELY
stmt_one ='''
dftips['Tip Quality'] = dftips[['total_bill', 'tip']].apply(lambda dftips: qualitytimely(dftips['total_bill'], dftips['tip']), axis=1)
'''
stmt_two = '''
dftips['Tip Quality'] = np.vectorize(qualitytimely)(dftips['total_bill'], dftips['tip'])
'''
timelyone = timeit.timeit(setup=pdtimely,stmt=stmt_one,number=1000)
timelytwo = timeit.timeit(setup=pdtimely,stmt=stmt_two,number=1000)
print('\nUsing timely: ',timelyone)
print('Using timely with vectorize: ',timelytwo)
""" # TIMELY CODE ENDS HERE

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
# SAME FORMAT (*SIMPLEST) A CONCATENATION (PAST TWO DF TOGETHER) THROUSH pd.concat() IS ALL THAT IS NEEDED
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

# COMBINING DATAFRAMES
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

