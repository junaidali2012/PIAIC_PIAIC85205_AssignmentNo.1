#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np #import numpy python library with import command


# 2. Create a null vector of size 10 

# In[4]:


arrnull = np.zeros((10)) # it create a null vector of size 10


# In[5]:


arrnull


# 3. Create a vector with values ranging from 10 to 49

# In[7]:


vector = np.arange(10, 50) # this command create a vector with values from 10 to 49


# In[9]:


vector # executing variable for checking its stored values


# 4. Find the shape of previous array in question 3

# In[10]:


vector.shape # command for find the shape of any array


# In[11]:


np.shape(vector) # 2nd procedure to find the shape of array.


# 5. Print the type of the previous array in question 3

# In[12]:


type(vector) # return the array type


# 6. Print the numpy version and the configuration
# 

# In[13]:


print(np.__version__) # used for getting the version of numpy


# 7. Print the dimension of the array in question 3
# 

# In[14]:


vector.ndim # it gives us the dimention of the array


# 8. Create a boolean array with all the True values

# In[15]:


np.full((3, 3), True, dtype=bool) # create a 3x3 numpy array of all the true's values


# In[16]:


np.ones((3,3), dtype=bool) # 2nd procedure to creat the above same


# 9. Create a two dimensional array
# 
# 
# 

# In[28]:


TwoDarray =  np.arange(4).reshape(2,2) # create two dimentional array


# In[29]:


TwoDarray


# 10. Create a three dimensional array
# 
# 

# In[30]:


ThreeDarray =  np.arange(9).reshape(3,3) # create three wo dimentional array


# In[32]:


ThreeDarray


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[34]:


vector = np.arange(9) # array of nine elements present init 
vector[::-1] #reverse the vector 


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[36]:


vector_null = np.zeros(10) # create the null vector of size 10
vector_null[4] = 1 # index 4 means fifth position equals to one as desired
vector_null 


# 13. Create a 3x3 identity matrix

# In[37]:


matrix_identity = np.ones((3,3)) # create a 3x3 identity matrix
matrix_identity


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[39]:


arr = np.array([1, 2, 3, 4, 5], dtype=float) # convert the data type of int into float
arr


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[40]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr1 * arr2  # this command used to multiply the array 1 into 2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[42]:


arr_compare = (arr1 > arr2) #compare two array with each others and returns T/F values
arr_compare


# 17. Extract all odd numbers from arr with values(0-9)

# In[48]:


arr = np.arange(10) # create the array 
num_odd = arr[ arr % 2 != 0] # extract all odd numbers 
num_odd


# 18. Replace all odd numbers to -1 from previous array

# In[49]:


arr = np.arange(10)      #create the array of 10 elements
arr[ arr % 2 != 0] = -1  #replace all odd numbers to -1 from previous array
arr


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[50]:


arr = np.arange(10)      #create the array of 10 elements
arr[5:9] = 12            #replace the values of indexes of places 5 to 8 equals to 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[54]:


TwoDayArray = np.ones((5,5)) # print original array
TwoDayArray[1:-1,1:-1] = 0 # put 1's on the border and o inside the arraay
TwoDayArray


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[56]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1][1] = 12  # replace the number inside array from 5 to 12 which is locate at array position of 5X5
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[58]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0][0] = 64 # convert all the values of 1st array to 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[61]:


array2d = np.arange(10).reshape(2,5) # this gives us 2-D array with 0-9 values
array2d[0] # slice out the first 1-D array from it.


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[64]:


array2d = np.arange(10).reshape(2,5) # this gives us 2-D array with 0-9 values
array2d
array2d[1][1] # slice out the second value from 2nd 2-D array from it.


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[65]:


array2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
array2d[:,2][0:2] # slice out the third column but only the first two rows


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[66]:


ran_value_array = np.random.randn(100).reshape(10,10) # create 10X10 array with random values
ran_value_array


# In[67]:


ran_value_array.max() #give us max value among random values array


# In[68]:


ran_value_array.min() #give us min value among random values array


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[74]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
a[a == b] # find the common items between a & b arrays 


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[76]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
c = np.where(a==b) # this cmd used to find the positions of matched element at a & b 
c


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[78]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data[names!='Will'] #Find all the values from array data where the values from array names are not equal to Will


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[82]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
result = (names!='Will') & (names!='Joe') #Find all the values from array data where the values from array names are not equal to Will and Joe
data[result]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[83]:


array_2d = np.random.randint(low=1,high=15,size=(5,3))
array_2d


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[84]:


array_shape = np.random.randint(low=1,high=16,size=(2,2,4)) # Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16
array_shape 


# 33. Swap axes of the array you created in Question 32

# In[85]:


np.swapaxes(array_shape,0,2) #Swap axes of the array


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[87]:


array_ten = np.arange(10) # create an array of size 10
array_ten_squared = np.sqrt(array_ten) #find the sqroot of every element
np.where(array_ten_squared<0.5,0,array_ten_squared) # checking the condition
# if the values less than 0.5, replace them with 0


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[94]:


array_one = np.random.randn(12) #Create random arrays of range 12
array_two = np.random.randn(12) #Create random arrays of range 12
maximum = np.maximum(array_one , array_two) #make an array with the maximum values between each element of the two arrays
maximum


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[96]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
unique_names = np.unique(names) #Find the unique names and sort them out!
unique_names


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[105]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
array = np.setdiff1d(a, b)
array


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[110]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray[1:-1] # delete column two
sampleArray[:,1][:] = [10,10,10] # insert new column in its deleted column
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[112]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y) # dot product of the above two matrix


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[115]:


matrixes = np.random.randn(20) # create a matrix of 20 random variable
commulative_sum = matrixes.cumsum() # find its cumulative sum
commulative_sum

