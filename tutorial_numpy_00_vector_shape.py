import numpy as np
a1_vect = np.array([1, 2, 3, 4])
print("a1_vect")
print(a1_vect.shape) 

a2_vect = np.array([[1, 2, 3, 4]])
print("a2_vect")
print(a2_vect.shape)
#  [1, 2, 3, 4]  
a3_vect = np.array([[1], [2], [3], [4]]) 

print("a3_vect")
print(a3_vect.shape)
#  [1,
#   2,
#   3,
#   4]  

##################################################
# So reading two pairs of brackets
# The num elemments in inner bracket  => columns
# The num elemments in outer bracket  => rows
##################################################
# 
# 
# More example ...  
a4_vect = np.array([[1,2], [3,4], [5,6]])  # Read 3x2
print("a4_vect")
print(a4_vect.shape)

# What about 2x3
a5_vect = np.array([[1,2,3], [4,5,6]])
print("a5_vect")
print(a5_vect.shape)

# What about 1x2x3
a6_vect = np.array([[[1,2,3], [4,5,6]]])
print("a6_vect")
print(a6_vect.shape)

# What about 2x2x3
a7_vect = np.array([[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]])
print("a7_vect")
print(a7_vect.shape)

# What about 2x3x3
a8_vect = np.array([[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]])
print("a8_vect")
print(a8_vect.shape)