import numpy as np
A = np.array([1, 2, 3, 4])
B = np.ones((4,1)) 

breakpoint() 
C = A+B
breakpoint()

####################################
del A, B, C
# How to fix 
# 1. Fix A. A = A.reshape(4,1)
A = np.array([1, 2, 3, 4])
B = np.ones((4,1)) 

A = A.reshape(4,1)
breakpoint()
C = A+B
print(C.shape)
breakpoint()

# 2. Fix B. B = B.reshape(4,) 
del A, B, C

A = np.array([1, 2, 3, 4])
B = np.ones((4,1)) 
B = B.reshape(4)
breakpoint()
C = A+B
print(C.shape)
breakpoint()

