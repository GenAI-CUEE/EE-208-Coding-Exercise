import numpy as np

# Matrix multiplication

A = np.array([[1, 2], [3, 4]]) 
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
breakpoint()

C1 = A @ B
C2 = np.dot(A,B)
C3 = np.matmul(A,B)

assert (C1 == C2).all() and (C1== C3).all()
print("They give the same results")
breakpoint

##################################

# Matrix-vector multiplication

A = np.array([[1, 2], [3, 4]]) 
print(A.shape)
b_vector = np.array([[5], [6]])
print(b_vector.shape)
breakpoint()

C1 = A @ b_vector
C2 = np.dot(A,b_vector)
C3 = np.matmul(A,b_vector)

assert (C1 == C2).all() and (C1== C3).all()
print("They give the same results")
breakpoint