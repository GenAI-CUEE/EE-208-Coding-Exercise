import numpy as np
A = np.array([[1, 2], [3, 4]]) 
B = np.array([[5, 6],[7, 8]])

C1 = A @ B
C2 = np.dot(A,B)
C3 = np.matmul(A,B)

assert (C1 == C2).all() and (C1== C3).all()
print("They give the same results")
breakpoint