import numpy as np
import time

n = 1000

rng = np.random.default_rng()
A = rng.random(size=(n,n))
 
# Compute the multiplicative inverse
A_inv = np.linalg.inv(A)

# --------------------------------------------------------- #
# What if we are computing the inverse of diagonal matrix ? 
 
random_vector = rng.random(size=(n,))
B = np.diag(random_vector) # The diagonal matrix

# In this case there are two ways to do the inversion
B1_inv = np.diag(1/np.diag(B)) 
B2_inv = np.linalg.inv(B)


#################################################################
# Measure the processing time 

start_time = time.perf_counter()
B1_inv = np.diag(1/np.diag(B)) 
end_time = time.perf_counter()
duration1 = end_time - start_time
print( "Execution time  np.diag(1/np.diag(B)): %.6f seconds" % duration1)

start_time = time.perf_counter()
B2_inv = np.linalg.inv(B)
end_time = time.perf_counter()
duration2 = end_time - start_time
print( "Execution time  np.linalg.inv(B) :  %.6f seconds" % duration2)

