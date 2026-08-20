import numpy as np

print("Sec 1. ###### Without .copy() ######")

a = [1, 2] 
b = a
a[0] = 2  
print(a)
print(b)  

aa = np.array([1, 2])
bb = aa
aa[0] = 2 
print(aa)
print(bb)  

breakpoint()

##########################################
print("Sec 2. ###### With .copy() ######")

del a, b, aa, bb
a = [1, 2] 
b = a.copy()
a[0] = 2  
print(a)
print(b) 

aa = np.array([1, 2])
bb = aa.copy()
aa[0] = 2 
print(aa)
print(bb) 
 
breakpoint()