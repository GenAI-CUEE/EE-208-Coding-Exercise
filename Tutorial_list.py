a = "apple"
b = "banana"
c = "grape"

str_list = [a, b, c] 
new_list = []

i = 0
for var_ in str_list:  
    print(i) 
    print(var_)
    i+=1
 
out = str_list.pop() 
new_list.append(out)
breakpoint() # print(str_list)

out = str_list.pop()
new_list.append(out)
breakpoint() # print(str_list)

str_list.pop()
new_list.append(out)
breakpoint() # print(str_list)
 
new_list  # print(new_list)
breakpoint()
 
 