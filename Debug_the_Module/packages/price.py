
def calculate_markup(input1, input2):  
    #breakpoint() 
    key1 = list(input1.keys())[0]
    key2 = list(input2.keys())[0]   
    assert (key1 == key2) and (len(list(input1.keys())) == 1)
    markup = input1[key1]  - input2[key1] 
    return {key1: markup}
