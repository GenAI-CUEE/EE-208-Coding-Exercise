# Given the selling price in the cart...
# 2 apples, each has the price of $1.5. Total is $3.00
# 1 bread, each has the price of $3.00. Total is $3.00
# 1 milk, each has the price of $2.25. Total is $6.75

cart = [("apple", 2, 1.50), ("bread", 1, 3.00), ("milk",  3, 2.25)]  



# ---------------------------------------------------------------------------------------------- #
#  --------------------------- Problem During Break   ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #

# Suppose that we know the cost of each product 
# The cost of each product are 
# 1 apple, each has the cost of $1
# 1 bread, each has the cost of $2.50
# 1 milk, each has the cost of $2.25

# Goal: we want to compute the markup : the margin between scalling price and the cost.
# the calculation for markup price done by a function called `calculate_markup(input1, input2)`, 
# e.g. 
# output = calculate_markup(input1, input2)
#
# input1: the dict for selling price, e.g., {"product_name1": selling_price1}
# input2: the dict for cost, e.g., {"product_name1": cost1}
# output: the dict for markup, e.g., {"product_name1": markup}

# Extra 
# Let us create a new python file inside  packages/ folder called price.py that contain `calculate_markup` function
# ---------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #
 
## Plan

## Simple idea : 1 product ##

# Step 1. Generate the dict for selling price from the list of tuple (defined above). Let's define an input1 
 
input1 = {product_name1: product_sell1}

# Step 2. Suppose that input2 for each product is {"apple", 1}, {"bread", 2.50}, {"milk", 2.25}. Let's define an input2 

input2 = {"apple": 1}

# Step 4. Let's do a simple calculation without Extra
# def calculate_markup(input1, input2)
 





##########################################################################################################


## Extend the calculation for all the product sold to the customer ##

# Step 5. >> Get a list that dicts, i.e., [ {"product_name1": selling_price1}, {"product_name2": selling_price2}, {"product_name3": selling_price3}]

# Step 6. >> Turn above info into list of dict , i.e., [{"apple": 1},  {"bread": 2.50}, {"milk": 2.25}]

# Step 7. >> Does it work correctly ? 


##########################################################################################################


## Do the extra ##
# Step 8. Let us create a new python file inside  packages/ folder called price.py that contain `calculate_markup` function

# Step 9. Make sure the function is linked, and can be executed. 