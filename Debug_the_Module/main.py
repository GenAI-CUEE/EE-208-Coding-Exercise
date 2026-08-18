from shop import calculate_total
cart = [("apple", 2, 1.50), ("bread", 1, 3.00), ("milk",  3, 2.25)] 
# e.g.,  ("apple", 2, 1.50) : (product, amount, price/piece)...
# Customer get 2 apples, each has the price of $1.5. Total is $3.00
print(calculate_total(cart))

