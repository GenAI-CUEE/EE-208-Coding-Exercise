def calculate_total(cart):
    total = 0
    for name, qty, price in cart:
        print(name)
        print(qty)
        print(price)
        total += qty*price        
        print(total) 
        breakpoint()
    return total