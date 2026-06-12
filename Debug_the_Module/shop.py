def calculate_total(cart):
    total = 0
    for name, qty, price in cart:
        total += price
    return total