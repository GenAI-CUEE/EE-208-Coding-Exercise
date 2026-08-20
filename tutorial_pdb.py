import pdb

def calculate_total(prices):
    total = 0
    for price in prices:
        total += price  
        breakpoint()
    return total

if __name__ == "__main__" :

    prices = [10, 20, "oops", 40]
    result = calculate_total(prices)
