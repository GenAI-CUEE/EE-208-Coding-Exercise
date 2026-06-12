scores = [70, 85, 60, 95]
bonus  = 5
total  = 0
for i, s in enumerate(scores): 
    boosted = s + bonus
    total += boosted
    breakpoint() # ← pauses

