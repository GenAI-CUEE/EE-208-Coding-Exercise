def grade(score):
    if score >= 90:
        return "A"
    elif (score >= 80)  and (score < 90):
        return "B"
    elif (score >= 70)  and (score < 80):
        return "C"
    elif (score >= 60)  and (score < 80):
        return "D"
    else:
        return "F" 

scores = [95, 82, "71", "oops", 55, -1]

for s in scores:
    try:
        result = int(s)
        if (result < 0) or (result > 100):
            raise ValueError("Out of range")
        print("Score %d → Grade %s" % (result, grade(result) ))
    except(ValueError, TypeError) as e:
        print(f"Skipping {s}: {e}")
        


