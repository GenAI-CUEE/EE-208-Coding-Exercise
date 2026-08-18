scores = ['10', '20', 'Oops', '40']
for s in scores:
    try:
        result = int(s)
        print("result %d" % result)
    except:
        "do nothing" 