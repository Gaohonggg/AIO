def check_int(value):
    try:
        return int(value)
    except ValueError:
        return False
tp = input("Input for tp: ")
tp = check_int(tp)
if tp == False:
    print("tp must be int")
fp = input("Input for fp: ")
fp = check_int(fp)
if fp == False:
    print("fp must be int")
fn = input("Input for fn: ")
fn = check_int(fn)
if fn == False:
    print("fn must be int")

if tp<=0 or fp<=0 or fn<=0:
    print("tp and fp and fn must be greater then zero")
    exit()

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_core = 2*(precision*recall)/(precision+recall)

print("precision is: {}".format(precision))
print("recall is: {}".format(recall))
print("f1_core is: {}".format(f1_core))
