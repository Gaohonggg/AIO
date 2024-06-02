import math
def sigmoid(x):
    return 1/(1+math.exp(-x))

def relu(x):
    if x<=0:
        return 0
    return x

def elu(x):
    if x<=0:
        return 0.01*( math.exp(x) - 1)
    return x

def is_number(n):
    try:
        return float(n)
    except ValueError:
        return False

x = input("Nhap so x: ")
x = is_number(x)
if x == False:
    print("x must be a number")
    exit()
func = input("Nhap function muon dung: ")
if func == "sigmoid":
    print("{}: f({}) = {}".format(func,x,sigmoid(x)))
elif func == "relu":
    print("{}: f({}) = {}".format(func,x,relu(x)))
elif func == "elu":
    print("{}: f({}) = {}".format(func,x,elu(x)))
else:
    print("{} is not supported".format(func))