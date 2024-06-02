factorial = {1:1,2:2}
def fact(n):
    global factorial
    if n<=0:
        return 1
    else:
        if factorial.get(n)==None:
            result = n*fact(n-1)
            factorial[n] = result
            return result
        else:
            return factorial[n]

def approx_sin(x,n):
    result = 0
    for i in range(n):
        result += pow(-1,i)*(pow(x,2*i+1))/fact(2*i+1)
    return result
def approx_cos(x,n):
    result = 0
    for i in range(n):
        result += pow(-1,i)*(pow(x,2*i))/fact(2*i)
    return result
def approx_sinh(x,n):
    result = 0
    for i in range(n):
        result += (pow(x,2*i+1))/fact(2*i+1)
    return result
def approx_cosh(x,n):
    result = 0
    for i in range(n):
        result += (pow(x,2*i))/fact(2*i)
    return result

x = float(input("Nhap x theo do: "))
x = x/180*3.14
n = int(input("Nhap n: "))
print("approx_sin(x={}, n={}) = {}".format(x,n,approx_sin(x,n)))
print("approx_cos(x={}, n={}) = {}".format(x,n,approx_cos(x,n)))
print("approx_sinh(x={}, n={}) = {}".format(x,n,approx_sinh(x,n)))
print("approx_cosh(x={}, n={}) = {}".format(x,n,approx_cosh(x,n)))