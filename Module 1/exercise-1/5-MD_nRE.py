def md_nre(y,y_hat,n,p):
    result = y**(1/n) - y_hat**(1/n)
    return result**p

y = float(input("Nhap y: "))
y_hat = float(input("Nhap y_hat: "))
n = int(input("Nhap n: "))
p = int(input("Nhap p: "))
print( md_nre(y,y_hat,n,p) )