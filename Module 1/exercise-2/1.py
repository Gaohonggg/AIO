def slidingwindow(l,k):
    result = []
    for i in range(len(l)-k+1):
        result = result + [max(l[i:i+k])]
    return result

l = []
print("Enter the size of list: ",end="")
s = int(input())
for i in range(s):
    print("Enter the {}th element: ".format(i+1),end="")
    temp = int(input())
    l = l + [temp]
print( l )
k = int(input("Enter window size: "))
print("the result is: {}".format(slidingwindow(l,k)))
