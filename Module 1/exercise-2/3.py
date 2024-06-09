with open("P1_data.txt","r") as f:
    s = f.read()
    f.close()
print( s )

s = s.lower()
l = s.replace("\n"," ").split(" ")
dic = {l[i]:l.count(l[i]) for i in range(len(l))}
print( dic )