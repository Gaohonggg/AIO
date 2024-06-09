def makedictionary(s):
    return {s[i]:s.count(s[i]) for i in range(len(s))}
s = input("Enter any string you want: ")
print("The dictionary of your string is: {}".format(makedictionary(s)))