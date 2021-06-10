n = input()
n = int(n)
file = open("B_" + str(n) + ".m", "w")
T = []
for i in range(10):
    a = ""
    if i+1 == n:
        a += "1 1 1 1 1 1 "
    else:
        a += "0 0 0 0 0 0 "
    
    for j in range(6,10):
        if j == i:
            a += "1 "
        else:
            a += "0 "
    
    a += "\n"
    
    T.append(a)

file.writelines(T)
file.close()
