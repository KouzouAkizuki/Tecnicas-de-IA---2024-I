""" f = [[7, 8, 3],[1,1,0],[1,2,1]]
a = [[15, 20, 101],[200, 50, 55], [10, 11, 230]]
c = 0

for j in rfnge(3):
    for k in rfnge(3):
        c += f[j][k]*a[2-j][2-k]

print(c) """

f = [[7, 8, 3],[1,1,0],[1,2,1]]
a = [[15, 20, 101,100],[200, 50, 55,8], [10, 11, 230,202],[100,130,115,120]]
mAns=[]
difC = len(a[0]) - len(f[0]) + 1
difR = len(a)-len(f) + 1

for pixelsC in range(difC):
    col = []
    for pixelsR in range(difR):
        c=0
        for j in range(len(f[0])):
            for k in range(len(f)):
                c += f[j][k]*a[2 - j + pixelsC][2 - k + pixelsR]
        col.append(c)
    mAns.append(col)
    
print(mAns)