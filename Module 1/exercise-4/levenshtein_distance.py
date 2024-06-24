def buildmatrix(s,t):
    row = len(s) + 1
    col = len(t) + 1
    m = [[]]

    for i in range(col):
        m[0].append(i)

    for i in range(1,row):
        m = m + [[]]
        m[i].append(i)

    for i in range(1,col):
        for j in range(1,row):
            cost = 0 if s[j-1] == t[i-1] else 1
            m[j].append( min( m[j-1][i] + 1,
                              m[j][i-1] + 1,
                              m[j-1][i-1] + cost) )
    print( m[row-1][col-1] )
    return m[row-1][col-1]