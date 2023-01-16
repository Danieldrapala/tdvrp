import allthedata
def find_subpath( S, C ):
    Rb=[]
    Re=[]
    Rb[1] = 1
    Re[1] = 1
    s = 1
    n = 1
    Q = allthedata.qarray[s]
    N = 1
    while s < S:
        Q = Q + allthedata.qarray[s+1]
        if Q > C:
            Re[n] = s
            Q = 0
            N = n + 1
            Rb[n] = s + 1
        s += 1
    Re[n] = s
    return {Re, Rb, N}
