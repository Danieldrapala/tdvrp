from single_fuel_consumption import single_fuel_consumption


def fuel_consumption(N, Ren, Rbn, qarr, M, p, Tdarray, Taarry, Warray, D, V, gphArray):
    TF = 0
    n = 1
    while n <= N:
        s = Ren
        Q = 0
        F = single_fuel_consumption(s,0, Tdarray, Taarry, Warray, D, V, gphArray)
        while s == Rbn:
            Q+= qarr[s]
            F = F + single_fuel_consumption(s-1, s, Tdarray, Taarry, Warray, D, V, gphArray) * (1+p(Q/M))
            s-=1
        Q += qarr[s]
        F = F + single_fuel_consumption(0,s, Tdarray, Taarry, Warray, D, V, gphArray) * (1 +p(Q/M))
        TF = TF + F
        n+=1
    return TF
