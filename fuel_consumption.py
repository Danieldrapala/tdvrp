import allthedata
from single_fuel_consumption import single_fuel_consumption


def fuel_consumption(N, Re, Rb, M, p, fuelDict, qarray):
    TF = 0
    n = 1
    while n <= N:
        s = Re[n]
        Q = 0
        F = fuelDict[s,0]
        while s == Rb[n]:
            Q+= qarray[s].q
            F = F + fuelDict[s-1,s] * (1+p(Q/M))
            s-=1
        Q += qarray[s].q
        F = F + fuelDict[0,s] * (1 +p(Q/M))
        TF = TF + F
        n+=1
    return TF
