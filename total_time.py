def total_time():
    tt=0
    s=1
    n=1
    while n < N:
        Tn=0
        Td=0
        T[0][s] = single_fuel_consumption(Td)[1]
        Tn= Tn + T[0][s]
        Tn = Tn + Ps
        while s = Re[n]:
            Td =Tn
            T[s][s+1] = single_fuel_consumption(Td)[1]
            Tn = Tn + T[s][s+1]
            s+=1
            Tn =T +Ps
        Td = Tn
        T[s][0] = single_fuel_consumption(Td)[1]
        Tn = Tn + T[s][0]
        TT = TT + Tn
        n+=1
        s+=1
    return TT