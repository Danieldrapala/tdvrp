def single_fuel_consumption(i, j, Tdarray, Taarry, Warray, D, V, gphArray):
    k = 1
    Fk= 0
    while Tdarray[i][j] < Warray[k+1]:
        k+=1
    Taarry[i][j] = Tdarray[i][j] + D[i][j]/V[k][i][j]
    F= gphArray[k][i][j] *  D[i][j]/V[k][i][j]
    while Taarry[i][j] >= Warray[k+1]:
        Taarry[i][j] = (Taarry[i][j] - Warray[k+1]) * V[k][i][j]/ V[k+1][i][j] + Warray[k+1]
        Fk = Fk + (Warray[k+1] - max(Tdarray[i][j], Warray[k]) *gphArray[k][i][j])
        F = Fk + (Taarry[i][j] -  Warray[k+1]) * gphArray[k+1][i][j]
        k+=1
    T = Taarry[i][j] - Tdarray[i][j]
    return {F,T}