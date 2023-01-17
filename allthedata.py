import find_subpath
import fuel_consumption
import total_time


def whole_process(M, p, C, S, ps, qarray, W, V, mpg):
    Re, Rb, N = find_subpath.find_subpath(S, C, qarray)
    tt, fuelDict = total_time.total_time(ps, Re, N, W, V, mpg, qarray)
    tf = fuel_consumption.fuel_consumption(N, Re, Rb, M, p, fuelDict, qarray)
    return tt, tf
