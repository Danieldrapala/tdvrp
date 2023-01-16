import find_subpath
import fuel_consumption
import total_time

qarray = [1,3,4,5,6,8]
routePlan = []



def all_process(routePlan):
    find_subpath.find_subpath()
    TT, Td, F = total_time.total_time()
    TF = fuel_consumption.fuel_consumption()
    return TT, TF

