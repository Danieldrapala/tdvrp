from matplotlib import pyplot as plt

from jsp_fwk import JSProblem
from jsp_fwk.solver import GoogleORCPSolver

# load benchmark problem
problem = JSProblem(benchmark='ft10')

# solve problem with user defined solver
s = GoogleORCPSolver()
s.solve(problem=problem, interval=100)