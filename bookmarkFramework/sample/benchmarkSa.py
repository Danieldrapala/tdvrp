import logging
from jsp_fwk import (JSProblem, JSSolution, BenchMark)
from jsp_fwk.solver import (GoogleORCPSolver, PriorityDispatchSolver, PuLPSolver)
from jsp_fwk.solver.simulatedAnealing import SimulatedAnnealingSolver
from jsp_fwk.solver.simulatedAnealingITEr import SimulatedAnnealingSolverIter
from jsp_fwk.solver.simulatedAnealingITErRANDOM import SimulatedAnnealingSolverIterRANDOM
from jsp_fwk.solver.simulatedAnealingRANDOM import SimulatedAnnealingSolverRANDOM


def print_intermediate_solution(solution:JSSolution):
    logging.info(f'Makespan: {solution.makespan}')


if __name__=='__main__':

    # ----------------------------------------
    # create problem from benchmark
    # ----------------------------------------
    names = ['abz5','abz6','abz7','abz8','abz9']
    problems = [JSProblem(benchmark=name) for name in names]

    # ----------------------------------------
    # test built-in solver
    # ----------------------------------------
    solvers = []

    # googl or-tools
    s = GoogleORCPSolver(max_time=300)
    solvers.append(s)
    # priority dispatching
    solvers.append(PriorityDispatchSolver(rule='mtwr', name='mtwr'.upper()))

    # PuLP solver
    solvers.append(SimulatedAnnealingSolver(n_iterations=100, temp=400))
    solvers.append(SimulatedAnnealingSolver(n_iterations=10, temp=400))
    solvers.append(SimulatedAnnealingSolver(n_iterations=1, temp=400))
    solvers.append(SimulatedAnnealingSolverIter(n_iterations=16100, temp=200))
    solvers.append(SimulatedAnnealingSolverIter(n_iterations=1610, temp=200))
    solvers.append(SimulatedAnnealingSolverIter(n_iterations=161, temp=200))
    solvers.append(SimulatedAnnealingSolverIter(n_iterations=1000, temp=300))
    solvers.append(SimulatedAnnealingSolverIter(n_iterations=1000, temp=200))
    solvers.append(SimulatedAnnealingSolverIter(n_iterations=1000, temp=100))
    solvers.append(SimulatedAnnealingSolverIter(n_iterations=1000, temp=20))
    solvers.append(SimulatedAnnealingSolverIterRANDOM(n_iterations=10000, temp=400))
    solvers.append(SimulatedAnnealingSolverIterRANDOM(n_iterations=5000, temp=400))
    solvers.append(SimulatedAnnealingSolverIterRANDOM(n_iterations=1000, temp=400))
    solvers.append(SimulatedAnnealingSolverRANDOM(n_iterations=10, temp=400))
    solvers.append(SimulatedAnnealingSolverRANDOM(n_iterations=100, temp=400))
    solvers.append(SimulatedAnnealingSolverRANDOM(n_iterations=1, temp=400))
    # ----------------------------------------
    # solve and result
    # ----------------------------------------
    benchmark = BenchMark(problems=problems, solvers=solvers, num_threads=6)
    benchmark.run(show_info=True)