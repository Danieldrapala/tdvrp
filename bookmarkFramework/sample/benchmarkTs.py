import logging
from jsp_fwk import (JSProblem, JSSolution, BenchMark)
from jsp_fwk.solver import (GoogleORCPSolver, PriorityDispatchSolver, PuLPSolver)
from jsp_fwk.solver.simulatedAnealing import SimulatedAnnealingSolver
from jsp_fwk.solver.simulatedAnealingITEr import SimulatedAnnealingSolverIter
from jsp_fwk.solver.simulatedAnealingITErRANDOM import SimulatedAnnealingSolverIterRANDOM
from jsp_fwk.solver.simulatedAnealingRANDOM import SimulatedAnnealingSolverRANDOM
from jsp_fwk.solver.tabuSearch import TabuSearchSolver


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
    solvers.append(TabuSearchSolver(n_iterations=10000, num_solutions_to_find=50, tabu_list_size=500, neighborhood_size=10,
                         reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=50, tabu_list_size=500, neighborhood_size=10,reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=100, num_solutions_to_find=50, tabu_list_size=500, neighborhood_size=10,reset_threshold=50))

    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=50, tabu_list_size=50, neighborhood_size=10,reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=50, tabu_list_size=100, neighborhood_size=10,reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=50, tabu_list_size=200, neighborhood_size=10,reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=10, tabu_list_size=100, neighborhood_size=10,reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=50, tabu_list_size=100, neighborhood_size=10,reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=100, tabu_list_size=100, neighborhood_size=10,reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=50, tabu_list_size=100, neighborhood_size=5,reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=50, tabu_list_size=100, neighborhood_size=10,reset_threshold=50))
    solvers.append(TabuSearchSolver(n_iterations=1000, num_solutions_to_find=50, tabu_list_size=100, neighborhood_size=20,reset_threshold=50))

    # ----------------------------------------
    # solve and result
    # ----------------------------------------
    benchmark = BenchMark(problems=problems, solvers=solvers, num_threads=6)
    benchmark.run(show_info=True)