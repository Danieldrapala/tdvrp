import logging
from jsp_fwk import (JSProblem, JSSolution, BenchMark)
from jsp_fwk.solver import (GoogleORCPSolver, PriorityDispatchSolver, PuLPSolver)
from jsp_fwk.solver.gaUX import GeneticAlgorithmSolverUX
from jsp_fwk.solver.gaUXRANDOM import GeneticAlgorithmSolverUXRANDOM
from jsp_fwk.solver.geneticAlgorithm import GeneticAlgorithmSolver
from jsp_fwk.solver.geneticAlgorithmNOWY import GeneticAlgorithmSolverNowy
from jsp_fwk.solver.geneticAlgorithmOXPERMRANDOM import GeneticAlgorithmOXPERMRANDOM
from jsp_fwk.solver.geneticAlgorithmRS import GeneticAlgorithmSolverRS
from jsp_fwk.solver.geneticAlgorithmRSUX import GeneticAlgorithmSolverRSUX
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
    names = ['ft10']
    problems = [JSProblem(benchmark=name) for name in names]

    # ----------------------------------------
    # test built-in solver
    # ----------------------------------------
    solvers = []

    # # googl or-tools
    # s = GoogleORCPSolver(max_time=300)
    # solvers.append(s)
    # # priority dispatching
    # solvers.append(PriorityDispatchSolver(rule='mtwr', name='mtwr'.upper()))

    # PuLP solver
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=50, n_iterations=100, selection_size=25))
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=50, n_iterations=500, selection_size=25))
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=200, n_iterations=1000, selection_size=100))
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=400, n_iterations=1000, selection_size=200))
    solvers.append(GeneticAlgorithmSolverUX(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    solvers.append(GeneticAlgorithmSolverUXRANDOM(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    solvers.append(GeneticAlgorithmOXPERMRANDOM(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    solvers.append(GeneticAlgorithmSolver(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    #ox
    solvers.append(GeneticAlgorithmSolverRS(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    solvers.append(GeneticAlgorithmSolverRSUX(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    solvers.append(GeneticAlgorithmSolverRS(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    solvers.append(GeneticAlgorithmSolver(mutation_probability=0.15, population_size=50, n_iterations=1000, selection_size=25))

    # ----------------------------------------
    # solve and result
    # ----------------------------------------
    benchmark = BenchMark(problems=problems, solvers=solvers, num_threads=6)
    benchmark.run(show_info=True)