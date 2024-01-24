import logging
from jsp_fwk import (JSProblem, JSSolution, BenchMark)
from jsp_fwk.solver import (GoogleORCPSolver, PriorityDispatchSolver, PuLPSolver)
from jsp_fwk.solver.gaUX import GeneticAlgorithmSolverUX
from jsp_fwk.solver.geneticAlgorithm import GeneticAlgorithmSolver
from jsp_fwk.solver.geneticAlgorithmNOWY import GeneticAlgorithmSolverNowy
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
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=50, n_iterations=500, selection_size=25))
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=50, n_iterations=100, selection_size=25))
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=50, n_iterations=500, selection_size=25))
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=100, n_iterations=500, selection_size=50))
    solvers.append(GeneticAlgorithmSolverNowy(mutation_probability=0.05, population_size=150, n_iterations=500, selection_size=75))
    solvers.append(GeneticAlgorithmSolverUX(mutation_probability=0.05, population_size=50, n_iterations=1000, selection_size=25))
    solvers.append(GeneticAlgorithmSolverUX(mutation_probability=0.05, population_size=50, n_iterations=500, selection_size=25))
    solvers.append(GeneticAlgorithmSolverUX(mutation_probability=0.05, population_size=50, n_iterations=100, selection_size=25))
    solvers.append(GeneticAlgorithmSolverUX(mutation_probability=0.05, population_size=200, n_iterations=500, selection_size=100))
    solvers.append(GeneticAlgorithmSolverUX(mutation_probability=0.05, population_size=100, n_iterations=500, selection_size=50))
    solvers.append(GeneticAlgorithmSolver(mutation_probability=0.05, population_size=50, n_iterations=7000, selection_size=25))
    solvers.append(GeneticAlgorithmSolver(mutation_probability=0.05, population_size=50, n_iterations=7000, selection_size=25))
    solvers.append(GeneticAlgorithmSolver(mutation_probability=0.05, population_size=50, n_iterations=7000, selection_size=25))
    solvers.append(GeneticAlgorithmSolver(mutation_probability=0.05, population_size=50, n_iterations=7000, selection_size=25))
    solvers.append(GeneticAlgorithmSolver(mutation_probability=0.05, population_size=50, n_iterations=7000, selection_size=25))

    # ----------------------------------------
    # solve and result
    # ----------------------------------------
    benchmark = BenchMark(problems=problems, solvers=solvers, num_threads=6)
    benchmark.run(show_info=True)