import logging
from jsp_fwk import (JSProblem, JSSolution)
from jsp_fwk.solver import PuLPSolver, GoogleORCPSolver, PriorityDispatchSolver
from jsp_fwk.solver.gaUX import GeneticAlgorithmSolverUX
from jsp_fwk.solver.geneticAlgorithm import GeneticAlgorithmSolver
from jsp_fwk.solver.geneticAlgorithmNOWY import GeneticAlgorithmSolverNowy
from jsp_fwk.solver.simulatedAnealing import SimulatedAnnealingSolver
from jsp_fwk.solver.tabuSearch import TabuSearchSolver


def print_intermediate_solution(solution:JSSolution):
    logging.info(f'Makespan: {solution.makespan}')


class SimulatedAnealingSolver:
    pass


if __name__=='__main__':

    # ----------------------------------------
    # create problem from benchmark
    # ----------------------------------------
    problem = JSProblem(benchmark='ft10')

###########
    #GENETYCZNY
    #MTWR vs RANDOM
    #CHROMOSOM zwykły vs biased genetic key
    # tournament vs random selection
    # ----------------------------------------
    # test built-in solver
    # ----------------------------------------
    # google or-tools
    # s = GoogleORCPSolver()

    # priority dispatching
    rules = ['MTWR']
    # s = PriorityDispatchSolver(rule=rules[-1])
    #
    s = GeneticAlgorithmSolverUX(mutation_probability=0.05, population_size=30, n_iterations= 2500, selection_size=10)

    # pulp solver
    # s = PuLPSolver(max_time=60)

    # ----------------------------------------
    # solve and result
    # ----------------------------------------
    s.solve(problem=problem, callback=print_intermediate_solution)
    s.wait()
    print('----------------------------------------')

    if s.status:
        print(f'Problem: {len(problem.jobs)} jobs, {len(problem.machines)} machines')
        print(f'Optimum: {problem.optimum}')
        print(f'Solution: {problem.solution.makespan}')
        print(f'Terminate successfully in {s.user_time} sec.')
    else:
        print(f'Solving process failed in {s.user_time} sec.')