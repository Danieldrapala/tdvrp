import logging
from jsp_fwk import (JSProblem, JSSolution)
from jsp_fwk.solver import PuLPSolver, GoogleORCPSolver
from jsp_fwk.solver.geneticAlgorithm import GeneticAlgorithmSolver
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

    # ----------------------------------------
    # test built-in solver
    # ----------------------------------------
    # google or-tools
    # s = GoogleORCPSolver()

    # priority dispatching
    # rules = ['ct']
    # s = PriorityDispatchSolver(rule=rules[-1])
    #
    # s = GeneticAlgorithmSolver(mutation_probability=0.1, population_size=25, n_iterations=10000)
    s = SimulatedAnnealingSolver(n_iterations=10000, temp=200)
    # s = TabuSearchSolver(n_iterations=10000,num_solutions_to_find=1, tabu_list_size=25)

    # pulp solver
    # s = PuLPSolver(max_time=60)

    # ----------------------------------------
    # solve and result
    # ----------------------------------------
    s.solve(problem=problem, interval=2000, callback=print_intermediate_solution)
    s.wait()
    print('----------------------------------------')

    logging.info(f'Makespan: {problem.solution.machine_ops}')
    logging.info(f'Makespan: {problem.solution.job_ops}')
    logging.info(f'Makespan: {problem.solution.sorted_ops}')
    logging.info(f'Makespan: {problem.solution.ops}')

    if s.status:
        print(f'Problem: {len(problem.jobs)} jobs, {len(problem.machines)} machines')
        print(f'Optimum: {problem.optimum}')
        print(f'Solution: {problem.solution.makespan}')
        print(f'Terminate successfully in {s.user_time} sec.')
    else:
        print(f'Solving process failed in {s.user_time} sec.')