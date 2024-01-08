
import random

from jsp_fwk import JSSolution, JSProblem
from jsp_fwk.common.exception import JSPException


class SimulatedAnnealingSolver(JSSolver):
    '''Simulated Annealing Solver.'''

    def __init__(self, name :str =None, n_iterations :int =100, mutation_probability :float =None) -> None:
        '''Simulated Annealing.

        Args:
            name (str, optional): Solver name.
        '''
        super().__init__(name)
        self.mutation_probability = mutation_probability
        self.n_iterations = n_iterations



    def do_solve(self, problem: JSProblem, population_size: int):
        solution = JSSolution(problem=problem)
        population = [self.get_random_solution(problem) for _ in range(population_size)]
        best_solution = min(population)
        iterations = 0

        # get static data
        data = population[0].data
        dependency_matrix_index_encoding = data.population
        usable_machines_matrix = data.usable_machines_matrix

        not_done = True
        while not iterations >= self.iterations:
            next_population = []
            while len(population) > self.selection_size and not_done:

                parent1 = self.random_method(population, self.selection_size)
                parent2 = self.random_method(population, self.selection_size)

                # breed the parents to produce child1 (parent1 cross parent2)
                # Note mutation happens in crossover function
                feasible_child = False
                while not feasible_child:
                    # the try except block is because sometimes the crossover operation results in a setup of -1
                    # which then produces an infeasible solution. This is due to the sequence dependency setup times matrix not allowing for wait time.
                    try:
                        child1 = self.crossover(problem, parent1, parent2,
                                           self.mutation_probability, dependency_matrix_index_encoding,
                                           usable_machines_matrix)
                        if child1 != parent1 and child1 != parent2:
                            feasible_child = True
                    except JSPException:
                        if iterations >= self.iterations:
                            not_done = False
                            break

                # breed the parents to produce child2 (parent2 cross parent1)
                feasible_child = False
                while not feasible_child:
                    try:
                        child2 = self.crossover(problem, parent2, parent1,
                                           self.mutation_probability, dependency_matrix_index_encoding,
                                           usable_machines_matrix)
                        if child2 != parent1 and child2 != parent2:
                            feasible_child = True
                    except JSPException:
                        if iterations >= self.iterations:
                            not_done = False
                            break

                # add best 2 individuals to next generation if they are not already in the next generation (elitist strategy)
                if not_done:
                    sorted_individuals = sorted([parent1, parent2, child1, child2])
                    added = 0
                    index = 0
                    while added < 2 and index < len(sorted_individuals):
                        if sorted_individuals[index] not in next_population:
                            next_population.append(sorted_individuals[index])
                            added += 1
                        index += 1

                    # if parent1, parent2, child1, and child2 are all in next_population, add random solutions
                    while added < 2:
                        next_population.append(solution.get_random_solution())
                        added += 1
                else:
                    next_population.append(parent1)
                    next_population.append(parent2)

                # check for better solution than best_solution
                if min(child1, child2) < best_solution:
                    best_solution = min(child1, child2)
            iterations += 1
            next_population += population
            population = next_population

        self.best_solution = best_solution
        self.result_population = next_population
        problem.update_solution(best_solution)
    def get_random_solution(self, solution :JSSolution):
            '''One iteration applying priority dispatching rule.'''
            # move form
            # collect imminent operations in the processing queue
            head_ops = solution.imminent_ops

            # dispatch operation by priority
            while head_ops:
                # dispatch operation with the first priority
                op = min(head_ops, key=lambda op: op.source.duration)
                solution.dispatch(op)

                # update imminent operations
                pos = head_ops.index(op)
                next_job_op = op.next_job_op
                if next_job_op is None:
                    head_ops = head_ops[0:pos] + head_ops[pos +1:]
                else:
                    head_ops[pos] = next_job_op
    def crossover(self, problem, parent1, parent2, mutation_probability, dependency_matrix_index_encoding, usable_machines_matrix):
        # TODO
        neighbour_solution = JSSolution(problem=problem)
        return neighbour_solution

    def random_sol(self, population):
        # TODO
        return population.pop(random.randrange(0, len(population)))