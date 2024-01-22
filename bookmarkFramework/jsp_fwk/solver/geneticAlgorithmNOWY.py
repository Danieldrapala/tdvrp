import random
import time
from copy import copy
from typing import Tuple, Any

import plotly.graph_objs as go
from jsp_fwk import JSSolution, JSProblem, JSSolver, OperationStep
from jsp_fwk.common.exception import JSPException
from operator import attrgetter

from jsp_fwk.solver.dispatching_rule import DisPatchingRules


class GeneticAlgorithmSolverNowy(JSSolver):
    """Simulated Annealing Solver."""

    def __init__(self, name: str = None, mutation_probability: float = None,
                 population_size: int = 10, selection_size: int = 2, n_iterations: int = 100) -> None:
        """Simulated Annealing.

        Args:
            name (str, optional): Solver name.
        """
        super().__init__(name)
        self.best_solution = None
        self.result_population = None
        self.selection_size = selection_size
        self.mutation_probability = mutation_probability
        self.n_iterations = n_iterations
        self.population_size = population_size

    def do_solve(self, problem: JSProblem):

        GAP = 2
        # population = [self.generate_chromosome_solution(solution, self.generate_chromosome_by_tail(solution, i, GAP)) for i in range(self.population_size)]
        population = [self.generate_chromosome_solution(problem, i=i, gap=GAP) for i in range(self.population_size)]

        best_solution = min(population, key=attrgetter('makespan'))
        problem.update_solution(best_solution)
        iterations = 0
        # get static data
        next_population = []
        x_values = []
        j_values = []
        while not iterations >= self.n_iterations:
            next_population = []
            parent1 = self.random_solution(population)
            parent2 = self.random_solution(population)
            # parent1 = self.tournament_solution(population, self.selection_size)
            # parent2 = self.tournament_solution(population, self.selection_size)
            # #

            child1 = self.crossoverox(problem, parent1.chromosome, parent2.chromosome, self.mutation_probability)
            child2 = self.crossoverox(problem, parent2.chromosome, parent1.chromosome, self.mutation_probability)


                # add best 2 individuals to next generation if they are not already in the next generation (elitist strategy)
            sorted_individuals = sorted([parent1, parent2, child1, child2], key=attrgetter("makespan"))
            added = 0
            index = 0
            while added < 2 and index < len(sorted_individuals):
                if sorted_individuals[index] not in next_population:
                    next_population.append(sorted_individuals[index])
                    added += 1
                index += 1
            # if parent1, parent2, child1, and child2 are all in next_population, add random solutions
            while added < 2:
                print("ile razy do tego dochodzi")
                # next_population.append(self.generate_chromosome_solution(problem, i=random.randrange(len(next_population)), gap=GAP))
                next_population.append(self.generate_chromosome_solution(problem,self.sorted_ordered_ops_list(problem), i=random.randrange(len(next_population)), gap=GAP))
                added += 1
            j_values.append(min(child1.makespan, child2.makespan))
            # check for better solution than best_solution
            if min(child1.makespan, child2.makespan) < best_solution.makespan:
                best_solution = min([child1, child2], key=attrgetter("makespan"))
                print('>%d  = %.5f' % (iterations, best_solution.makespan))
                problem.update_solution(best_solution)
            iterations += 1
            next_population += population
            population = next_population

        x_values = list(range(len(j_values)))
        # Create a trace for the scatter plot
        trace = go.Scatter(x=x_values, y=j_values, mode='markers', marker=dict(size=2))

        # Create a layout for the plot
        layout = go.Layout(title='Plot of number iterations and makespan output', xaxis=dict(title='Index of j_values'),
                           yaxis=dict(title='j values'))

        # Create a figure with the trace and layout
        fig = go.Figure(data=[trace], layout=layout)

        # Show the plot
        fig.show()
        self.best_solution = best_solution
        self.result_population = next_population
        problem.update_solution(best_solution)
        print("koniec")

    def createChild(self, problem, parent1, parent2, ux, mutation):
        child1 = []
        for idx, allele in enumerate(ux):
            if allele >= 0.75:
                child1.append((parent1[idx][0],parent2[idx][1]))
            else:
                child1.append((parent1[idx][0],parent1[idx][1]))
        if random.random() < mutation:
            pos1 = random.randrange(len(child1))
            pos2 = random.randrange(len(child1))
            child1[pos1], child1[pos2] = child1[pos2], child1[pos1]
        return self.generate_chromosome_solution(problem, child1)

    def crossoverox(self, problem, parent1, parent2, mutation):
        child1 = parent1.copy()
        ra = random.randrange(2, int(0.4*len(parent1)))
        pos = random.sample(range(len(parent1)),ra)
        infected_values=[parent1[posx] for posx in pos]
        indexes_of_infected_values_parent2 = sorted([parent2.index(infected_value) for infected_value in infected_values])
        i = 0
        for gnome in range(len(parent2)):
            if i < len(indexes_of_infected_values_parent2) and gnome == indexes_of_infected_values_parent2[i]:
                child1[gnome] =infected_values[i]
                i+=1
            else:
                child1[gnome] = parent2[gnome]
        if random.random() < mutation:
            pos1 = random.randrange(len(child1))
            pos2 = random.randrange(len(child1))
            child1[pos1], child1[pos2] = child1[pos2], child1[pos1]
        return self.generate_chromosome_solution(problem, child1)

    def random_solution(self, popchrom):
        value = random.randrange(0, len(popchrom))
        return popchrom.pop(value)

    def tournament_solution(self, popchrom, tor_size):
        sel_ind= random.sample(range(len(popchrom)),tor_size)
        selection_sorted = sorted([index for index in sel_ind], key=lambda index: popchrom[index].makespan)
        return popchrom.pop(selection_sorted[0])

    def generate_ux_crossover(self, size):
        return [random.random() for _ in range(size)]

    def generate_chromosome_solution(self, problem, chromosome = None, i=None, gap=None):
        solution = JSSolution(problem)
        if chromosome is None:
            chromosome = random.sample(range(len(problem.ops)), len(problem.ops))
            ORDEREDLIST = self.sorted_ordered_ops_list(problem, chromosome)
            # ORDEREDLIST = self.generate_chromosome_by_tail(solution, i, gap)
        else:
            ORDEREDLIST = self.sorted_ordered_ops_list(problem, chromosome)
        solution.chromosome = chromosome
        '''One iteration applying priority dispatching rule.'''
        # move form
        # collect imminent operations in the processing queue
        head_ops = solution.imminent_ops
        # dispatch operation by priority
        OrdersQue=copy(ORDEREDLIST)
        while head_ops:
            # dispatch operation with the first priority
            job_id =OrdersQue.pop(0)[0]
            op: OperationStep = [op for op in head_ops if op.source.job.id == job_id][0]
            solution.dispatch(op)
            # update imminent operations
            pos = head_ops.index(op)
            next_job_op = op.next_job_op
            if next_job_op is None:
                head_ops = head_ops[0:pos] + head_ops[pos + 1:]
            else:
                head_ops[pos] = next_job_op
        return solution

    def generate_chromosome_by_tail(self, solution, i, GAP):
        lista_prio = [(op.source.job.id, - random.randrange(op.tail, (op.tail + 1) + i * GAP) / (max(solution.ops, key=lambda op: op.tail).tail + i * GAP)) for op in solution.ops]
        return sorted(lista_prio, key=lambda tuple:tuple[1])

    def generate_random_chromosome(self, solsize):
        return [random.random() for i in range(len(solsize.ops))]

    def sorted_ordered_ops_list(self, solsize, chromosome):
        return sorted([(i % len(solsize.jobs), chromosome[i]) for i in range(len(solsize.ops))],key=lambda tuple: tuple[1])