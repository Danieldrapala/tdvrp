import random
import time
import plotly.graph_objs as go
from jsp_fwk import JSSolution, JSProblem, JSSolver
from jsp_fwk.common.exception import JSPException
from operator import attrgetter

from jsp_fwk.solver.dispatching_rule import DisPatchingRules


class GeneticAlgorithmSolver(JSSolver):
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
        GAP = 5
        # population = [self.generate_chromosome_solution(solution, self.generate_chromosome_by_tail(solution, i, GAP)) for i in range(self.population_size)]
        popchrom = list(map(list, zip(*[self.generate_chromosome_solution(problem,i=i,gap=GAP) for i in range(self.population_size)])))
        population = popchrom[0]
        best_solution = min(population, key=attrgetter('makespan'))
        iterations = 0

        # get static data
        next_population = []
        not_done = True
        j_values = []
        while not iterations >= self.n_iterations:
            next_population = []
            while len(population) > self.selection_size and not_done:

                parent1 = self.random_solution(population)
                parent2 = self.random_solution(population)
                ux = self.generate_ux_crossover(len(parent1.ops))
                # breed the parents to produce child1 (parent1 cross parent2)
                # Note mutation happens in crossover function
                child1 = self.createChild(problem, parent1.chromosome, parent2.chromosome, ux)
                # breed the parents to produce child2 (parent2 cross parent1)

                child2 = self.createChild(problem, parent2.chromosome, parent1.chromosome, ux)

                # add best 2 individuals to next generation if they are not already in the next generation (elitist strategy)
                sorted_individuals = sorted([parent1, parent2, child1[0], child2[0]], key=attrgetter("makespan"))
                added = 0
                index = 0
                while added < 2 and index < len(sorted_individuals):
                    if sorted_individuals[index] not in next_population:
                        next_population.append(sorted_individuals[index])
                        added += 1
                    index += 1
                # if parent1, parent2, child1, and child2 are all in next_population, add random solutions
                while added < 2:
                    # next_population.append(self.generate_chromosome_solution(randomSolution, self.generate_chromosome_by_tail(solution,len(next_population), GAP)))
                    next_population.append(self.generate_chromosome_solution(problem, len(next_population), GAP))
                    added += 1
                j_values.append(min(child1[0].makespan, child2[0].makespan))
                # check for better solution than best_solution
                if min(child1[0].makespan, child2[0].makespan) < best_solution.makespan:
                    best_solution = min([child1[0], child2[0]], key=attrgetter("makespan"))
                    problem.update_solution(best_solution)
            iterations += 1
            next_population += population
            population = next_population
        x_values = list(range(len(j_values)))

        # Create a trace for the scatter plot
        trace = go.Scatter(x=x_values, y=j_values, mode='markers', marker=dict(size=10))

        # Create a layout for the plot
        layout = go.Layout(title='Scatter Plot of i and j', xaxis=dict(title='Index of j_values'),
                           yaxis=dict(title='j values'))

        # Create a figure with the trace and layout
        fig = go.Figure(data=[trace], layout=layout)

        # Show the plot
        fig.show()
        self.best_solution = best_solution
        self.result_population = next_population
        problem.update_solution(best_solution)
        print("koniec")

    def createChild(self, problem, parent1, parent2, ux):
        child1 = []
        for idx, allele in enumerate(ux):
            if allele >= 0.75:
                child1.append(parent2[idx])
            else:
                child1.append(parent1[idx])
        return self.generate_chromosome_solution(problem, child1)

    def random_solution(self, popchrom):
        value = random.randrange(0, len(popchrom))
        return popchrom.pop(value)

    def generate_ux_crossover(self, size):
        return [random.random() for _ in range(size)]

    def generate_chromosome_solution(self, problem, chromosome=None, i=None, gap=None):

        solution = JSSolution(problem)
        if chromosome is None:
            chromosome = self.generate_chromosome_by_tail(solution,i,gap)
        '''One iteration applying priority dispatching rule.'''
        # move form
        # collect imminent operations in the processing queue
        solution.chromosome = chromosome
        head_ops = solution.imminent_ops
        initial_heads = head_ops
        # dispatch operation by priority

        while head_ops:
            # dispatch operation with the first priority
            if initial_heads:
                op = min(initial_heads, key=lambda op: DisPatchingRules.MTWR(op, solution))
            else:
                op = max(head_ops, key=lambda op: chromosome[op.source.id])
            solution.dispatch(op)

            if initial_heads:
                pos = initial_heads.index(op)
                initial_heads = initial_heads[0:pos] + initial_heads[pos + 1:]
            # update imminent operations
            pos = head_ops.index(op)
            next_job_op = op.next_job_op
            if next_job_op is None:
                head_ops = head_ops[0:pos] + head_ops[pos + 1:]
            else:
                head_ops[pos] = next_job_op
        return solution, chromosome

    def generate_chromosome_by_tail(self, solution, i, GAP):
        return [random.randrange(op.tail, (op.tail +1)+ i * GAP) / (max(solution.ops, key=lambda op: op.tail).tail + i * GAP) for
                op in solution.ops]

    def generate_random_chromosome(self, solsize):
        return [random.random() for _ in range(solsize)]
