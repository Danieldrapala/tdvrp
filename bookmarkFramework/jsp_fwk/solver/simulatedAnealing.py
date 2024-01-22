import random
from math import exp, floor

from numpy.random import rand

from jsp_fwk import JSSolution, JSProblem, JSSolver, OperationStep
from jsp_fwk.solver.dispatching_rule import DisPatchingRules
import plotly.graph_objs as go


class SimulatedAnnealingSolver(JSSolver):
    '''Simulated Annealing Solver.'''

    def __init__(self, name :str =None, n_iterations :int =100, temp :int =2000) -> None:
        '''Simulated Annealing.

        Args:
            name (str, optional): Solver name.
        '''
        super().__init__(name)
        self.temp = temp
        self.n_iterations = n_iterations

    # def do_solveA(self, problem: JSProblem):
    #     solution, permutation = self.generate_solution(problem, self.generate_random_permutation(problem))
    #     # solution, permutation = self.generate_solution(problem)
    #     best = solution
    #     problem.update_solution(best)
    #     best_permutation = permutation
    #     curr_permutation = best_permutation
    #     curr = solution
    #     for i in range(self.n_iterations):
    #         # candidate_permutation = self.getNeighbour(curr_permutation)
    #         candidate_permutation = self.getNeighbourClose(curr_permutation)
    #         # candidate_permutation = self.getNeighbourTenPercent(curr_permutation)
    #         candidate = self.generate_solution(problem, candidate_permutation)[0]
    #         if candidate.makespan < best.makespan:
    #             best_permutation, best = candidate_permutation, candidate
    #             problem.update_solution(best)
    #             print('>%d f(%s) = %.5f' % (i, best, best.makespan))
    #             # difference between candidate and current point evaluation
    #         # t = t * min(float(1- (i+1)/self.n_iterations),0.99)
    #         t = self.temp / (float(i+1))
    #         # calculate metropolis acceptance criterion
    #         diff = candidate.makespan - curr.makespan
    #         print("candidate.makespan", candidate.makespan)
    #         if diff < 0:
    #             curr_permutation, curr = candidate_permutation, candidate
    #         else:
    #             metropolis = exp(-diff / t)
    #             if rand() < metropolis:
    #                 curr_permutation, curr = candidate_permutation, candidate
    #         # calculate temperature for current epoch
    #         print("iteracja", i)
    #         print("temperatura",t)
    #         if t == 0:
    #             return [best_permutation, best]
    #         # check if we should keep the new point
    def do_solve(self, problem: JSProblem):
        solution, permutation = self.generate_solution(problem, self.generate_random_permutation(problem))
        # solution, permutation = self.generate_solution(problem)
        best = solution
        problem.update_solution(best)
        best_permutation = permutation
        curr_permutation = best_permutation
        curr_makespan = best.makespan
        # run the algorithm
        t = self.temp
        tk = 10e-3
        j_values=[]
        while t > tk:
            for i in range(self.n_iterations):
                # take a step
                candidate_permutation = self.getNeighbour(curr_permutation, len(problem.machines), len(problem.jobs))
                # candidate_permutation = self.getNeighbourJustGoDeeper(curr_permutation)
                # candidate_permutation = self.getNeighbourClose(curr_permutation)
                # evaluate candidate point
                candidate = self.generate_solution(problem, candidate_permutation)[0]

                j_values.append(candidate.makespan)
                # check for new best solution
                if candidate.makespan < best.makespan:
                    # store new best point
                    # report progress
                    # difference between candidate and current point evaluation
                    best_permutation, best = candidate_permutation, candidate
                    problem.update_solution(best)
                    print('>%d f(%s) = %.5f' % (i, t, best.makespan))
                diff = candidate.makespan - curr_makespan

                # calculate temperature for current epoch
                # t = self.temp / (float(i+1))
                # calculate metropolis acceptance criterion
                # check if we should keep the new point
                if diff < 0:
                    curr_permutation, curr_makespan = candidate_permutation, candidate.makespan
                elif diff > 0:
                    metropolis = exp(-diff / t)
                    if rand() < metropolis:
                        curr_permutation, curr_makespan = candidate_permutation, candidate.makespan
            # t = self.temp /float(iterations)
            # iterations+=1
            t = 0.5 * t
        print(t)
        x_values = list(range(len(j_values)))
        trace = go.Scatter(x=x_values, y=j_values, mode='markers', marker=dict(size=2))
        layout = go.Layout(title='Scatter Plot of i and j', xaxis=dict(title='Index of j_values'),
                           yaxis=dict(title='j values'))
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()
        return [best_permutation, best]

    def getNeighbourTenPercent(self, permutation):
        mutated_permutation = permutation.copy()
        percent = random.randrange(floor(len(permutation)*0.1))/2
        for _ in range(int(percent)):
            # Wybierz losowo dwie pozycje
            pos1, pos2 = random.sample(range(len(permutation)), 2)
            mutated_permutation[pos1], mutated_permutation[pos2] = mutated_permutation[pos2], mutated_permutation[pos1]
        return mutated_permutation

    def getNeighbourClose(self, permutation):
        mutated_permutation = permutation.copy()
        mutated_permutation_sorted = sorted(mutated_permutation)
        pos1 = random.randrange(len(permutation))
        get_index1 = mutated_permutation.index(mutated_permutation_sorted[pos1])
        if pos1 == len(permutation) - 1:
            get_index3 = mutated_permutation.index(mutated_permutation_sorted[pos1 - 1])
            mutated_permutation[get_index1], mutated_permutation[get_index3] = mutated_permutation[get_index3], mutated_permutation[get_index1]
        elif pos1 == 0:
            get_index2 = mutated_permutation.index(mutated_permutation_sorted[pos1 + 1])
            mutated_permutation[get_index1], mutated_permutation[get_index2] = mutated_permutation[get_index2], mutated_permutation[
                get_index1]
        else:
            get_index3 = mutated_permutation.index(mutated_permutation_sorted[pos1 + 1])
            mutated_permutation[get_index1], mutated_permutation[get_index3] = mutated_permutation[get_index3], mutated_permutation[
                get_index1]
        return mutated_permutation


    def getNeighbour(self, permutation, machines, jobs):
        mutated_permutation = permutation.copy()
        pos1 = random.randrange(len(permutation))
        x = random.randrange(jobs)
        y = int(random.randint(-1,1))
        pos2 = ((pos1 + x * machines) % (machines * jobs) + y) % (machines * jobs)
        mutated_permutation[pos1], mutated_permutation[pos2] = mutated_permutation[pos2], mutated_permutation[pos1]
        return mutated_permutation

    def getNeighbourJustGoDeeper(self, permutation):
        mutated_permutation = permutation.copy()
        pos1, pos2 = random.sample(range(len(permutation)), 2)
        temp_val = mutated_permutation[pos1]
        mutated_permutation.pop(pos1)
        mutated_permutation.insert(pos2,temp_val)
        return mutated_permutation

    def generate_solution(self, problem, permutation=None):
        solution = JSSolution(problem)
        if permutation is None:
            permutation = self.generate_initial_solution_permutation(solution)
        '''One iteration applying priority dispatching rule.'''
        # move form
        # collect imminent operations in the processing queue
        head_ops = solution.imminent_ops

        orderedList= [(i%problem.jobs, permutation[i]) for i in range(problem.ops)]
        while head_ops:
            # dispatch operation with the first priority
            job_id =orderedList.pop(0)[0]
            op: OperationStep = [op for op in head_ops if op.source.job.id == job_id][0]
            solution.dispatch(op)
            # update imminent operations
            pos = head_ops.index(op)
            next_job_op = op.next_job_op
            if next_job_op is None:
                head_ops = head_ops[0:pos] + head_ops[pos + 1:]
            else:
                head_ops[pos] = next_job_op
        return solution, permutation
        # dispatch operation by priority
        # while head_ops:
        #     # dispatch operation with the first priority
        #     op = max(head_ops, key=lambda op: permutation[op.source.id])
        #     solution.dispatch(op)
        #     # update imminent operations
        #     pos = head_ops.index(op)
        #     next_job_op = op.next_job_op
        #     if next_job_op is None:
        #         head_ops = head_ops[0:pos] + head_ops[pos + 1:]
        #     else:
        #         head_ops[pos] = next_job_op
        # return solution, permutation

    def generate_random_permutation(self, problem):
        return random.sample(range(0, len(problem.ops)), len(problem.ops))

    def generate_initial_solution_permutation(self, solution):
        return [op.tail for op in solution.ops]

