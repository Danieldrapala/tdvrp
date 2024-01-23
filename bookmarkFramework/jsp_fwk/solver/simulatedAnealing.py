import random
from copy import copy
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
        solution, permutation = self.generate_solution(problem, permutation=self.generate_random_permutation(problem))
        # solution, permutation = self.generate_solution(problem)
        print(solution.orderedseq)
        best = solution
        problem.update_solution(best)
        best_permutation = permutation
        curr_permutation = best_permutation
        curr_candidate = solution
        curr_makespan = best.makespan
        # run the algorithm
        t = self.temp
        tk = 10e-3
        j_values=[]
        while t > tk:
            print(t, curr_makespan)
            for i in range(self.n_iterations):
                # take a step
                candidate_sortedList = self.getNeighbourFromSolutionCriticalPath(solution= curr_candidate)
                # candidate_permutation = self.getNeighbourJustGoDeeper(curr_permutation)
                # candidate_permutation = self.getNeighbourClose(curr_permutation)
                # evaluate candidate point
                candidate = self.generate_solution(problem, sortedList= candidate_sortedList)[0]
                j_values.append(candidate.makespan)
                # check for new best solution
                if candidate.makespan < best.makespan:
                    # store new best point
                    # report progress
                    # difference between candidate and current point evaluation
                    best =  candidate
                    problem.update_solution(best)
                    print('>%d f(%s) = %.5f' % (i, t, best.makespan))
                diff = candidate.makespan - curr_makespan

                # calculate temperature for current epoch
                # t = self.temp / (float(i+1))
                # calculate metropolis acceptance criterion
                # check if we should keep the new point
                if diff < 0:
                    curr_candidate, curr_makespan = candidate, candidate.makespan
                elif diff > 0:
                    metropolis = exp(-diff / t)
                    if rand() < metropolis:
                        curr_candidate,curr_makespan = candidate, candidate.makespan
            # t = self.temp /float(iterations)
            # iterations+=1
            t = 0.9 * t
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

    def getNeighbourFromSolutionCriticalPath(self, solution):
        x = random.randrange(len(solution.criticalPath))
        position=solution.criticalPath[x].source.id
        job_id = solution.criticalPath[x].source.job.id
        lower_index_bound = 0
        lower_index_counter = 0
        while lower_index_counter == (position % len(solution.machine_ops)) - 1:
            if solution.orderedseq[lower_index_bound][0] == job_id:
                lower_index_counter+=1
            lower_index_bound+=1
        actualSelectedXonOrderedSeq = lower_index_bound
        for i in range(lower_index_bound, len(solution.orderedseq)):
            if solution.orderedseq[i][0] == job_id:
                actualSelectedXonOrderedSeq = i
        top_index_bound = len(solution.orderedseq)
        top_index_counter = 0
        while top_index_counter == len(solution.machine_ops) - position % len(solution.machine_ops) + 1:
            if solution.orderedseq[top_index_bound][0] == job_id:
                top_index_counter+=1
            top_index_bound-=1
        if top_index_bound == len(solution.orderedseq):
            top_index_bound -=1
        placement_index = random.sample(range(lower_index_bound, top_index_bound), top_index_bound - lower_index_bound)
        counter = 0
        if len(placement_index)== 0:
            return self.getNeighbourFromSolutionCriticalPath(solution)
        idx = placement_index[random.randrange(len(placement_index))]
        while counter < len(placement_index):
            if actualSelectedXonOrderedSeq == placement_index:
                continue
            op_id = solution.orderedseq[placement_index[counter]][0]
            for i in range(len(solution.criticalPath)):
                if solution.criticalPath[i].source.id == op_id:
                    idx = placement_index[counter]
                    break
            counter+=1
        temp = solution.orderedseq[actualSelectedXonOrderedSeq]
        solution.orderedseq[actualSelectedXonOrderedSeq] = solution.orderedseq[idx]
        solution.orderedseq[idx] = temp
        return solution.orderedseq
    def getNeighbourFromSolution(self, solution):
        x = random.randrange(len(solution.orderedseq))
        job_id = int(solution.orderedseq[x][0]/len(solution.machine_ops))
        lower_index_bound = x - 1
        while lower_index_bound >0:
            if int(solution.orderedseq[lower_index_bound][0]/len(solution.machine_ops)) == job_id:
                break
            lower_index_bound-=1
        top_index_bound = x + 1
        while top_index_bound < len(solution.orderedseq):
            if int(solution.orderedseq[top_index_bound][0]/len(solution.machine_ops)) == job_id:
                break
            top_index_bound+=1
        if top_index_bound == len(solution.orderedseq):
            top_index_bound -=1
        placement_index = x
        while placement_index == x:
            placement_index = random.randint(lower_index_bound, top_index_bound)
        temp = solution.orderedseq[x]
        solution.orderedseq[x] = solution.orderedseq[placement_index]
        solution.orderedseq[placement_index] = temp
        return solution.orderedseq


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

    def generate_solution(self, problem, permutation=None, sortedList=None):
        solution = JSSolution(problem)

        if permutation is None and sortedList is None:
            sortedList = sorted([(op.source.job.id, -op.tail) for op in solution.ops], key= lambda tuple:tuple[1])
        '''One iteration applying priority dispatching rule.'''
        # move form
        # collect imminent operations in the processing queue
        head_ops = solution.imminent_ops
        if sortedList is None:
            orderedList = sorted([(i %len(problem.machines), permutation[i]) for i in range(len(problem.ops))], key= lambda tuple:tuple[1])
        else:
            for idx,tup in enumerate(sortedList):
                temp = list(tup)
                temp[0] = int(temp[0] / len(problem.machines))
                sortedList[idx] = temp
            orderedList = sortedList
        inter = 0
        while head_ops:
            # dispatch operation with the first priority
            toList = list(orderedList[inter])
            job_id = toList[0]
            op: OperationStep = [op for op in head_ops if op.source.job.id == job_id][0]
            solution.dispatch(op)
            # update imminent operations
            pos = head_ops.index(op)
            next_job_op = op.next_job_op
            if next_job_op is None:
                head_ops = head_ops[0:pos] + head_ops[pos + 1:]
            else:
                head_ops[pos] = next_job_op
            toList[0]=job_id*len(problem.machines) + op.source.id % len(problem.machines)
            orderedList[inter] = tuple(toList)
            inter+=1
        solution.forward()
        solution.backward()
        solution.computeCriticalPath()
        solution.orderedseq = orderedList
        return solution, solution.orderedseq
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

