import random
from math import exp, floor

from numpy.random import rand

from jsp_fwk import JSSolution, JSProblem, JSSolver, OperationStep
from jsp_fwk.solver.dispatching_rule import DisPatchingRules
import plotly.graph_objs as go


class SimulatedAnnealingSolverIterRANDOM(JSSolver):
    '''Simulated Annealing Solver.'''

    def __init__(self, name: str = None, n_iterations: int = 100, temp: int = 2000) -> None:
        '''Simulated Annealing.

        Args:
            name (str, optional): Solver name.
        '''
        super().__init__(name)
        self.temp = temp
        self.n_iterations = n_iterations

    def do_solve(self, problem: JSProblem):
        solution, permutation = self.generate_solutionA(problem, self.generate_random_permutation(problem))
        # solution, permutation = self.generate_solutionA(problem)
        best = solution
        problem.update_solution(best)
        best_permutation = permutation
        curr_permutation = solution
        curr = solution
        for i in range(self.n_iterations):
            # candidate_permutation = self.getNeighbourFromSolutionCriticalPath(solution=curr)
            candidate_permutation = self.getNeighbourFromSolution(solution=curr)
            # candidate_permutation = self.getNeighbourClose(curr_permutation)
            # candidate_permutation = self.getNeighbourTenPercent(curr_permutation)
            candidate = self.generate_solutionA(problem, candidate_permutation)[0]
            t = self.temp / (float(i + 1))
            if candidate.makespan < best.makespan:
                best_permutation, best = candidate_permutation, candidate
                problem.update_solution(best)
                print('>%d t %s makespan= %.5f' % (i, t, best.makespan))
                # difference between candidate and current point evaluation
            # t = t * min(float(1- (i+1)/self.n_iterations),0.99)

            # calculate metropolis acceptance criterion
            diff = candidate.makespan - curr.makespan
            if diff < 0:
                curr_permutation, curr = candidate, candidate
            else:
                metropolis = exp(-diff / (100*t))
                if rand() < metropolis:
                    curr_permutation, curr = candidate, candidate
            # calculate temperature for current epoch
            if t == 0:
                return [best_permutation, best]
            # check if we should keep the new point


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
        # dispatch operation by priority
        while head_ops:
            # dispatch operation with the first priority
            op = max(head_ops, key=lambda op: permutation[op.source.id])
            solution.dispatch(op)
            # update imminent operations
            pos = head_ops.index(op)
            next_job_op = op.next_job_op
            if next_job_op is None:
                head_ops = head_ops[0:pos] + head_ops[pos + 1:]
            else:
                head_ops[pos] = next_job_op
        return solution, permutation

    def generate_random_permutation(self, problem):
        return random.sample(range(0, len(problem.ops)), len(problem.ops))

    def generate_initial_solution_permutation(self, solution):
        return [op.tail for op in solution.ops]

    def neighbour(self, solution: JSSolution):
        ran = random.randrange(0,len(solution.machine_ops))

    def getNeighbourFromSolution(self, solution):
        x = random.randrange(len(solution.orderedseq))
        job_id = int(solution.orderedseq[x][0] / len(solution.machine_ops))
        lower_index_bound = x - 1
        while lower_index_bound > 0:
            if int(solution.orderedseq[lower_index_bound][0] / len(solution.machine_ops)) == job_id:
                break
            lower_index_bound -= 1
        top_index_bound = x + 1
        while top_index_bound < len(solution.orderedseq):
            if int(solution.orderedseq[top_index_bound][0] / len(solution.machine_ops)) == job_id:
                break
            top_index_bound += 1
        if top_index_bound == len(solution.orderedseq):
            top_index_bound -= 1
        placement_index = x
        while placement_index == x:
            placement_index = random.randint(lower_index_bound, top_index_bound)
        temp = solution.orderedseq[x]
        solution.orderedseq[x] = solution.orderedseq[placement_index]
        solution.orderedseq[placement_index] = temp
        return solution.orderedseq


    def getNeighbourFromSolutionCriticalPath(self, solution):
            x = random.randrange(len(solution.criticalPath))
            position = solution.criticalPath[x].source.id
            job_id = solution.criticalPath[x].source.job.id
            lower_index_bound = 0
            lower_index_counter = 0
            while lower_index_counter == (position % len(solution.machine_ops)) - 1:
                if solution.orderedseq[lower_index_bound][0] == job_id:
                    lower_index_counter += 1
                lower_index_bound += 1
            actualSelectedXonOrderedSeq = lower_index_bound
            for i in range(lower_index_bound, len(solution.orderedseq)):
                if solution.orderedseq[i][0] == job_id:
                    actualSelectedXonOrderedSeq = i
            top_index_bound = len(solution.orderedseq)
            top_index_counter = 0
            while top_index_counter == len(solution.machine_ops) - position % len(solution.machine_ops) + 1:
                if solution.orderedseq[top_index_bound][0] == job_id:
                    top_index_counter += 1
                top_index_bound -= 1
            if top_index_bound == len(solution.orderedseq):
                top_index_bound -= 1
            placement_index = random.sample(range(lower_index_bound, top_index_bound),top_index_bound - lower_index_bound)
            counter = 0
            if len(placement_index) == 0:
                print("tu")
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
                counter += 1
            temp = solution.orderedseq[actualSelectedXonOrderedSeq]
            solution.orderedseq[actualSelectedXonOrderedSeq] = solution.orderedseq[idx]
            solution.orderedseq[idx] = temp
            return solution.orderedseq

    def generate_solutionA(self, problem, permutation=None, sortedList=None):
        solution = JSSolution(problem)
        orderedList = list()
        if permutation is None and sortedList is None:
            orderedList = sorted([(op.source.job.id, -op.tail) for op in solution.ops], key=lambda tuple: tuple[1])
        '''One iteration applying priority dispatching rule.'''
        # move form
        # collect imminent operations in the processing queue
        head_ops = solution.imminent_ops
        if sortedList is None and len(orderedList) is 0:
            orderedList = sorted([(i % len(problem.machines), permutation[i]) for i in range(len(problem.ops))],
                                 key=lambda tuple: tuple[1])
        elif sortedList is not None and len(orderedList) is 0:
            for idx, tup in enumerate(sortedList):
                temp = list(tup)
                temp[0] = int(temp[0] / len(problem.machines))
                sortedList[idx] = temp
            orderedList = sortedList
        inter = 0
        while head_ops and inter<len(orderedList):
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
            toList[0] = job_id * len(problem.machines) + op.source.id % len(problem.machines)
            orderedList[inter] = tuple(toList)
            inter += 1
        solution.forward()
        solution.backward()
        solution.computeCriticalPath()
        solution.orderedseq = orderedList
        return solution, solution.orderedseq
