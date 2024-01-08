from math import exp

from numpy.random import rand

from jsp_fwk import JSSolution, JSProblem


class SimulatedAnnealingSolver(JSSolver):
    '''Simulated Annealing Solver.'''

    def __init__(self, name :str =None, n_iterations :int =100, temp :int =200) -> None:
        '''Simulated Annealing.

        Args:
            name (str, optional): Solver name.
        '''
        super().__init__(name)
        self.temp = temp
        self.n_iterations = n_iterations


    def do_solve(self, problem: JSProblem):
        solution = JSSolution(problem=problem)
        self.solving_iteration(solution=solution)
        problem.update_solution(solution=solution)

        best = problem.solution
        best_eval = best.makespan
        curr = best
        curr_eval = best_eval
        # run the algorithm
        for i in range(self.n_iterations):
            # take a step
            candidate = self.getNeighbour(curr, problem)
            # evaluate candidate point
            candidate_eval = candidate.makespan
            # check for new best solution
            if candidate_eval < best_eval:
                # store new best point
                best, best_eval = candidate, candidate_eval
                # report progress
                print('>%d f(%s) = %.5f' % (i, best, best_eval))
            # difference between candidate and current point evaluation
            diff = candidate_eval - curr_eval
            # calculate temperature for current epoch
            t = self.temp / float(i + 1)
            # calculate metropolis acceptance criterion
            metropolis = exp(-diff / t)
            # check if we should keep the new point
            if diff < 0 or rand() < metropolis:
                # store the new current point
                curr, curr_eval = candidate, candidate_eval
        return [best, best_eval]

    def solving_iteration(self, solution :JSSolution):
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
    def getNeighbour(self, solution :JSSolution, problem :JSProblem):
        # TODO
        neighbour_solution = JSSolution(problem=problem)
        return neighbour_solution