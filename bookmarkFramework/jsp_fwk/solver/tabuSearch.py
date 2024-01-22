import heapq
import random
from queue import Queue


from jsp_fwk import JSProblem, JSSolver, JSSolution
from jsp_fwk.common.exception import JSPException


class TabuSearchSolver(JSSolver):
    '''Tabu Search Solver.'''

    def __init__(self, name: str = None, n_iterations: int = 100, num_solutions_to_find: int = 1,
                 tabu_list_size: int = 10, reset_threshold: int = 0, neighborhood_size: int = 0,
                 neighborhood_wait: int = 0) -> None:
        '''Simulated Annealing.

        Args:
            name (str, optional): Solver name.
        '''
        super().__init__(name)
        self.n_iterations = n_iterations
        self.num_solutions_to_find = num_solutions_to_find
        self.tabu_list_size = tabu_list_size
        self.reset_threshold = reset_threshold
        self.neighborhood_wait = neighborhood_wait
        self.neighborhood_size = neighborhood_size

    def do_solve(self, problem: JSProblem):

        solution = self.generate_solution(problem)
        problem.update_solution(solution)
        absoluteBest = solution
        # ts variables
        tabu_list = _TabuList()
        seed_solution = solution
        best_solutions_heap = Heap(max_heap=True)
        for _ in range(self.num_solutions_to_find):
            best_solutions_heap.push(solution)

        # variables used for restarts
        lacking_solution = seed_solution
        counter = 0
        iterations = 0

        while not iterations >= self.n_iterations:
            neighborhood = self._generate_neighborhood(seed_solution, problem, self.n_iterations, iterations)
            sorted_neighborhood = sorted(neighborhood.solutions.items())
            break_boolean = False
            for makespan, lst in sorted_neighborhood:  # sort neighbors in increasing order by makespan
                for neighbor in lst:  # sort subset of neighbors with the same makespans
                    if neighbor not in tabu_list:
                        # if new seed solution is not better than current seed solution add it to the tabu list
                        if neighbor.makespan >= seed_solution.makespan:
                            tabu_list.put(seed_solution)
                            if len(tabu_list) > self.tabu_list_size:
                                tabu_list.get()

                        seed_solution = neighbor
                        break_boolean = True
                        break
                if break_boolean:
                    break

            if seed_solution.makespan < best_solutions_heap[0].makespan:
                lacking_solution = best_solutions_heap.pop()  # remove the worst best solution from the heap
                best_solutions_heap.push(seed_solution)  # add the new best solution to the heap
                counter=0
                if seed_solution.makespan <absoluteBest.makespan:
                    absoluteBest = seed_solution
                    problem.update_solution(seed_solution)

            # if solution is not being improved after a number of iterations, force a move to a worse one
            counter += 1
            if counter > self.reset_threshold:
                if not lacking_solution.makespan > seed_solution.makespan and len(sorted_neighborhood) > 1:
                    # add the seed solution to the tabu list
                    tabu_list.put(seed_solution)
                    if len(tabu_list) > self.tabu_list_size:
                        tabu_list.get()
                    # choose a worse solution from the neighborhood
                    seed_solution = sorted_neighborhood[random.randint(1, int(len(sorted_neighborhood))-1)][1][0]
                counter = 0
                lacking_solution = seed_solution
            iterations += 1

        # convert best_solutions_heap to a sorted list
        best_solutions_list = []
        while len(best_solutions_heap) > 0:
            sol = best_solutions_heap.pop()
            best_solutions_list.append(sol)

        self.all_solutions = best_solutions_list
        self.best_solution = absoluteBest

        return self.best_solution

    def getNeighbourTenPercent(self, solution):
        mutated_permutation = solution.chromosome.copy()
        percent = random.randrange(int(len(mutated_permutation)*0.1))/2
        for _ in range(int(percent)):
            # Wybierz losowo dwie pozycje
            pos1, pos2 = random.sample(range(len(mutated_permutation)), 2)
            mutated_permutation[pos1], mutated_permutation[pos2] = mutated_permutation[pos2], mutated_permutation[pos1]
        return mutated_permutation

    def getNeighbour(self, sol):
        mutated_permutation = sol.chromosome
        pos1, pos2 = random.sample(range(len(mutated_permutation)), 2)
        mutated_permutation[pos1], mutated_permutation[pos2] = mutated_permutation[pos2], mutated_permutation[pos1]
        return mutated_permutation

    def getNeighbour(self, permutation, machines, jobs):
        mutated_permutation = permutation.chromosome
        pos1 = random.randrange(len(mutated_permutation))
        x = random.randrange(jobs)
        y = int(random.randint(-1,1))
        pos2 = ((pos1 + x * machines) % (machines * jobs) + y) % (machines * jobs)
        mutated_permutation[pos1], mutated_permutation[pos2] = mutated_permutation[pos2], mutated_permutation[pos1]
        return mutated_permutation


    def _generate_neighborhood(self, solution, problem, iterations, iteration):
        neighborhood = _SolutionSet()
        while neighborhood.size < self.neighborhood_size:
            try:
                permutation = self.getNeighbour(solution, len(problem.machines), len(problem.jobs))
                # permutation = self.getNeighbourTenPercent(solution)
                neighbor = self.generate_solution(problem, permutation)
                if neighbor not in neighborhood:
                    neighborhood.add(neighbor)
            except JSPException:
                pass
        return neighborhood

    def generate_solution(self, problem, permutation=None):
        solution = JSSolution(problem)
        if permutation is None:
            # permutation = self.generate_random_permutation(problem)
            permutation = self.generate_initial_solution_permutation(solution)
        '''One iteration applying priority dispatching rule.'''
        # move form
        # collect imminent operations in the processing queue
        head_ops = solution.imminent_ops
        # dispatch operation by priority
        solution.chromosome = permutation
        while head_ops:
            # dispatch operation with the first priority
            op = min(head_ops, key=lambda op: permutation[op.source.id])
            solution.dispatch(op)
            # update imminent operations
            pos = head_ops.index(op)
            next_job_op = op.next_job_op
            if next_job_op is None:
                head_ops = head_ops[0:pos] + head_ops[pos + 1:]
            else:
                head_ops[pos] = next_job_op
        return solution

    def generate_random_permutation(self, problem):
        return random.sample(range(0, len(problem.ops)), len(problem.ops))

    def generate_initial_solution_permutation(self, solution):
        return [-op.tail for op in solution.ops]


'''
TS data structures
'''


class _TabuList(Queue):
    """
    Queue for containing Solution instances.
    """

    def __init__(self, max_size=0):
        super().__init__(max_size)
        self.solutions = _SolutionSet()

    def put(self, solution, block=True, timeout=None):
        super().put(solution, block, timeout)
        self.solutions.add(solution)

    def get(self, block=True, timeout=None):
        result = super().get(block, timeout)
        self.solutions.remove(result)
        return result

    def __contains__(self, solution):
        return solution in self.solutions

    def __len__(self):
        return self.solutions.size


class _SolutionSet:
    def __init__(self):
        self.size = 0
        self.solutions = {}

    def add(self, solution):
        if solution.makespan not in self.solutions:
            self.solutions[solution.makespan] = [solution]
        else:
            self.solutions[solution.makespan].append(solution)

        self.size += 1

    def remove(self, solution):
        if len(self.solutions[solution.makespan]) == 1:
            del self.solutions[solution.makespan]
        else:
            self.solutions[solution.makespan].remove(solution)

        self.size -= 1

    def __contains__(self, solution):
        return solution.makespan in self.solutions and solution in self.solutions[solution.makespan]


class Heap:
    """
    Heap data structure.
    """

    def __init__(self, max_heap=False):
        self._heap = []
        self._is_max_heap = max_heap

    def push(self, obj):
        if self._is_max_heap:
            heapq.heappush(self._heap, MaxHeapObj(obj))
        else:
            heapq.heappush(self._heap, obj)

    def pop(self):
        if self._is_max_heap:
            return heapq.heappop(self._heap).val
        else:
            return heapq.heappop(self._heap)

    def __getitem__(self, i):
        return self._heap[i].val

    def __len__(self):
        return len(self._heap)


class MaxHeapObj:
    """
    Wrapper class used for max heaps.
    """

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val.makespan > other.val.makespan

    def __gt__(self, other):
        return self.val.makespan < other.val.makespan

    def __eq__(self, other):
        return self.val.makespan == other.val.makespan
