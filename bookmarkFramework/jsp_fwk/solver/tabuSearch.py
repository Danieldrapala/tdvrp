from queue import Queue

from math import exp

from numpy.random import rand

from jsp_fwk import JSSolution, JSProblem
from jsp_fwk.common.exception import JSPException


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
        # get static data
        dependency_matrix_index_encoding = self.initial_solution.data.job_task_index_matrix
        usable_machines_matrix = self.initial_solution.data.usable_machines_matrix

        # ts variables
        tabu_list = _TabuList()
        seed_solution = self.initial_solution
        best_solutions_heap = Heap(max_heap=True)
        for _ in range(self.num_solutions_to_find):
            best_solutions_heap.push(self.initial_solution)

        # variables used for restarts
        lacking_solution = seed_solution
        counter = 0
        iterations = 0

        while not iterations >= self.iterations:
            neighborhood = self._generate_neighborhood(seed_solution,
                                                       dependency_matrix_index_encoding,
                                                       usable_machines_matrix)

            sorted_neighborhood = sorted(neighborhood.solutions.items())
            break_boolean = False

            for makespan, lst in sorted_neighborhood:  # sort neighbors in increasing order by makespan
                for neighbor in sorted(lst):  # sort subset of neighbors with the same makespans
                    if neighbor not in tabu_list:
                        # if new seed solution is not better than current seed solution add it to the tabu list
                        if neighbor >= seed_solution:
                            tabu_list.put(seed_solution)
                            if len(tabu_list) > self.tabu_list_size:
                                tabu_list.get()

                        seed_solution = neighbor
                        break_boolean = True
                        break

                if break_boolean:
                    break

            if seed_solution < best_solutions_heap[0]:
                best_solutions_heap.pop()  # remove the worst best solution from the heap
                best_solutions_heap.push(seed_solution)  # add the new best solution to the heap
                if self.benchmark and seed_solution.makespan < absolute_best_solution_makespan:
                    absolute_best_solution_makespan = seed_solution.makespan
                    absolute_best_solution_iteration = iterations

            # if solution is not being improved after a number of iterations, force a move to a worse one
            counter += 1
            if counter > self.reset_threshold:
                if not lacking_solution > seed_solution and len(sorted_neighborhood) > 10:
                    # add the seed solution to the tabu list
                    tabu_list.put(seed_solution)
                    if len(tabu_list) > self.tabu_list_size:
                        tabu_list.get()
                    # choose a worse solution from the neighborhood
                    seed_solution = sorted_neighborhood[random.randint(1, int(0.2 * len(sorted_neighborhood)))][1][0]

                counter = 0
                lacking_solution = seed_solution
            iterations += 1

        # convert best_solutions_heap to a sorted list
        best_solutions_list = []
        while len(best_solutions_heap) > 0:
            sol = best_solutions_heap.pop()
            sol.machine_makespans = np.asarray(sol.machine_makespans)
            best_solutions_list.append(sol)

        self.all_solutions = best_solutions_list
        self.best_solution = min(best_solutions_list)

        return self.best_solution

    def solving_iteration(self, solution :JSSolution):

    def _generate_neighborhood(self, seed_solution, dependency_matrix_index_encoding, usable_machines_matrix):
            stop_time = time.time() + self.neighborhood_wait
            neighborhood = _SolutionSet()
            while neighborhood.size < self.neighborhood_size and time.time() < stop_time:
                try:
                    neighbor = generate_neighbor(seed_solution, self.probability_change_machine,
                                                 dependency_matrix_index_encoding, usable_machines_matrix)

                    if neighbor not in neighborhood:
                        neighborhood.add(neighbor)

                except JSPException:
                    pass
            return neighborhood
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
