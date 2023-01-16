import math
import random

import numpy as np
from numpy.random import rand


class Point:
  def __init__(self, cords, s):
    self.cords = cords
    self.s = s

retailStories = [Point((0,20),0), Point((10, 30), 0), Point((0, 40), 0), Point((0, 50), 0), Point((20, 20), 0), Point((50, 20), 0)]
# define range for input



def calculate_distance(point1, point2):
    return math.sqrt(pow(point2[0]- point1[0],2) + pow(point2[1]-point1[1],2))

def closest_node(node, nodes):
    dist_2 = []
    for nodee in nodes:
        if nodee.s != 0:
            dist_2.append(99999999999999999)
        else:
            dist_2.append(calculate_distance(node, nodee.cords))
    min = np.argsort(dist_2)
    for i in range(len(nodes)):
        if nodes[min[i]].s ==0:
            return min[i]

def initial_solution(S, retailStories):
    point = (0, 0)
    s = 1
    while s <= S:
        closest_node_index = closest_node(point, retailStories)
        point = retailStories[closest_node_index].cords
        retailStories[closest_node_index].s = s
        s += 1
    return retailStories


def randomize_solution(curr):
    seqlist = []
    for s in curr:
        seqlist.append(s.s)
    min = np.argsort(seqlist)
    sample = random.sample(range(1, len(curr)), 3)
    print(sample)
    for i in curr:
        print(i.s)
    curr[min[sample[0]-1]].s = sample[1]
    curr[min[sample[1]-1]].s = sample[2]
    curr[min[sample[2]-1]].s = sample[0]
    for i in curr:
        print(i.s)
    return curr

def simulated_annealing(objective, A, tempstart, tempEnd, Zmax, Zmin):
    # generate an initial point
    # first point pierwsza siatka współrzednych poprawna
    best = initial_solution(len(retailStories), retailStories)
    beta = tempstart - tempEnd / (A-1) * tempstart * tempEnd
    temp = tempstart
    # evaluate the initial point
    best_eval = objective(best)
    # current working solution
    curr, curr_eval = best, best_eval
    # run the algorithm
    for i in range(A):
        # take a step  TUTAJ NIE WIEM CO TO ZNACZY TAKE A STEP W NASZYM przykładzie
        candidate = randomize_solution(curr)
        # evaluate candidate point
        candidate_eval = objective(candidate)
        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        temp = temp / float(temp * beta + 1)
        # calculate metropolis acceptance criterion
        metropolis = math.exp(-diff / temp)
        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
    return [best, best_eval]


randomize_solution(initial_solution(6,retailStories))