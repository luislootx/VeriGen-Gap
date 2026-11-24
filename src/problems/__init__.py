from .base import BaseProblem
from .sat import SAT3Problem
from .tsp import TSPDecisionProblem
from .graph_coloring import GraphColoringProblem
from .sorting import SortingProblem
from .shortest_path import ShortestPathProblem
from .matching import BipartiteMatchingProblem

P_PROBLEMS = [SortingProblem, ShortestPathProblem, BipartiteMatchingProblem]
NP_PROBLEMS = [SAT3Problem, TSPDecisionProblem, GraphColoringProblem]
ALL_PROBLEMS = P_PROBLEMS + NP_PROBLEMS
