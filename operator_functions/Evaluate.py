import numpy as np
import ioh 

def convert_to_bipolar(individual):
    bi_individual = 1 - 2 * individual
    return tuple(bi_individual.astype(np.int32))


class Evaluate():
    def __init__(self, problem, budget=5000):
        self.memory_dict = {}
        self.problem = problem
        self.eval_count = 0 
        self.last_score = 0 # Safety fallback
        self.budget = budget
        self.iterations = 0

    def eval(self, individuals):
        return np.array([self(ind) for ind in individuals])

    def __call__(self, individual):
        # 1. N-QUEENS SYMMETRY
        # Safer way to check problem type in IOH
        if "NQueens" in str(self.problem):
            board_dim = int(np.sqrt(len(individual)))
            board = np.asarray(individual, dtype=np.int32).reshape((board_dim, board_dim))
            variants = [
                board, 
                np.rot90(board, 1), 
                np.rot90(board, 2), 
                np.rot90(board, 3),
                np.flip(board, 0), 
                np.flip(board, 1), 
                board.T, 
                np.fliplr(board).T
            ]
            key = min(v.tobytes() for v in variants)

            if key not in self.memory_dict:
                if self.eval_count < self.budget:
                    self.memory_dict[key] = self.problem(individual)
                    self.eval_count += 1
                else:
                    return self.last_score # Return previous best if out of budget
            
            self.last_score = self.memory_dict[key]
            self.iterations += 1
            return self.memory_dict[key]

        # 2. LABS SYMMETRY
        else: 
            bi = convert_to_bipolar(individual)
            variants = [
                bi, 
                bi[::-1], 
                tuple(-x for x in bi), 
                tuple(-x for x in bi[::-1])
                ]
            
            for v in variants:
                if v in self.memory_dict:
                    return self.memory_dict[v]

            if self.eval_count < self.budget:
                score = self.problem(individual)
                self.memory_dict[bi] = score
                self.eval_count += 1
                self.last_score = score
                self.iterations += 1
                return score
            else:
                return self.last_score # Budget exhausted