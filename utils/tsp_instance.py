from typing import List, Tuple
import numpy as np


class TSPInstance:
    def __init__(
        self, 
        name: str = None, 
        file_path: str = None,
        comment: str = None, 
        tsp_type: str = None,
        cities_number: int= None, 
        cities_coords: List[Tuple[int, float, float]] = None, 
        best_solution_cost: int = None, 
    ):
        self.name = name
        self.file_path = file_path
        self.comment = comment
        self.tsp_type = tsp_type
        self.cities_number = cities_number
        self.cities_coords = cities_coords
        self.best_solution_cost = best_solution_cost

        self.matrix = None

    def generate_adjacency_matrix(self):
        self.matrix = np.zeros((self.cities_number, self.cities_number))
        for i in range(self.cities_number):
            for j in range(self.cities_number):
                if i == j:
                    self.matrix[i, j] = -np.inf
                else:
                    self.matrix[i, j] = np.sqrt(
                        (self.cities_coords[i][1] - self.cities_coords[j][1])**2 + (self.cities_coords[i][2] - self.cities_coords[j][2])**2
                    )
        return self.matrix
