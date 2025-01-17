from typing import List, Tuple, Dict
import numpy as np
import networkx as nx

class TwiceAroundTheTree():
    def __init__(self, number_of_cities:int, cost_matrix:np.ndarray, solution_cost:int=None):
        self.number_of_cities:int = number_of_cities
        self.cost_matrix:np.ndarray = cost_matrix
        self.nx_graph:nx.Graph = nx.Graph()
        self.minimum_spanning_tree:Dict[int, List[int]]

        self.solution_cost:int = 0
        self.solution_path:List[int] = []

        self.partial_path:List[int] = []
        self.best_solution_cost:int = solution_cost #Global Optimum

    def create_nx_graph(self)->None:
        for i in range(self.number_of_cities):
            for j in range(i + 1, self.number_of_cities):
                if i != j:
                    self.nx_graph.add_edge(i, j, weight=self.cost_matrix[i][j])


    def get_minimun_spanning_tree(self)->Dict[int, List[int]]:
        mst = nx.minimum_spanning_tree(self.nx_graph, weight='weight')
        graph_dict = {k: []for k in range(self.number_of_cities)}
        for u, v, data in mst.edges(data=True):
            graph_dict[u].append([v, data['weight']])
            graph_dict[v].append([u, data['weight']])
        for i in range(self.number_of_cities):
            graph_dict[i] = sorted(graph_dict[i], key=lambda x: x[1], reverse=False)
            graph_dict[i] = [j[0] for j in graph_dict[i]]
        self.minimum_spanning_tree = graph_dict
        return self.minimum_spanning_tree
    

    def explore(self, node:int, parent:int=None)->None:
        self.partial_path.append(node)
        for neighbor in self.minimum_spanning_tree[node]:
            if neighbor != parent:
                self.explore(neighbor, node)
                self.partial_path.append(node)


    def clean_solution(self)->List[int]:
        visited = [False] * self.number_of_cities
        final_path = []
        for node in self.partial_path:
            if not visited[node]:
                final_path.append(node)
                visited[node] = True
        final_path.append(self.partial_path[0])
        return final_path

    def compute_solution_cost(self, path:List[int])->int:
        total_cost = 0
        for i in range(self.number_of_cities):
            total_cost += self.cost_matrix[path[i]][path[i + 1]]
        return total_cost
    
    def optimize(self)->Tuple[List[int], int]:
        self.create_nx_graph()
        self.get_minimun_spanning_tree()
        self.explore(0, None)
        final_path = self.clean_solution()
        total_cost = self.compute_solution_cost(final_path)
        
        self.solution_path = final_path
        self.solution_cost = total_cost

        return self.solution_path, self.solution_cost