import re
import os
import sys
import numpy as np
from datetime import datetime
from typing import Literal
from typing import List, Tuple, Callable, Dict, Any,Literal
from IPython.display import clear_output

class StackQueue():
    def __init__(self):
        self.memory:List[Any] = []
        self.city_number: int = 0

    def is_empty(self)->bool:
        return self.memory is None or len(self.memory)==0

    def add_item(self, element:Any)->None:
        self.memory.append(element)

    def pop_by_maximum(self, key:Callable[[Any], Any])->Any:
        value = -np.inf
        index = None
        if not self.is_empty():
            for i in range(len(self.memory)):
                if key(self.memory[i]) > value:
                    value = key(self.memory[i])
                    index = i
        return self.memory.pop(index)

    def add_prioritized_item(self, element:Any)->None:
        self.memory = [element] + self.memory

    def pop_end(self)->Any:
        response = None
        if not self.is_empty(): response = self.memory.pop(-1)
        return response

    def pop_start(self)->Any:
        response = None
        if not self.is_empty(): response = self.memory.pop(0)
        return response

    def get_size(self)->int:
        return 0 if self.memory is None else len(self.memory)

    def shuffle(self)->None:
        np.random.shuffle(self.memory)

    def sort(self, key:Callable[[Any], Any], reverse:bool=True)->None:
        self.memory = sorted(self.memory, key=key, reverse=reverse)

    def total_reprune(self, best_solution_cost:int)->None:
        # init_size = self.get_size()
        self.memory = [i for i in self.memory if i.cost + i.current_weight <= best_solution_cost]
        # end_size = self.get_size()
        # return init_size - end_size

class BnBNode:
    def __init__(self, parent:int, level:int, cost:int, current_weight:int,visited_cities:int, current_path:List[int]):
        self.parent:int = parent
        self.level:int = level
        self.cost:int = cost
        self.current_weight:int = current_weight
        self.visited_cities:int = visited_cities
        self.current_path:List[int] = current_path

class BranchAndBound():
    def __init__(self, number_of_cities:int, cost_matrix:np.ndarray, solution_coast:int=None, policy:str='d80b20'):
        self.number_of_cities:int = number_of_cities
        self.cost_matrix:np.ndarray = cost_matrix
        self.minimal_table:List[Dict[str, int]] = []

        self.current_solution_cost:int = np.inf
        self.current_solution_path:List[int] = self.get_new_path()

        self.best_solution_cost:int = solution_coast #Global Optimum
        self.policy:Literal['depth_first', 'breadth_first', 'd80b20', ] = policy
        self.memory:StackQueue = StackQueue()

        self.achieved_leafes:int = 0
        self.minimum_achieved_leafes:int = self.number_of_cities*10
        self.depth_counter: int = 0

        self.kill_mem_opt:int = 0
        self.kill_mem_opt_total:int = 0
        self.maximum_iters_without_change:int = 1_000_000
        self.iters_without_change:int = 0
        self.new_best_it:int = 0

        self.execution_metrics:Dict[str, Any] = {}


    def compute_solution_cost(self, path:List[int])->int:
        total_cost = 0
        for i in range(self.number_of_cities):
            total_cost += self.cost_matrix[path[i]][path[i + 1]]
        return total_cost
        
    def compile_solution(self, path:List[int])->List[int]:
        solution_path:List[int] = path[:]
        if len(path) == self.number_of_cities + 1:
            solution_path[self.number_of_cities] = path[0]
        else:
            solution_path.append(path[0])
        return solution_path

    def get_new_visited_path(self)->int:
        return 0

    def get_new_path(self)->List[int]:
        return [-1]

    def has_next(self)->bool:
        has_next:bool = True
        if (
            self.memory.is_empty() or
            (self.best_solution_cost is not None and self.current_solution_cost <= self.best_solution_cost)
        ):
            print('LOWER COST', self.current_solution_cost)
            has_next = False
        return has_next

    def get_position_permutation(self)->List[int]:
        return np.random.permutation(self.number_of_cities).tolist()

    def get_greedy_positions(self, parent:int)->List[int]:
        return self.minimal_table[parent]['rank'][0:]

    def next(self, it:int)->BnBNode:
        next_node:BnBNode = None
        if self.kill_mem_opt_total > 0 and self.kill_mem_opt_total < self.memory.get_size():
            if self.kill_mem_opt < self.kill_mem_opt_total:
                self.kill_mem_opt+=1
                next_node = self.memory.pop_start()
            else:
                next_node = self.memory.pop_start()
                self.kill_mem_opt = 0
                self.kill_mem_opt_total = 0
            return next_node
        elif it <= self.number_of_cities**2:
            next_node = self.memory.pop_start()
        else:
            if self.achieved_leafes < self.minimum_achieved_leafes:
                next_node = self.memory.pop_end()
            else:
                if self.policy == 'depth_first':
                    next_node = self.memory.pop_end()
                elif self.policy == 'random':
                    if np.random.rand() > .5:
                        next_node = self.memory.pop_end()
                    else:
                        next_node = self.memory.pop_start()
                elif self.policy == 'breadth_first':
                    next_node = self.memory.pop_start()
                elif self.policy == 'd80b20':
                    if self.depth_counter == 0:
                        p:float = np.random.rand()
                        if p> .2:
                            next_node = self.memory.pop_end()
                            self.depth_counter=1%(self.number_of_cities+1)
                        else:
                            next_node =self.memory.pop_start()
                    else:
                        next_node = self.memory.pop_end()
                        self.depth_counter = (self.depth_counter +1) % (self.number_of_cities+1)
                elif self.policy == 'd70b30':
                    if self.depth_counter == 0:
                        p:float = np.random.rand()
                        if p> .3:
                            next_node = self.memory.pop_end()
                            self.depth_counter=1%(self.number_of_cities+1)
                        else:
                            next_node =self.memory.pop_start()
                    else:
                        next_node = self.memory.pop_end()
                        self.depth_counter = (self.depth_counter +1) % (self.number_of_cities+1)
                elif self.policy == 'd92b08':
                    if self.depth_counter == 0:
                        p:float = np.random.rand()
                        if p> .08:
                            next_node = self.memory.pop_end()
                            self.depth_counter=1%(self.number_of_cities+1)
                        else:
                            next_node =self.memory.pop_start()
                    else:
                        next_node = self.memory.pop_end()
                        self.depth_counter = (self.depth_counter +1) % (self.number_of_cities+1)
                elif self.policy == 'd50b50':
                    if self.depth_counter == 0:
                        p:float = np.random.rand()
                        if p> .5:
                            next_node = self.memory.pop_end()
                            self.depth_counter=1%(self.number_of_cities+1)
                        else:
                            next_node =self.memory.pop_start()
                    else:
                        next_node = self.memory.pop_end()
                        self.depth_counter = (self.depth_counter +1) % (self.number_of_cities+1)
                elif self.policy == 'd20b80':
                    if np.random.rand() > .8:
                        next_node = self.memory.pop_end()
                    else:
                        next_node = self.memory.pop_start()
                else:
                    next_node = self.memory.pop_end()
        return next_node

    def get_minimum_edges(self, row:int)->Tuple[int, int]:
        return self.minimal_table[row]['first'], self.minimal_table[row]['second']

    def compute_minimal_edges(self)->Tuple[int, int]:
        for row in range(self.number_of_cities):
            total_sort:List[int] = np.argsort(self.cost_matrix[row]).tolist()
            _,first_min_index,second_min_index = total_sort[:3]
            self.minimal_table.append({
                'first': self.cost_matrix[row][first_min_index],
                'second': self.cost_matrix[row][second_min_index],
                'rank': total_sort
            })

    def set_best_current_solution(self, path:List[int], coast:int)->None:
        self.current_solution_cost = coast
        self.current_solution_path = path[:]

    def optimization_end_case(self, solution:BnBNode, logging:bool=False, it:int = 0)->None:
        if solution.level == self.number_of_cities:
            if self.cost_matrix[solution.parent, solution.current_path[0]] > 0:
                self.achieved_leafes+=1
                # if logging: print('Achieved Leafs: {}'.format(self.achieved_leafes))
                current_coast:int = solution.current_weight + self.cost_matrix[solution.parent, solution.current_path[0]]
                solution_path:List[int] = self.compile_solution(solution.current_path)
                if current_coast < self.current_solution_cost:
                    if logging: print(f"New Best Solution Found: EC: {current_coast}")
                    self.set_best_current_solution(solution_path, current_coast)
                    self.execution_metrics['tracking'].append([datetime.now(), it, current_coast])
                    self.new_best_it = it
                    if self.memory.get_size() >= self.number_of_cities*1000:
                        self.memory.total_reprune(self.current_solution_cost)


    def visit(self, visited_mask, index):
        visited_mask |= (1 << index)
        return visited_mask

    def is_visited(self, visited_mask, index:int):
        return (visited_mask & (1<< index)) !=0

    def optimize(self, logging:bool=False, max_time:int=30)->Tuple[List[int], int, Dict[str, Any]]:
        init_dt:datetime = datetime.now()
        self.execution_metrics = {
            'init_time':init_dt, 'end_time':None, 'execution_time':None, 'iterations':0, 'best_solution_cost':None, 'best_solution_path':None,
            'tracking': [], 'policy': self.policy, 'memory_size': self.memory.get_size() 
        }
        current_city:int = 0
        current_path:List[int] = self.get_new_path()
        visited_cities:int = self.get_new_visited_path()
        
        self.compute_minimal_edges()
        initial_bound:int = int(np.ceil(sum([np.sum(self.get_minimum_edges(i)) for i in range(self.number_of_cities)])/2))
        if logging: print('Initial LB: {}'.format(initial_bound))
        self.current_solution_cost = np.inf
        visited_cities = self.visit(visited_cities, current_city)
        current_path[0] = current_city
        

        # print(visited_cities)
        # exit()
        self.memory.add_item(
            BnBNode(parent=current_city, level=1, cost=initial_bound,current_weight=0, visited_cities=visited_cities, current_path=current_path)
        )

        iteration_condition:bool = True
        counter = 0
        self.iters_without_change = 0
        while(iteration_condition):
            solution:BnBNode = self.next(counter)
            
            if solution.level == self.number_of_cities:
                self.optimization_end_case(solution, logging, counter)
                self.iters_without_change = counter
            else:
                for i in self.get_position_permutation():
                    if (
                        solution.parent != i and not self.is_visited(solution.visited_cities,i) and self.cost_matrix[solution.parent, i] > 0
                    ):
                        current_weight:int = solution.current_weight + self.cost_matrix[solution.parent, i]
                        current_bound:int  = solution.cost
                        if solution.level == 1:
                            first_min_parent, _ = self.get_minimum_edges(solution.parent)
                            first_min, _ = self.get_minimum_edges(i)
                            current_bound -= (first_min_parent + first_min)/2
                        else:
                            _, second_min_parent = self.get_minimum_edges(solution.parent)
                            first_min, _ = self.get_minimum_edges(i)
                            current_bound -= (second_min_parent + first_min)/2
                        if current_bound + current_weight < self.current_solution_cost:
                            new_path:List[int] = solution.current_path[:]
                            new_path.append(i)

                            new_item:BnBNode = BnBNode(
                                parent=i,
                                level=solution.level+1,
                                cost=current_bound,
                                current_weight=current_weight,
                                visited_cities=self.visit(solution.visited_cities, i),
                                current_path=new_path
                            )

                            if solution.level + 1 == self.number_of_cities:
                                self.optimization_end_case(new_item, logging, counter)
                                self.iters_without_change = counter
                            else:
                                self.memory.add_item(new_item)
            iteration_condition = self.has_next()
            counter+=1

            current_dt = datetime.now()

            minutes_used = (current_dt - init_dt).total_seconds()/60
            if minutes_used >= max_time:
                if logging: print('Execution Time Limit Reached! {} min!'.format(max_time))
                iteration_condition = False
                self.execution_metrics['end_time'] = current_dt
                self.execution_metrics['execution_time'] = minutes_used
                self.execution_metrics['iterations'] = counter
                self.execution_metrics['best_solution_cost'] = self.current_solution_cost
                self.execution_metrics['best_solution_path'] = self.current_solution_path
                self.execution_metrics['memory_size'] = self.memory.get_size()
 
            if (counter - self.iters_without_change) > self.maximum_iters_without_change and self.kill_mem_opt_total == 0:
                if logging: print('Maximum Iterations Without Change Reached!')
                self.iters_without_change = counter + 5*(self.number_of_cities**2)
                self.kill_mem_opt_total = 5*(self.number_of_cities**2)
                self.kill_mem_opt = 0

            if counter% 200_000 == 0 and (counter - self.new_best_it) > 200_000:
                if self.kill_mem_opt_total == 0: self.memory.sort(key=lambda x: x.cost + x.current_weight, reverse=True)
                if counter% 1_000_000 == 0 and logging:
                    os.system('clear')
                    clear_output(wait=True)
                    print('- Init Execution DT [{}]'.format(init_dt.strftime('%Y-%m-%d %H:%M:%S')))
                    print('- Execution DT [{}] ~ [{}] '.format(current_dt.strftime('%Y-%m-%d %H:%M:%S'),minutes_used))
                    print('Current Best Score: {} | Achieved Leafs: {}'.format(self.current_solution_cost, self.achieved_leafes))

    
        return self.current_solution_path, self.current_solution_cost, self.execution_metrics
