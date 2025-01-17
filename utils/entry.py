from typing import List, Tuple
import numpy as np
import gzip
from utils.tsp_instance import TSPInstance

def read_tsp_file(file_path: str, solution_cost:int = None) -> TSPInstance:
    with gzip.open(file_path, 'rt') as f:
        lines = f.readlines()

    name = None
    comment = None
    dimension = None
    edge_weight_type = None
    node_coord_section = False
    node_coords = []

    for line in lines:
        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("COMMENT"):
            comment = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            node_coord_section = True
        elif line.startswith("EOF"):
            break
        elif node_coord_section:
            parts = line.split()
            node_coords.append((int(parts[0]), float(parts[1]), float(parts[2])))

    return TSPInstance(
        name=name,
        file_path=file_path,
        comment=comment,
        tsp_type=edge_weight_type,
        cities_number=dimension,
        cities_coords=node_coords,
        best_solution_cost=solution_cost,
    )