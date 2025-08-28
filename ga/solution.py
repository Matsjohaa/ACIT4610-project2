import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.vrp_dataset_util import load_data_instance, dataset_paths
from utils.vrp_customer_util import Customer
from .fitness import fitness

import random

class Solution:
    #Each solution is a list of routes for every vehicle
    #Goal is to find the list of routes messured by cost 
    def __init__(self, list_of_routes ):
        self.routes = list_of_routes
    
    def fitness(self):
        return fitness(self)
    
    def __str__(self):
        return "\n".join([f"Route {i+1}: {[c.id for c in r.customers]}" for i, r in enumerate(self.routes)])

class Route: 
    def __init__(self, depot, customers):
        self.depot = depot
        self.customers = customers

    def calculate_distance(self):
        distance = 0.0
        current = self.depot
        for customer in self.customers:
            distance += current.calculate_distance_to_next(customer)
            current = customer
        distance += current.calculate_distance_to_next(self.depot)
        return distance



#Creates a solution of random routes based on the dataset chosen. 
def create_random_solution(dataset_instance):
    #Step 1: retrieve and create an instance of the dataset
    dataset = load_data_instance(dataset_paths[dataset_instance])

    customers = dataset.customers
    depot = Customer(0, dataset.depot["x"], dataset.depot["y"])

    customer_pool = customers[:]
    
    #Step 2: Shuffle the customer_ids to ensure we get a random starting point
    random.shuffle(customer_pool)

    #Step 3: Retrieve the number of vehicles and create a list of lists for each vehicle available in the dataset
    num_vehicles = dataset.vehicles
    route_lists = [[] for _ in range(num_vehicles)]

    #Step 4: round robin concept - interates and places a customerid into each list after one another
    for idx, customer in enumerate(customer_pool):
        vehicle_index = idx % num_vehicles
        route_lists[vehicle_index].append(customer)

    #Step 5: Places the depot at the start and end of each route
    routes = [Route(depot, route) for route in route_lists]

    return Solution(routes)

#Example usage
solution = create_random_solution("small_1")
print(f"Fitness: {solution.fitness():.4f}")