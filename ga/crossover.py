import random
from .solution import Solution, Route
from utils.vrp_customer_util import Customer
from .constants import CROSSOVER_RATE

def route_based_crossover(parent1: Solution, parent2: Solution, depot: Customer, crossover_rate: float = CROSSOVER_RATE) -> Solution:
    # Flip a coin: apply crossover or not
    if random.random() > crossover_rate:
        # Just clone the better parent (e.g. parent1 here)
        return Solution([Route(depot, r.customers[:]) for r in parent1.routes])

    # Step 1: Copy some routes from parent1
    max_routes_to_copy = len(parent1.routes) - 1  # leave at least 1 route free
    num_routes_to_copy = random.randint(1, max_routes_to_copy)
    copied_routes = [Route(depot, route.customers[:]) for route in parent1.routes[:num_routes_to_copy]]

    # Collect visited customer IDs
    visited_ids = set()
    for route in copied_routes:
        visited_ids.update(c.id for c in route.customers)

    # Step 2: Get remaining customers from parent2
    remaining_customers = []
    for route in parent2.routes:
        for customer in route.customers:
            if customer.id not in visited_ids:
                remaining_customers.append(customer)

    # Step 3: Distribute remaining customers across remaining vehicles
    num_vehicles = len(parent1.routes)
    num_remaining_routes = num_vehicles - len(copied_routes)
    route_lists = [[] for _ in range(num_remaining_routes)]

    for idx, customer in enumerate(remaining_customers):
        route_index = idx % num_remaining_routes
        route_lists[route_index].append(customer)

    # Step 4: Convert to Route objects
    new_routes = copied_routes.copy()
    for route in route_lists:
        new_routes.append(Route(depot, route))

    return Solution(new_routes)
