import random
from .solution import Solution
from .constants import MUTATION_RATE

def mutate(solution: Solution, mutation_rate: float = MUTATION_RATE) -> Solution:
    # Copy routes (shallow copy is enough since we replace customer lists)
    new_routes = []
    for route in solution.routes:
        new_routes.append(route)

    # Flatten all customers into a list with (route_idx, customer_idx) for mutation
    all_customers = []
    for route_idx, route in enumerate(new_routes):
        for customer_idx, customer in enumerate(route.customers):
            all_customers.append((route_idx, customer_idx, customer))

    # With some probability, swap two customers
    if random.random() < mutation_rate and len(all_customers) >= 2:
        (r1, i1, c1), (r2, i2, c2) = random.sample(all_customers, 2)

        # Swap them
        new_routes[r1].customers[i1], new_routes[r2].customers[i2] = c2, c1

    return Solution(new_routes)
