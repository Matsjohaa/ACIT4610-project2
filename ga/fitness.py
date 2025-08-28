def calculate_total_distance(solution):
    return sum(route.calculate_distance() for route in solution.routes)

def fitness(solution):
    # Minimize distance → higher fitness = lower distance
    distance = calculate_total_distance(solution)
    if distance == 0:
        return float('inf')  # avoid division by zero
    return 1 / distance