import random
from typing import List, Tuple, Dict

# Type aliases for clarity
Location = Tuple[float, float]
Scenario = Dict[str, any]


def get_scenarios() -> List[Scenario]:
    # Set a local random seed for reproducible scenarios only
    state = random.getstate()
    #remove this for new map every time
    random.seed(42)
    scenarios = []
    # Small: 2-10 vehicles, 10-20 customers
    scenarios.append(generate_scenario("small_1", 3, 12))
    scenarios.append(generate_scenario("small_2", 7, 18))
    # Medium: 11-25 vehicles, 15-30 customers
    scenarios.append(generate_scenario("medium_1", 12, 20))
    scenarios.append(generate_scenario("medium_2", 20, 28))
    # Large: 26-50 vehicles, 20-50 customers
    scenarios.append(generate_scenario("large_1", 30, 35))
    scenarios.append(generate_scenario("large_2", 45, 48))
    random.setstate(state)
    return scenarios

def generate_customers(num_customers: int, xlim: Tuple[int, int], ylim: Tuple[int, int]) -> List[Location]:
    return [
        (random.uniform(*xlim), random.uniform(*ylim))
        for _ in range(num_customers)
    ]

def generate_scenario(
    name: str,
    num_vehicles: int,
    num_customers: int,
    xlim: Tuple[int, int] = (0, 100),
    ylim: Tuple[int, int] = (0, 100)
) -> Scenario:
    depot = (random.uniform(*xlim), random.uniform(*ylim))
    customers = generate_customers(num_customers, xlim, ylim)
    return {
        "name": name,
        "depot": depot,
        "customers": customers,
        "num_vehicles": num_vehicles
    }



# if this script is run, the scenarios will be generated with customer locations
if __name__ == "__main__":
    scenarios = get_scenarios()
    for s in scenarios:
        print(f"Scenario: {s['name']}")
        print(f"  Depot: {s['depot']}")
        print(f"  Vehicles: {s['num_vehicles']}")
        print(f"  Customers: {len(s['customers'])}")
        print(f"  Customer locations: {s['customers']}")
        print()
