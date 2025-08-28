import json
from .vrp_customer_util import Customer, turn_elements_to_customers

#Dataset instance
class Instance:
    def __init__(self, depot, vehicles, customers):
        self.depot = depot
        self.vehicles = vehicles
        self.customers = customers

#Load dataset and create instance
def load_data_instance(filename: str) -> Instance:
    with open(filename) as f:
        data = json.load(f)
        return Instance(
            data["depot"], 
            data["vehicles"], 
            turn_elements_to_customers(data["customers"])
            )
    
#Dataset dict for easier reference through the program
dataset_paths = {
    "small_2": "data/small_instance_2.json",
    "small_1": "data/small_instance_1.json",
    "medium_2": "data/medium_instance_2.json",
    "medium_1": "data/medium_instance_1.json",
    "large_2": "data/large_instance_2.json",
    "large_1": "data/large_instance_1.json"
}


""" Example usage:
my_instance = load_data_instance(dataset_paths["medium_1"])
print(my_instance.vehicles)
print(len(my_instance.customers))
"""


