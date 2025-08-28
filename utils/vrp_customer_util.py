
#Customer instance
class Customer: 
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
    
    def calculate_distance_to_next(self, next_point):
        return ((self.x - next_point.x)**2 + (self.y - next_point.y)**2)**0.5
    
#Helper function for dataset instance creation
def turn_elements_to_customers(customer_list):
    return [Customer(e["id"], e["x"], e["y"]) for e in customer_list]
    