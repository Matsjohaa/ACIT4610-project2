import matplotlib.pyplot as plt
import os

def plot_vrp_map(depot, customers, routes, scenario_name, save_dir="data"):
    """
    Plots the VRP solution and saves it as an image file.
    depot: tuple (x, y)
    customers: list of tuples [(x, y), ...]
    routes: list of lists of customer indices [[...], ...]
    scenario_name: string for filename
    save_dir: directory to save the image
    """
    plt.figure(figsize=(8, 8))
    # Plot depot
    plt.scatter(*depot, c='red', marker='s', s=100, label='Depot')
    # Plot customers
    cx, cy = zip(*customers)
    plt.scatter(cx, cy, c='blue', s=40, label='Customers')
    # Plot routes
    # Use a list of distinct colors for up to 20 vehicles, then fallback to colormap
    distinct_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
        '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
        '#808000', '#ffd8b1', '#000075', '#808080'
    ]
    num_routes = len(routes)
    for i, route in enumerate(routes):
        if not route:
            continue
        route_points = [depot] + [customers[idx] for idx in route] + [depot]
        rx, ry = zip(*route_points)
        if i < len(distinct_colors):
            color = distinct_colors[i]
        else:
            color = plt.cm.tab20(i % 20)
        plt.plot(rx, ry, '-', color=color, label=f'Vehicle {i+1}')
        plt.scatter([customers[idx][0] for idx in route], [customers[idx][1] for idx in route], color=color)
    plt.title(f'VRP Solution: {scenario_name}')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.tight_layout()
    # Save file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"vrp_map_{scenario_name}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Map saved to {filename}")
