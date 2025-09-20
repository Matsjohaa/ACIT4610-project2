import matplotlib.pyplot as plt
from .models import VRPInstance, Solution
from typing import Iterable


def plot_points(inst: VRPInstance, ax=None):
    """
    Plot the depot and all customer points (without drawing routes).
    Depot is drawn as a red star, customers as blue circles.
    """
    if ax is None:
        fig, ax = plt.subplots()
    x0, y0 = inst.depot
    ax.scatter([x0], [y0], s=100, c="red", marker="*", label="Depot")

    # customer coordinates
    xs = [c.x for c in inst.customers]
    ys = [c.y for c in inst.customers]

    ax.scatter(xs, ys, c="blue", label="Customers")
    ax.legend()
    return ax


def plot_solution(inst: VRPInstance, routes: Solution, ax=None, title=None):
    """
    Plot a full VRP solution: depot, customers, and vehicle routes.
    Each route is drawn in a different color.
    """
    if ax is None:
        fig, ax = plt.subplots()
    plot_points(inst, ax=ax)

    # points list: depot first, then all customers
    pts = [inst.depot] + [(c.x, c.y) for c in inst.customers]
    colors = plt.cm.get_cmap("tab10", len(routes))

    for i, r in enumerate(routes):
        if not r:
            continue
        col = colors(i)

        """Deprecated visualization module.

        All previous plotting helpers removed. Use plot_utils.plot_pareto for
        current visualization needs.
        """

        __all__: list[str] = []
        # last customer -> depot
