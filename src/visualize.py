import matplotlib.pyplot as plt
from .models import VRPInstance, Solution


def plot_points(inst: VRPInstance, ax=None):
    """Plot depot and customers without routes."""
    if ax is None:
        fig, ax = plt.subplots()
    x0, y0 = inst.depot
    ax.scatter([x0], [y0], s=100, c="red", marker="*", label="Depot")
    xs = [c[0] for c in inst.customers]
    ys = [c[1] for c in inst.customers]
    ax.scatter(xs, ys, c="blue", label="Customers")
    ax.legend()
    return ax


def plot_solution(inst: VRPInstance, routes: Solution, ax=None, title=None):
    """Plot depot, customers, and routes (different colors per vehicle)."""
    if ax is None:
        fig, ax = plt.subplots()
    plot_points(inst, ax=ax)

    pts = [inst.depot] + inst.customers
    colors = plt.cm.get_cmap("tab10", len(routes))

    for i, r in enumerate(routes):
        if not r:
            continue
        col = colors(i)
        # depot -> first
        ax.plot([pts[0][0], pts[r[0]][0]], [pts[0][1], pts[r[0]][1]], c=col)
        # between customers
        for a, b in zip(r, r[1:]):
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], c=col)
        # last -> depot
        ax.plot([pts[r[-1]][0], pts[0][0]], [pts[r[-1]][1], pts[0][1]], c=col, label=f"Vehicle {i + 1}")

    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return ax


def compare_solutions(inst: VRPInstance, routes_a: Solution, routes_b: Solution,
                      title_a="Baseline", title_b="GA Best"):
    """Side-by-side comparison of two solutions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_solution(inst, routes_a, ax=axes[0], title=title_a)
    plot_solution(inst, routes_b, ax=axes[1], title=title_b)
    plt.show()
