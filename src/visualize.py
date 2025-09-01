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

        # depot -> first customer
        ax.plot([pts[0][0], pts[r[0]][0]], [pts[0][1], pts[r[0]][1]], c=col)

        # between customers
        for a, b in zip(r, r[1:]):
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], c=col)

        # last customer -> depot
        ax.plot([pts[r[-1]][0], pts[0][0]], [pts[r[-1]][1], pts[0][1]],
                c=col, label=f"Vehicle {i + 1}")

    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return ax


def compare_solutions(inst: VRPInstance, routes_a: Solution, routes_b: Solution,
                      title_a="Baseline", title_b="GA Best"):
    """
    Plot two solutions side by side for visual comparison.
    Left panel shows routes_a, right panel shows routes_b.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_solution(inst, routes_a, ax=axes[0], title=title_a)
    plot_solution(inst, routes_b, ax=axes[1], title=title_b)
    plt.show()


def plot_convergence(fitness_history: list[float], title: str | None = None):
    """
    Plot the convergence curve (best fitness over generations).
    """
    xs = list(range(1, len(fitness_history) + 1))
    plt.figure()
    plt.plot(xs, fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    if title:
        plt.title(title)
    plt.show()


def print_rule(width: int = 72) -> None:
    """Print a horizontal rule."""
    print("─" * width)


def print_kv(label: str, value, sep: str = ": ") -> None:
    """Print a left-aligned key–value line."""
    left = f"{label}{sep}"
    print(f"{left:<28}{value}")


def print_row(cols: Iterable[str], widths: Iterable[int]) -> None:
    """Print a table row with fixed column widths."""
    cells = []
    for text, w in zip(cols, widths):
        cells.append(f"{text:<{w}}")
    print("  " + " | ".join(cells))


def print_table(header: Iterable[str], rows: Iterable[Iterable[str]], widths: Iterable[int]) -> None:
    """Print a simple text table: header + rows with column widths."""
    print_row(header, widths)
    # header underline
    print_row(["-" * w for w in widths], widths)
    for r in rows:
        print_row(r, widths)
