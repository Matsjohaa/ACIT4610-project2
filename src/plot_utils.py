import matplotlib.pyplot as plt
from typing import Sequence
from pathlib import Path
from .moea import Individual
from .constants import MOEA_ALGORITHM
from .split import dp_split_capacity
from .distances import distance_matrix, route_length
from .instances import InstanceData
import numpy as np


def plot_pareto(pareto: Sequence[Individual], inst: InstanceData, show: bool = False, out_dir: str | Path = "visualisations") -> Path:
    # sort by first objective for line connection to improve readability
    pareto_sorted = sorted(pareto, key=lambda ind: ind.objectives[0])
    xs = [ind.objectives[0] for ind in pareto_sorted]
    ys = [ind.objectives[1] for ind in pareto_sorted]
    fig, ax = plt.subplots(figsize=(7,5), dpi=140)
    sc = ax.scatter(xs, ys, c=range(len(xs)), cmap='viridis', s=55, edgecolor='k', linewidths=0.4)
    ax.plot(xs, ys, color='gray', alpha=0.35, linewidth=1)
    ax.set_xlabel('Total Distance', fontsize=11)
    ax.set_ylabel('Route Distance Std Dev', fontsize=11)
    ax.set_title(f'Pareto Front ({MOEA_ALGORITHM}) - {inst.name}', fontsize=13)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Index along distance-sorted front')
    fig.tight_layout()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"pareto_{MOEA_ALGORITHM.lower()}_{inst.name}.png"
    fig.savefig(file_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)
    return file_path


def plot_routes(ind: Individual, inst: InstanceData, M: np.ndarray, show: bool = False, out_dir: str | Path = "visualisations", filename: str | None = None) -> Path:
    # split routes using provided instance & matrix
    try:
        routes = dp_split_capacity(ind.perm, inst.n_vehicles, M, inst.demands, inst.capacity)
    except ValueError:
        routes = []
    fig, ax = plt.subplots(figsize=(6,6), dpi=140)
    depot_coord = inst.coords[0]
    ax.scatter([depot_coord[0]], [depot_coord[1]], c='red', s=120, marker='s', label='Depot')
    cust_x = [inst.coords[i][0] for i in range(1, len(inst.demands)+1)]
    cust_y = [inst.coords[i][1] for i in range(1, len(inst.demands)+1)]
    ax.scatter(cust_x, cust_y, c='black', s=55, label='Customers')
    for idx, (x, y) in enumerate(zip(cust_x, cust_y), start=1):
        ax.annotate(f"{idx} ({inst.demands[idx-1]})", (x, y), textcoords='offset points', xytext=(4,4), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.65))
    colors = plt.cm.tab10(np.linspace(0,1,max(1,len(routes))))
    total_dist = 0.0
    for r_i, r in enumerate(routes):
        if not r:
            continue
        col = colors[r_i % len(colors)]
        path = [0] + r + [0]
        xs = [inst.coords[n][0] for n in path]
        ys = [inst.coords[n][1] for n in path]
        ax.plot(xs, ys, '-o', color=col, label=f'Route {r_i+1}: {r}')
        total_dist += route_length(r, M)
    ax.set_title(f'Routes {inst.name} (Total Dist={total_dist:.2f}, Std={ind.objectives[1]:.2f})')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = f"routes_{MOEA_ALGORITHM.lower()}_{inst.name}.png"
    file_path = out_path / filename
    fig.savefig(file_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)
    return file_path

def plot_route_comparison(best_distance: Individual, best_std: Individual, inst: InstanceData, M: np.ndarray, show: bool=False, out_dir: str | Path = "visualisations") -> Path:
    def _split(ind):
        try:
            return dp_split_capacity(ind.perm, inst.n_vehicles, M, inst.demands, inst.capacity)
        except ValueError:
            return []
    routes_dist = _split(best_distance)
    routes_std = _split(best_std)
    depot = inst.coords[0]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12,6), dpi=140)
    titles = [
        f"Best Distance (Routes={len([r for r in routes_dist if r])}, D={best_distance.objectives[0]:.2f}, Std={best_distance.objectives[1]:.2f})",
        f"Best Std (Routes={len([r for r in routes_std if r])}, D={best_std.objectives[0]:.2f}, Std={best_std.objectives[1]:.2f})"
    ]
    for ax, routes, ind, title in zip(axes, [routes_dist, routes_std], [best_distance, best_std], titles):
        ax.scatter([depot[0]], [depot[1]], c='red', s=120, marker='s', label='Depot')
        cust_x = [inst.coords[i][0] for i in range(1, len(inst.demands)+1)]
        cust_y = [inst.coords[i][1] for i in range(1, len(inst.demands)+1)]
        ax.scatter(cust_x, cust_y, c='black', s=45, label='Customers')
        colors = plt.cm.tab10(np.linspace(0,1,max(1,len(routes))))
        for r_i, r in enumerate(routes):
            if not r:
                continue
            col = colors[r_i % len(colors)]
            path = [0] + r + [0]
            xs = [inst.coords[n][0] for n in path]
            ys = [inst.coords[n][1] for n in path]
            ax.plot(xs, ys, '-o', color=col, linewidth=1, markersize=4)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal', adjustable='box')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=7, loc='upper left')
    fig.suptitle(f"Route Comparison - {inst.name} ({MOEA_ALGORITHM})")
    fig.tight_layout(rect=[0,0,1,0.95])
    file_path = out_path / f"routes_comparison_{MOEA_ALGORITHM.lower()}_{inst.name}.png"
    fig.savefig(file_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)
    return file_path

__all__ = ["plot_pareto", "plot_routes", "plot_route_comparison"]
