from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv


@dataclass
class GAMetrics:
    """
    Container for storing GA performance metrics for one experiment run.
    """
    preset: str          # Which GA preset was used (fast, balanced, thorough)
    instance: str        # Instance name (e.g., small_01)
    n_customers: int     # Number of customers in the instance
    n_vehicles: int      # Number of vehicles available
    best_fitness: float  # Best fitness value achieved
    best_distance: float # Corresponding total distance
    runtime_s: float     # Runtime in seconds
    generations: int     # Number of generations executed
    pop_size: int        # Population size
    evaluations: int     # Total number of fitness evaluations
    evals_per_sec: float # Evaluations per second (efficiency measure)


def save_convergence_csv(path: Path, fitness_history: list[float]) -> None:
    """
    Save the fitness history (convergence curve) to a CSV file.
    Each row = generation number and best fitness at that generation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_fitness"])
        for g, val in enumerate(fitness_history, start=1):
            w.writerow([g, val])


def save_metrics_csv(path: Path, rows: Iterable[GAMetrics]) -> None:
    """
    Save a list of GA experiment results (GAMetrics) to a CSV file.
    Each row = summary metrics for one run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Write header
        w.writerow([
            "preset", "instance", "n_customers", "n_vehicles",
            "best_fitness", "best_distance", "runtime_s",
            "generations", "pop_size", "evaluations", "evals_per_sec"
        ])
        # Write rows for each GAMetrics entry
        for m in rows:
            w.writerow([
                m.preset, m.instance, m.n_customers, m.n_vehicles,
                m.best_fitness, m.best_distance, m.runtime_s,
                m.generations, m.pop_size, m.evaluations, m.evals_per_sec
            ])
