from .evolve import evolve

if __name__ == "__main__":
    best = evolve("medium_1")
    print("\nBest final solution:")
    print(best)
    print(f"Fitness: {best.fitness():.4f}")
