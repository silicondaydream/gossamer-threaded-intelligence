#!/usr/bin/env python3
"""
Run a simple task allocation simulation demo.
"""
import numpy as np

from gossamer.algorithms.coordination.task_allocation import allocate_tasks


def main():
    # Simulation parameters
    n_agents = 10
    n_tasks = 7
    n_dims = 2
    # Generate random positions
    agents = np.random.rand(n_agents, n_dims) * 100.0
    tasks = np.random.rand(n_tasks, n_dims) * 100.0

    # Compute assignments
    assignments = allocate_tasks(agents, tasks)

    # Display results
    print("Agent -> Task assignments:")
    for i, t in enumerate(assignments):
        if t is not None:
            dist = np.linalg.norm(agents[i] - tasks[t])
            print(f"  Agent {i} -> Task {t} (distance {dist:.2f})")
        else:
            print(f"  Agent {i} -> None")

    # Summary
    assigned = [a for a in assignments if a is not None]
    print(f"\nTotal agents: {n_agents}, tasks: {n_tasks}, assigned: {len(assigned)}")


if __name__ == "__main__":
    main()