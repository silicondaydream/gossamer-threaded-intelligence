"""
Task allocation strategies for swarm intelligence.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def allocate_tasks(agent_positions, task_positions):
    """
    Allocate tasks to agents by solving the assignment problem (minimize total distance).

    Each agent is assigned at most one task, and each task is assigned to at most one agent.
    If there are more agents than tasks, some agents receive None.
    If there are more tasks than agents, only len(agent_positions) tasks are assigned.

    Parameters:
        agent_positions: array_like of shape (n_agents, n_dims)
        task_positions: array_like of shape (n_tasks, n_dims)

    Returns:
        assignments: list of length n_agents, where assignments[i] is the index of the task
                     assigned to agent i, or None if no task assigned.
    """
    agents = np.asarray(agent_positions, dtype=float)
    tasks = np.asarray(task_positions, dtype=float)
    if agents.ndim != 2 or tasks.ndim != 2:
        raise ValueError("agent_positions and task_positions must be 2D arrays")
    if agents.shape[1] != tasks.shape[1]:
        raise ValueError("agent_positions and task_positions must have same dimensionality")

    n_agents = agents.shape[0]
    n_tasks = tasks.shape[0]
    if n_agents == 0 or n_tasks == 0:
        return [None] * n_agents

    # Compute cost matrix (distance)
    cost = cdist(agents, tasks)
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build assignments list
    assignments = [None] * n_agents
    for ai, ti in zip(row_ind, col_ind):
        assignments[ai] = int(ti)
    return assignments
 
def auction_allocate_tasks(agent_positions, task_positions, epsilon=1e-3):
    """
    Market-based (auction) assignment: agents bid for tasks to minimize travel cost.

    Parameters:
        agent_positions: array_like, shape (n_agents, n_dims)
        task_positions: array_like, shape (n_tasks, n_dims)
        epsilon: float, minimal bid increment

    Returns:
        assignments: list of length n_agents, with task index or None
    """
    agents = np.asarray(agent_positions, dtype=float)
    tasks = np.asarray(task_positions, dtype=float)
    if agents.ndim != 2 or tasks.ndim != 2:
        raise ValueError("agent_positions and task_positions must be 2D arrays")
    if agents.shape[1] != tasks.shape[1]:
        raise ValueError("agent_positions and task_positions must have same dimensionality")
    n_agents = agents.shape[0]
    n_tasks = tasks.shape[0]
    if n_agents == 0:
        return []
    if n_tasks == 0:
        return [None] * n_agents
    # Compute cost (distance)
    cost = cdist(agents, tasks)
    # Initialize prices and assignments
    prices = np.zeros(n_tasks, dtype=float)
    assignments = -np.ones(n_agents, dtype=int)
    unassigned = list(range(n_agents))
    # Auction loop
    while unassigned:
        i = unassigned.pop(0)
        # Compute utilities: negative cost minus price
        utilities = -cost[i] - prices
        j_best = int(np.argmax(utilities))
        u_best = utilities[j_best]
        # Compute second best utility
        utilities[j_best] = -np.inf
        u_second = np.max(utilities)
        bid = u_best - u_second + epsilon
        prices[j_best] += bid
        # Assign agent i to task j_best, freeing previous owner
        prev = np.where(assignments == j_best)[0]
        assignments[i] = j_best
        if prev.size > 0:
            prev_agent = int(prev[0])
            assignments[prev_agent] = -1
            unassigned.append(prev_agent)
    # Convert to list with None
    return [int(a) if a >= 0 else None for a in assignments]