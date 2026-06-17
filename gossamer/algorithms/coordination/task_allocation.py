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
    # Pad with dummy objects so surplus agents have somewhere to go. Without
    # this, an over-subscribed auction (more agents than tasks) never
    # terminates: displaced agents keep re-bidding for the scarce tasks. Each
    # dummy carries a dominating constant cost, so real tasks are always
    # preferred and exactly the surplus agents settle on dummies (-> None).
    n_dummy = max(0, n_agents - n_tasks)
    if n_dummy > 0:
        dummy_cost = (float(cost.max()) if cost.size else 1.0) * 10.0 + 1.0
        cost = np.hstack([cost, np.full((n_agents, n_dummy), dummy_cost)])
    n_obj = cost.shape[1]
    # Initialize prices and assignments
    prices = np.zeros(n_obj, dtype=float)
    assignments = -np.ones(n_agents, dtype=int)
    unassigned = list(range(n_agents))
    # Auction loop. Dummy padding guarantees a feasible square assignment and
    # therefore termination; the iteration cap is a defensive backstop against
    # pathological price cycling.
    max_iters = 1000 * max(1, n_agents)
    iters = 0
    while unassigned and iters < max_iters:
        iters += 1
        i = unassigned.pop(0)
        # Compute utilities: negative cost minus price
        utilities = -cost[i] - prices
        j_best = int(np.argmax(utilities))
        u_best = utilities[j_best]
        # Compute second best utility (guard the single-object case)
        if utilities.size > 1:
            utilities[j_best] = -np.inf
            u_second = np.max(utilities)
        else:
            u_second = u_best
        bid = u_best - u_second + epsilon
        prices[j_best] += bid
        # Assign agent i to object j_best, freeing previous owner
        prev = np.where(assignments == j_best)[0]
        assignments[i] = j_best
        if prev.size > 0:
            prev_agent = int(prev[0])
            assignments[prev_agent] = -1
            unassigned.append(prev_agent)
    # Real tasks are indices [0, n_tasks); dummy objects map to None.
    return [int(a) if (0 <= a < n_tasks) else None for a in assignments]