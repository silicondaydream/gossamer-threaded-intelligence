import numpy as np
import pytest

from gossamer.algorithms.coordination.task_allocation import allocate_tasks


def test_equal_agents_and_tasks():
    agents = np.array([[0.0, 0.0], [10.0, 0.0]])
    tasks = np.array([[1.0, 0.0], [9.0, 0.0]])
    assignments = allocate_tasks(agents, tasks)
    assert assignments == [0, 1]


def test_more_agents_than_tasks():
    agents = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]])
    tasks = np.array([[1.0, 0.0], [9.0, 0.0]])
    assignments = allocate_tasks(agents, tasks)
    # Two tasks, three agents: two get tasks, one None
    assert assignments.count(None) == 1
    # Check specific optimal assignment
    assert assignments[0] == 0
    assert assignments[1] == 1


def test_more_tasks_than_agents():
    agents = np.array([[0.0, 0.0], [10.0, 0.0]])
    tasks = np.array([[1.0, 0.0], [9.0, 0.0], [5.0, 5.0]])
    assignments = allocate_tasks(agents, tasks)
    # Only two tasks assigned
    assert len(assignments) == 2
    # Optimal assignment: agents 0->task0, 1->task1
    assert assignments == [0, 1]


def test_empty_agents_or_tasks():
    # No agents
    assignments = allocate_tasks(np.empty((0, 2)), np.array([[1.0, 1.0]]))
    assert assignments == []
    # No tasks
    assignments = allocate_tasks(np.array([[0.0, 0.0]]), np.empty((0, 2)))
    assert assignments == [None]


def test_invalid_shapes():
    with pytest.raises(ValueError):
        allocate_tasks([1, 2, 3], [[1, 2]])
    with pytest.raises(ValueError):
        allocate_tasks(np.zeros((2, 2)), np.zeros((2, 3)))