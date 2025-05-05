# Gossamer Threaded Intelligence

> Gossamer Threaded Intelligence is a suite of proprietary algorithms for swarm intelligence, distributed decision-making, and robust autonomous agent coordination.

## Installation

```bash
git clone <repo-url>
cd gossamer-intelligence
pip install -r requirements.txt
pip install -e .
```

## Structure

- gossamer/: main package
-   communication/: communication protocols (BaseProtocol, BroadcastProtocol, GossipProtocol)
- examples/: demonstration scripts (run_flocking_demo.py, run_task_allocation_demo.py)
- tests/: unit tests

## Usage Example

```python
from gossamer.simulator import SwarmSimulator

def log(step, positions, velocities, metrics):
    print(
        f"Step {step}: cohesion={metrics['cohesion']:.3f}, "
        f"alignment={metrics['alignment']:.3f}, "
        f"separation={metrics['separation']:.3f}"
    )

sim = SwarmSimulator(n_agents=30, n_dims=2, dt=0.1)
sim.run(50, callback=log)
```

For full documentation, see the `docs/` directory:

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html source build/html
```

## Gossamer in the Arboria Research Workflow

Gossamer provides the modular framework for defining the "intelligence" of distributed agents simulated within the `Leviathan Engine`. Its primary roles are:

1.  **Algorithm Implementation:** Providing a structured way to implement swarm intelligence algorithms (e.g., flocking, task allocation, consensus, navigation, resilience strategies).
2.  **Agent Behavior Definition:** Defining how individual agents perceive their environment, make decisions, communicate, and act based on the implemented algorithms.
3.  **Interfacing with Simulation:** Defining a clear contract for how agent logic receives state information from and sends action commands back to a simulator like `Leviathan`.

### Connecting with Leviathan Engine

`Gossamer`-based agent logic is designed to be dynamically loaded and executed by the `Leviathan` simulator.

*   **Mechanism:** Researchers implement agent behaviors by subclassing a base agent class provided by `Gossamer` (e.g., `gossamer.agents.BaseAgent`). `Leviathan` uses its Python bindings to instantiate and call methods on these agent classes during the simulation loop.
*   **Required Implementation:** A typical `Gossamer` agent class callable by `Leviathan` must implement specific methods defined by the `BaseAgent` interface.

    ```python
    # Example Agent Implementation (e.g., my_agent_logic.py)
    import gossamer

    class MyResearchAgent(gossamer.agents.BaseAgent):
        def __init__(self, agent_id, config_params):
            """ Initialize agent state, potentially load parameters """
            super().__init__(agent_id)
            self.target_direction = None
            # Load algorithm parameters from config if needed
            self.flocking_weight = config_params.get("flocking_weight", 1.0)

        def step(self, simulation_state):
            """
            The core logic executed each simulation step.
            Receives state information from Leviathan.
            Must return an action dictionary.
            """
            # 1. Perceive: Access data provided by Leviathan
            my_pos = simulation_state.get_position()
            my_vel = simulation_state.get_velocity()
            my_energy = simulation_state.get_energy()
            visible_neighbors = simulation_state.get_neighbors(radius=50.0) # Example sensor query

            # 2. Decide: Apply Gossamer algorithms
            # (e.g., calculate alignment, cohesion, separation vectors from neighbors)
            flocking_vector = self._calculate_flocking(my_pos, my_vel, visible_neighbors)
            energy_management_decision = self._check_energy_levels(my_energy)

            # 3. Act: Determine action(s) to return to Leviathan
            action = {
                "move_vector": flocking_vector * self.flocking_weight,
                "set_state": energy_management_decision, # e.g., 'hibernate', 'active'
                "broadcast_message": None # Or some data object
            }
            return action

        def _calculate_flocking(self, pos, vel, neighbors):
            # --- Implementation of flocking logic ---
            # (Uses functions potentially from gossamer.algorithms.coordination)
            pass

        def _check_energy_levels(self, energy):
            # --- Implementation of energy management ---
            pass

    # --- Helper functions or other classes ---
    ```

*   **Simulation State API:** The `simulation_state` object passed to the `step` method provides access to information simulated by `Leviathan`. The exact structure and available methods need to be kept consistent between `Leviathan`'s API definition and `Gossamer`'s agent implementations. It might include methods like:
    *   `get_position()`
    *   `get_velocity()`
    *   `get_energy()`
    *   `get_internal_state()`
    *   `get_neighbors(radius)` / `get_messages()`
    *   `get_environment_data(position)`
    *   `get_current_time()`
*   **Action Return Values:** The dictionary returned by the `step` method tells `Leviathan` what the agent wants to do (e.g., apply a movement force/vector, change its internal state, send a message).

### Typical Research Workflow using Gossamer

1.  **Design Algorithm:** Conceptualize the swarm behavior or algorithm needed for the research question.
2.  **Implement Agent:** Create a new Python module (e.g., `my_research_agent.py`) within a suitable project structure.
    *   Import necessary `Gossamer` components (`BaseAgent`, algorithm modules).
    *   Define a custom agent class inheriting from `BaseAgent`.
    *   Implement the `__init__` and `step` methods, utilizing the `simulation_state` object provided by `Leviathan` and returning valid action dictionaries.
    *   Leverage or develop reusable algorithm components within the `gossamer.algorithms` structure.
3.  **Unit Test (Recommended):** Write unit tests for the agent logic where possible, mocking the `simulation_state` object.
4.  **Integrate with Leviathan:** Configure a `Leviathan` simulation scenario (`config.yaml`) to use your new agent class by specifying the `gossamer_module` and `gossamer_class`.
5.  **Run & Iterate:** Execute the simulation via `Leviathan`. Analyze the results (potentially using `Maneuver.Map`) and iterate on the agent logic in `Gossamer` based on observed behavior and metrics.

*(Ensure the installed Gossamer library version is compatible with the Leviathan version being used, particularly regarding the simulation_state API and expected action formats.)*
