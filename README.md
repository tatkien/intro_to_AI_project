# AI Taxi Project

Interactive visualization of AI search algorithms on a custom Taxi environment built with Pygame.

## What This Project Does

- Simulates a grid-based taxi task with walls, mud tiles, and multiple passengers.
- Lets you run BFS, DFS, and A* from the same initial state.
- Animates each solution path step-by-step.
- Shows live metrics: expanded nodes, path length, path cost, execution time, and progress.

## Features

- Pygame UI with scene-based navigation (menu and taxi simulation)
- Three levels with increasing map size and task count
- Manual play controls (arrow keys + space)
- Search playback controls (run, pause/resume, reset)
- Terrain-aware cost model (mud is more expensive than road)

## Requirements

- Python 3.10+
- See dependencies in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Start the application:

```bash
python main.py
```

## Controls

### Menu Scene

- Mouse click: choose level or exit
- Mouse wheel: scroll info panel
- ESC: quit application

### Taxi Scene

- Mouse click: choose algorithm and control buttons
- Arrow keys: manual movement (when search playback is not running)
  - Down = South
  - Up = North
  - Right = East
  - Left = West
- Space: smart pickup/dropoff action in manual mode
- Mouse wheel: scroll right-side stats panel
- ESC: quit application

## Search Algorithms

Implemented in `src/search/algorithms.py`:

- BFS (Breadth-First Search)
  - Uses FIFO frontier
  - Complete for finite state space
  - Good baseline for shallow solutions
- DFS (Depth-First Search)
  - Uses LIFO frontier
  - Can be fast for lucky search order, not path-optimal
- A* (A-Star Search)
  - Uses `f(n) = g(n) + h(n)`
  - Heuristic comes from environment (`TaxiEnv.heuristic`)
  - Tracks best known path cost per state

## Environment Model

Main environment class: `TaxiEnv` in `src/scenes/taxi.py`.

### State Representation

```text
[taxi_row, taxi_col, passenger_states..., current_passenger]
```

- `passenger_state`: `0 = waiting`, `1 = in taxi`, `2 = delivered`
- `current_passenger`: `-1` if taxi is empty, otherwise passenger index

### Actions

- `0`: South
- `1`: North
- `2`: East
- `3`: West
- `4`: Pickup
- `5`: Dropoff

### Terrain Costs

Configured in `config.py`:

- Road (`EMPTY`): walkable, cost `1`
- Mud (`MUD`): walkable, cost `3`
- Wall (`WALL`): not walkable

Search uses `cost = -reward`, so lower movement reward corresponds to higher path cost.

### Levels

Configured inside `TaxiEnv._configure_level`:

- Level 1: `10x10`, 4 locations, 1 passenger task
- Level 2: `15x15`, 6 locations, 2 passenger tasks
- Level 3: `20x20`, 10 locations, 3 passenger tasks

## Project Structure

```text
intro_to_AI_project/
|-- main.py
|-- settings.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- src/
|   |-- scene_manager.py
|   |-- scenes/
|   |   |-- scene.py
|   |   |-- menu.py
|   |   |-- taxi.py
|   |   `-- element.py
|   `-- search/
|       |-- algorithms.py
|       `-- __init__.py
|-- images/
|   |-- logo.jpg
|   |-- road.jpg
|   |-- mud.jpg
|   |-- wall.png
|   |-- taxi.png
|   |-- passenger.png
|   `-- dest.png
```

## Notes About Extra Scripts

- `test.py`: standalone Gymnasium-based taxi environment prototype/manual test.
- `boxing_test.py`: unrelated Atari Boxing RAM experiment.

The main playable app is launched from `main.py`.

## Troubleshooting

- If images fail to load, verify files exist under `images/` and keep filenames unchanged.
- If installation fails, update pip first:

```bash
python -m pip install --upgrade pip
```

- On Windows, DPI scaling is handled in `main.py` via `SetProcessDPIAware()`.

## Learning Extensions

Good next improvements for this project:

1. Add UCS or Greedy Best-First for comparison.
2. Add seeded resets for reproducible algorithm benchmarks.
3. Export run metrics to CSV for offline analysis.
